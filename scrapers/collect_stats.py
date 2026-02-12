import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .constants import BASE
from .scraper_utils import ScraperConfig, ScraperSession


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_norm = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
        rename_map[col] = col_norm
    return df.rename(columns=rename_map)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def clean_stat_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)

    # Require player id
    if "ncaa_id" in df.columns:
        df["ncaa_id"] = pd.to_numeric(df["ncaa_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ncaa_id"])

    # Drop junk columns if present
    if "gdp" in df.columns:
        df = df.drop(columns=["gdp"])

    # Normalize known column names
    df = df.rename(columns={
        "slgpct": "slg_pct",
        "obpct": "ob_pct",
        "player": "player_name",
        "team": "team_name",
        "yr": "class",
    })

    return df


def parse_table(page, table, year, school, conference, div, team_id) -> List[Dict[str, Any]]:
    headers = [th.inner_text().strip() for th in table.query_selector_all("thead th")]
    headers = [re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_").lower() for h in headers]

    rows: List[Dict[str, Any]] = []

    for tr in table.query_selector_all("tbody tr"):
        tds = tr.query_selector_all("td")
        if not tds:
            continue

        values = [td.inner_text().strip() for td in tds]
        row_dict = dict(zip(headers, values, strict=False))

        a = tr.query_selector("a[href*='/players/']")
        ncaa_id, player_url = None, None
        if a:
            href = a.get_attribute("href") or ""
            if "/players/" in href:
                try:
                    ncaa_id = int(href.split("/")[-1].split("?")[0])
                except ValueError:
                    ncaa_id = None
                player_url = BASE + href

        row_dict.update({
            "division": div,
            "year": year,
            "team_name": school,
            "conference": conference,
            "team_id": team_id,
            "ncaa_id": ncaa_id,
            "player_url": player_url,
        })
        rows.append(row_dict)

    return rows


@dataclass
class TeamProgress:
    team_id: int
    team_name: str
    conference: str
    batting_done: bool = False
    pitching_done: bool = False
    last_status_batting: int = 0
    last_status_pitching: int = 0
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "conference": self.conference,
            "batting_done": self.batting_done,
            "pitching_done": self.pitching_done,
            "last_status_batting": self.last_status_batting,
            "last_status_pitching": self.last_status_pitching,
            "updated_at": self.updated_at,
        }


def progress_path(outdir: Path, div: int, year: int) -> Path:
    return outdir / "_progress" / f"d{div}_{year}_team_progress.json"


def load_progress(outdir: Path, div: int, year: int) -> Dict[int, TeamProgress]:
    p = progress_path(outdir, div, year)
    raw = read_json(p)
    if not raw:
        return {}
    prog: Dict[int, TeamProgress] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
            prog[tid] = TeamProgress(
                team_id=tid,
                team_name=v.get("team_name", ""),
                conference=v.get("conference", ""),
                batting_done=bool(v.get("batting_done", False)),
                pitching_done=bool(v.get("pitching_done", False)),
                last_status_batting=int(v.get("last_status_batting", 0) or 0),
                last_status_pitching=int(v.get("last_status_pitching", 0) or 0),
                updated_at=v.get("updated_at", ""),
            )
        except Exception:
            continue
    return prog


def save_progress(outdir: Path, div: int, year: int, prog: Dict[int, TeamProgress]) -> None:
    payload = {str(tid): tp.to_dict() for tid, tp in prog.items()}
    safe_write_json(progress_path(outdir, div, year), payload)


def is_hard_block(status: int) -> bool:
    return status == 403


def short_break_every(n: int, idx: int) -> None:
    if n > 0 and idx > 0 and idx % n == 0:
        time.sleep(60)


def scrape_stats(
    team_ids_file: str,
    year: int,
    divisions: List[int],
    outdir: str,
    batch_size: int = 10,
    base_delay: float = 10.0,
    daily_budget: int = 20000,
    rest_every: int = 12,
):
    teams = pd.read_csv(team_ids_file)

    teams["year"] = pd.to_numeric(teams["year"], errors="coerce").astype("Int64")
    teams["division"] = pd.to_numeric(teams["division"], errors="coerce").astype("Int64")
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce").astype("Int64")
    teams = teams.dropna(subset=["year", "division", "team_id"])

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        block_resources=False,
        daily_request_budget=daily_budget,
    )

    with ScraperSession(config) as session:
        for div in divisions:
            teams_div = teams.query("year == @year and division == @div").copy()
            if teams_div.empty:
                continue

            teams_div["team_id"] = teams_div["team_id"].astype(int)
            total_teams = len(teams_div)

            print(f"\n=== d{div} {year} team stats — {total_teams} teams ===")
            print(f"    (budget remaining: {session.requests_remaining} requests)")

            batting_out = outdir_path / f"d{div}_batting_{year}.csv"
            pitching_out = outdir_path / f"d{div}_pitching_{year}.csv"

            prog = load_progress(outdir_path, div, year)

            for start in range(0, total_teams, batch_size):
                if session.requests_remaining <= 0:
                    print("[budget] daily request budget exhausted, stopping")
                    break

                end = min(start + batch_size, total_teams)
                batch = teams_div.iloc[start:end]

                print(f"\n[batch] teams {start+1}-{end} (budget: {session.requests_remaining})")

                for i, row in enumerate(batch.itertuples(index=False), start=1):
                    if session.requests_remaining <= 0:
                        break

                    team_id = int(row.team_id)
                    team_name = getattr(row, "team_name", "")
                    conference = getattr(row, "conference", "")

                    tp = prog.get(team_id) or TeamProgress(
                        team_id=team_id, team_name=team_name, conference=conference
                    )

                    if tp.batting_done and tp.pitching_done:
                        continue

                    print(f"[team] {team_name} ({team_id})")

                    short_break_every(rest_every, (start + i))

                    if not tp.batting_done:
                        url = f"{BASE}/teams/{team_id}/season_to_date_stats?year={year}"
                        html, status = session.fetch(url, wait_selector="#stat_grid tbody tr", wait_timeout=15000)
                        tp.last_status_batting = status
                        tp.updated_at = now_iso()

                        if is_hard_block(status):
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            print("[STOP] got 403 (hard block). Saved progress and stopping for safety.")
                            return

                        if not html or status >= 400:
                            print(f"FAILED batting: HTTP {status}")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        table = session.page.query_selector("#stat_grid")
                        if not table:
                            print("FAILED batting: no #stat_grid")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        rows_bat = parse_table(session.page, table, year, team_name, conference, div, team_id)
                        if rows_bat:
                            df_bat = clean_stat_df(pd.DataFrame(rows_bat))
                            if not df_bat.empty:
                                append_csv(batting_out, df_bat)
                                print(f"OK batting -> wrote {len(df_bat)} rows")

                        tp.batting_done = True
                        prog[team_id] = tp
                        save_progress(outdir_path, div, year, prog)

                    if not tp.pitching_done:
                        pitch_link = session.page.query_selector("a.nav-link:has-text('Pitching')")
                        if not pitch_link:
                            print("FAILED pitching: no Pitching tab")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        href = pitch_link.get_attribute("href") or ""
                        if not href:
                            print("FAILED pitching: tab missing href")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        pitch_url = BASE + href
                        html2, status2 = session.fetch(pitch_url, wait_selector="#stat_grid tbody tr", wait_timeout=15000)
                        tp.last_status_pitching = status2
                        tp.updated_at = now_iso()

                        if is_hard_block(status2):
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            print("[STOP] got 403 (hard block). Saved progress and stopping for safety.")
                            return

                        if not html2 or status2 >= 400:
                            print(f"FAILED pitching: HTTP {status2}")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        table2 = session.page.query_selector("#stat_grid")
                        if not table2:
                            print("FAILED pitching: no #stat_grid")
                            prog[team_id] = tp
                            save_progress(outdir_path, div, year, prog)
                            continue

                        rows_pit = parse_table(session.page, table2, year, team_name, conference, div, team_id)
                        if rows_pit:
                            df_pit = clean_stat_df(pd.DataFrame(rows_pit))
                            if not df_pit.empty:
                                append_csv(pitching_out, df_pit)
                                print(f"OK pitching -> wrote {len(df_pit)} rows")

                        tp.pitching_done = True
                        prog[team_id] = tp
                        save_progress(outdir_path, div, year, prog)

            print(f"\n[done] division d{div} {year}")
            print(f"  batting file:  {batting_out}")
            print(f"  pitching file: {pitching_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--team_ids_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/stats")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    parser.add_argument("--rest_every", type=int, default=12, help="Rest 60s every N teams")
    args = parser.parse_args()

    scrape_stats(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
        rest_every=args.rest_every,
    )
