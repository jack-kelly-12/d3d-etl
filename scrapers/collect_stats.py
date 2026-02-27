import argparse
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import BASE
from .scraper_utils import HardBlockError, ScraperConfig, ScraperSession

HITTING_CATEGORY_ID_2026 = 15867
PITCHING_CATEGORY_ID_2026 = 15868


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_norm = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
        rename_map[col] = col_norm
    return df.rename(columns=rename_map)


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def clean_stat_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)

    if "ncaa_id" in df.columns:
        df["ncaa_id"] = pd.to_numeric(df["ncaa_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ncaa_id"])

    if "gdp" in df.columns:
        df = df.drop(columns=["gdp"])

    df = df.rename(
        columns={
            "slgpct": "slg_pct",
            "obpct": "ob_pct",
            "player": "player_name",
            "team": "team_name",
            "yr": "class",
        }
    )

    return df


def parse_table(page, table, year, school, conference, div, team_id) -> list[dict[str, Any]]:
    headers = [th.inner_text().strip() for th in table.query_selector_all("thead th")]
    headers = [re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_").lower() for h in headers]

    rows: list[dict[str, Any]] = []

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

        row_dict.update(
            {
                "division": div,
                "year": year,
                "team_name": school,
                "conference": conference,
                "team_id": team_id,
                "ncaa_id": ncaa_id,
                "player_url": player_url,
            }
        )
        rows.append(row_dict)

    return rows


def is_hard_block(status: int) -> bool:
    return status in (403, 429, 430)


def short_break_every(n: int, idx: int) -> None:
    if n > 0 and idx > 0 and idx % n == 0:
        time.sleep(60)


def human_pause(min_s: float = 2.5, max_s: float = 7.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def hard_block_cooldown(min_minutes: int = 30, max_minutes: int = 90) -> None:
    cooldown_seconds = int(random.uniform(min_minutes * 60, max_minutes * 60))
    print(f"[cooldown] hard block detected, sleeping {cooldown_seconds}s before exit")
    time.sleep(cooldown_seconds)


def played_team_ids_path(played_team_ids_dir: Path, div: int, year: int) -> Path:
    return played_team_ids_dir / "_tmp" / f"d{div}_teams_played_{year}.csv"


def season_to_date_stats_url(
    team_id: int, year: int, year_stat_category_id: int | None = None
) -> str:
    url = f"{BASE}/teams/{team_id}/season_to_date_stats?year={year}"
    if year_stat_category_id is not None:
        return f"{url}&year_stat_category_id={year_stat_category_id}"
    return url


def load_played_team_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "team_id" not in df.columns:
        return set()
    s = pd.to_numeric(df["team_id"], errors="coerce").dropna().astype(int)
    return set(s.tolist())


def load_processed_team_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["team_id"])
    except Exception:
        return set()
    s = pd.to_numeric(df["team_id"], errors="coerce").dropna().astype(int)
    return set(s.tolist())


def scrape_stats(
    team_ids_file: str,
    year: int,
    divisions: list[int],
    outdir: str,
    played_team_ids_dir: str | None = None,
    run_all: bool = False,
    run_remaining: bool = False,
    batch_size: int = 10,
    base_delay: float = 10.0,
    jitter_pct: float | None = None,
    random_delay_min: float = 1.0,
    random_delay_max: float = 30.0,
    daily_budget: int = 20000,
    batch_cooldown_s: int = 90,
):
    teams = pd.read_csv(team_ids_file)

    teams["year"] = pd.to_numeric(teams["year"], errors="coerce").astype("Int64")
    teams["division"] = pd.to_numeric(teams["division"], errors="coerce").astype("Int64")
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce").astype("Int64")
    teams = teams.dropna(subset=["year", "division", "team_id"])

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    played_ids_dir_path = Path(played_team_ids_dir) if played_team_ids_dir else None

    config = ScraperConfig(
        base_delay=base_delay,
        jitter_pct=0.0 if jitter_pct is None else jitter_pct,
        min_request_delay=random_delay_min,
        max_request_delay=random_delay_max,
        block_resources=False,
        daily_request_budget=daily_budget,
    )

    division_jobs = []
    for div in divisions:
        teams_div = teams.query("year == @year and division == @div").copy()
        if teams_div.empty:
            continue

        teams_div["team_id"] = teams_div["team_id"].astype(int)
        total_teams = len(teams_div)
        batting_out = outdir_path / f"d{div}_batting_{year}.csv"
        pitching_out = outdir_path / f"d{div}_pitching_{year}.csv"
        played_ids_file = None
        if played_ids_dir_path is not None and not run_all:
            played_ids_file = played_team_ids_path(played_ids_dir_path, div, year)
            played_ids = load_played_team_ids(played_ids_file)
            if played_ids:
                teams_div = teams_div[teams_div["team_id"].isin(played_ids)].copy()
            else:
                teams_div = teams_div.iloc[0:0].copy()

        done_batting_ids = load_processed_team_ids(batting_out) if run_remaining else set()
        done_pitching_ids = load_processed_team_ids(pitching_out) if run_remaining else set()
        done_ids = done_batting_ids & done_pitching_ids
        scoped = (
            teams_div[~teams_div["team_id"].isin(done_ids)].copy() if run_remaining else teams_div
        )
        scoped = scoped.sample(frac=1.0, random_state=None).reset_index(drop=True)

        print(
            f"\n=== d{div} {year} team stats — total {total_teams} | done {len(done_ids)} | remaining {len(scoped)} ==="
        )

        division_jobs.append(
            {
                "div": div,
                "teams_div": scoped,
                "batting_out": batting_out,
                "pitching_out": pitching_out,
                "done_batting_ids": done_batting_ids,
                "done_pitching_ids": done_pitching_ids,
                "played_ids_file": played_ids_file,
            }
        )

    if not any(len(job["teams_div"]) > 0 for job in division_jobs):
        return

    with ScraperSession(config) as session:
        try:
            for job in division_jobs:
                div = job["div"]
                teams_div = job["teams_div"]
                batting_out = job["batting_out"]
                pitching_out = job["pitching_out"]
                done_batting_ids = job["done_batting_ids"]
                done_pitching_ids = job["done_pitching_ids"]
                played_ids_file = job["played_ids_file"]
                total_teams = len(teams_div)
                exhausted_budget = False
                hard_blocked = False
                teams_processed = 0
                next_long_break_at = random.randint(15, 25)
                next_page_reset_at = random.randint(20, 35)

                if total_teams == 0:
                    if played_ids_file and played_ids_file.exists():
                        played_ids_file.unlink()
                    continue

                print(f"    (budget remaining: {session.requests_remaining} requests)")

                for start in range(0, total_teams, batch_size):
                    if session.requests_remaining <= 0:
                        print("[budget] daily request budget exhausted, stopping")
                        exhausted_budget = True
                        break

                    end = min(start + batch_size, total_teams)
                    batch = teams_div.iloc[start:end]

                    print(
                        f"\n[batch] teams {start + 1}-{end} (budget: {session.requests_remaining})"
                    )
                    batch_batting_frames: list[pd.DataFrame] = []
                    batch_pitching_frames: list[pd.DataFrame] = []

                    for row in batch.itertuples(index=False):
                        if session.requests_remaining <= 0:
                            exhausted_budget = True
                            break

                        team_id = int(row.team_id)
                        team_name = getattr(row, "team_name", "")
                        conference = getattr(row, "conference", "")
                        batting_done = run_remaining and team_id in done_batting_ids
                        pitching_done = run_remaining and team_id in done_pitching_ids

                        print(f"[team] {team_name} ({team_id})")
                        teams_processed += 1

                        if not batting_done:
                            batting_category = HITTING_CATEGORY_ID_2026 if year == 2026 else None
                            url = season_to_date_stats_url(team_id, year, batting_category)
                            html, status = session.fetch(
                                url, wait_selector="#stat_grid", wait_timeout=15000
                            )

                            if is_hard_block(status):
                                print(f"[STOP] got hard block status {status}. Stopping for safety.")
                                hard_block_cooldown()
                                hard_blocked = True
                                break

                            if not html or status >= 400:
                                print(f"FAILED batting: HTTP {status}")
                                continue

                            human_pause(0.8, 2.2)
                            table = session.page.query_selector("#stat_grid")
                            if not table:
                                print("FAILED batting: no #stat_grid")
                                continue

                            rows_bat = parse_table(
                                session.page, table, year, team_name, conference, div, team_id
                            )
                            if rows_bat:
                                df_bat = clean_stat_df(pd.DataFrame(rows_bat))
                                if not df_bat.empty:
                                    batch_batting_frames.append(df_bat)
                                    print(f"OK batting -> queued {len(df_bat)} rows")
                            human_pause()

                        if not pitching_done:
                            if year == 2026:
                                pitch_url = season_to_date_stats_url(
                                    team_id, year, PITCHING_CATEGORY_ID_2026
                                )
                            else:
                                pitch_link = None
                                if not batting_done:
                                    pitch_link = session.page.query_selector(
                                        "a.nav-link:has-text('Pitching')"
                                    )
                                if batting_done or not pitch_link:
                                    base_url = season_to_date_stats_url(team_id, year)
                                    html_base, status_base = session.fetch(
                                        base_url,
                                        wait_selector="a.nav-link",
                                        wait_timeout=15000,
                                    )
                                    if is_hard_block(status_base):
                                        print(
                                            f"[STOP] got hard block status {status_base}. Stopping for safety."
                                        )
                                        hard_block_cooldown()
                                        hard_blocked = True
                                        break
                                    if not html_base or status_base >= 400:
                                        print(f"FAILED pitching: HTTP {status_base}")
                                        continue
                                    pitch_link = session.page.query_selector(
                                        "a.nav-link:has-text('Pitching')"
                                    )
                                if not pitch_link:
                                    print("FAILED pitching: no Pitching tab")
                                    continue

                                href = pitch_link.get_attribute("href") or ""
                                if not href:
                                    print("FAILED pitching: tab missing href")
                                    continue

                                pitch_url = BASE + href
                            html2, status2 = session.fetch(
                                pitch_url, wait_selector="#stat_grid", wait_timeout=15000
                            )

                            if is_hard_block(status2):
                                print(f"[STOP] got hard block status {status2}. Stopping for safety.")
                                hard_block_cooldown()
                                hard_blocked = True
                                break

                            if not html2 or status2 >= 400:
                                print(f"FAILED pitching: HTTP {status2}")
                                continue

                            human_pause(0.8, 2.2)
                            table2 = session.page.query_selector("#stat_grid")
                            if not table2:
                                print("FAILED pitching: no #stat_grid")
                                continue

                            rows_pit = parse_table(
                                session.page, table2, year, team_name, conference, div, team_id
                            )
                            if rows_pit:
                                df_pit = clean_stat_df(pd.DataFrame(rows_pit))
                                if not df_pit.empty:
                                    batch_pitching_frames.append(df_pit)
                                    print(f"OK pitching -> queued {len(df_pit)} rows")
                            human_pause()

                        if teams_processed >= next_long_break_at:
                            coffee_break = random.uniform(60, 180)
                            print(f"[break] sleeping {coffee_break:.1f}s after {teams_processed} teams")
                            time.sleep(coffee_break)
                            next_long_break_at += random.randint(15, 25)

                        if teams_processed >= next_page_reset_at:
                            print(f"[rotate] resetting browser page after {teams_processed} teams")
                            session.reset_page(clear_cookies=False)
                            next_page_reset_at += random.randint(20, 35)

                        human_pause(3.0, 10.0)

                    if batch_batting_frames:
                        append_csv(batting_out, pd.concat(batch_batting_frames, ignore_index=True))
                        print(
                            f"[batch-write] batting -> appended {sum(len(x) for x in batch_batting_frames)} rows"
                        )
                    if batch_pitching_frames:
                        append_csv(
                            pitching_out, pd.concat(batch_pitching_frames, ignore_index=True)
                        )
                        print(
                            f"[batch-write] pitching -> appended {sum(len(x) for x in batch_pitching_frames)} rows"
                        )
                    if hard_blocked:
                        return

                    if end < total_teams:
                        cooldown = int(batch_cooldown_s)
                        print(f"\n[cooldown] sleeping {cooldown}s before next batch")
                        time.sleep(cooldown)

                print(f"\n[done] division d{div} {year}")
                print(f"  batting file:  {batting_out}")
                print(f"  pitching file: {pitching_out}")
                if played_ids_file and played_ids_file.exists() and not exhausted_budget:
                    played_ids_file.unlink()
                    print(f"  removed filter file: {played_ids_file}")
        except HardBlockError as exc:
            print(str(exc))
            print("[STOP] hard block detected. Stopping for safety.")
            return


if __name__ == "__main__":
    print("[start] scrapers.collect_stats", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--team_ids_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv"
    )
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/stats")
    parser.add_argument("--played_team_ids_dir", default=None)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--run_remaining", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--jitter_pct", type=float, default=None)
    parser.add_argument("--random_delay_min", type=float, default=1.0)
    parser.add_argument("--random_delay_max", type=float, default=30.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_stats(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        played_team_ids_dir=args.played_team_ids_dir,
        run_all=args.run_all,
        run_remaining=args.run_remaining,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        jitter_pct=args.jitter_pct,
        random_delay_min=args.random_delay_min,
        random_delay_max=args.random_delay_max,
        daily_budget=args.daily_budget,
        batch_cooldown_s=args.batch_cooldown_s,
    )
