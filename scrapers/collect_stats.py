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

    return df.rename(
        columns={
            "slgpct": "slg_pct",
            "obpct": "ob_pct",
            "player": "player_name",
            "team": "team_name",
            "yr": "class",
        }
    )


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


def human_pause(min_s: float = 1.0, max_s: float = 3.0) -> None:
    time.sleep(random.uniform(min_s, max_s))


def warm_session(session: ScraperSession) -> None:
    print("[warmup] visiting stats.ncaa.org")
    try:
        session.page.goto(BASE, timeout=20000, wait_until="domcontentloaded")
        time.sleep(random.uniform(3.0, 5.0))
    except Exception:
        pass


def played_team_ids_path(played_team_ids_dir: Path, div: int, year: int) -> Path:
    return played_team_ids_dir / "_tmp" / f"d{div}_teams_played_{year}.csv"


def season_to_date_stats_url(
    team_id: int, year: int, year_stat_category_id: int | None = None
) -> str:
    url = f"{BASE}/teams/{team_id}/season_to_date_stats?year={year}"
    if year_stat_category_id is not None:
        return f"{url}&year_stat_category_id={year_stat_category_id}"
    return url


def load_team_ids_from_csv(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if "team_id" not in df.columns:
        return set()
    return set(pd.to_numeric(df["team_id"], errors="coerce").dropna().astype(int).tolist())


def _scrape_batting(
    session: ScraperSession,
    team_id: int,
    year: int,
    team_name: str,
    conference: str,
    div: int,
) -> pd.DataFrame | None:
    category = HITTING_CATEGORY_ID_2026 if year == 2026 else None
    url = season_to_date_stats_url(team_id, year, category)
    html, status = session.fetch(url, wait_selector="#stat_grid", wait_timeout=15000)

    if not html or status >= 400:
        print(f"FAILED batting: HTTP {status}")
        return None

    human_pause(0.3, 0.8)

    table = session.page.query_selector("#stat_grid")
    if not table:
        print("FAILED batting: no #stat_grid")
        return None

    rows = parse_table(session.page, table, year, team_name, conference, div, team_id)
    if not rows:
        return None
    return clean_stat_df(pd.DataFrame(rows))


def _scrape_pitching(
    session: ScraperSession,
    team_id: int,
    year: int,
    team_name: str,
    conference: str,
    div: int,
    batting_was_skipped: bool,
) -> pd.DataFrame | None:
    if year == 2026:
        pitch_url = season_to_date_stats_url(team_id, year, PITCHING_CATEGORY_ID_2026)
    else:
        pitch_link = None
        if not batting_was_skipped:
            pitch_link = session.page.query_selector("a.nav-link:has-text('Pitching')")

        if batting_was_skipped or not pitch_link:
            base_url = season_to_date_stats_url(team_id, year)
            html_base, status_base = session.fetch(
                base_url, wait_selector="a.nav-link", wait_timeout=15000
            )
            if not html_base or status_base >= 400:
                print(f"FAILED pitching: HTTP {status_base}")
                return None
            pitch_link = session.page.query_selector("a.nav-link:has-text('Pitching')")

        if not pitch_link:
            print("FAILED pitching: no Pitching tab")
            return None

        href = pitch_link.get_attribute("href") or ""
        if not href:
            print("FAILED pitching: tab missing href")
            return None

        pitch_url = BASE + href

    html, status = session.fetch(pitch_url, wait_selector="#stat_grid", wait_timeout=15000)

    if not html or status >= 400:
        print(f"FAILED pitching: HTTP {status}")
        return None

    human_pause(0.3, 0.8)

    table = session.page.query_selector("#stat_grid")
    if not table:
        print("FAILED pitching: no #stat_grid")
        return None

    rows = parse_table(session.page, table, year, team_name, conference, div, team_id)
    if not rows:
        return None
    return clean_stat_df(pd.DataFrame(rows))


def _process_division(
    session: ScraperSession,
    job: dict,
    year: int,
    batch_size: int,
    batch_cooldown_s: int,
    run_remaining: bool,
) -> None:
    div = job["div"]
    teams_div = job["teams_div"]
    batting_out = job["batting_out"]
    pitching_out = job["pitching_out"]
    done_batting = job["done_batting"]
    done_pitching = job["done_pitching"]
    played_ids_file = job["played_ids_file"]
    total = len(teams_div)

    if total == 0:
        if played_ids_file and played_ids_file.exists():
            played_ids_file.unlink()
        return

    print(f"    (budget remaining: {session.requests_remaining} requests)")

    teams_processed = 0
    next_long_break = random.randint(20, 30)
    next_page_reset = random.randint(30, 50)
    budget_exhausted = False

    for start in range(0, total, batch_size):
        if session.requests_remaining <= 0:
            print("[budget] daily request budget exhausted, stopping")
            budget_exhausted = True
            break

        end = min(start + batch_size, total)
        batch = teams_div.iloc[start:end]
        print(f"\n[batch] teams {start + 1}-{end} (budget: {session.requests_remaining})")

        batch_batting: list[pd.DataFrame] = []
        batch_pitching: list[pd.DataFrame] = []

        for row in batch.itertuples(index=False):
            if session.requests_remaining <= 0:
                budget_exhausted = True
                break

            team_id = int(row.team_id)
            team_name = getattr(row, "team_name", "")
            conference = getattr(row, "conference", "")
            skip_batting = run_remaining and team_id in done_batting
            skip_pitching = run_remaining and team_id in done_pitching

            print(f"[team] {team_name} ({team_id})")
            teams_processed += 1

            if not skip_batting:
                df_bat = _scrape_batting(session, team_id, year, team_name, conference, div)
                if df_bat is not None and not df_bat.empty:
                    batch_batting.append(df_bat)
                    print(f"OK batting -> queued {len(df_bat)} rows")
                human_pause(1.0, 2.5)

            if not skip_pitching:
                df_pit = _scrape_pitching(
                    session, team_id, year, team_name, conference, div, skip_batting
                )
                if df_pit is not None and not df_pit.empty:
                    batch_pitching.append(df_pit)
                    print(f"OK pitching -> queued {len(df_pit)} rows")
                human_pause(1.0, 2.5)

            if teams_processed >= next_long_break:
                coffee = random.uniform(30, 90)
                print(f"[break] sleeping {coffee:.1f}s after {teams_processed} teams")
                time.sleep(coffee)
                warm_session(session)
                next_long_break += random.randint(20, 30)

            if teams_processed >= next_page_reset:
                print(f"[rotate] resetting browser page after {teams_processed} teams")
                session.reset_page(clear_cookies=False)
                warm_session(session)
                next_page_reset += random.randint(30, 50)

            human_pause(1.5, 4.0)

        if batch_batting:
            append_csv(batting_out, pd.concat(batch_batting, ignore_index=True))
            print(f"[batch-write] batting -> appended {sum(len(x) for x in batch_batting)} rows")
        if batch_pitching:
            append_csv(pitching_out, pd.concat(batch_pitching, ignore_index=True))
            print(
                f"[batch-write] pitching -> appended {sum(len(x) for x in batch_pitching)} rows"
            )

        if end < total:
            cooldown = int(batch_cooldown_s * random.uniform(0.6, 1.0))
            print(f"\n[cooldown] sleeping {cooldown}s before next batch")
            time.sleep(cooldown)

    print(f"\n[done] division d{div} {year}")
    print(f"  batting file:  {batting_out}")
    print(f"  pitching file: {pitching_out}")
    if played_ids_file and played_ids_file.exists() and not budget_exhausted:
        played_ids_file.unlink()
        print(f"  removed filter file: {played_ids_file}")


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
    random_delay_min: float = 1.0,
    random_delay_max: float = 30.0,
    daily_budget: int = 20000,
    batch_cooldown_s: int = 90,
):
    teams = pd.read_csv(team_ids_file)
    for col in ("year", "division", "team_id"):
        teams[col] = pd.to_numeric(teams[col], errors="coerce").astype("Int64")
    teams = teams.dropna(subset=["year", "division", "team_id"])

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    played_ids_dir_path = Path(played_team_ids_dir) if played_team_ids_dir else None

    config = ScraperConfig(
        base_delay=base_delay,
        min_request_delay=random_delay_min,
        max_request_delay=random_delay_max,
        block_resources=False,
        daily_request_budget=daily_budget,
    )

    jobs = []
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
            played_ids = load_team_ids_from_csv(played_ids_file)
            if played_ids:
                teams_div = teams_div[teams_div["team_id"].isin(played_ids)].copy()
            else:
                teams_div = teams_div.iloc[0:0].copy()

        done_batting = load_team_ids_from_csv(batting_out) if run_remaining else set()
        done_pitching = load_team_ids_from_csv(pitching_out) if run_remaining else set()
        done = done_batting & done_pitching
        scoped = teams_div[~teams_div["team_id"].isin(done)].copy() if run_remaining else teams_div
        scoped = scoped.sample(frac=1.0, random_state=None).reset_index(drop=True)

        print(
            f"\n=== d{div} {year} team stats — "
            f"total {total_teams} | done {len(done)} | remaining {len(scoped)} ==="
        )

        jobs.append({
            "div": div,
            "teams_div": scoped,
            "batting_out": batting_out,
            "pitching_out": pitching_out,
            "done_batting": done_batting,
            "done_pitching": done_pitching,
            "played_ids_file": played_ids_file,
        })

    if not any(len(j["teams_div"]) > 0 for j in jobs):
        return

    with ScraperSession(config) as session:
        try:
            warm_session(session)
            for job in jobs:
                _process_division(session, job, year, batch_size, batch_cooldown_s, run_remaining)
        except HardBlockError as exc:
            print(str(exc))
            print("[STOP] hard block detected. Stopping for safety.")


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
    parser.add_argument("--random_delay_min", type=float, default=1.0)
    parser.add_argument("--random_delay_max", type=float, default=15.0)
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
        random_delay_min=args.random_delay_min,
        random_delay_max=args.random_delay_max,
        daily_budget=args.daily_budget,
        batch_cooldown_s=args.batch_cooldown_s,
    )
