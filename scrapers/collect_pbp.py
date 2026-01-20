import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import ScraperConfig, ScraperSession


def year_to_season(year: int) -> str:
    return f"{year-1}-{str(year)[-2:]}"


def load_existing(outdir, div, year):
    fpath = Path(outdir) / f"d{div}_pbp_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        if "contest_id" in df.columns:
            return df, set(df["contest_id"].unique())
    return pd.DataFrame(), set()


def get_schedules(indir, div, year):
    fpath = Path(indir) / f"d{div}_schedules_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={'contest_id': 'Int64'})
        if "contest_id" in df.columns:
            df = df.drop_duplicates(subset=["contest_id"])
            return df
    return pd.DataFrame()


def scrape_game_pbp(session: ScraperSession, contest_id, div, year):
    url = f"{BASE}/contests/{contest_id}/play_by_play"

    html, status = session.fetch(url, wait_selector="table", wait_timeout=10000)
    if not html or status >= 400:
        print(f"  failed game {contest_id}: HTTP {status}")
        return pd.DataFrame()

    page = session.page

    away_team_id, home_team_id = None, None
    team_links = page.query_selector_all("a[href*='/teams/']")
    team_ids = []
    for link in team_links:
        href = link.get_attribute("href")
        if "/teams/" in href:
            m = re.search(r"/teams/(\d+)", href)
            if m:
                tid = int(m.group(1))
                if tid not in team_ids:
                    team_ids.append(tid)
            if len(team_ids) >= 2:
                break
    if len(team_ids) >= 2:
        away_team_id = team_ids[0]
        home_team_id = team_ids[1]

    tables = page.query_selector_all("table")
    if len(tables) < 4:
        return pd.DataFrame()

    inning_tables = tables[3:]
    frames = []
    for i, tbl in enumerate(inning_tables, start=1):
        html_content = tbl.inner_html()
        df = pd.read_html(StringIO(f"<table>{html_content}</table>"))[0]
        if df.shape[1] >= 3:
            df = df.rename(columns={df.columns[0]: "away_des", df.columns[2]: "home_des"})
            df["inning"] = i
            df["contest_id"] = contest_id
            df["division"] = div
            df["year"] = year
            df["away_team_id"] = away_team_id
            df["home_team_id"] = home_team_id
            frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["away_text"] = out["away_des"].fillna("").str.strip()
        out["home_text"] = out["home_des"].fillna("").str.strip()
        out["away_score"] = out["Score"].astype(str).str.extract(r"^(\d+)-")[0]
        out["home_score"] = out["Score"].astype(str).str.extract(r"-(\d+)$")[0]

        out = out[~((out.away_text != "") & (out.home_text != ""))]
        out["away_team_id"] = out["away_team_id"].astype("Int64")
        out["home_team_id"] = out["home_team_id"].astype("Int64")

        return out[[
            "division", "year", "contest_id", "inning",
            "away_team_id", "home_team_id",
            "away_text", "home_text", "away_score", "home_score"
        ]]

    return pd.DataFrame()


def scrape_pbp(
    indir,
    outdir,
    year,
    divisions,
    missing_only=False,
    batch_size=50,
    headless=True,
    base_delay=2.0,
    daily_budget=20000
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        headless=headless,
        block_resources=False,
        daily_request_budget=daily_budget,
    )

    with ScraperSession(config) as session:
        for div in divisions:
            sched = get_schedules(indir, div, year)
            if sched.empty:
                print(f"no schedule for d{div} {year}")
                continue

            existing, done_ids = load_existing(outdir, div, year)

            sched = sched[sched["contest_id"].notna()]

            if missing_only:
                to_scrape = sched[~sched["contest_id"].isin(done_ids)]
                skipped = len(sched) - len(to_scrape)
                if skipped > 0:
                    print(f"d{div} {year}: skipping {skipped} already-scraped games")
                if to_scrape.empty:
                    print(f"d{div} {year}: all {len(sched)} games already scraped")
                    continue
            else:
                to_scrape = sched
                existing = pd.DataFrame()

            rows = []
            total_games = len(to_scrape)
            print(f"\n=== d{div} {year} pbp — {total_games} games to scrape ===")
            print(f"    (budget remaining: {session.requests_remaining} requests)")

            games_scraped = 0
            fpath = outdir / f"d{div}_pbp_{year}.csv"

            for start in range(0, total_games, batch_size):
                if session.requests_remaining <= 0:
                    print("[budget] daily request budget exhausted, stopping")
                    break

                end = min(start + batch_size, total_games)
                batch = to_scrape.iloc[start:end]

                for _, r in batch.iterrows():
                    if session.requests_remaining <= 0:
                        break

                    gid = r["contest_id"]
                    df = scrape_game_pbp(session, gid, div, year)
                    games_scraped += 1

                    if not df.empty:
                        rows.append(df)
                        print(f"[{games_scraped}/{total_games}] game {gid}: {len(df)} rows")
                    else:
                        print(f"[{games_scraped}/{total_games}] game {gid}: no data")

                print(f"batch {start+1}-{end} done (budget: {session.requests_remaining})")

                if rows:
                    new_df = pd.concat(rows, ignore_index=True)
                    if not existing.empty:
                        out = pd.concat([existing, new_df], ignore_index=True)
                    else:
                        out = new_df
                    out.to_csv(fpath, index=False)
                    print(f"  [checkpoint] saved {len(out)} rows")

            if rows:
                new_df = pd.concat(rows, ignore_index=True)
                if not existing.empty:
                    out = pd.concat([existing, new_df], ignore_index=True)
                else:
                    out = new_df
                out.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(out)} total rows, {len(new_df)} new)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--missing_only", action="store_true",
                        help="Only scrape games in schedule but not yet in PBP data")
    parser.add_argument("--indir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/pbp")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (may trigger bot detection)")
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_pbp(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions,
        missing_only=args.missing_only,
        batch_size=args.batch_size,
        headless=args.headless,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
