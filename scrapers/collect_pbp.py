import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import (
    HardBlockError,
    ScraperConfig,
    ScraperSession,
    cooldown_between_batches,
    get_scraper_logger,
    short_break_every,
)

logger = get_scraper_logger(__name__)


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
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64"})
        if "contest_id" in df.columns:
            df = df.drop_duplicates(subset=["contest_id"])
            df = df[df['game_url'].str.contains('box_score')]
            return df
    return pd.DataFrame()


def scrape_game_pbp(session: ScraperSession, contest_id, div, year):
    url = f"{BASE}/contests/{contest_id}/play_by_play"
    html, status = session.fetch(url, wait_selector="table", wait_timeout=10000)

    if not html or status >= 400:
        logger.info(f"  failed game {contest_id}: HTTP {status}")
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

        return out[
            [
                "division",
                "year",
                "contest_id",
                "inning",
                "away_team_id",
                "home_team_id",
                "away_text",
                "home_text",
                "away_score",
                "home_score",
            ]
        ]

    return pd.DataFrame()


def scrape_pbp(
    indir,
    outdir,
    year,
    divisions,
    missing_only=False,
    batch_size=50,
    base_delay=10.0,
    rest_every=12,
    batch_cooldown_s=90,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    division_jobs = []
    for div in divisions:
        sched = get_schedules(indir, div, year)
        if sched.empty:
            logger.info(f"no schedule for d{div} {year}")
            continue

        existing, done_ids = load_existing(outdir, div, year)
        sched = sched[sched["contest_id"].notna()]

        if missing_only:
            to_scrape = sched[~sched["contest_id"].isin(done_ids)]
            skipped = len(sched) - len(to_scrape)
            if skipped > 0:
                logger.info(f"d{div} {year}: skipping {skipped} already-scraped games")
            if to_scrape.empty:
                logger.info(f"d{div} {year}: all {len(sched)} games already scraped")
                continue
        else:
            to_scrape = sched
            existing = pd.DataFrame()

        total_games = len(to_scrape)
        logger.info(f"\n=== d{div} {year} pbp — {total_games} games to scrape ===")
        division_jobs.append({"div": div, "existing": existing, "to_scrape": to_scrape, "total_games": total_games})

    if not division_jobs:
        return

    config = ScraperConfig(base_delay=base_delay, block_resources=False)

    with ScraperSession(config) as session:
        try:
            for job in division_jobs:
                div = job["div"]
                existing = job["existing"]
                to_scrape = job["to_scrape"]
                total_games = job["total_games"]

                rows = []
                games_scraped = 0
                fpath = outdir / f"d{div}_pbp_{year}.csv"

                for start in range(0, total_games, batch_size):
                    end = min(start + batch_size, total_games)
                    batch = to_scrape.iloc[start:end]

                    for _, r in batch.iterrows():
                        gid = r["contest_id"]
                        short_break_every(rest_every, games_scraped + 1, sleep_s=60.0)
                        df = scrape_game_pbp(session, gid, div, year)
                        games_scraped += 1

                        if not df.empty:
                            rows.append(df)
                            logger.info(f"[{games_scraped}/{total_games}] game {gid}: {len(df)} rows")
                        else:
                            logger.info(f"[{games_scraped}/{total_games}] game {gid}: no data")

                    logger.info(f"batch {start + 1}-{end} done")
                    cooldown_between_batches(end, total_games, float(batch_cooldown_s))

                    if rows:
                        new_df = pd.concat(rows, ignore_index=True)
                        out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
                        out.to_csv(fpath, index=False)
                        logger.info(f"  [checkpoint] saved {len(out)} rows")

                if rows:
                    new_df = pd.concat(rows, ignore_index=True)
                    out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
                    out.to_csv(fpath, index=False)
                    logger.info(f"saved {fpath} ({len(out)} total rows, {len(new_df)} new)")
        except HardBlockError as exc:
            logger.error(str(exc))
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--missing_only", action="store_true", help="Only scrape games in schedule but not yet in PBP data")
    parser.add_argument("--indir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/pbp")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--rest_every", type=int, default=12)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_pbp(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions,
        missing_only=args.missing_only,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        rest_every=args.rest_every,
        batch_cooldown_s=args.batch_cooldown_s,
    )
