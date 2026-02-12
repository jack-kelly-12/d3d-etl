import argparse
import re
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


def scrape_schedules(
    team_ids_file,
    year,
    divisions,
    outdir,
    batch_size=10,
    base_delay=10.0,
    rest_every=12,
    batch_cooldown_s=90,
):
    teams = pd.read_csv(team_ids_file)

    teams["year"] = pd.to_numeric(teams["year"], errors="coerce").astype("Int64")
    teams["division"] = pd.to_numeric(teams["division"], errors="coerce").astype("Int64")
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce").astype("Int64")

    teams = teams.dropna(subset=["year", "division", "team_id"])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        block_resources=True,
    )

    with ScraperSession(config) as session:
        for div in divisions:
            teams_div = teams.query("year == @year and division == @div").copy()
            if teams_div.empty:
                continue

            teams_div["team_id"] = teams_div["team_id"].astype(int)
            total_teams = len(teams_div)
            logger.info(f"\n=== {year} D{div} schedules — {total_teams} teams ===")
            logger.info(f"    (budget remaining: {session.requests_remaining} requests)")

            rows = []
            try:
                for start in range(0, total_teams, batch_size):
                    if session.requests_remaining <= 0:
                        logger.info("[budget] daily request budget exhausted, stopping")
                        break

                    end = min(start + batch_size, total_teams)
                    batch = teams_div.iloc[start:end]

                    for i, row in enumerate(batch.itertuples(index=False), start=1):
                        if session.requests_remaining <= 0:
                            break

                        short_break_every(rest_every, start + i, sleep_s=60.0)
                        team_id = row.team_id
                        school = row.team_name
                        conference = row.conference
                        url = f"{BASE}/teams/{team_id}"

                        html, status = session.fetch(
                            url,
                            wait_selector="div.card-header:has-text('Schedule/Results')",
                            wait_timeout=10000
                        )

                        if not html or status >= 400:
                            logger.info(f"failed {school} ({team_id}): HTTP {status}")
                            continue

                        trs = session.page.query_selector_all("div.card-body table tbody tr")

                        for tr in trs:
                            tds = tr.query_selector_all("td")
                            if not tds or len(tds) < 3:
                                continue

                            date = tds[0].inner_text().strip()[:10]

                            opponent_raw = tds[1].inner_text().strip()
                            opponent_link = tds[1].query_selector("a[href*='/teams/']")
                            opponent_team_id = None
                            opponent_name = opponent_raw
                            away = opponent_raw.startswith("@")

                            if opponent_link:
                                href = opponent_link.get_attribute("href")
                                m = re.search(r"/teams/(\d+)", href)
                                if m:
                                    opponent_team_id = int(m.group(1))
                                opponent_name = opponent_link.inner_text().strip()

                            result_raw = tds[2].inner_text().strip()
                            game_result = None
                            team_score, opp_score = None, None
                            if result_raw:
                                game_result = result_raw.split()[0]
                                score_match = re.search(r"(\d+)-(\d+)", result_raw)
                                if score_match:
                                    team_score, opp_score = map(int, score_match.groups())

                            game_link = tds[2].query_selector("a[href*='/contests/']")
                            game_url, contest_id = None, None
                            if game_link:
                                href = game_link.get_attribute("href")
                                game_url = BASE + href
                                m = re.search(r"/contests/(\d+)", href)
                                if m:
                                    contest_id = int(m.group(1))

                            rows.append({
                                "year": year,
                                "division": div,
                                "team_id": team_id,
                                "team_name": school,
                                "conference": conference,
                                "date": date,
                                "opponent": opponent_name,
                                "opponent_team_id": opponent_team_id,
                                "away": away,
                                "game_result": game_result,
                                "team_score": team_score,
                                "opponent_score": opp_score,
                                "game_url": game_url,
                                "contest_id": contest_id
                            })

                        logger.info(f"success {school} ({team_id})")

                    logger.info(f"batch {start+1}-{end} done (budget: {session.requests_remaining})")
                    cooldown_between_batches(end, total_teams, float(batch_cooldown_s))
            except HardBlockError as exc:
                logger.error(str(exc))
                logger.info("[STOP] hard block detected. Saving collected schedule rows and exiting.")

            if rows:
                df = pd.DataFrame(rows).dropna(subset=['game_url'])
                df['opponent_team_id'] = df['opponent_team_id'].astype("Int64")
                for col in ["contest_id", "opponent_team_id", "team_id", "division", "year"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                df = df[df['division'] == div].copy()

                fname = f"d{div}_schedules_{year}.csv"
                fpath = outdir / fname
                df.to_csv(fpath, index=False)
                logger.info(f"saved {fpath} ({len(df)} rows) for division {div}")
            if session.hard_blocked:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--team_ids_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--rest_every", type=int, default=12)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_schedules(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        rest_every=args.rest_every,
        batch_cooldown_s=args.batch_cooldown_s,
    )
