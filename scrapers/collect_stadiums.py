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


def get_most_recent_team_ids(team_ids_file):
    teams = pd.read_csv(team_ids_file)
    teams = teams.dropna(subset=["team_id", "year", "division"])
    teams = teams.sort_values(["org_id", "year"], ascending=[True, False])
    most_recent = teams.groupby("org_id").first().reset_index()
    return most_recent[["org_id", "team_name", "team_id", "year"]]


def scrape_stadiums(
    team_ids_file,
    outdir,
    batch_size=25,
    base_delay=10.0,
    rest_every=12,
    batch_cooldown_s=90,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    team_data = get_most_recent_team_ids(team_ids_file)
    total_teams = len(team_data)
    logger.info(f"Found {total_teams} unique orgs with most recent team IDs")

    config = ScraperConfig(base_delay=base_delay, block_resources=True)
    rows = []

    with ScraperSession(config) as session:
        try:
            logger.info("Browser ready, starting scraping...")
            for idx, row in team_data.iterrows():
                short_break_every(rest_every, idx + 1, sleep_s=60.0)
                org_id = row["org_id"]
                team_name = row["team_name"]
                team_id = int(row["team_id"])
                year = int(row["year"])

                url = f"{BASE}/teams/{team_id}"
                html, status = session.fetch(
                    url, wait_selector="div.card-header", wait_timeout=15000
                )
                if not html or status >= 400:
                    logger.info(f"failed {team_name} ({team_id}): HTTP {status}")
                    continue

                page = session.page
                stadium_card = page.query_selector(f"div#team_venues_{team_id}")
                if not stadium_card:
                    for card in page.query_selector_all("div.card"):
                        header = card.query_selector("div.card-header")
                        if header and "Stadium" in header.inner_text():
                            stadium_card = card
                            break

                if not stadium_card:
                    logger.info(f"no stadium section found {team_name} ({team_id})")
                    continue

                card_body = stadium_card.query_selector("div.card-body")
                if not card_body:
                    logger.info(f"no card-body in stadium section {team_name} ({team_id})")
                    continue

                venue_cards = card_body.query_selector_all(
                    "div.card[id^='team_page_season_venue_']"
                )
                if not venue_cards:
                    logger.info(f"no venues found {team_name} ({team_id})")
                    continue

                for venue_card in venue_cards:
                    venue_id = venue_card.get_attribute("id")
                    venue_id_num = re.search(r"venue_(\d+)", venue_id) if venue_id else None
                    venue_id_num = int(venue_id_num.group(1)) if venue_id_num else None

                    dl = venue_card.query_selector("dl.row")
                    if not dl:
                        continue

                    stadium_data = {
                        "org_id": org_id,
                        "team_name": team_name,
                        "year": year,
                        "venue_id": venue_id_num,
                        "stadium_name": None,
                        "capacity": None,
                        "year_built": None,
                    }

                    dts = dl.query_selector_all("dt")
                    dds = dl.query_selector_all("dd")
                    for dt, dd in zip(dts, dds, strict=False):
                        label = dt.inner_text().strip().rstrip(":")
                        value = dd.inner_text().strip()
                        if label == "Name":
                            stadium_data["stadium_name"] = value
                        elif label == "Capacity":
                            try:
                                stadium_data["capacity"] = int(re.sub(r"[^\d]", "", value))
                            except (ValueError, AttributeError):
                                stadium_data["capacity"] = None
                        elif label == "Year Built":
                            try:
                                stadium_data["year_built"] = int(value)
                            except (ValueError, AttributeError):
                                stadium_data["year_built"] = None

                    if stadium_data["stadium_name"]:
                        rows.append(stadium_data)

                logger.info(f"success {team_name} ({team_id}) - {len(venue_cards)} venues")
                if (idx + 1) % batch_size == 0:
                    logger.info(f"Processed {idx + 1}/{total_teams} teams...")
                    cooldown_between_batches(idx + 1, total_teams, float(batch_cooldown_s))
        except HardBlockError as exc:
            logger.error(str(exc))
            return

    if rows:
        df = pd.DataFrame(rows)
        fpath = outdir / "stadiums.csv"
        df.to_csv(fpath, index=False)
        logger.info(f"saved {fpath} ({len(df)} rows)")
    else:
        logger.info("No stadium data collected")


if __name__ == "__main__":
    print("[start] scrapers.collect_stadiums", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_ids_file", default="./data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="./data/stadiums")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--rest_every", type=int, default=12)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_stadiums(
        team_ids_file=args.team_ids_file,
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        rest_every=args.rest_every,
        batch_cooldown_s=args.batch_cooldown_s,
    )
