import argparse
import re
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import ScraperConfig, ScraperSession


def get_most_recent_team_ids(team_ids_file):
    teams = pd.read_csv(team_ids_file)
    teams = teams.dropna(subset=['team_id', 'year', 'division'])
    teams = teams.sort_values(['org_id', 'year'], ascending=[True, False])
    most_recent = teams.groupby('org_id').first().reset_index()
    return most_recent[['org_id', 'school_name', 'team_id', 'year']]


def scrape_stadiums(
    team_ids_file,
    outdir,
    batch_size=25,
    headless=True,
    base_delay=2.0,
    daily_budget=20000
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    team_data = get_most_recent_team_ids(team_ids_file)
    total_teams = len(team_data)
    print(f"Found {total_teams} unique orgs with most recent team IDs")

    config = ScraperConfig(
        base_delay=base_delay,
        headless=headless,
        block_resources=True,
        daily_request_budget=daily_budget,
    )

    rows = []

    with ScraperSession(config) as session:
        print(f"Browser ready, starting scraping... (budget: {session.requests_remaining})")

        for idx, row in team_data.iterrows():
            if session.requests_remaining <= 0:
                print("[budget] daily request budget exhausted, stopping")
                break

            org_id = row['org_id']
            school_name = row['school_name']
            team_id = int(row['team_id'])
            year = int(row['year'])

            url = f"{BASE}/teams/{team_id}"

            html, status = session.fetch(url, wait_selector="div.card-header", wait_timeout=15000)

            if not html or status >= 400:
                print(f"failed {school_name} ({team_id}): HTTP {status}")
                continue

            page = session.page

            stadium_card_id = f"team_venues_{team_id}"
            stadium_card = page.query_selector(f"div#{stadium_card_id}")

            if not stadium_card:
                all_cards = page.query_selector_all("div.card")
                for card in all_cards:
                    header = card.query_selector("div.card-header")
                    if header and "Stadium" in header.inner_text():
                        stadium_card = card
                        break

            if not stadium_card:
                print(f"no stadium section found {school_name} ({team_id})")
                continue

            card_body = stadium_card.query_selector("div.card-body")
            if not card_body:
                print(f"no card-body in stadium section {school_name} ({team_id})")
                continue

            venue_cards = card_body.query_selector_all("div.card[id^='team_page_season_venue_']")

            if not venue_cards:
                print(f"no venues found {school_name} ({team_id})")
                continue

            for venue_card in venue_cards:
                venue_id = venue_card.get_attribute("id")
                venue_id_num = re.search(r'venue_(\d+)', venue_id) if venue_id else None
                venue_id_num = int(venue_id_num.group(1)) if venue_id_num else None

                dl = venue_card.query_selector("dl.row")
                if not dl:
                    continue

                stadium_data = {
                    'org_id': org_id,
                    'school_name': school_name,
                    'year': year,
                    'venue_id': venue_id_num,
                    'stadium_name': None,
                    'capacity': None,
                    'year_built': None
                }

                dts = dl.query_selector_all("dt")
                dds = dl.query_selector_all("dd")

                for dt, dd in zip(dts, dds, strict=False):
                    label = dt.inner_text().strip().rstrip(':')
                    value = dd.inner_text().strip()

                    if label == "Name":
                        stadium_data['stadium_name'] = value
                    elif label == "Capacity":
                        try:
                            stadium_data['capacity'] = int(re.sub(r'[^\d]', '', value))
                        except (ValueError, AttributeError):
                            stadium_data['capacity'] = None
                    elif label == "Year Built":
                        try:
                            stadium_data['year_built'] = int(value)
                        except (ValueError, AttributeError):
                            stadium_data['year_built'] = None

                if stadium_data['stadium_name']:
                    rows.append(stadium_data)

            print(f"success {school_name} ({team_id}) - {len(venue_cards)} venues")

            if (idx + 1) % batch_size == 0:
                print(f"Processed {idx + 1}/{total_teams} teams... (budget: {session.requests_remaining})")

    if rows:
        df = pd.DataFrame(rows)
        fpath = outdir / "stadiums.csv"
        df.to_csv(fpath, index=False)
        print(f"saved {fpath} ({len(df)} rows)")
    else:
        print("No stadium data collected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_ids_file", default="./data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="./data/stadiums")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_stadiums(
        team_ids_file=args.team_ids_file,
        outdir=args.outdir,
        batch_size=args.batch_size,
        headless=args.headless,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
