import argparse
import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://stats.ncaa.org"

def get_most_recent_team_ids(team_ids_file):
    teams = pd.read_csv(team_ids_file)
    teams = teams.dropna(subset=['team_id', 'year', 'division'])
    teams = teams.sort_values(['org_id', 'year'], ascending=[True, False])
    most_recent = teams.groupby('org_id').first().reset_index()
    return most_recent[['org_id', 'school_name', 'team_id', 'year']]

def scrape_stadiums(team_ids_file, outdir, batch_size=25, max_retries=2, pause_between_batches=2):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    team_data = get_most_recent_team_ids(team_ids_file)
    total_teams = len(team_data)
    print(f"Found {total_teams} unique orgs with most recent team IDs")

    rows = []

    print("Starting Playwright browser...")
    with sync_playwright() as p:
        print("Launching Chromium...")
        browser = p.chromium.launch(headless=False)
        print("Creating new page...")
        page = browser.new_page()
        print("Browser ready, starting scraping...")

        for idx, row in team_data.iterrows():
            org_id = row['org_id']
            school_name = row['school_name']
            team_id = int(row['team_id'])
            year = int(row['year'])

            url = f"{BASE}/teams/{team_id}"
            print(f"Processing {school_name} ({team_id}) - {url}")
            success = False

            for retry in range(1, max_retries + 1):
                try:
                    print(f"  Navigating to {url} (attempt {retry})...")
                    page.goto(url, timeout=5000, wait_until="domcontentloaded")
                    print("  Page loaded, waiting for card-header...")
                    page.wait_for_selector("div.card-header", timeout=15000)
                    print("  Found card-header, parsing...")

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
                        success = True
                        break

                    card_body = stadium_card.query_selector("div.card-body")
                    if not card_body:
                        print(f"no card-body in stadium section {school_name} ({team_id})")
                        success = True
                        break

                    print("  Found stadium card-body, looking for venues...")
                    venue_cards = card_body.query_selector_all("div.card[id^='team_page_season_venue_']")
                    print(f"  Found {len(venue_cards)} venue cards")

                    if not venue_cards:
                        print(f"no venues found {school_name} ({team_id})")
                        success = True
                        break

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

                        print(f"    Found {len(dts)} dt elements and {len(dds)} dd elements")

                        for dt, dd in zip(dts, dds):
                            label = dt.inner_text().strip().rstrip(':')
                            value = dd.inner_text().strip()

                            print(f"    Label: '{label}', Value: '{value}'")

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

                        print(f"    Parsed stadium_data: {stadium_data}")

                        if stadium_data['stadium_name']:
                            rows.append(stadium_data)

                    print(f"success {school_name} ({team_id}) - {len(venue_cards)} venues")
                    success = True
                    break

                except Exception as e:
                    if retry == max_retries:
                        print(f"failed {school_name} ({team_id}): {e}")
                    time.sleep(2 ** (retry - 1))

            if not success:
                print(f"no data {school_name} ({team_id})")

            if (idx + 1) % batch_size == 0:
                print(f"Processed {idx + 1}/{total_teams} teams...")
                time.sleep(pause_between_batches)

        browser.close()

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
    parser.add_argument("--pause_between_batches", type=int, default=2)
    args = parser.parse_args()

    scrape_stadiums(
        team_ids_file=args.team_ids_file,
        outdir=args.outdir,
        batch_size=args.batch_size,
        pause_between_batches=args.pause_between_batches
    )

