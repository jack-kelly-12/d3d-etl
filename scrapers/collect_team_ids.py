from playwright.sync_api import sync_playwright
import pandas as pd
import time
from pathlib import Path
import argparse

BASE = "https://stats.ncaa.org"
START_URL = f"{BASE}/teams/history?org_id=141&sport_code=MBA"

def scrape_team_history(outdir, batch_size=25, pause_between_batches=1):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=r"C:\\Users\\jackkelly\\AppData\\Local\\Google\\Chrome\\User Data\\Default",
            headless=False,
            channel="chrome"
        )
        page = browser.new_page()
        page.goto(START_URL, timeout=60000)

        page.wait_for_selector("#org_id_select", state="attached")
        options = page.query_selector_all("#org_id_select option")

        orgs = []
        for opt in options:
            val = opt.get_attribute("value")
            txt = opt.inner_text().strip()
            if val and not txt.lower().startswith("select"):
                orgs.append({"org_id": int(val), "school_name": txt})

        print(f"Found {len(orgs)} orgs")

        rows = []
        for i, org in enumerate(orgs, 1):
            url = f"{BASE}/teams/history?org_id={org['org_id']}&sport_code=MBA"
            try:
                page.goto(url, timeout=60000)
                page.wait_for_selector("#team_history_data_table tbody tr", timeout=15000)
                trs = page.query_selector_all("#team_history_data_table tbody tr")

                for tr in trs:
                    tds = [c.inner_text().strip() for c in tr.query_selector_all("td")]
                    if not tds or "No data available" in tds[0]:
                        continue

                    season = tds[0]
                    try:
                        season_year = int(season.split("-")[0])
                    except ValueError:
                        continue

                    # keep >= 2020 season so we include 2020-21 and newer
                    if season_year < 2020:
                        continue

                    link = tr.query_selector("td a[href^='/teams/']")
                    team_id = int(link.get_attribute("href").split("/")[2]) if link else None

                    rows.append({
                        "org_id": org["org_id"],
                        "school_name": org["school_name"],
                        "season": season,
                        "coach": tds[1],
                        "division": tds[2],
                        "conference": tds[3],
                        "wins": tds[4],
                        "losses": tds[5],
                        "ties": tds[6],
                        "wl_pct": tds[7],
                        "team_id": team_id
                    })
                print(f"success {org['school_name']} ({org['org_id']})")
            except Exception as e:
                print(f"failed {org['school_name']} ({org['org_id']}): {e}")

            if i % batch_size == 0:
                print(f"Processed {i}/{len(orgs)} schools…")
                time.sleep(pause_between_batches)

        df = pd.DataFrame(rows)
        fpath = outdir / "ncaa_team_history.csv"
        df.to_csv(fpath, index=False)
        print(f"saved {fpath} ({len(df)} rows)")

        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="../new_data")
    args = parser.parse_args()

    scrape_team_history(
        outdir=args.outdir
    )
