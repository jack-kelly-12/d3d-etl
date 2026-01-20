import argparse
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import ScraperConfig, ScraperSession

START_URL = f"{BASE}/teams/history?org_id=141&sport_code=MBA"


def scrape_team_history(
    outdir,
    batch_size=25,
    headless=True,
    base_delay=2.0,
    daily_budget=20000
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        headless=headless,
        block_resources=True,
        daily_request_budget=daily_budget,
    )

    with ScraperSession(config) as session:
        html, status = session.fetch(START_URL, wait_selector="#org_id_select", wait_timeout=60000)
        if not html or status >= 400:
            print(f"failed to load start page: HTTP {status}")
            return

        options = session.page.query_selector_all("#org_id_select option")

        orgs = []
        for opt in options:
            val = opt.get_attribute("value")
            txt = opt.inner_text().strip()
            if val and not txt.lower().startswith("select"):
                orgs.append({"org_id": int(val), "school_name": txt})

        print(f"Found {len(orgs)} orgs (budget: {session.requests_remaining})")

        rows = []
        for i, org in enumerate(orgs, 1):
            if session.requests_remaining <= 0:
                print("[budget] daily request budget exhausted, stopping")
                break

            url = f"{BASE}/teams/history?org_id={org['org_id']}&sport_code=MBA"
            html, status = session.fetch(url, wait_selector="#team_history_data_table tbody tr", wait_timeout=15000)

            if not html or status >= 400:
                print(f"failed {org['school_name']} ({org['org_id']}): HTTP {status}")
                continue

            trs = session.page.query_selector_all("#team_history_data_table tbody tr")

            for tr in trs:
                tds = [c.inner_text().strip() for c in tr.query_selector_all("td")]
                if not tds or "No data available" in tds[0]:
                    continue

                season = tds[0]
                try:
                    season_year = int(season.split("-")[0])
                except ValueError:
                    continue

                if season_year < 2020:
                    continue

                year = int(season.split("-")[1]) + 2000

                div_str = tds[2]
                div_map = {"D-I": 1, "D-II": 2, "D-III": 3}
                division = div_map.get(div_str, "-")

                link = tr.query_selector("td a[href^='/teams/']")
                team_id = int(link.get_attribute("href").split("/")[2]) if link else None

                rows.append({
                    "org_id": org["org_id"],
                    "school_name": org["school_name"],
                    "year": year,
                    "division": division,
                    "conference": tds[3],
                    "wins": tds[4],
                    "losses": tds[5],
                    "ties": tds[6],
                    "wl_pct": tds[7],
                    "coach": tds[1],
                    "coach_id": None,
                    "team_id": team_id
                })

            print(f"success {org['school_name']} ({org['org_id']})")

            if i % batch_size == 0:
                print(f"Processed {i}/{len(orgs)} schools... (budget: {session.requests_remaining})")

        df = pd.DataFrame(rows)
        fpath = outdir / "ncaa_team_history.csv"
        df.to_csv(fpath, index=False)
        print(f"saved {fpath} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="../new_data")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_team_history(
        outdir=args.outdir,
        batch_size=args.batch_size,
        headless=args.headless,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
