import argparse
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

START_URL = f"{BASE}/teams/history?org_id=141&sport_code=MBA"

logger = get_scraper_logger(__name__)


def scrape_team_history(
    outdir,
    batch_size=25,
    base_delay=10.0,
    rest_every=12,
    batch_cooldown_s=90,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        block_resources=True,
    )

    with ScraperSession(config) as session:
        try:
            html, status = session.fetch(START_URL, wait_selector="#org_id_select", wait_timeout=60000)
        except HardBlockError as exc:
            logger.error(str(exc))
            logger.info("[STOP] hard block detected before start page. Exiting.")
            return
        if not html or status >= 400:
            logger.info(f"failed to load start page: HTTP {status}")
            return

        options = session.page.query_selector_all("#org_id_select option")

        orgs = []
        for opt in options:
            val = opt.get_attribute("value")
            txt = opt.inner_text().strip()
            if val and not txt.lower().startswith("select"):
                orgs.append({"org_id": int(val), "team_name": txt})

        logger.info(f"Found {len(orgs)} orgs (budget: {session.requests_remaining})")

        rows = []
        try:
            for i, org in enumerate(orgs, 1):
                if session.requests_remaining <= 0:
                    logger.info("[budget] daily request budget exhausted, stopping")
                    break

                short_break_every(rest_every, i, sleep_s=60.0)
                url = f"{BASE}/teams/history?org_id={org['org_id']}&sport_code=MBA"
                html, status = session.fetch(url, wait_selector="#team_history_data_table tbody tr", wait_timeout=15000)

                if not html or status >= 400:
                    logger.info(f"failed {org['team_name']} ({org['org_id']}): HTTP {status}")
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
                        "team_name": org["team_name"],
                        "year": year,
                        "division": division,
                        "conference": tds[3],
                        "coach": tds[1],
                        "coach_id": None,
                        "team_id": team_id
                    })

                logger.info(f"success {org['team_name']} ({org['org_id']})")

                if i % batch_size == 0:
                    logger.info(f"Processed {i}/{len(orgs)} schools... (budget: {session.requests_remaining})")
                    cooldown_between_batches(i, len(orgs), float(batch_cooldown_s))
        except HardBlockError as exc:
            logger.error(str(exc))
            logger.info("[STOP] hard block detected. Saving collected team history rows and exiting.")

        df = pd.DataFrame(rows)
        fpath = outdir / "ncaa_team_history.csv"
        df.to_csv(fpath, index=False)
        logger.info(f"saved {fpath} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--rest_every", type=int, default=12)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_team_history(
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        rest_every=args.rest_every,
        batch_cooldown_s=args.batch_cooldown_s,
    )
