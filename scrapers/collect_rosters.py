import argparse
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import HardBlockError, ScraperConfig, ScraperSession


def scrape_rosters(
    team_ids_file, year, divisions, outdir, batch_size=10, base_delay=2.0, daily_budget=20000
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
        daily_request_budget=daily_budget,
    )

    with ScraperSession(config) as session:
        for div in divisions:
            teams_div = teams.query("year == @year and division == @div").copy()
            if teams_div.empty:
                continue

            teams_div["team_id"] = teams_div["team_id"].astype(int)
            total_teams = len(teams_div)
            print(f"\n=== {year} D{div} rosters — {total_teams} teams ===")
            print(f"    (budget remaining: {session.requests_remaining} requests)")

            rows = []
            try:
                for idx, row in enumerate(teams_div.itertuples(index=False), 1):
                    if session.requests_remaining <= 0:
                        print("[budget] daily request budget exhausted, stopping")
                        break

                    team_id = row.team_id
                    school = row.team_name
                    conference = row.conference

                    url = f"{BASE}/teams/{team_id}/roster"
                    html, status = session.fetch(
                        url,
                        wait_selector="table[id^='rosters_form_players_'] tbody tr",
                        wait_timeout=15000,
                    )

                    if not html or status >= 400:
                        print(f"failed {school} ({team_id}): HTTP {status}")
                        continue

                    trs = session.page.query_selector_all(
                        "table[id^='rosters_form_players_'] tbody tr"
                    )

                    for tr in trs:
                        tds = [td.inner_text().strip() for td in tr.query_selector_all("td")]
                        a = tr.query_selector("a[href^='/players/']")
                        player_id, player_name = None, None
                        if a:
                            href = a.get_attribute("href")
                            if href and "/players/" in href:
                                try:
                                    player_id = int(href.split("/")[-1])
                                except ValueError:
                                    pass
                            player_name = a.inner_text().strip()
                        rows.append(
                            {
                                "org_id": row.org_id,
                                "team_name": school,
                                "year": year,
                                "division": div,
                                "team_id": team_id,
                                "conference": conference,
                                "player_id": player_id,
                                "player_name": player_name,
                                "games_played": tds[0] if len(tds) > 0 else None,
                                "games_started": tds[1] if len(tds) > 1 else None,
                                "number": tds[2] if len(tds) > 2 else None,
                                "class": tds[4] if len(tds) > 4 else None,
                                "position": tds[5] if len(tds) > 5 else None,
                                "height": tds[6] if len(tds) > 6 else None,
                                "bats": tds[7] if len(tds) > 7 else None,
                                "throws": tds[8] if len(tds) > 8 else None,
                                "hometown": tds[9] if len(tds) > 9 else None,
                                "high_school": tds[10] if len(tds) > 10 else None,
                            }
                        )
                    print(f"success {school} ({team_id})")

                    if idx % batch_size == 0:
                        print(
                            f"processed {idx}/{total_teams} (budget: {session.requests_remaining})"
                        )
            except HardBlockError as exc:
                print(str(exc))
                print("[STOP] hard block detected. Saving collected roster rows and exiting.")

            if rows:
                df = pd.DataFrame(rows)
                fpath = outdir / f"d{div}_rosters_{year}.csv"
                df.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(df)} rows)")
            if session.hard_blocked:
                return


if __name__ == "__main__":
    print("[start] scrapers.collect_rosters", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--team_ids_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv"
    )
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/rosters")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_rosters(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
