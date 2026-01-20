import argparse
import re
from pathlib import Path

import pandas as pd

from .constants import BASE
from .scraper_utils import ScraperConfig, ScraperSession


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_norm = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
        rename_map[col] = col_norm
    return df.rename(columns=rename_map)


def parse_table(session: ScraperSession, table, year, school, conference, div, team_id):
    headers = [th.inner_text().strip() for th in table.query_selector_all("thead th")]
    headers = [re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_").lower() for h in headers]
    rows = []
    for tr in table.query_selector_all("tbody tr"):
        tds = tr.query_selector_all("td")
        if not tds:
            continue
        values = [td.inner_text().strip() for td in tds]
        row_dict = dict(zip(headers, values, strict=False))
        a = tr.query_selector("a[href*='/players/']")
        ncaa_id, player_url = None, None
        if a:
            href = a.get_attribute("href")
            if "/players/" in href:
                try:
                    ncaa_id = int(href.split("/")[-1].split("?")[0])
                except ValueError:
                    ncaa_id = None
                player_url = BASE + href
        row_dict.update({
            "division": div,
            "year": year,
            "team_name": school,
            "conference": conference,
            "team_id": team_id,
            "ncaa_id": ncaa_id,
            "player_url": player_url,
        })
        rows.append(row_dict)
    return rows


def scrape_stats(
    team_ids_file,
    year,
    divisions,
    outdir,
    batch_size=10,
    headless=True,
    base_delay=2.0,
    daily_budget=20000
):
    teams = pd.read_csv(team_ids_file)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        headless=headless,
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
            print(f"\n=== d{div} {year} stats — {total_teams} teams ===")
            print(f"    (budget remaining: {session.requests_remaining} requests)")

            results = {"batting": [], "pitching": []}

            for start in range(0, total_teams, batch_size):
                if session.requests_remaining <= 0:
                    print("[budget] daily request budget exhausted, stopping")
                    break

                end = min(start + batch_size, total_teams)
                batch = teams_div.iloc[start:end]

                for _, row in batch.iterrows():
                    if session.requests_remaining <= 0:
                        break

                    team_id = row.team_id
                    school = row.school_name
                    conference = row.conference

                    url = f"{BASE}/teams/{team_id}/season_to_date_stats?year={year}"
                    html, status = session.fetch(url, wait_selector="#stat_grid tbody tr", wait_timeout=15000)

                    if html and status < 400:
                        table = session.page.query_selector("#stat_grid")
                        if table:
                            rows = parse_table(session, table, year, school, conference, div, team_id)
                            results["batting"].extend(rows)
                        print(f"success {school} ({team_id}) batting")

                        pitch_link = session.page.query_selector("a.nav-link:has-text('Pitching')")
                        if pitch_link:
                            href = pitch_link.get_attribute("href")
                            pitch_url = BASE + href
                            html2, status2 = session.fetch(pitch_url, wait_selector="#stat_grid tbody tr", wait_timeout=15000)
                            if html2 and status2 < 400:
                                table = session.page.query_selector("#stat_grid")
                            if table:
                                    rows = parse_table(session, table, year, school, conference, div, team_id)
                                    results["pitching"].extend(rows)
                            print(f"success {school} ({team_id}) pitching")
                        else:
                            print(f"failed {school} ({team_id}) pitching: HTTP {status2}")
                    else:
                        print(f"no pitching link for {school} ({team_id})")
                else:
                    print(f"failed {school} ({team_id}) batting: HTTP {status}")

                print(f"batch {start+1}-{end} done (budget: {session.requests_remaining})")

            for stat_type, rows in results.items():
                if rows:
                    df = pd.DataFrame(rows)
                    df = normalize_cols(df)
                    df = df.dropna(subset=['ncaa_id'])
                    if "ncaa_id" in df.columns:
                        df["ncaa_id"] = pd.to_numeric(df["ncaa_id"], errors="coerce").astype("Int64")
                    if "gdp" in df.columns:
                        df = df.drop(columns=['gdp'])
                    df = df.rename(columns={
                        'slgpct': 'slg_pct',
                        'obpct': 'ob_pct',
                        'player': 'player_name',
                        'team': 'team_name',
                        'hbp': 'hbp',
                        'yr': 'class'})
                    fname = f"d{div}_{stat_type}_{year}.csv"
                    fpath = outdir / fname
                    df.to_csv(fpath, index=False)
                    print(f"saved {fpath} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--team_ids_file", default="../data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="../data/stats")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_stats(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        batch_size=args.batch_size,
        headless=args.headless,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
