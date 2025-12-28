import argparse
import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://stats.ncaa.org"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_norm = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
        rename_map[col] = col_norm
    return df.rename(columns=rename_map)

def parse_table(table, year, school, conference, div, team_id):
    headers = [th.inner_text().strip() for th in table.query_selector_all("thead th")]
    headers = [re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_").lower() for h in headers]
    rows = []
    for tr in table.query_selector_all("tbody tr"):
        tds = tr.query_selector_all("td")
        if not tds:
            continue
        values = [td.inner_text().strip() for td in tds]
        row_dict = dict(zip(headers, values))
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

def scrape_stats(team_ids_file, year, divisions, outdir, batch_size=10, max_retries=3):
    teams = pd.read_csv(team_ids_file)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        for div in divisions:
            teams_div = teams.query("year == @year and division == @div").copy()
            if teams_div.empty:
                continue
            teams_div["team_id"] = teams_div["team_id"].astype(int)
            total_teams = len(teams_div)
            results = {"batting": [], "pitching": []}
            for start in range(0, total_teams, batch_size):
                end = min(start + batch_size, total_teams)
                batch = teams_div.iloc[start:end]
                for _, row in batch.iterrows():
                    team_id = row.team_id
                    school = row.school_name
                    conference = row.conference

                    url = f"{BASE}/teams/{team_id}/season_to_date_stats?year={year}"
                    success = False
                    for retry in range(1, max_retries+1):
                        try:
                            page.goto(url, timeout=45000)
                            page.wait_for_selector("#stat_grid tbody tr", timeout=15000)
                            table = page.query_selector("#stat_grid")
                            if table:
                                rows = parse_table(table, year, school, conference, div, team_id)
                                results["batting"].extend(rows)
                            print(f"success {school} ({team_id}) batting")
                            success = True
                            break
                        except Exception as e:
                            if retry == max_retries:
                                print(f"failed {school} ({team_id}) batting: {e}")
                            time.sleep(2 ** (retry-1))
                    if not success:
                        print(f"no data {school} ({team_id}) batting")

                    try:
                        pitch_link = page.query_selector("a.nav-link:has-text('Pitching')")
                        if pitch_link:
                            href = pitch_link.get_attribute("href")
                            pitch_url = BASE + href
                            page.goto(pitch_url, timeout=45000)
                            page.wait_for_selector("#stat_grid tbody tr", timeout=15000)
                            table = page.query_selector("#stat_grid")
                            if table:
                                rows = parse_table(table, year, school, conference, div, team_id)
                                results["pitching"].extend(rows)
                            print(f"success {school} ({team_id}) pitching")
                        else:
                            print(f"no pitching link for {school} ({team_id})")
                    except Exception as e:
                        print(f"failed {school} ({team_id}) pitching: {e}")

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
        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--team_ids_file", default="../data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="../data/stats")
    args = parser.parse_args()
    scrape_stats(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir
    )
