import argparse
import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://stats.ncaa.org"

def scrape_schedules(team_ids_file, year, divisions, outdir, batch_size=10, max_retries=3):
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
            print(f"\n=== {year} D{div} schedules — {total_teams} teams ===")

            rows = []

            for start in range(0, total_teams, batch_size):
                end = min(start + batch_size, total_teams)
                batch = teams_div.iloc[start:end]

                for _, row in batch.iterrows():
                    team_id = row.team_id
                    school = row.school_name
                    conference = row.conference
                    url = f"{BASE}/teams/{team_id}"
                    success = False

                    for retry in range(1, max_retries+1):
                        try:
                            page.goto(url, timeout=45000)
                            page.wait_for_selector("div.card-header:has-text('Schedule/Results')", timeout=10000)
                            trs = page.query_selector_all("div.card-body table tbody tr")

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

                            print(f"success {school} ({team_id})")
                            success = True
                            break
                        except Exception as e:
                            if retry == max_retries:
                                print(f"failed {school} ({team_id}): {e}")
                            time.sleep(2 ** (retry-1))

                    if not success:
                        print(f"no data {school} ({team_id})")

            if rows:
                df = pd.DataFrame(rows).dropna(subset=['game_url'])
                df['opponent_team_id'] = df['opponent_team_id'].fillna(0).astype(int)
                for col in ["contest_id", "opponent_team_id", "team_id", "division", "year"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                df = df[df['division'] == div].copy()

                fname = f"d{div}_schedules_{year}.csv"
                fpath = outdir / fname
                df.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(df)} rows) for division {div}")


        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--team_ids_file", default="../data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="../data/schedules")
    args = parser.parse_args()

    scrape_schedules(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir
    )
