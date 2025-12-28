import argparse
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://stats.ncaa.org"

def year_to_season(year: int) -> str:
    return f"{year-1}-{str(year)[-2:]}"

def load_existing(outdir, div, year):
    fpath = Path(outdir) / f"d{div}_pbp_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        if "contest_id" in df.columns:
            return df, set(df["contest_id"].unique())
    return pd.DataFrame(), set()

def get_schedules(indir, div, year):
    fpath = Path(indir) / f"d{div}_schedules_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={'contest_id': 'Int64'})
        if "contest_id" in df.columns:
            df = df.drop_duplicates(subset=["contest_id"])
            return df
    return pd.DataFrame()


def scrape_game_pbp(page, contest_id, div, year, max_retries=3):
    url = f"{BASE}/contests/{contest_id}/play_by_play"
    for retry in range(1, max_retries+1):
        try:
            page.goto(url, timeout=45000)
            tables = page.query_selector_all("table")
            if len(tables) < 4:
                return pd.DataFrame()
            inning_tables = tables[3:]
            frames = []
            for i, tbl in enumerate(inning_tables, start=1):
                html = tbl.inner_html()
                df = pd.read_html(f"<table>{html}</table>")[0]
                if df.shape[1] >= 3:
                    df = df.rename(columns={df.columns[0]:"away_des", df.columns[2]:"home_des"})
                    df["inning"] = i
                    df["contest_id"] = contest_id
                    df["division"] = div
                    df["year"] = year
                    frames.append(df)
            if frames:
                out = pd.concat(frames, ignore_index=True)
                out["away_text"] = out["away_des"].fillna("").str.strip()
                out["home_text"] = out["home_des"].fillna("").str.strip()
                out["away_score"] = out["Score"].astype(str).str.extract(r"^(\d+)-")[0]
                out["home_score"] = out["Score"].astype(str).str.extract(r"-(\d+)$")[0]

                out = out[~((~out.away_text.isna()) & (~out.home_text.isna()))]

                return out[[
                    "division","year","contest_id","inning",
                    "away_text","home_text","away_score","home_score"
                ]]
            return pd.DataFrame()
        except Exception as e:
            if retry == max_retries:
                print(f"failed game {contest_id}: {e}")
            time.sleep(2 ** (retry-1))
    return pd.DataFrame()

def scrape_pbp(indir, outdir, year, divisions, batch_size=50, pause_between_games=0.5, pause_between_batches=5):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        for div in divisions:
            sched = get_schedules(indir, div, year)
            if sched.empty:
                print(f"no schedule for d{div} {year}")
                continue

            existing, done_ids = load_existing(outdir, div, year)
            rows = []
            total_games = len(sched)
            print(f"\n=== d{div} {year} pbp — {total_games} games ===")

            for start in range(0, total_games, batch_size):
                end = min(start + batch_size, total_games)
                batch = sched.iloc[start:end]

                for _, r in batch.iterrows():
                    gid = r["contest_id"]
                    if pd.isna(gid) or gid in done_ids:
                        print(f"skip game {gid}")
                        continue
                    df = scrape_game_pbp(page, gid, div, year)
                    if not df.empty:
                        rows.append(df)
                        print(f"success game {gid} ({len(df)} rows)")
                    else:
                        print(f"no data game {gid}")
                    time.sleep(pause_between_games)

                print(f"batch {start+1}-{end} done")
                time.sleep(pause_between_batches)

            if rows:
                new_df = pd.concat(rows, ignore_index=True)
                out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
                fpath = outdir / f"d{div}_pbp_{year}.csv"
                out.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(out)} rows)")

        browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--indir", default="../data/schedules")
    parser.add_argument("--outdir", default="../data/pbp")
    args = parser.parse_args()

    scrape_pbp(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions
    )
