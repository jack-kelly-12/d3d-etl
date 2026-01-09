import argparse
import html
import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = "https://stats.ncaa.org"


def load_existing(outdir, div, year, kind):
    fpath = Path(outdir) / f"d{div}_lineups_{kind}_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64", "ncaa_id": "Int64", "team_id": "Int64"})
        if "contest_id" in df.columns:
            return df, set(df["contest_id"].dropna().unique())
    return pd.DataFrame(), set()


def get_schedules(indir, div, year):
    fpath = Path(indir) / f"d{div}_schedules_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64"})
        if "contest_id" in df.columns:
            return df.drop_duplicates(subset=["contest_id"])
    return pd.DataFrame()


def _parse_team_header_text(card_header_el):
    # header contains: <a href="/teams/597018">...Rose-Hulman</a> Hitting
    a = card_header_el.query_selector("a[href*='/teams/']")
    team_id = None
    team_name = None
    if a:
        href = a.get_attribute("href") or ""
        m = re.search(r"/teams/(\d+)", href)
        if m:
            team_id = int(m.group(1))
        team_name = (a.inner_text() or "").strip()
    header_text = (card_header_el.inner_text() or "").strip()
    return team_id, team_name, header_text


def _is_indented_sub(td_html: str) -> bool:
    # NCAA uses literal &nbsp; in markup; when we read inner_html we will still see it.
    # Some pages may render unicode NBSP too, so catch both.
    if td_html is None:
        return False
    return ("&nbsp;" in td_html) or ("\u00a0" in td_html)


def _extract_player_link_data(name_td_el):
    a = name_td_el.query_selector("a[href*='/players/']")
    if not a:
        return None
    href = a.get_attribute("href") or ""
    m = re.search(r"/players/(\d+)", href)
    if not m:
        return None
    ncaa_id = int(m.group(1))
    name_text = (a.inner_text() or "").strip()
    name_text = html.unescape(name_text)
    return ncaa_id, name_text


def scrape_game_lineups(page, contest_id, div, year, max_retries=3):
    """
    Returns (hitting_df, pitching_df) for a contest_id.
    Each df has one row per player appearance in that game-table (starters + subs + pitchers).
    """
    url = f"{BASE}/contests/{contest_id}/individual_stats"

    for retry in range(1, max_retries + 1):
        try:
            page.goto(url, timeout=10000)
            page.wait_for_selector("div.card", timeout=10000)

            cards = page.query_selector_all("div.card")
            if not cards:
                return pd.DataFrame(), pd.DataFrame()

            hit_rows = []
            pit_rows = []

            for card in cards:
                header = card.query_selector("div.card-header")
                body = card.query_selector("div.card-body")
                tbl = body.query_selector("table") if body else None
                if not header or not tbl:
                    continue

                team_id, team_name, header_text = _parse_team_header_text(header)
                header_lower = header_text.lower()

                is_hitting = "hitting" in header_lower
                is_pitching = "pitching" in header_lower
                if not (is_hitting or is_pitching):
                    continue

                # Get table header labels so we can map columns robustly
                ths = tbl.query_selector_all("thead tr th")
                col_names = [(th.inner_text() or "").strip() for th in ths]

                # Find indices we care about
                # Hitting: ["#", "Name", "P", ...]
                # Pitching: usually ["#", "Name", "IP", "H", "R", ...] (varies)
                idx_num = None
                idx_name = None
                idx_pos = None  # hitting position column ("P") if present

                for i, c in enumerate(col_names):
                    c_norm = c.replace("\n", " ").strip()
                    if c_norm == "#":
                        idx_num = i
                    if c_norm.lower() == "name":
                        idx_name = i
                    if is_hitting and c_norm == "P":
                        idx_pos = i

                # Iterate player rows in tbody; skip totals rows without player links
                trs = tbl.query_selector_all("tbody tr")
                for tr in trs:
                    tds = tr.query_selector_all("td")
                    if not tds:
                        continue
                    if idx_name is None or idx_name >= len(tds):
                        continue

                    name_td = tds[idx_name]
                    name_td_html = name_td.inner_html() or ""
                    link_data = _extract_player_link_data(name_td)
                    if not link_data:
                        continue  # totals/team rows, inning period rows, etc.

                    ncaa_id, player_name = link_data
                    is_sub = _is_indented_sub(name_td_html)

                    bat_order = None
                    if idx_num is not None and idx_num < len(tds):
                        raw_num = (tds[idx_num].inner_text() or "").strip()
                        raw_num = raw_num.replace("\u00a0", "").strip()
                        if raw_num != "":
                            # some rows may have non-numeric, but usually numeric
                            mnum = re.search(r"\d+", raw_num)
                            bat_order = int(mnum.group(0)) if mnum else None

                    pos = None
                    if is_hitting and idx_pos is not None and idx_pos < len(tds):
                        pos = (tds[idx_pos].inner_text() or "").strip()
                        pos = pos.replace("\u00a0", "").strip() or None

                    base = {
                        "division": div,
                        "year": year,
                        "contest_id": int(contest_id),
                        "team_id": team_id,
                        "team_name": team_name,
                        "ncaa_id": ncaa_id,
                        "name": player_name,
                        "is_sub": bool(is_sub),
                    }

                    if is_hitting:
                        hit_rows.append(
                            {
                                **base,
                                "bat_order": bat_order,
                                "pos": pos,
                            }
                        )
                    elif is_pitching:
                        # For pitching we still keep bat_order as None (not meaningful)
                        # and store a "role_pos" as whatever the table shows in the 3rd column if it exists.
                        # But most NCAA pitching tables don't have a position column like "P", so keep it simple.
                        pit_rows.append(
                            {
                                **base,
                                "pitch_order": bat_order,  # this is often jersey slot or appearance order; still useful
                            }
                        )

            hit_df = pd.DataFrame(hit_rows)
            pit_df = pd.DataFrame(pit_rows)

            # Normalize types
            for df in (hit_df, pit_df):
                if df.empty:
                    continue
                for c in ("team_id", "ncaa_id", "contest_id"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            return hit_df, pit_df

        except Exception as e:
            if retry == max_retries:
                print(f"failed lineup game {contest_id}: {e}")
            time.sleep(2 ** (retry - 1))

    return pd.DataFrame(), pd.DataFrame()


def scrape_lineups(
    indir,
    outdir,
    year,
    divisions,
    batch_size=50,
    pause_between_games=0.5,
    pause_between_batches=5,
    headless=False,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        for div in divisions:
            sched = get_schedules(indir, div, year)
            if sched.empty:
                print(f"no schedule for d{div} {year}")
                continue

            existing_hit, done_hit = load_existing(outdir, div, year, kind="hitting")
            existing_pit, done_pit = load_existing(outdir, div, year, kind="pitching")

            done_ids = done_hit.intersection(done_pit) if (done_hit and done_pit) else (done_hit | done_pit)

            rows_hit = []
            rows_pit = []

            total_games = len(sched)
            print(f"\n=== d{div} {year} lineups — {total_games} games ===")

            for start in range(0, total_games, batch_size):
                end = min(start + batch_size, total_games)
                batch = sched.iloc[start:end]

                for _, r in batch.iterrows():
                    gid = r["contest_id"]
                    if pd.isna(gid) or int(gid) in done_ids:
                        print(f"skip game {gid}")
                        continue

                    hit_df, pit_df = scrape_game_lineups(page, int(gid), div, year)
                    if not hit_df.empty:
                        rows_hit.append(hit_df)
                    if not pit_df.empty:
                        rows_pit.append(pit_df)

                    print(f"game {gid}: hit={len(hit_df)} pit={len(pit_df)}")
                    time.sleep(pause_between_games)

                print(f"batch {start+1}-{end} done")
                time.sleep(pause_between_batches)

            # Save outputs
            if rows_hit:
                new_hit = pd.concat(rows_hit, ignore_index=True)
                out_hit = (
                    pd.concat([existing_hit, new_hit], ignore_index=True)
                    if not existing_hit.empty
                    else new_hit
                )
                fpath = outdir / f"d{div}_lineups_hitting_{year}.csv"
                out_hit.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(out_hit)} rows)")

            if rows_pit:
                new_pit = pd.concat(rows_pit, ignore_index=True)
                out_pit = (
                    pd.concat([existing_pit, new_pit], ignore_index=True)
                    if not existing_pit.empty
                    else new_pit
                )
                fpath = outdir / f"d{div}_lineups_pitching_{year}.csv"
                out_pit.to_csv(fpath, index=False)
                print(f"saved {fpath} ({len(out_pit)} rows)")

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--indir", default="../data/schedules")
    parser.add_argument("--outdir", default="../data/lineups")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    scrape_lineups(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions,
        batch_size=args.batch_size,
        headless=args.headless,
    )
