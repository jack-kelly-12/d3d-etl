import argparse
import html
import re
from pathlib import Path

import pandas as pd

from .scraper_utils import ScraperConfig, ScraperSession

BASE = "https://stats.ncaa.org"


def load_existing(outdir, div, year, kind):
    fpath = Path(outdir) / f"d{div}_{kind}_lineups_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64", "ncaa_id": "Int64", "team_id": "Int64"})
        if "contest_id" in df.columns:
            done = set(pd.to_numeric(df["contest_id"], errors="coerce").dropna().astype(int).unique())
            return df, done
    return pd.DataFrame(), set()


def get_schedules(indir, div, year):
    fpath = Path(indir) / f"d{div}_schedules_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64"})
        if "contest_id" in df.columns:
            return df.drop_duplicates(subset=["contest_id"])
    return pd.DataFrame()


def _parse_team_header_text(card_header_el):
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


def _atomic_write_csv(df: pd.DataFrame, fpath: Path):
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(fpath)


def _dedupe_hit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["division", "year", "contest_id", "team_id", "ncaa_id", "is_sub", "bat_order", "pos"] if c in df.columns]
    if not key:
        return df
    return df.drop_duplicates(subset=key, keep="last")


def _dedupe_pit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["division", "year", "contest_id", "team_id", "ncaa_id", "is_sub", "pitch_order"] if c in df.columns]
    if not key:
        return df
    return df.drop_duplicates(subset=key, keep="last")


def scrape_game_lineups(session: ScraperSession, contest_id, div, year):
    url = f"{BASE}/contests/{contest_id}/individual_stats"

    html_content, status = session.fetch(url, wait_selector="div.card", wait_timeout=20000)

    if not html_content or status >= 400:
        print(f"  failed lineup game {contest_id}: HTTP {status}")
        return pd.DataFrame(), pd.DataFrame()

    page = session.page
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

        is_hitting = "hitting" in header_lower or "batting" in header_lower
        is_pitching = "pitching" in header_lower
        if not (is_hitting or is_pitching):
            continue

        ths = tbl.query_selector_all("thead tr th")
        col_names = [(th.inner_text() or "").strip() for th in ths]

        idx_num = None
        idx_name = None
        idx_pos = None

        for i, c in enumerate(col_names):
            c_norm = c.replace("\n", " ").strip()
            if c_norm == "#":
                idx_num = i
            if c_norm.lower() == "name":
                idx_name = i
            if is_hitting and c_norm in ("P", "Pos"):
                idx_pos = i

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
                continue

            ncaa_id, player_name = link_data
            is_sub = _is_indented_sub(name_td_html)

            order_num = None
            if idx_num is not None and idx_num < len(tds):
                raw_num = (tds[idx_num].inner_text() or "").strip()
                raw_num = raw_num.replace("\u00a0", "").strip()
                if raw_num != "":
                    mnum = re.search(r"\d+", raw_num)
                    order_num = int(mnum.group(0)) if mnum else None

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
                hit_rows.append({**base, "bat_order": order_num, "pos": pos})
            else:
                pit_rows.append({**base, "pitch_order": order_num})

    hit_df = pd.DataFrame(hit_rows)
    pit_df = pd.DataFrame(pit_rows)

    for df in (hit_df, pit_df):
        if df.empty:
            continue
        for c in ("team_id", "ncaa_id", "contest_id"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return hit_df, pit_df


def scrape_lineups(
    indir,
    outdir,
    year,
    divisions,
    batch_size=50,
    headless=True,
    base_delay=2.0,
    daily_budget=20000,
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
        for div in divisions:
            sched = get_schedules(indir, div, year)
            if sched.empty:
                print(f"no schedule for d{div} {year}")
                continue

            existing_hit, done_hit = load_existing(outdir, div, year, kind="batting")
            existing_pit, done_pit = load_existing(outdir, div, year, kind="pitching")
            done_ids = done_hit.intersection(done_pit) if (done_hit and done_pit) else (done_hit | done_pit)

            total_games = len(sched)
            sched = sched[~sched["contest_id"].isin(list(done_ids))]
            games_to_scrape = len(sched)

            print(f"\n=== d{div} {year} lineups — {total_games} games - already scraped {len(done_ids)} games - to scrape {games_to_scrape} ===")
            print(f"    (budget remaining: {session.requests_remaining} requests)")

            if games_to_scrape == 0:
                continue

            rows_hit = []
            rows_pit = []

            for start in range(0, games_to_scrape, batch_size):
                if session.requests_remaining <= 0:
                    print("[budget] daily request budget exhausted, stopping")
                    break

                end = min(start + batch_size, games_to_scrape)
                batch = sched.iloc[start:end]

                for _, r in batch.iterrows():
                    if session.requests_remaining <= 0:
                        break

                    gid = r["contest_id"]
                    if pd.isna(gid):
                        continue

                    hit_df, pit_df = scrape_game_lineups(session, int(gid), div, year)

                    if not hit_df.empty:
                        rows_hit.append(hit_df)
                    if not pit_df.empty:
                        rows_pit.append(pit_df)

                    print(f"game {int(gid)}: hit={len(hit_df)} pit={len(pit_df)}")

                print(f"batch {start+1}-{end} done (budget: {session.requests_remaining})")

                if rows_hit:
                    new_hit = pd.concat(rows_hit, ignore_index=True)
                    out_hit = pd.concat([existing_hit, new_hit], ignore_index=True) if not existing_hit.empty else new_hit
                    out_hit = _dedupe_hit(out_hit)
                    fpath = outdir / f"d{div}_batting_lineups_{year}.csv"
                    _atomic_write_csv(out_hit, fpath)

                if rows_pit:
                    new_pit = pd.concat(rows_pit, ignore_index=True)
                    out_pit = pd.concat([existing_pit, new_pit], ignore_index=True) if not existing_pit.empty else new_pit
                    out_pit = _dedupe_pit(out_pit)
                    fpath = outdir / f"d{div}_pitching_lineups_{year}.csv"
                    _atomic_write_csv(out_pit, fpath)

            if rows_hit:
                new_hit = pd.concat(rows_hit, ignore_index=True)
                out_hit = pd.concat([existing_hit, new_hit], ignore_index=True) if not existing_hit.empty else new_hit
                out_hit = _dedupe_hit(out_hit)
                fpath = outdir / f"d{div}_batting_lineups_{year}.csv"
                _atomic_write_csv(out_hit, fpath)
                print(f"saved {fpath} ({len(out_hit)} rows)")

            if rows_pit:
                new_pit = pd.concat(rows_pit, ignore_index=True)
                out_pit = pd.concat([existing_pit, new_pit], ignore_index=True) if not existing_pit.empty else new_pit
                out_pit = _dedupe_pit(out_pit)
                fpath = outdir / f"d{div}_pitching_lineups_{year}.csv"
                _atomic_write_csv(out_pit, fpath)
                print(f"saved {fpath} ({len(out_pit)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--indir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/lineups")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    scrape_lineups(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions,
        batch_size=args.batch_size,
        headless=args.headless,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
    )
