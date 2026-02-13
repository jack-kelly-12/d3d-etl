import argparse
import html
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .constants import BASE
from .scraper_utils import HardBlockError, ScraperConfig, ScraperSession


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def is_hard_block(status: int) -> bool:
    return status == 403


def looks_blocked(html_text: str) -> bool:
    t = (html_text or "").lower()
    pats = [
        "access denied",
        "request rejected",
        "forbidden",
        "service unavailable",
        "unusual traffic",
        "blocked",
    ]
    return any(p in t for p in pats)


def get_schedules(indir, div, year):
    fpath = Path(indir) / f"d{div}_schedules_{year}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, dtype={"contest_id": "Int64"})
        if "contest_id" in df.columns:
            df = df.drop_duplicates(subset=["contest_id"])
            df = df[df['game_url'].str.contains('box_score')]
            return df
    return pd.DataFrame()


def _parse_team_header_text(card_header_el) -> Tuple[Optional[int], str]:
    a = card_header_el.query_selector("a[href*='/teams/']")
    team_id = None
    if a:
        href = a.get_attribute("href") or ""
        m = re.search(r"/teams/(\d+)", href)
        if m:
            team_id = int(m.group(1))
    header_text = (card_header_el.inner_text() or "").strip()
    return team_id, header_text


def _is_indented_sub(td_html: str) -> bool:
    if td_html is None:
        return False
    return ("&nbsp;" in td_html) or ("\u00a0" in td_html)


def _extract_player_link_data(name_td_el) -> Optional[Tuple[int, str]]:
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


def _dedupe_hit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["division", "year", "contest_id", "team_id", "ncaa_id", "is_sub", "jersey", "position"] if c in df.columns]
    if not key:
        return df
    return df.drop_duplicates(subset=key, keep="last")


def _dedupe_pit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["division", "year", "contest_id", "team_id", "ncaa_id", "is_sub", "jersey"] if c in df.columns]
    if not key:
        return df
    return df.drop_duplicates(subset=key, keep="last")


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in ("team_id", "ncaa_id", "contest_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "jersey" in df.columns:
        df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    if "is_sub" in df.columns:
        df["is_sub"] = df["is_sub"].astype(bool)
    return df


def _existing_contest_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["contest_id"], dtype={"contest_id": "Int64"})
        if "contest_id" not in df.columns:
            return set()
        s = pd.to_numeric(df["contest_id"], errors="coerce").dropna().astype(int)
        return set(s.tolist())
    except Exception:
        return set()


def completed_game_ids_from_csv(hit_out: Path, pit_out: Path) -> Tuple[Set[int], Set[int], Set[int]]:
    hit_done = _existing_contest_ids(hit_out)
    pit_done = _existing_contest_ids(pit_out)
    both_done = hit_done & pit_done
    return hit_done, pit_done, both_done


def _add_is_starter(pit_df: pd.DataFrame) -> pd.DataFrame:
    if pit_df.empty:
        return pit_df

    df = pit_df.copy()

    if "is_starter" in df.columns:
        df = df.drop(columns=["is_starter"])

    df["is_starter"] = 0
    df["_row"] = range(len(df))

    starters = (
        df.sort_values(["contest_id", "team_id", "_row"])
        .groupby(["contest_id", "team_id"], as_index=False)
        .head(1)[["_row"]]
    )

    if not starters.empty:
        df.loc[starters["_row"].values, "is_starter"] = 1

    df = df.drop(columns=["_row"])
    return df


def scrape_game_lineups(
    session: ScraperSession,
    contest_id: int,
    div: int,
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, str]:
    url = f"{BASE}/contests/{contest_id}/individual_stats"
    html_content, status = session.fetch(url, wait_selector="div.card", wait_timeout=20000)

    if not html_content:
        return pd.DataFrame(), pd.DataFrame(), status, ""
    if looks_blocked(html_content):
        return pd.DataFrame(), pd.DataFrame(), 403, html_content
    if status >= 400:
        return pd.DataFrame(), pd.DataFrame(), status, html_content

    page = session.page
    cards = page.query_selector_all("div.card")
    if not cards:
        return pd.DataFrame(), pd.DataFrame(), status, html_content

    hit_rows: List[Dict[str, Any]] = []
    pit_rows: List[Dict[str, Any]] = []

    for card in cards:
        header = card.query_selector("div.card-header")
        body = card.query_selector("div.card-body")
        tbl = body.query_selector("table") if body else None
        if not header or not tbl:
            continue

        team_id, header_text = _parse_team_header_text(header)
        header_lower = header_text.lower()

        is_hitting = ("hitting" in header_lower) or ("batting" in header_lower)
        is_pitching = ("pitching" in header_lower)
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

            jersey = None
            if idx_num is not None and idx_num < len(tds):
                raw_num = (tds[idx_num].inner_text() or "").strip()
                raw_num = raw_num.replace("\u00a0", "").strip()
                if raw_num != "":
                    mnum = re.search(r"\d+", raw_num)
                    jersey = int(mnum.group(0)) if mnum else None

            pos = None
            if is_hitting and idx_pos is not None and idx_pos < len(tds):
                pos = (tds[idx_pos].inner_text() or "").strip()
                pos = pos.replace("\u00a0", "").strip() or None

            base = {
                "division": div,
                "year": year,
                "contest_id": int(contest_id),
                "team_id": team_id,
                "ncaa_id": ncaa_id,
                "player_name": player_name,
                "is_sub": is_sub,
            }

            if is_hitting:
                hit_rows.append({**base, "jersey": jersey, "position": pos})
            else:
                pit_rows.append({**base, "jersey": jersey})

    hit_df = _normalize_types(pd.DataFrame(hit_rows))
    pit_df = _normalize_types(pd.DataFrame(pit_rows))
    pit_df = _add_is_starter(pit_df)

    return hit_df, pit_df, status, html_content


def scrape_lineups(
    indir: str,
    outdir: str,
    year: int,
    divisions: List[int],
    missing_only: bool = False,
    batch_size: int = 25,
    base_delay: float = 10.0,
    daily_budget: int = 20000,
    batch_cooldown_s: int = 90,
):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    division_jobs = []
    for div in divisions:
        sched = get_schedules(indir, div, year)
        if sched.empty:
            print(f"no schedule for d{div} {year}")
            continue

        hit_out = outdir_path / f"d{div}_batting_lineups_{year}.csv"
        pit_out = outdir_path / f"d{div}_pitching_lineups_{year}.csv"
        hit_done, pit_done, both_done = completed_game_ids_from_csv(hit_out, pit_out)

        all_games = sched["contest_id"].tolist()
        games = [gid for gid in all_games if gid not in both_done] if missing_only else all_games

        total_games = len(all_games)
        remaining = len(games)

        division_jobs.append(
            {
                "div": div,
                "hit_out": hit_out,
                "pit_out": pit_out,
                "games": games,
                "total_games": total_games,
                "remaining": remaining,
                "both_done_count": len(both_done),
                "hit_done_count": len(hit_done),
                "pit_done_count": len(pit_done),
            }
        )

    for job in division_jobs:
        print(
            f"\n=== d{job['div']} {year} lineups — total {job['total_games']} | done {job['both_done_count']} | remaining {job['remaining']} ==="
        )
        print(
            f"    (hit done: {job['hit_done_count']} | pit done: {job['pit_done_count']})"
        )

    if not any(job["remaining"] > 0 for job in division_jobs):
        return

    config = ScraperConfig(
        base_delay=base_delay,
        block_resources=False,
        daily_request_budget=daily_budget,
        jitter_pct=0.4,
    )

    with ScraperSession(config) as session:
        try:
            for job in division_jobs:
                div = job["div"]
                hit_out = job["hit_out"]
                pit_out = job["pit_out"]
                games = job["games"]
                remaining = job["remaining"]

                if remaining <= 0:
                    continue

                hit_done, pit_done, both_done = completed_game_ids_from_csv(hit_out, pit_out)
                print(f"    (budget remaining: {session.requests_remaining})")

                for batch_start in range(0, remaining, batch_size):
                    if session.requests_remaining <= 0:
                        print("[budget] daily request budget exhausted, stopping")
                        break

                    batch = games[batch_start:batch_start + batch_size]
                    batch_end = batch_start + len(batch)

                    print(f"\n[batch] games {batch_start+1}-{batch_end} (budget: {session.requests_remaining})")

                    hit_done, pit_done, both_done = completed_game_ids_from_csv(hit_out, pit_out)

                    for gid in batch:
                        if session.requests_remaining <= 0:
                            break

                        if missing_only and gid in both_done:
                            continue

                        print(f"\n[game] {gid}")

                        hit_df, pit_df, status, html_content = scrape_game_lineups(session, gid, div, year)

                        if is_hard_block(status):
                            print("[STOP] got 403 (hard block). Stopping for safety (CSV progress preserved).")
                            return

                        if status >= 400:
                            print(f"  failed: HTTP {status}")
                            continue

                        wrote_hit = 0
                        wrote_pit = 0

                        if gid not in hit_done and not hit_df.empty:
                            hit_df = _dedupe_hit(hit_df)
                            append_csv(hit_out, hit_df)
                            wrote_hit = len(hit_df)
                            hit_done.add(gid)

                        if gid not in pit_done and not pit_df.empty:
                            pit_df = _dedupe_pit(pit_df)
                            append_csv(pit_out, pit_df)
                            wrote_pit = len(pit_df)
                            pit_done.add(gid)

                        if gid in hit_done and gid in pit_done:
                            both_done.add(gid)

                        print(f"  ok: hit={wrote_hit} pit={wrote_pit} (budget: {session.requests_remaining})")

                    if batch_end < remaining:
                        cooldown = int(batch_cooldown_s)
                        print(f"\n[cooldown] sleeping {cooldown}s before next batch")
                        time.sleep(cooldown)

                print(f"\n[done] division d{div} {year}")
                print(f"  batting file:  {hit_out}")
                print(f"  pitching file: {pit_out}")
        except HardBlockError as exc:
            print(str(exc))
            print("[STOP] hard block detected. Stopping scraper (CSV progress preserved).")
            return


if __name__ == "__main__":
    print("[start] scrapers.collect_lineups", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--missing_only", action="store_true", help="Only scrape contests missing from both lineup outputs")
    parser.add_argument("--indir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/lineups")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_lineups(
        indir=args.indir,
        outdir=args.outdir,
        year=args.year,
        divisions=args.divisions,
        missing_only=args.missing_only,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        daily_budget=args.daily_budget,
        batch_cooldown_s=args.batch_cooldown_s,
    )
