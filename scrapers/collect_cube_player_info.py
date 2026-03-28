import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

from scripts.hash_player_ids import SALT, hash_player_id

BASE_URL = "https://thebaseballcube.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

COLUMNS = [
    "cube_player_id",
    "player_id",
    "proper_name",
    "high_level",
    "years",
    "positions",
    "height",
    "weight",
    "bats",
    "throws",
    "place",
    "high_school",
    "high_school_id",
    "high_school_location",
    "colleges",
    "college_ids",
]


def _collect_player_ids(
    data_dir: Path,
    divisions: list[str] | None = None,
    years: list[int] | None = None,
) -> list[int]:
    ids: set[int] = set()
    cube_stats_dir = data_dir / "cube_stats"
    if not cube_stats_dir.exists():
        cube_stats_dir = data_dir
    for csv_file in sorted(cube_stats_dir.glob("*.csv")):
        if csv_file.name == "cube_player_info.csv":
            continue
        if divisions is not None or years is not None:
            m = re.match(r"^(.+?)_(batting|pitching)_(\d{4})\.csv$", csv_file.name)
            if not m:
                continue
            file_div, _, file_year = m.group(1), m.group(2), int(m.group(3))
            if divisions is not None and file_div not in divisions:
                continue
            if years is not None and file_year not in years:
                continue
        for col in ("cube_player_id", "player_id"):
            try:
                df = pd.read_csv(csv_file, usecols=[col], dtype=str)
                numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                ids.update(numeric.astype(int).unique())
                break
            except Exception:
                continue
    return sorted(ids)


def _done_player_ids(out_path: Path) -> tuple[set[int], set[str]]:
    """Return (done cube_player_ids, done player_ids) already in the output file.

    Checks both columns so stubs added by reconcile_players (which have
    cube_player_id=NaN but a valid player_id) are recognised as already done.
    """
    if not out_path.exists():
        return set(), set()
    try:
        df = pd.read_csv(out_path, usecols=["cube_player_id", "player_id"], dtype=str)
        cube_ids = set(pd.to_numeric(df["cube_player_id"], errors="coerce").dropna().astype(int))
        player_ids = set(df["player_id"].dropna().unique())
        return cube_ids, player_ids
    except Exception:
        return set(), set()


def _extract_college_id(href: str) -> str | None:
    if not href:
        return None
    m = re.search(r"/college_history/(\d+)", href)
    return m.group(1) if m else None


def _empty_row(player_id: int) -> dict:
    row = {col: None for col in COLUMNS}
    row["cube_player_id"] = player_id
    return row


def _parse_player_page(soup: BeautifulSoup, player_id: int) -> dict:
    row = _empty_row(player_id)

    field_map = {
        "High Level": "high_level",
        "Years": "years",
        "Proper Name": "player_name",
        "Positions": "positions",
        "Height / Weight": "height_weight",
        "Bats / Throws": "b_t",
        "Place": "hometown",
        "High School": "high_school",
        "Colleges": "colleges",
    }

    pi_info = soup.find("div", class_="pi-info")
    if not pi_info:
        return row

    for pi_row in pi_info.find_all("div", class_="pi-row"):
        subject_div = pi_row.find("div", class_="pi-subject")
        value_div = pi_row.find("div", class_="pi-value")
        if not subject_div or not value_div:
            continue

        label = subject_div.get_text(strip=True)
        col = field_map.get(label)
        if col is None:
            continue

        if col == "colleges":
            links = value_div.find_all("a", href=re.compile(r"college_history"))
            names = [a.get_text(strip=True) for a in links]
            college_ids = [_extract_college_id(a.get("href", "")) for a in links]
            college_ids = [c for c in college_ids if c]
            row["colleges"] = ";".join(names)
            row["college_ids"] = ";".join(college_ids)
        elif col == "high_school":
            a = value_div.find("a")
            hs_name = a.get_text(strip=True) if a else value_div.get_text(strip=True)
            hs_id = None
            if a and a.get("href"):
                m = re.search(r"/hs_team/(\d+)", a["href"])
                hs_id = m.group(1) if m else None
            paren = value_div.get_text(strip=True)
            m = re.search(r"\((.+?)\)", paren)
            hs_location = m.group(1) if m else ""
            row["high_school"] = hs_name
            row["high_school_id"] = hs_id
            row["high_school_location"] = hs_location
        elif col == "height_weight":
            text = value_div.get_text(strip=True)
            parts = [p.strip() for p in text.split("/")]
            row["height"] = parts[0] if len(parts) > 0 else ""
            row["weight"] = parts[1] if len(parts) > 1 else ""
        elif col == "bats_throws":
            text = value_div.get_text(strip=True)
            parts = [p.strip() for p in text.split("/")]
            row["bats"] = parts[0] if len(parts) > 0 else ""
            row["throws"] = parts[1] if len(parts) > 1 else ""
        elif col == "place":
            a = value_div.find("a")
            row["place"] = a.get_text(strip=True) if a else value_div.get_text(strip=True)
        else:
            row[col] = value_div.get_text(strip=True)

    return row


def _fetch_player(session, player_id: int) -> dict | None:
    url = f"{BASE_URL}/content/player/{player_id}/"
    try:
        resp = session.get(url, timeout=20)
    except Exception as e:
        print(f"  ERROR player {player_id}: {e}")
        return None

    if resp.status_code in (403, 429):
        return {"__rate_limited": True, "cube_player_id": player_id}
    if resp.status_code >= 400:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    return _parse_player_page(soup, player_id)


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    df = df[COLUMNS]
    if path.exists() and path.stat().st_size > 0:
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def scrape_cube_player_info(
    data_dir: str,
    out_file: str,
    run_remaining: bool = False,
    workers: int = 8,
    divisions: list[str] | None = None,
    years: list[int] | None = None,
):
    data_path = Path(data_dir)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_ids = _collect_player_ids(data_path, divisions=divisions, years=years)
    filter_desc = ""
    if divisions:
        filter_desc += f" divisions={divisions}"
    if years:
        filter_desc += f" years={years}"
    print(f"[info] {len(all_ids)} unique player_ids found across data files{filter_desc}")

    if run_remaining:
        done_cube, done_player = _done_player_ids(out_path)
        ids = [
            pid for pid in all_ids
            if pid not in done_cube and hash_player_id(pid, SALT) not in done_player
        ]
        print(f"[info] {len(done_cube)} already scraped, {len(ids)} remaining")
    else:
        ids = all_ids

    if not ids:
        print("[done] no remaining players")
        return

    total = len(ids)
    print(f"[info] scraping {total} players with {workers} threads")

    def _new_session():
        s = cloudscraper.create_scraper()
        s.headers.update(HEADERS)
        return s

    state = {
        "session": _new_session(),
        "count": 0,
    }
    write_lock = threading.Lock()
    pause_lock = threading.Lock()
    pause_until = {"time": 0.0}

    def _process(player_id: int) -> None:
        while True:
            wait_secs = pause_until["time"] - time.time()
            if wait_secs > 0:
                time.sleep(wait_secs + 0.1)

            row = _fetch_player(state["session"], player_id)
            if row is None:
                return

            if row.get("__rate_limited"):
                with pause_lock:
                    if pause_until["time"] <= time.time():
                        print(
                            f"\n[RATE LIMIT] player {player_id} — recycling session, waiting 30s..."
                        )
                        try:
                            state["session"].close()
                        except Exception:
                            pass
                        state["session"] = _new_session()
                        pause_until["time"] = time.time() + 30
                continue

            row["player_id"] = hash_player_id(player_id, SALT)

            with write_lock:
                _append_csv(out_path, pd.DataFrame([row]))
                state["count"] += 1
                c = state["count"]
                if c % 100 == 0:
                    print(f"[progress] {c}/{total} saved ({c * 100 / total:.1f}%)")
            return

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, pid): pid for pid in ids}
            for f in as_completed(futures):
                f.result()
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        try:
            state["session"].close()
        except Exception:
            pass

    print(f"\n[done] {state['count']}/{total} players saved to {out_path}")


if __name__ == "__main__":
    print("[start] scrapers.collect_cube_player_info", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/Users/jackkelly/Desktop/d3d-etl/data/",
    )
    parser.add_argument(
        "--out_file",
        default="/Users/jackkelly/Desktop/d3d-etl/data/cube_stats/cube_player_info.csv",
    )
    parser.add_argument("--run_remaining", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--divisions", nargs="+", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    args = parser.parse_args()

    scrape_cube_player_info(
        data_dir=args.data_dir,
        out_file=args.out_file,
        run_remaining=args.run_remaining,
        workers=args.workers,
        divisions=args.divisions,
        years=args.years,
    )
