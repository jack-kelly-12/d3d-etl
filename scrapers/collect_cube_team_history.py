import argparse
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.thebaseballcube.com/content/college_seasons"
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _normalize_division(raw: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()


def _scrape_season(session: requests.Session, year: int) -> list[dict]:
    url = f"{BASE_URL}/{year}/"
    print(f"[{year}] fetching {url}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict] = []

    current_division = None
    for el in soup.find_all(["b", "div"]):
        if el.name == "b":
            text = el.get_text(strip=True)
            if text:
                current_division = _normalize_division(text)
            continue

        if current_division is None:
            continue

        links = el.find_all("a")
        if not links:
            continue

        college_link = None
        conf_link = None
        for a in links:
            if a.parent != el:
                continue
            href = a.get("href", "")
            if "stats_college" in href:
                college_link = a
            elif "college_summary" in href or "blacklink" in a.get("class", []):
                conf_link = a

        if college_link is None:
            continue

        college_name = college_link.get_text(strip=True)
        href = college_link.get("href", "")
        m = re.search(r"~(\d+)", href)
        college_id = int(m.group(1)) if m else None

        conference = conf_link.get_text(strip=True) if conf_link else ""

        rows.append({
            "division": current_division,
            "college_id": college_id,
            "college_name": college_name,
            "conference": conference,
            "year": year,
        })

    print(f"  found {len(rows)} teams across divisions")
    return rows


def scrape_cube_team_history(
    out_file: str,
    years: list[int] | None = None,
):
    if years is None:
        years = list(YEARS)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: list[dict] = []
    try:
        for year in years:
            rows = _scrape_season(session, year)
            all_rows.extend(rows)
            time.sleep(2.0)
    finally:
        session.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_file, index=False)
    print(f"\n[done] saved {len(df)} rows to {out_file}")

    for div in sorted(df["division"].unique()):
        count = len(df[df["division"] == div])
        print(f"  {div}: {count} team-years")


if __name__ == "__main__":
    print("[start] scrapers.collect_cube_team_history", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_file",
        default="/Users/jackkelly/Desktop/d3d-etl/data/cube_team_history.csv",
    )
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()

    scrape_cube_team_history(
        out_file=args.out_file,
        years=args.years,
    )
