import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

DIVISION_URLS = {
    1: "https://thebaseballcube.com/content/schools/ncaa-1/",
    2: "https://thebaseballcube.com/content/schools/ncaa-2/",
    3: "https://thebaseballcube.com/content/schools/ncaa-3/",
}


def abs_url(page_url, href):
    if not href:
        return ""
    return urljoin(page_url, href)


def scrape_division(division: int, url: str) -> pd.DataFrame:
    print(f"\n--- Scraping Division {division}: {url} ---")
    scraper = cloudscraper.create_scraper()
    scraper.headers.update(HEADERS)
    resp = scraper.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "grid1"})
    if table is None:
        print(f"  [!] Could not find #grid1 table. The page may require a login.")
        print(f"  [!] Page snippet:\n{soup.get_text()[:500]}")
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells or "header-row" in tr.get("class", []):
            continue
        if "record(s)" in tr.get_text(strip=True).lower():
            continue

        college_td = cells[0]
        college_a = college_td.find("a")
        college_name = college_a.get_text(strip=True) if college_a else college_td.get_text(strip=True)
        college_url = abs_url(url, college_a["href"]) if (college_a and college_a.get("href")) else ""

        nick_td = cells[1] if len(cells) > 1 else None
        nick_a = nick_td.find("a") if nick_td else None
        nickname = nick_a.get_text(strip=True) if nick_a else (nick_td.get_text(strip=True) if nick_td else "")

        conf_td = cells[2] if len(cells) > 2 else None
        conf_a = conf_td.find("a") if conf_td else None
        conference = conf_a.get_text(strip=True) if conf_a else (conf_td.get_text(strip=True) if conf_td else "")
        conference_url = abs_url(url, conf_a["href"]) if (conf_a and conf_a.get("href")) else ""

        loc_td = cells[3] if len(cells) > 3 else None
        loc_a = loc_td.find("a") if loc_td else None
        location = loc_a.get_text(strip=True) if loc_a else (loc_td.get_text(strip=True) if loc_td else "")

        full_name_td = cells[4] if len(cells) > 4 else None
        full_name = full_name_td.get_text(strip=True) if full_name_td else ""

        year_td = cells[5] if len(cells) > 5 else None
        year_a = year_td.find("a") if year_td else None
        year = year_a.get_text(strip=True) if year_a else (year_td.get_text(strip=True) if year_td else "")
        year_url = abs_url(url, year_a["href"]) if (year_a and year_a.get("href")) else ""

        rows.append({
            "division": division,
            "college": college_name,
            "college_url": college_url,
            "nickname": nickname,
            "conference": conference,
            "conference_url": conference_url,
            "location": location,
            "full_name": full_name,
            "latest_year": year,
            "year_stats_url": year_url,
        })

    df = pd.DataFrame(rows)
    print(f"  ✓ Found {len(df)} schools")
    return df

if __name__ == "__main__":
    all_frames = []

    OUT_FILE = "/Users/jackkelly/Desktop/d3d-etl/data/baseball_cube_schools.csv"

    for div, url in DIVISION_URLS.items():
        df_div = scrape_division(div, url)
        all_frames.append(df_div)
        time.sleep(1.5)

    df_all = pd.concat(all_frames, ignore_index=True)

    print(f"\n{'='*50}")
    print(f"Total schools scraped: {len(df_all)}")
    print(f"Breakdown by division:\n{df_all['division'].value_counts().sort_index().to_string()}")

    df_all.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to: baseball_cube_schools.csv")