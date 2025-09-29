import asyncio
import re
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import aiohttp
from pathlib import Path
from typing import List, Tuple, Optional

DIRECTORY_API = "https://web3.ncaa.org/directory/api/directory/memberList?type=12&sportCode=MBA"

async def fetch_school_list():
    async with aiohttp.ClientSession() as session:
        async with session.get(DIRECTORY_API) as resp:
            resp.raise_for_status()
            return await resp.json()

def ensure_https(base: str) -> str:
    if not base:
        return ""
    base = base.strip()
    if not base.startswith("http"):
        base = "https://" + base
    return base.rstrip("/")

def presto_season_slug(season: int) -> str:
    prev = season - 1
    yy = str(season)[-2:]  # 2026 -> "26"
    return f"{prev}-{yy}"

def build_candidate_urls(base: str, season: int) -> List[Tuple[str, str]]:
    """
    Returns [(url, hint)], where hint is the URL family we tried.
    Order matters: try Sidearm first, then Presto.
    """
    base = ensure_https(base)
    return [
        (f"{base}/sports/baseball/roster/{season}", "sidearm"),
        (f"{base}/sports/bsb/{presto_season_slug(season)}/roster", "presto"),
    ]

def detect_cms(html: str) -> Optional[str]:
    """
    Lightweight detection based on distinctive assets/classes.
    Returns "sidearm", "presto", or None.
    """
    h = html.lower()
    if "presto-sport-static" in h or "theme-assets.prestosports.com" in h:
        return "presto"
    if "class=\"sidearm-" in h or "sidearm-" in h:
        return "sidearm"
    return None


def _txt(el):
    return el.get_text(strip=True) if el else None

def parse_sidearm(html: str, team: str, season: int, url: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for p in soup.select(".sidearm-roster-player"):
        name_el = p.select_one(".sidearm-roster-player-name a, .sidearm-roster-player-name")
        number_el = p.select_one(".sidearm-roster-player-jersey-number")
        pos_el = p.select_one(".sidearm-roster-player-position-long-short, .sidearm-roster-player-position")
        height_el = p.select_one(".sidearm-roster-player-height")
        weight_el = p.select_one(".sidearm-roster-player-weight")
        year_el = p.select_one(".sidearm-roster-player-academic-year")
        hometown_el = p.select_one(".sidearm-roster-player-hometown")
        hs_el = p.select_one(".sidearm-roster-player-highschool")
        prev_el = p.select_one(".sidearm-roster-player-previous-school")
        bt_el = p.select_one(".sidearm-roster-player-custom1, .sidearm-roster-player-bats-throws")

        img_el = p.select_one("img")
        img_url = None
        if img_el:
            src = img_el.get("data-src") or img_el.get("src")
            if src:
                img_url = src.split("?")[0]

        rows.append({
            "team": team,
            "year": season,
            "roster_url": url,
            "name": name_el.get_text(strip=True) if name_el else None,
            "number": number_el.get_text(strip=True) if number_el else None,
            "position": pos_el.get_text(strip=True) if pos_el else None,
            "height": height_el.get_text(strip=True) if height_el else None,
            "weight": weight_el.get_text(strip=True) if weight_el else None,
            "class": year_el.get_text(strip=True) if year_el else None,
            "b_t": bt_el.get_text(strip=True) if bt_el else None,
            "hometown": hometown_el.get_text(strip=True) if hometown_el else None,
            "highschool": hs_el.get_text(strip=True) if hs_el else None,
            "previous_school": prev_el.get_text(strip=True) if prev_el else None,
            "img_url": img_url,
        })

    if not rows:
        for tr in soup.select("table.sidearm-table tbody tr"):
            name_el = tr.select_one(".sidearm-table-player-name a, .sidearm-table-player-name")
            num_el = tr.select_one(".roster_jerseynum")
            year_el = tr.select_one(".roster_class")
            pos_el = tr.select_one(".rp_position_short")
            bt_el = tr.select_one(".sidearm-table-custom-field, .rp_custom1")
            ht_el = tr.select_one(".height")
            wt_el = tr.select_one(".rp_weight")
            home_el = tr.select_one(".hometownhighschool")
            prev_el = tr.select_one(".player_previous_school")

            img_el = tr.select_one("img")
            img_url = None
            if img_el:
                src = img_el.get("data-src") or img_el.get("src")
                if src:
                    img_url = src.split("?")[0]

            rows.append({
                "team": team,
                "year": season,
                "roster_url": url,
                "name": name_el.get_text(strip=True) if name_el else None,
                "number": num_el.get_text(strip=True) if num_el else None,
                "position": pos_el.get_text(strip=True) if pos_el else None,
                "height": ht_el.get_text(strip=True) if ht_el else None,
                "weight": wt_el.get_text(strip=True) if wt_el else None,
                "class": year_el.get_text(strip=True) if year_el else None,
                "b_t": bt_el.get_text(strip=True) if bt_el else None,
                "hometown": home_el.get_text(strip=True) if home_el else None,
                "highschool": None,
                "previous_school": prev_el.get_text(strip=True) if prev_el else None,
                "img_url": img_url,
            })

    return rows

def parse_presto(html: str, team: str, season: int, url: str) -> List[dict]:
    """
    Presto templates vary a bit, so we try a few common patterns.
    Fallback: find player links containing '/players/'.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # Heuristic 1: Card/list items commonly used by Presto themes
    blocks = soup.select(
        ".ps-roster__item, .ps_roster__item, .roster__player, .roster-player, "
        "li[class*='roster'], .athlete, .person, .card:has(a[href*='/players/'])"
    )
    if not blocks:
        # Heuristic 2: any anchor that looks like a player bio link
        cand_links = soup.select("a[href*='/players/']")
        for a in cand_links:
            name = _txt(a)
            if not name:
                continue
            # try to find an image nearby (parent card/list item)
            parent = a
            for _ in range(3):
                parent = parent.parent
                if parent is None:
                    break
                img_el = parent.select_one("img")
                if img_el:
                    img = img_el.get("data-src") or img_el.get("src")
                    img = img.split("?")[0] if img else None
                    break
            else:
                img = None

            rows.append({
                "team": team,
                "year": season,
                "roster_url": url,
                "name": name,
                "number": None,
                "position": None,
                "height": None,
                "weight": None,
                "class": None,
                "b_t": None,
                "hometown": None,
                "highschool": None,
                "previous_school": None,
                "img_url": img,
            })
        return rows

    for b in blocks:
        name_el = b.select_one("a[href*='/players/'], .name, .person__name, .athlete__name, h3, h4")
        pos_el = b.select_one(".position, .pos, .athlete__position, .roster__position")
        number_el = b.select_one(".number, .no, .roster__number")
        height_el = b.select_one(".height, .ht, .roster__height")
        weight_el = b.select_one(".weight, .wt, .roster__weight")
        year_el = b.select_one(".year, .class, .academic-year, .roster__year")
        bt_el = b.select_one(".bats-throws, .bats_throws, .bt, .roster__bats-throws")
        hometown_el = b.select_one(".hometown, .home-town, .roster__hometown")
        hs_el = b.select_one(".highschool, .roster__highschool, .hs")
        prev_el = b.select_one(".previous-school, .prev-school, .roster__previous-school")

        img_el = b.select_one("img")
        img_url = None
        if img_el:
            src = img_el.get("data-src") or img_el.get("src")
            if src:
                img_url = src.split("?")[0]

        name = _txt(name_el)
        if not name:
            continue

        rows.append({
            "team": team,
            "year": season,
            "roster_url": url,
            "name": name,
            "number": _txt(number_el),
            "position": _txt(pos_el),
            "height": _txt(height_el),
            "weight": _txt(weight_el),
            "class": _txt(year_el),
            "b_t": _txt(bt_el),
            "hometown": _txt(hometown_el),
            "highschool": _txt(hs_el),
            "previous_school": _txt(prev_el),
            "img_url": img_url,
        })
    return rows

async def fetch_html(browser, url: str) -> Tuple[Optional[str], int]:
    page = await browser.new_page()
    try:
        resp = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        status = resp.status if resp else 0
        html = await page.content()
        await page.close()
        return html, status
    except Exception:
        await page.close()
        return None, 0

async def scrape_team(browser, base: str, team_name: str, season: int) -> List[dict]:
    candidates = build_candidate_urls(base, season)
    for url, hint in candidates:
        print(f"🔎 {team_name}: trying {hint} → {url}")
        html, status = await fetch_html(browser, url)
        if not html or status >= 400:
            print(f"  ❌ {team_name}: HTTP {status} or no HTML on {hint}")
            continue

        cms = detect_cms(html) or hint
        if cms == "sidearm":
            rows = parse_sidearm(html, team_name, season, url)
        elif cms == "presto":
            rows = parse_presto(html, team_name, season, url)
        else:
            rows = parse_sidearm(html, team_name, season, url)
            if not rows:
                rows = parse_presto(html, team_name, season, url)

        if rows:
            print(f"  ✅ {team_name}: {len(rows)} players ({cms})")
            return rows
        else:
            print(f"  ⚠️ {team_name}: no players parsed on {hint}; trying next…")

    print(f"  🚫 {team_name}: no working roster URL")
    return []

async def main(season=2026, limit=None, batch_size=10, missing_only=False, outdir="../data"):
    schools = await fetch_school_list()

    roster_targets = []
    for s in schools:
        base = ensure_https(s.get("athleticWebUrl"))
        if not base:
            continue
        roster_targets.append((base, s["nameOfficial"]))

    if limit:
        roster_targets = roster_targets[:limit]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"team_headshots_{season}.csv"

    skip = set()
    if missing_only and outpath.exists():
        existing = pd.read_csv(outpath)
        done = existing.loc[existing["season"] == season, "team"].dropna().unique().tolist()
        skip = set(done)
        print(f"⏭️  Skipping {len(skip)} teams already saved for {season}")

    filtered = [(b, t) for (b, t) in roster_targets if t not in skip]
    print(f"🎯 {len(filtered)} teams to scrape for {season}")

    all_players = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for i in range(0, len(filtered), batch_size):
            batch = filtered[i:i+batch_size]
            tasks = [scrape_team(browser, base, team, season) for base, team in batch]
            results = await asyncio.gather(*tasks)
            for rows in results:
                all_players.extend(rows)

        await browser.close()

    df_new = pd.DataFrame(all_players)

    if outpath.exists() and missing_only:
        existing = pd.read_csv(outpath)
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(outpath, index=False)
    print(f"📦 Saved {len(df)} rows across {df['team'].nunique() if not df.empty else 0} teams → {outpath}")
    return df

if __name__ == "__main__":
    for season in range(2021, 2026):
        df = asyncio.run(main(season=season, batch_size=10, missing_only=True, outdir="../data"))
