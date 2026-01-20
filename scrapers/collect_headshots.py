import argparse
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from .scraper_utils import BLOCKED_RESOURCE_TYPES, ScraperConfig

OUTDIR = "/Users/jackkelly/Desktop/d3d-etl/data/headshots"


async def fetch_school_list():
    with open("/Users/jackkelly/Desktop/d3d-etl/data/school_websites.json") as f:
        return json.load(f)


def ensure_https(base: str) -> str:
    if not base:
        return ""
    base = base.strip()
    if not base.startswith("http"):
        base = "https://" + base
    return base.rstrip("/")


def presto_season_slug(season: int) -> str:
    prev = season - 1
    yy = str(season)[-2:]
    return f"{prev}-{yy}"


def build_candidate_urls(base: str, season: int) -> list[tuple[str, str]]:
    base = ensure_https(base)
    return [
        (f"{base}/sports/baseball/roster/{season}", "sidearm"),
        (f"{base}/sports/bsb/{presto_season_slug(season)}/roster", "presto"),
    ]


def detect_cms(html: str) -> str | None:
    h = (html or "").lower()
    if "presto-sport-static" in h or "theme-assets.prestosports.com" in h:
        return "presto"
    if "class=\"sidearm-" in h or "sidearm-" in h:
        return "sidearm"
    return None


def _txt(el) -> str | None:
    if not el:
        return None
    s = " ".join(el.get_text(" ", strip=True).split())
    return s if s else None


def _strip_query(url: str | None) -> str | None:
    if not url:
        return None
    return url.split("?")[0]


def _strip_sidearm_crop(url: str | None) -> str | None:
    if not url:
        return None
    if "images.sidearmdev.com/crop" in url and "url=" in url:
        try:
            q = parse_qs(urlparse(url).query)
            raw = q.get("url", [None])[0]
            if raw:
                return unquote(raw)
        except Exception:
            pass
    return _strip_query(url)


def _clean_field(x: str | None) -> str | None:
    if not x:
        return None
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None


def parse_sidearm(html: str, team: str, season: int, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

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
            img_url = _strip_sidearm_crop(src)

        name = _txt(name_el)
        if not name:
            continue

        rows.append({
            "team": team,
            "year": season,
            "roster_url": url,
            "name": name,
            "number": _clean_field(_txt(number_el)),
            "position": _clean_field(_txt(pos_el)),
            "height": _clean_field(_txt(height_el)),
            "weight": _clean_field(_txt(weight_el)),
            "class": _clean_field(_txt(year_el)),
            "b_t": _clean_field(_txt(bt_el)),
            "hometown": _clean_field(_txt(hometown_el)),
            "highschool": _clean_field(_txt(hs_el)),
            "previous_school": _clean_field(_txt(prev_el)),
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
                img_url = _strip_sidearm_crop(src)

            name = _txt(name_el)
            if not name:
                continue

            rows.append({
                "team": team,
                "year": season,
                "roster_url": url,
                "name": name,
                "number": _clean_field(_txt(num_el)),
                "position": _clean_field(_txt(pos_el)),
                "height": _clean_field(_txt(ht_el)),
                "weight": _clean_field(_txt(wt_el)),
                "class": _clean_field(_txt(year_el)),
                "b_t": _clean_field(_txt(bt_el)),
                "hometown": _clean_field(_txt(home_el)),
                "highschool": None,
                "previous_school": _clean_field(_txt(prev_el)),
                "img_url": img_url,
            })

    if not rows:
        for card in soup.select('div[data-test-id="s-person-card-list__root"]'):
            name_el = card.select_one("h3")
            name = _txt(name_el)
            if not name:
                continue

            number = None
            stamp_sr = card.select_one('[data-test-id="s-person-thumbnail__stamp-sr-only-text"]')
            if stamp_sr and stamp_sr.parent:
                raw = _txt(stamp_sr.parent)
                if raw:
                    number = raw.replace("Jersey Number", "").strip()
            if not number:
                stamp_text = card.select_one(".s-stamp__text")
                number = _txt(stamp_text)

            pos_el = card.select_one('[data-test-id="s-person-details__bio-stats-person-position-short"]')
            class_el = card.select_one('[data-test-id="s-person-details__bio-stats-person-title"]')
            height_el = card.select_one('[data-test-id="s-person-details__bio-stats-person-season"]')
            weight_el = card.select_one('[data-test-id="s-person-details__bio-stats-person-weight"]')

            hometown_el = card.select_one('[data-test-id="s-person-card-list__content-location-person-hometown"]')
            hs_el = card.select_one('[data-test-id="s-person-card-list__content-location-person-high-school"]')

            img_el = card.select_one('img[data-test-id="s-image-resized__img"], img')
            img_url = None
            if img_el:
                src = img_el.get("src") or img_el.get("data-src")
                img_url = _strip_sidearm_crop(src)

            rows.append({
                "team": team,
                "year": season,
                "roster_url": url,
                "name": name,
                "number": _clean_field(number),
                "position": _clean_field(_txt(pos_el)),
                "height": _clean_field(_txt(height_el)),
                "weight": _clean_field(_txt(weight_el)),
                "class": _clean_field(_txt(class_el)),
                "b_t": None,
                "hometown": _clean_field(_txt(hometown_el)),
                "highschool": _clean_field(_txt(hs_el)),
                "previous_school": None,
                "img_url": img_url,
            })

    return rows


def parse_presto(html: str, team: str, season: int, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    blocks = soup.select(
        ".ps-roster__item, .ps_roster__item, .roster__player, .roster-player, "
        "li[class*='roster'], .athlete, .person, .card:has(a[href*='/players/'])"
    )
    if not blocks:
        cand_links = soup.select("a[href*='/players/']")
        for a in cand_links:
            name = _txt(a)
            if not name:
                continue

            parent = a
            img = None
            for _ in range(3):
                parent = parent.parent
                if parent is None:
                    break
                img_el = parent.select_one("img")
                if img_el:
                    img = img_el.get("data-src") or img_el.get("src")
                    img = _strip_query(img) if img else None
                    break

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
            img_url = _strip_query(src) if src else None

        name = _txt(name_el)
        if not name:
            continue

        rows.append({
            "team": team,
            "year": season,
            "roster_url": url,
            "name": name,
            "number": _clean_field(_txt(number_el)),
            "position": _clean_field(_txt(pos_el)),
            "height": _clean_field(_txt(height_el)),
            "weight": _clean_field(_txt(weight_el)),
            "class": _clean_field(_txt(year_el)),
            "b_t": _clean_field(_txt(bt_el)),
            "hometown": _clean_field(_txt(hometown_el)),
            "highschool": _clean_field(_txt(hs_el)),
            "previous_school": _clean_field(_txt(prev_el)),
            "img_url": img_url,
        })

    return rows


async def fetch_html(page, url: str, config: ScraperConfig) -> tuple[str | None, int]:
    try:
        resp = await page.goto(url, timeout=config.timeout_ms, wait_until="domcontentloaded")
        status = resp.status if resp else 0

        try:
            await page.wait_for_selector(
                'div[data-test-id="s-person-card-list__root"], .sidearm-roster-player, table.sidearm-table',
                timeout=6000,
            )
        except Exception:
            pass

        html = await page.content()
        return html, status
    except Exception:
        return None, 0


async def scrape_team(browser, base: str, team_name: str, season: int, config: ScraperConfig) -> list[dict]:
    candidates = build_candidate_urls(base, season)

    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
    )
    page = await context.new_page()

    if config.block_resources:
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in BLOCKED_RESOURCE_TYPES
            else route.continue_()
        ))

    try:
        for url, hint in candidates:
            print(f"  {team_name}: trying {hint}")
            html, status = await fetch_html(page, url, config)
            if not html or status >= 400:
                print(f"    {team_name}: HTTP {status} on {hint}")
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
                print(f"    {team_name}: {len(rows)} players ({cms})")
                return rows
            else:
                print(f"    {team_name}: no players parsed on {hint}")

        print(f"    {team_name}: no working roster URL")
        return []
    finally:
        await page.close()
        await context.close()


async def main(
    season=2026,
    limit=None,
    batch_size=5,
    missing_only=False,
    outdir=OUTDIR,
    team_name=None,
    base_delay=3.0,
):
    schools = await fetch_school_list()

    roster_targets = []
    for s in schools:
        base = ensure_https(s.get("athleticWebUrl"))
        if not base:
            continue
        roster_targets.append((base, s["nameOfficial"]))

    if team_name:
        roster_targets = [(b, t) for (b, t) in roster_targets if t == team_name]
        if not roster_targets:
            print(f"Team '{team_name}' not found in school list")
            return pd.DataFrame()
    elif limit:
        roster_targets = roster_targets[:limit]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"team_headshots_{season}.csv"

    skip = set()
    if missing_only and outpath.exists():
        existing = pd.read_csv(outpath)
        existing["year"] = pd.to_numeric(existing["year"], errors="coerce")
        done = existing.loc[existing["year"] == int(season), "team"].dropna().unique().tolist()
        skip = set(done)
        print(f"Skipping {len(skip)} teams already saved for {season}")

    filtered = [(b, t) for (b, t) in roster_targets if t not in skip]
    print(f"{len(filtered)} teams to scrape for {season}")

    config = ScraperConfig(
        base_delay=base_delay,
        headless=True,
        block_resources=True,
    )

    all_players = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=config.headless)

        for i in range(0, len(filtered), batch_size):
            batch = filtered[i:i + batch_size]
            print(f"\nBatch {i//batch_size + 1}: {len(batch)} teams")

            tasks = [scrape_team(browser, base, team, season, config) for base, team in batch]
            results = await asyncio.gather(*tasks)
            for rows in results:
                all_players.extend(rows)

            if i + batch_size < len(filtered):
                await asyncio.sleep(base_delay)

        await browser.close()

    df_new = pd.DataFrame(all_players)

    if outpath.exists() and missing_only:
        existing = pd.read_csv(outpath)
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows across {df['team'].nunique() if not df.empty else 0} teams -> {outpath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape team roster headshots")
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument("--team", type=str, default=None, help="Specific team name to scrape")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of teams")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for concurrent requests")
    parser.add_argument("--missing-only", action="store_true", help="Only scrape missing teams")
    parser.add_argument("--outdir", type=str, default=OUTDIR, help="Output directory")
    parser.add_argument("--base-delay", type=float, default=3.0, help="Base delay between batches")
    args = parser.parse_args()

    df = asyncio.run(main(
        season=args.season,
        team_name=args.team,
        limit=args.limit,
        batch_size=args.batch_size,
        missing_only=args.missing_only,
        outdir=args.outdir,
        base_delay=args.base_delay,
    ))
