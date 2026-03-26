import argparse
import re
import time
from datetime import date, timedelta
from pathlib import Path

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

from .scraper_utils import get_scraper_logger
from .constants import CURRENT_YEAR

logger = get_scraper_logger(__name__)

SEASON_START = (1, 15)

DIVISION_CONFIGS = {
    "njcaa_1": {
        "base": "https://njcaastats.prestosports.com",
        "path": "sports/bsb/{season}/div1/composite",
    },
    "njcaa_2": {
        "base": "https://njcaastats.prestosports.com",
        "path": "sports/bsb/{season}/div2/composite",
    },
    "njcaa_3": {
        "base": "https://njcaastats.prestosports.com",
        "path": "sports/bsb/{season}/div3/composite",
    },
    "naia": {
        "base": "https://naiastats.prestosports.com",
        "path": "sports/bsb/scoreboard",
    },
}

OUT_COLS = [
    "year", "division", "contest_id", "team_name", "team_slug",
    "opponent_team_name", "opponent_team_slug", "date", "game_time",
    "game_url", "team_score", "opponent_score", "neutral_site",
    "is_neutral_site", "attendance",
]


def _season_str(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


def _parse_score(el):
    if el is None:
        return pd.NA
    m = re.search(r"\d+", el.get_text(strip=True))
    return int(m.group()) if m else pd.NA


def _extract_game_id(href: str) -> str | None:
    m = re.search(r"/boxscores/([^/?]+?)(?:\.xml)?(?:\?|$)", href)
    return m.group(1) if m else None


def _extract_team_slug(row) -> str:
    img = row.select_one(".team-logo img")
    if not img:
        return pd.NA
    m = re.search(r"/logos/id/([^/]+)\.png", img.get("src", ""))
    return m.group(1) if m else pd.NA


def _extract_game_time(card) -> str:
    link = card.find("a", attrs={"aria-label": True})
    if not link:
        return pd.NA
    m = re.search(r"\d{1,2}:\d{2}\s*[AP]M", link["aria-label"])
    return m.group().strip() if m else pd.NA


def _parse_participant(row):
    name_el = row.select_one(".team-name")
    raw = name_el.get_text(separator=" ", strip=True) if name_el else ""
    name = re.sub(r"^at\s+", "", raw).strip()
    score = _parse_score(row.select_one(".event-result"))
    slug = _extract_team_slug(row)
    return name, score, slug


def _parse_cards(soup: BeautifulSoup, base_url: str, division: str, date_str: str, year: int) -> list[dict]:
    rows = []
    for card in soup.select(".event-box"):
        sport_el = card.select_one(".list-event-sport a")
        if sport_el and "baseball" not in sport_el.get_text(strip=True).lower():
            continue

        p_rows = card.select(".list-events-participants")
        if len(p_rows) < 2:
            continue

        status_el = card.select_one(".cal-status")
        status = status_el.get_text(strip=True) if status_el else ""
        if "final" not in status.lower():
            continue

        away_name, away_score, away_slug = _parse_participant(p_rows[0])
        home_name, home_score, home_slug = _parse_participant(p_rows[1])

        box_link = card.find("a", href=re.compile(r"/boxscores/"))
        if box_link:
            href = box_link["href"]
            contest_id = _extract_game_id(href)
            game_url = href if href.startswith("http") else f"{base_url}{href}"
        else:
            contest_id = pd.NA
            game_url = pd.NA

        rows.append({
            "year":                year,
            "division":            division,
            "contest_id":          contest_id,
            "team_name":           home_name,
            "team_slug":           home_slug,
            "opponent_team_name":  away_name,
            "opponent_team_slug":  away_slug,
            "date":                date_str,
            "game_time":           _extract_game_time(card),
            "game_url":            game_url,
            "team_score":          home_score,
            "opponent_score":      away_score,
            "neutral_site":        pd.NA,
            "is_neutral_site":     "va text-muted" not in str(p_rows[1]),
            "attendance":          pd.NA,
        })
    return rows


def _get_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUT_COLS)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=OUT_COLS)
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS].copy()


def _resume_date(existing: pd.DataFrame, year: int) -> date:
    if not existing.empty and "date" in existing.columns:
        parsed = pd.to_datetime(existing["date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().date() + timedelta(days=1)
    return date(year, *SEASON_START)


def _save(path: Path, existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows, columns=OUT_COLS) if new_rows else pd.DataFrame(columns=OUT_COLS)
    out = pd.concat([existing, new_df], ignore_index=True)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUT_COLS].drop_duplicates(subset=["year", "division", "contest_id"], keep="last")
    out["_d"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["_d", "contest_id"], na_position="last").drop(columns=["_d"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def scrape_non_ncaa_schedules(year: int, divisions: list[str], outdir: str, base_delay: float = 1.5):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    today = date.today()
    is_current = (year == CURRENT_YEAR)
    season = _season_str(year)
    scraper = cloudscraper.create_scraper()

    for div in divisions:
        cfg = DIVISION_CONFIGS.get(div)
        if cfg is None:
            logger.info(f"Skipping unsupported division: {div}")
            continue

        base_url = cfg["base"]
        url = f"{base_url}/{cfg['path'].format(season=season)}"
        fpath = outdir_path / f"{div}_schedules_{year}.csv"

        existing = _get_existing(fpath) if is_current else pd.DataFrame(columns=OUT_COLS)
        seen_ids: set[str] = set(existing["contest_id"].dropna().astype(str))
        start = _resume_date(existing, year) if is_current else date(year, *SEASON_START)

        logger.info(f"=== {div} {year}: {start} → {today} ===")
        pending: list[dict] = []
        current = start

        while current <= today:
            date_str = current.strftime("%Y-%m-%d")
            try:
                resp = scraper.get(url, params={"d": date_str}, timeout=20)
                resp.raise_for_status()
                cards = _parse_cards(BeautifulSoup(resp.text, "html.parser"), base_url, div, date_str, year)
                new = [r for r in cards if str(r.get("contest_id", "")) not in seen_ids]
                for r in new:
                    seen_ids.add(str(r["contest_id"]))
                pending.extend(new)
                logger.info(f"{date_str} — {len(cards)} games, {len(new)} new")
            except Exception as e:
                logger.warning(f"{date_str} — error: {e}")

            current += timedelta(days=1)
            if current <= today:
                time.sleep(base_delay)

        if pending:
            existing = _save(fpath, existing, pending)
        logger.info(f"saved {fpath} ({len(existing)} rows)")


if __name__ == "__main__":
    print("[start] scrapers.collect_non_ncaa_schedules", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", default=list(DIVISION_CONFIGS))
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--base_delay", type=float, default=1.5)
    args = parser.parse_args()

    scrape_non_ncaa_schedules(
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        base_delay=args.base_delay,
    )
