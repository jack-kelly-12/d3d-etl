import argparse
import time
import urllib.parse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from .scraper_utils import get_scraper_logger
from .constants import CURRENT_YEAR

logger = get_scraper_logger(__name__)

SHA256 = "6b26e5cda954c1302873c52835bfd223e169e2068b12511e92b3ef29fac779c2"
SPORT = "baseball"
SCHEDULE_ALT_BASE = "https://ncaa-api.henrygd.me/schedule-alt"

DIVISION_CODES = {
    "ncaa_1": 1,
    "ncaa_2": 2,
    "ncaa_3": 3,
}

DIVISION_ALT_CODES = {
    "ncaa_1": "d1",
    "ncaa_2": "d2",
    "ncaa_3": "d3",
}

OUT_COLS = [
    "year",
    "division",
    "contest_id",
    "team_name",
    "team_slug",
    "opponent_team_name",
    "opponent_team_slug",
    "date",
    "game_time",
    "game_url",
    "team_score",
    "opponent_score",
    "neutral_site",
    "is_neutral_site",
    "attendance",
]


def _build_url(d: date, division: int, season_year: int) -> str:
    date_str = d.strftime("%m/%d/%Y")
    extensions = urllib.parse.quote(
        '{"persistedQuery":{"version":1,"sha256Hash":"' + SHA256 + '"}}',
        safe="",
    )
    variables = urllib.parse.quote(
        f'{{"sportCode":"MBA","division":{division},"seasonYear":{season_year},"contestDate":"{date_str}","week":null}}',
        safe="",
    )
    return f"https://sdataprod.ncaa.com/?meta=GetContests_web&extensions={extensions}&variables={variables}"


def _fetch_game_dates(div: str, season_year: int, session: requests.Session) -> list[date]:
    """Fetch the list of dates that have games from the schedule-alt endpoint."""
    alt_div = DIVISION_ALT_CODES[div]
    url = f"{SCHEDULE_ALT_BASE}/{SPORT}/{alt_div}/{season_year}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        games = r.json()["data"]["schedules"]["games"]
        return [datetime.strptime(g["contestDate"], "%m/%d/%Y").date() for g in games]
    except Exception as e:
        logger.warning(f"Failed to fetch game dates for {div} {season_year}: {e}")
        return []


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


def _get_start_date(existing: pd.DataFrame) -> date | None:
    """Return the day after the latest date in existing data, or None if empty."""
    if existing.empty or "date" not in existing.columns:
        return None
    parsed = pd.to_datetime(existing["date"], errors="coerce").dropna()
    if parsed.empty:
        return None
    from datetime import timedelta
    return parsed.max().date() + timedelta(days=1)


def _fetch_contests(d: date, division: int, season_year: int, session: requests.Session) -> list[dict]:
    url = _build_url(d, division, season_year)
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("data", {}).get("contests") or []
    except Exception as e:
        logger.warning(f"{d} — fetch error: {e}")
        return []


def _parse_date(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    for fmt, length in [("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d", 10), ("%m/%d/%Y", 10)]:
        try:
            return datetime.strptime(s[:length], fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s


def _parse_contests(contests: list[dict], div: str, year: int) -> list[dict]:
    rows: list[dict] = []
    for c in contests:
        if c.get("gameState") != "F":
            continue
        teams = c.get("teams") or []
        home = next((t for t in teams if t.get("isHome")), None)
        away = next((t for t in teams if not t.get("isHome")), None)
        contest_id = c["contestId"]
        game_date = _parse_date(c.get("startDate"))
        game_url = f"https://www.ncaa.com/game/{contest_id}"

        rows.append({
            "year": year,
            "division": div,
            "contest_id": contest_id,
            "team_name": home["nameShort"] if home else (away["nameShort"] if away else None),
            "team_slug": home["seoname"] if home else (away["seoname"] if away else None),
            "team_conference_slug": home["conferenceSeo"] if home else (away["conferenceSeo"] if away else None),
            "opponent_team_name": away["nameShort"] if away else None,
            "opponent_team_slug": away["seoname"] if away else (home["seoname"] if home else None),
            "opponent_conference_slug": away["conferenceSeo"] if away else (home["conferenceSeo"] if home else None),
            "date": game_date,
            "game_time": c.get("startTime"),
            "game_url": game_url,
            "team_score": home["score"] if home else None,
            "opponent_score": away["score"] if away else None,
            "neutral_site": c.get("neutralSite"),
            "is_neutral_site": bool(c.get("neutralSite")),
            "attendance": c.get("attendance", pd.NA),
        })
    return rows


def _save_schedule(path: Path, existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows, columns=OUT_COLS) if new_rows else pd.DataFrame(columns=OUT_COLS)
    out = pd.concat([existing, new_df], ignore_index=True)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUT_COLS].copy()
    out = out.drop_duplicates(subset=["year", "division", "contest_id"], keep="last")
    out["_date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["_date", "contest_id"], na_position="last").drop(columns=["_date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def scrape_schedules(year, divisions, outdir, base_delay=1.0, **_):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    season_year = year - 1
    today = date.today()
    is_current_year = (year == CURRENT_YEAR)

    with requests.Session() as http:
        http.headers.update({"User-Agent": "Mozilla/5.0"})

        for div in divisions:
            if div not in DIVISION_CODES:
                logger.info(f"Skipping unsupported division: {div}")
                continue

            api_div = DIVISION_CODES[div]
            fpath = outdir_path / f"{div}_schedules_{year}.csv"

            if is_current_year:
                existing = _get_existing(fpath)
                existing_ids: set[int] = set(
                    pd.to_numeric(existing["contest_id"], errors="coerce").dropna().astype(int)
                )
                start_date = _get_start_date(existing)
            else:
                existing = pd.DataFrame(columns=OUT_COLS)
                existing_ids = set()
                start_date = None

            # Get the list of dates that have games from the alt endpoint
            all_dates = _fetch_game_dates(div, season_year, http)
            if not all_dates:
                logger.warning(f"{div} {year}: no game dates returned from schedule-alt endpoint")
                continue

            # Filter dates
            if is_current_year:
                dates_to_fetch = [d for d in all_dates if d <= today]
                if start_date:
                    dates_to_fetch = [d for d in dates_to_fetch if d >= start_date]
            else:
                dates_to_fetch = all_dates

            if not dates_to_fetch:
                logger.info(f"{div} {year}: already up to date")
                continue

            logger.info(f"=== {div} {year}: {len(dates_to_fetch)} dates to fetch ===")

            pending: list[dict] = []
            for i, d in enumerate(dates_to_fetch):
                contests = _fetch_contests(d, api_div, season_year, http)
                final_rows = _parse_contests(contests, div, year)
                new_rows = [r for r in final_rows if int(r["contest_id"]) not in existing_ids]
                for r in new_rows:
                    existing_ids.add(int(r["contest_id"]))
                pending.extend(new_rows)
                logger.info(f"{d} — {len(contests)} fetched, {len(final_rows)} final, {len(new_rows)} new")

                if i < len(dates_to_fetch) - 1:
                    time.sleep(base_delay)

            if pending:
                existing = _save_schedule(fpath, existing, pending)
            logger.info(f"saved {fpath} ({len(existing)} rows)")


if __name__ == "__main__":
    print("[start] scrapers.collect_schedules", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=str, default=["ncaa_1", "ncaa_2", "ncaa_3"])
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--base_delay", type=float, default=1.0)
    parser.add_argument("--team_ids_file", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--rest_every", type=int, default=None)
    parser.add_argument("--batch_cooldown_s", type=int, default=None)
    args = parser.parse_args()

    scrape_schedules(
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        base_delay=args.base_delay,
    )
