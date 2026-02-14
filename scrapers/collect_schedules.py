import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

from .constants import BASE
from .scraper_utils import HardBlockError, ScraperConfig, ScraperSession, get_scraper_logger

logger = get_scraper_logger(__name__)

SCOREBOARD_PATH = "/contests/livestream_scoreboards"
SEASON_DIVISION_IDS = {
    1: 18783,
    2: 18784,
    3: 18783,
}

OUT_COLS = [
    "year",
    "division",
    "contest_id",
    "team",
    "team_id",
    "opponent",
    "opponent_team_id",
    "date",
    "game_url",
    "neutral_site",
    "is_neutral_site",
    "team_score",
    "opponent_score",
    "innings",
    "attendance",
]


def played_team_ids_path(outdir: Path, div: int, year: int) -> Path:
    return outdir / "_tmp" / f"d{div}_teams_played_{year}.csv"


def write_played_team_ids(path: Path, team_ids: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"team_id": sorted(team_ids)})
    df.to_csv(path, index=False)


def parse_int(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"\d+", text.replace(",", ""))
    return int(m.group(0)) if m else None


def clean_team_name(text: str) -> str:
    s = (text or "").strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def to_iso_date(raw: str, fallback: date) -> str:
    s = (raw or "").strip()
    if not s:
        return fallback.isoformat()
    for fmt in ("%m/%d/%Y %I:%M %p", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return fallback.isoformat()


def extract_game_datetime(text: str) -> str:
    s = text or ""
    m = re.search(r"\d{2}/\d{2}/\d{4}(?:\s+\d{1,2}:\d{2}\s*[AP]M)?", s)
    return m.group(0) if m else ""


def get_existing(path: Path) -> pd.DataFrame:
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


def get_start_date(existing: pd.DataFrame, year: int) -> date:
    if existing.empty or "date" not in existing.columns:
        return date(year, 2, 1)
    parsed = pd.to_datetime(existing["date"], errors="coerce").dropna()
    if parsed.empty:
        return date(year, 2, 1)
    return (parsed.max().date() + timedelta(days=1))


def iter_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_scoreboard_url(division: int, game_day: date) -> str:
    params = {
        "utf8": "✓",
        "season_division_id": SEASON_DIVISION_IDS[division],
        "game_date": game_day.strftime("%m/%d/%Y"),
        "conference_id": 0,
        "tournament_id": "",
        "commit": "Submit",
    }
    return f"{BASE}{SCOREBOARD_PATH}?{urlencode(params)}"


def parse_team_row(row) -> tuple[str, int | None, int | None]:
    link = row.select_one("a[href*='/teams/']")
    if link:
        raw_name = link.get_text(" ", strip=True)
    else:
        name_cell = row.select_one("td.opponents_min_width")
        raw_name = name_cell.get_text(" ", strip=True) if name_cell else ""
    team_name = clean_team_name(raw_name)
    team_id = None
    if link and link.get("href"):
        m = re.search(r"/teams/(\d+)", link["href"])
        if m:
            team_id = int(m.group(1))
    score_div = row.select_one("div[id^='score_']")
    score = parse_int(score_div.get_text(" ", strip=True) if score_div else "")
    return team_name, team_id, score


def is_team_row(row) -> bool:
    if row.select_one("a[href*='/teams/']"):
        return True
    name_cell = row.select_one("td.opponents_min_width")
    if not name_cell:
        return False
    txt = clean_team_name(name_cell.get_text(" ", strip=True))
    return bool(txt)


def parse_innings(table, contest_id: int) -> int | None:
    if table is None:
        return None
    ls = table.select_one(f"table#linescore_{contest_id}_table")
    if ls is None:
        return None
    rows = ls.select("tr")
    if len(rows) < 2:
        return None
    away_cells = [td.get_text(strip=True) for td in rows[0].select("td")]
    home_cells = [td.get_text(strip=True) for td in rows[1].select("td")]
    max_len = max(len(away_cells), len(home_cells))
    last_non_empty = 0
    for i in range(max_len):
        a = away_cells[i] if i < len(away_cells) else ""
        h = home_cells[i] if i < len(home_cells) else ""
        if a or h:
            last_non_empty = i + 1
    return last_non_empty or None


def _top_level_rows(table) -> list:
    body = table.find("tbody")
    if body:
        return body.find_all("tr", recursive=False)
    return table.find_all("tr", recursive=False)


def _table_metadata(rows: list, fallback_day: date) -> tuple[str, str, bool, int | None]:
    first_row_text = rows[0].get_text(" ", strip=True) if rows else ""
    date_iso = to_iso_date(extract_game_datetime(first_row_text), fallback_day)

    location = rows[1].get_text(" ", strip=True) if len(rows) > 1 else ""
    is_neutral = location.strip().startswith("@")
    neutral_site = location.strip().lstrip("@").strip() if is_neutral else ""

    att_match = re.search(r"Attend:\s*([0-9,]+)", first_row_text, flags=re.IGNORECASE)
    attendance = parse_int(att_match.group(1) if att_match else "")
    return date_iso, neutral_site, is_neutral, attendance


def parse_table_games(table, division: int, year: int, fallback_day: date) -> list[dict]:
    rows = _top_level_rows(table)
    if not rows:
        return []

    date_iso, neutral_site, is_neutral, attendance = _table_metadata(rows, fallback_day)
    out: list[dict] = []

    def flush_game(contest_id: int | None, pair_rows: list) -> None:
        if contest_id is None or len(pair_rows) < 2:
            return
        away_name, away_id, away_score = parse_team_row(pair_rows[0])
        home_name, home_id, home_score = parse_team_row(pair_rows[1])
        if not away_name and not home_name:
            return

        innings = parse_innings(table, contest_id)
        game_url = f"{BASE}/contests/{contest_id}"

        out.append(
            {
                "year": year,
                "division": division,
                "contest_id": contest_id,
                "team": away_name,
                "team_id": away_id,
                "opponent": home_name,
                "opponent_team_id": home_id,
                "date": date_iso,
                "game_url": game_url,
                "neutral_site": neutral_site,
                "is_neutral_site": is_neutral,
                "team_score": away_score,
                "opponent_score": home_score,
                "innings": innings,
                "attendance": attendance,
            }
        )
        out.append(
            {
                "year": year,
                "division": division,
                "contest_id": contest_id,
                "team": home_name,
                "team_id": home_id,
                "opponent": away_name,
                "opponent_team_id": away_id,
                "date": date_iso,
                "game_url": game_url,
                "neutral_site": neutral_site,
                "is_neutral_site": is_neutral,
                "team_score": home_score,
                "opponent_score": away_score,
                "innings": innings,
                "attendance": attendance,
            }
        )

    i = 0
    while i < len(rows):
        tr = rows[i]
        rid = tr.get("id", "")
        if not str(rid).startswith("contest_"):
            i += 1
            continue

        contest_id = parse_int(str(rid))
        away_row = tr
        home_row = None

        j = i + 1
        while j < len(rows):
            look = rows[j]
            look_id = str(look.get("id", ""))
            if look_id.startswith("contest_") and parse_int(look_id) != contest_id:
                break
            if is_team_row(look):
                home_row = look
                break
            j += 1

        if home_row is not None:
            flush_game(contest_id, [away_row, home_row])
            i = j + 1
        else:
            i += 1

    return out


def parse_scoreboard_page(html: str, division: int, year: int, game_day: date) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    tables = soup.select("div.table-responsive table")
    if not tables:
        tables = soup.select("table")

    rows: list[dict] = []
    for table in tables:
        tid = str(table.get("id", ""))
        if tid.startswith("linescore_"):
            continue
        rows.extend(parse_table_games(table, division, year, game_day))
    return rows


def save_schedule(path: Path, existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows, columns=OUT_COLS) if new_rows else pd.DataFrame(columns=OUT_COLS)
    out = pd.concat([existing, new_df], ignore_index=True)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUT_COLS].copy()
    out = out.drop_duplicates(subset=["year", "division", "contest_id", "team_id"], keep="last")
    out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["date_parsed", "contest_id", "team_id"], na_position="last").drop(columns=["date_parsed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def scrape_schedules(
    team_ids_file,
    year,
    divisions,
    outdir,
    batch_size=10,
    base_delay=10.0,
    rest_every=12,
    batch_cooldown_s=90,
):
    _ = team_ids_file, batch_size, rest_every, batch_cooldown_s

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    config = ScraperConfig(
        base_delay=base_delay,
        block_resources=True,
    )

    today = datetime.now(ZoneInfo("America/New_York")).date()
    with ScraperSession(config) as session:
        for div in divisions:
            if div not in SEASON_DIVISION_IDS:
                logger.info(f"Skipping unsupported division: {div}")
                continue

            fpath = outdir_path / f"d{div}_schedules_{year}.csv"
            played_ids_file = played_team_ids_path(outdir_path, div, year)
            if played_ids_file.exists():
                played_ids_file.unlink()
            existing = get_existing(fpath)
            start_day = get_start_date(existing, year)
            end_day = min(today, date(year, 12, 31))
            played_team_ids: set[int] = set()

            if start_day > end_day:
                logger.info(f"d{div} {year}: up to date (max date already >= today)")
                continue

            logger.info(f"\n=== d{div} {year} schedules ===")
            logger.info(f"Fetching dates {start_day.isoformat()} -> {end_day.isoformat()}")
            logger.info(f"    (budget remaining: {session.requests_remaining} requests)")

            pending_rows: list[dict] = []
            try:
                for game_day in iter_dates(start_day, end_day):
                    if session.requests_remaining <= 0:
                        logger.info("[budget] daily request budget exhausted, stopping")
                        break

                    url = build_scoreboard_url(div, game_day)
                    html, status = session.fetch(url, wait_selector="tr[id^='contest_']", wait_timeout=5000)
                    if not html or status >= 400:
                        logger.info(f"d{div} {game_day.isoformat()}: HTTP {status}")
                        continue

                    day_rows = parse_scoreboard_page(html, div, year, game_day)
                    if day_rows:
                        pending_rows.extend(day_rows)
                        for row in day_rows:
                            tid = row.get("team_id")
                            if tid is not None and not pd.isna(tid):
                                played_team_ids.add(int(tid))
                        logger.info(f"d{div} {game_day.isoformat()}: parsed {len(day_rows)} team rows")

                    if pending_rows and len(pending_rows) >= 500:
                        existing = save_schedule(fpath, existing, pending_rows)
                        logger.info(f"checkpoint saved {fpath} ({len(existing)} rows)")
                        pending_rows = []

            except HardBlockError as exc:
                logger.error(str(exc))
                logger.info("[STOP] hard block detected. Saving progress and exiting.")
                if pending_rows:
                    existing = save_schedule(fpath, existing, pending_rows)
                    logger.info(f"saved {fpath} ({len(existing)} rows)")
                if played_team_ids:
                    write_played_team_ids(played_ids_file, played_team_ids)
                    logger.info(f"saved {played_ids_file} ({len(played_team_ids)} team_ids)")
                return

            if pending_rows:
                existing = save_schedule(fpath, existing, pending_rows)

            logger.info(f"saved {fpath} ({len(existing)} rows)")
            if played_team_ids:
                write_played_team_ids(played_ids_file, played_team_ids)
                logger.info(f"saved {played_ids_file} ({len(played_team_ids)} team_ids)")


if __name__ == "__main__":
    print("[start] scrapers.collect_schedules", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--team_ids_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv")
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/schedules")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=10.0)
    parser.add_argument("--rest_every", type=int, default=12)
    parser.add_argument("--batch_cooldown_s", type=int, default=90)
    args = parser.parse_args()

    scrape_schedules(
        team_ids_file=args.team_ids_file,
        year=args.year,
        divisions=args.divisions,
        outdir=args.outdir,
        batch_size=args.batch_size,
        base_delay=args.base_delay,
        rest_every=args.rest_every,
        batch_cooldown_s=args.batch_cooldown_s,
    )
