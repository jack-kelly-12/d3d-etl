import argparse
import re
import time
from pathlib import Path

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

from .scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)

PBP_COLS = [
    "year", "division", "contest_id", "date",
    "inning", "home_team_id", "away_team_id",
    "home_text", "away_text", "home_score", "away_score",
]

BATTER_COLS = [
    "year", "division", "contest_id", "team_id",
    "player_name", "position", "number", "is_sub",
]

PITCHER_COLS = [
    "year", "division", "contest_id", "team_id",
    "player_name", "number", "is_starter", "is_reliever",
]


def _load_schedule(schedules_dir: Path, div: str, year: int) -> pd.DataFrame:
    fpath = schedules_dir / f"{div}_schedules_{year}.csv"
    if not fpath.exists():
        logger.warning(f"No schedule file: {fpath}")
        return pd.DataFrame()
    try:
        return pd.read_csv(fpath, dtype=str)
    except Exception as e:
        logger.warning(f"Could not read {fpath}: {e}")
        return pd.DataFrame()


def _load_done_ids(lineups_dir: Path, div: str, year: int) -> set[str]:
    done: set[str] = set()
    fpath = lineups_dir / f"{div}_batting_lineups_{year}.csv"
    if fpath.exists():
        try:
            df = pd.read_csv(fpath, usecols=["contest_id"], dtype=str)
            done.update(df["contest_id"].dropna())
        except Exception:
            pass
    return done


def _append_rows(path: Path, rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _parse_scoring_summary(soup: BeautifulSoup) -> list[tuple[str, int, int]]:
    for cap in soup.select("caption h2"):
        if "Scoring" in cap.get_text():
            table = cap.find_parent("table")
            if not table:
                continue
            plays = []
            for row in table.select("tbody tr"):
                play_td = row.select_one("td.text")
                total_td = row.select_one("td.total")
                if not play_td or not total_td:
                    continue
                m = re.match(r"(\d+)\s*-\s*(\d+)", total_td.get_text(strip=True))
                if m:
                    plays.append((
                        play_td.get_text(separator=" ", strip=True).lower(),
                        int(m.group(1)),
                        int(m.group(2)),
                    ))
            return plays
    return []


def _parse_pbp(
    soup: BeautifulSoup,
    contest_id: str,
    game_date: str,
    away_team_id: str,
    home_team_id: str,
    division: str,
    year: int,
) -> list[dict]:
    scoring_plays = _parse_scoring_summary(soup)
    scoring_idx = 0
    cur_away = cur_home = 0
    rows = []

    for section in soup.select("section[id^='pbp-inning-']"):
        inning_num = int(section["id"].replace("pbp-inning-", ""))
        for half_idx, half_table in enumerate(section.select(".table-responsive")):
            batting_team = "away" if half_idx == 0 else "home"
            for play_td in half_table.select("tr:not(.totals) td.text"):
                text = play_td.get_text(separator=" ", strip=True)
                if not text:
                    continue
                if scoring_idx < len(scoring_plays):
                    s_text, s_away, s_home = scoring_plays[scoring_idx]
                    if s_text in text.lower() or text.lower() in s_text:
                        cur_away, cur_home = s_away, s_home
                        scoring_idx += 1
                rows.append({
                    "year":         year,
                    "division":     division,
                    "contest_id":   contest_id,
                    "date":         game_date,
                    "inning":       inning_num,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_text":    text if batting_team == "home" else pd.NA,
                    "away_text":    text if batting_team == "away" else pd.NA,
                    "home_score":   cur_home,
                    "away_score":   cur_away,
                })
    return rows


def _parse_lineups(
    soup: BeautifulSoup,
    contest_id: str,
    away_team_id: str,
    home_team_id: str,
    division: str,
    year: int,
) -> tuple[list[dict], list[dict]]:
    batting_rows = []

    panels = soup.select(".player-stats .stats-box.half")
    for panel_idx, panel in enumerate(panels[:2]):
        team_id = away_team_id if panel_idx == 0 else home_team_id

        batter_table = None
        for cap in panel.select("table caption h2"):
            t = cap.find_parent("table")
            if t:
                batter_table = t
                break
        if not batter_table:
            batter_table = panel.select_one("table.striped")
        if not batter_table:
            continue

        seen_positions: set[str] = set()
        for row in batter_table.select("tbody tr:not(.totals)"):
            th = row.select_one("th.row-head")
            if not th:
                continue
            pos_el = th.select_one(".player-position")
            position = pos_el.get_text(strip=True).upper() if pos_el else ""
            name_el = th.select_one(".player-name")
            name = name_el.get_text(strip=True) if name_el else ""
            if not name:
                continue
            is_sub = position in seen_positions or (position == "" and name != "")
            if position:
                seen_positions.add(position)
            batting_rows.append({
                "year":        year,
                "division":    division,
                "contest_id":  contest_id,
                "team_id":     team_id,
                "player_name": name,
                "position":    position,
                "number":      pd.NA,
                "is_sub":      is_sub,
            })

    pitching_rows = []
    pitcher_tables = [
        cap.find_parent("table")
        for cap in soup.select(".stats-wrap .stats-box.half table caption h2")
        if "Pitcher" in cap.get_text() and cap.find_parent("table")
    ]
    for panel_idx, pitcher_table in enumerate(pitcher_tables[:2]):
        team_id = away_team_id if panel_idx == 0 else home_team_id
        data_rows = [r for r in pitcher_table.select("tbody tr:not(.totals)") if r.select_one("th.row-head")]
        for row_idx, row in enumerate(data_rows):
            th = row.select_one("th.row-head")
            name_el = th.select_one(".player-name")
            name = name_el.get_text(strip=True) if name_el else th.get_text(strip=True)
            if not name:
                continue
            pitching_rows.append({
                "year":        year,
                "division":    division,
                "contest_id":  contest_id,
                "team_id":     team_id,
                "player_name": name,
                "number":      pd.NA,
                "is_starter":  row_idx == 0,
                "is_reliever": row_idx != 0,
            })

    return batting_rows, pitching_rows


def _fetch_game(
    scraper,
    game_url: str,
    contest_id: str,
    game_date: str,
    away_team_id: str,
    home_team_id: str,
    division: str,
    year: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    resp = scraper.get(game_url, timeout=20)
    if resp.status_code >= 400:
        logger.warning(f"{contest_id} — HTTP {resp.status_code}")
        return [], [], []
    soup = BeautifulSoup(resp.text, "html.parser")
    pbp = _parse_pbp(soup, contest_id, game_date, away_team_id, home_team_id, division, year)
    batters, pitchers = _parse_lineups(soup, contest_id, away_team_id, home_team_id, division, year)
    return pbp, batters, pitchers


def scrape_non_ncaa_games(
    year: int,
    divisions: list[str],
    schedules_dir: str,
    pbp_outdir: str,
    lineups_outdir: str,
    base_delay: float = 1.5,
):
    sched_path = Path(schedules_dir)
    pbp_path = Path(pbp_outdir)
    lin_path = Path(lineups_outdir)
    pbp_path.mkdir(parents=True, exist_ok=True)
    lin_path.mkdir(parents=True, exist_ok=True)

    scraper = cloudscraper.create_scraper()

    for div in divisions:
        sched = _load_schedule(sched_path, div, year)
        if sched.empty:
            continue

        sched = sched.dropna(subset=["contest_id", "game_url"])
        done_ids = _load_done_ids(lin_path, div, year)
        to_do = sched[~sched["contest_id"].isin(done_ids)].reset_index(drop=True)

        if to_do.empty:
            logger.info(f"{div} {year}: all games already processed")
            continue

        logger.info(f"=== {div} {year}: {len(to_do)} games to process ===")

        out_pbp = pbp_path / f"{div}_pbp_{year}.csv"
        out_batters = lin_path / f"{div}_batting_lineups_{year}.csv"
        out_pitchers = lin_path / f"{div}_pitching_lineups_{year}.csv"

        for i, row in to_do.iterrows():
            contest_id = row["contest_id"]
            game_url = row["game_url"]
            game_date = row.get("date", "")
            away_team_id = row.get("opponent_team_slug", pd.NA)
            home_team_id = row.get("team_slug", pd.NA)

            try:
                pbp_rows, batter_rows, pitcher_rows = _fetch_game(
                    scraper, game_url, contest_id, game_date,
                    away_team_id, home_team_id, div, year,
                )
            except Exception as e:
                logger.warning(f"{contest_id} — error: {e}")
                continue

            if not pbp_rows and not batter_rows and not pitcher_rows:
                logger.info(f"{contest_id} — no data")
            else:
                _append_rows(out_pbp, pbp_rows, PBP_COLS)
                _append_rows(out_batters, batter_rows, BATTER_COLS)
                _append_rows(out_pitchers, pitcher_rows, PITCHER_COLS)
                logger.info(f"{contest_id} — pbp={len(pbp_rows)}, batters={len(batter_rows)}, pitchers={len(pitcher_rows)}")

            if i + 1 < len(to_do):
                time.sleep(base_delay)

        logger.info(f"{div} {year}: done")


if __name__ == "__main__":
    print("[start] scrapers.collect_non_ncaa_game", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", default=["njcaa_1", "njcaa_2", "njcaa_3", "naia"])
    parser.add_argument("--indir", required=True, help="Directory containing schedule CSVs")
    parser.add_argument("--pbp_outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/pbp")
    parser.add_argument("--lineups_outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/lineups")
    parser.add_argument("--base_delay", type=float, default=1.5)
    args = parser.parse_args()

    scrape_non_ncaa_games(
        year=args.year,
        divisions=args.divisions,
        schedules_dir=args.indir,
        pbp_outdir=args.pbp_outdir,
        lineups_outdir=args.lineups_outdir,
        base_delay=args.base_delay,
    )
