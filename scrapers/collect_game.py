import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from .scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)

BASE = "https://ncaa-api.henrygd.me"

PBP_COLS = [
    "year",
    "division",
    "contest_id",
    "date",
    "inning",
    "home_team_id",
    "away_team_id",
    "home_text",
    "away_text",
    "home_score",
    "away_score",
]

BATTER_COLS = [
    "year",
    "division",
    "contest_id",
    "team_id",
    "player_name",
    "position",
    "number",
    "is_sub",
]

PITCHER_COLS = [
    "year",
    "division",
    "contest_id",
    "team_id",
    "player_name",
    "number",
    "is_starter",
    "is_reliever"
]

def _load_contest_ids(schedules_dir: Path, div: str, year: int) -> tuple[set[int], dict[int, str]]:
    fpath = schedules_dir / f"{div}_schedules_{year}.csv"
    if not fpath.exists():
        logger.warning(f"No schedules file: {fpath}")
        return set(), {}
    try:
        df = pd.read_csv(fpath, usecols=["contest_id", "date"])
        df["contest_id"] = pd.to_numeric(df["contest_id"], errors="coerce")
        df = df.dropna(subset=["contest_id"])
        df["contest_id"] = df["contest_id"].astype(int)
        date_map = dict(zip(df["contest_id"], df["date"].fillna("")))
        return set(df["contest_id"]), date_map
    except Exception as e:
        logger.warning(f"Could not read {fpath}: {e}")
        return set(), {}


def _no_data_path(pbp_outdir: Path, div: str, year: int) -> Path:
    return pbp_outdir / "_tmp" / f"{div}_pbp_no_data_{year}.csv"

def _load_done_ids(pbp_outdir: Path, lineups_outdir: Path, div: str, year: int) -> set[int]:
    done: set[int] = set()
    for fpath in [
        pbp_outdir / f"{div}_pbp_{year}.csv",
        lineups_outdir / f"{div}_batting_lineups_{year}.csv",
    ]:
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, usecols=["contest_id"])
            done.update(pd.to_numeric(df["contest_id"], errors="coerce").dropna().astype(int))
        except Exception:
            pass
    no_data = _no_data_path(pbp_outdir, div, year)
    if no_data.exists():
        try:
            df = pd.read_csv(no_data)
            done.update(pd.to_numeric(df["contest_id"], errors="coerce").dropna().astype(int))
        except Exception:
            pass
    return done


def _flush_no_data(pbp_outdir: Path, div: str, year: int, new_ids: set[int]) -> None:
    if not new_ids:
        return
    p = _no_data_path(pbp_outdir, div, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: set[int] = set()
    if p.exists():
        try:
            df = pd.read_csv(p)
            existing.update(pd.to_numeric(df["contest_id"], errors="coerce").dropna().astype(int))
        except Exception:
            pass
    pd.DataFrame({"contest_id": sorted(existing | new_ids)}).to_csv(p, index=False)


def _append_rows(path: Path, rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _fetch_pbp(contest_id: int, div: str, year: int, date: str, session: requests.Session) -> list[dict]:
    try:
        r = session.get(f"{BASE}/game/{contest_id}/play-by-play", timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        pbp = data.get("playByPlay") or data
    except Exception as e:
        logger.warning(f"{contest_id} pbp fetch error: {e}")
        return []

    teams = pbp.get("teams") or []
    home_team_id = next((t.get("seoname") for t in teams if t.get("isHome")), None)
    away_team_id = next((t.get("seoname") for t in teams if not t.get("isHome")), None)
    team_side = {
        int(t["teamId"]): "home" if t.get("isHome") else "away"
        for t in teams
        if t.get("teamId") is not None
    }

    rows: list[dict] = []
    home_score = away_score = 0
    for period in pbp.get("periods") or []:
        inning = period.get("periodNumber")
        for event in period.get("playbyplayStats") or []:
            team_id = event.get("teamId")
            side = team_side.get(team_id) if team_id is not None else None
            for play in event.get("plays") or []:
                hs = play.get("homeScore")
                vs = play.get("visitorScore")
                if hs is not None:
                    home_score = hs
                if vs is not None:
                    away_score = vs
                rows.append({
                    "year":         year,
                    "division":     div,
                    "contest_id":   contest_id,
                    "date":         date,
                    "inning":       inning,
                    "home_team_id":    home_team_id,
                    "away_team_id":    away_team_id,
                    "home_text":    play.get("playText") if side == "home" else pd.NA,
                    "away_text":    play.get("playText") if side == "away" else pd.NA,
                    "home_score":   home_score,
                    "away_score":   away_score,
                })
    return rows


def _fetch_boxscore(contest_id: int, div: str, year: int, session: requests.Session) -> tuple[list[dict], list[dict]]:
    try:
        r = session.get(f"{BASE}/game/{contest_id}/boxscore", timeout=10)
        if r.status_code != 200:
            return [], []
        data = r.json()
        bs = data.get("boxScore") or data
    except Exception as e:
        logger.warning(f"{contest_id} boxscore fetch error: {e}")
        return [], []

    teams = bs.get("teams") or []
    home_team_id = next((t.get("seoname") for t in teams if t.get("isHome")), None)
    away_team_id = next((t.get("seoname") for t in teams if not t.get("isHome")), None)
    team_meta = {
        int(t["teamId"]): ("home" if t.get("isHome") else "away", t.get("seoname"))
        for t in teams
        if t.get("teamId") is not None
    }

    batter_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    for team_box in bs.get("teamBoxscore") or []:
        team_id = team_box.get("teamId")
        for p in team_box.get("playerStats") or []:
            base = {
                "year":        year,
                "contest_id":  contest_id,
                "division":    div,
                "team_id":   team_id,
                "player_name": p.get("firstName") + " " + p.get("lastName"),
                "position":    p.get("position"),
                "number":      p.get("number"),
            }
            bat = p.get("batterStats")
            if bat:
                batter_rows.append({ **base, 
                "is_sub":     not p.get("starter"), })
            pit = p.get("pitcherStats")
            if pit:
                pitcher_rows.append({ **base, "is_starter": p.get("starter"), "is_reliever": not p.get("starter") })
    return batter_rows, pitcher_rows


def scrape_games(year, divisions, schedules_dir, pbp_outdir, lineups_outdir, base_delay=0.2, **_):
    sched_path = Path(schedules_dir)
    pbp_path = Path(pbp_outdir)
    lin_path = Path(lineups_outdir)
    pbp_path.mkdir(parents=True, exist_ok=True)
    lin_path.mkdir(parents=True, exist_ok=True)

    with requests.Session() as http:
        http.headers.update({"User-Agent": "Mozilla/5.0"})

        for div in divisions:
            contest_ids, date_map = _load_contest_ids(sched_path, div, year)
            if not contest_ids:
                logger.info(f"{div} {year}: no schedule data, skipping")
                continue

            done_ids = _load_done_ids(pbp_path, lin_path, div, year)
            to_do = sorted(contest_ids - done_ids)

            if not to_do:
                logger.info(f"{div} {year}: all {len(contest_ids)} games already processed")
                continue

            logger.info(f"=== {div} {year}: {len(to_do)} games to process ===")

            out_pbp      = pbp_path / f"{div}_pbp_{year}.csv"
            out_batters  = lin_path / f"{div}_batting_lineups_{year}.csv"
            out_pitchers = lin_path / f"{div}_pitching_lineups_{year}.csv"

            no_data: set[int] = set()

            for i, cid in enumerate(to_do):
                pbp_rows = _fetch_pbp(cid, div, year, date_map.get(cid, ""), http)
                time.sleep(base_delay)
                batter_rows, pitcher_rows = _fetch_boxscore(cid, div, year, http)

                if not pbp_rows and not batter_rows and not pitcher_rows:
                    no_data.add(cid)
                    logger.info(f"{cid} — no data")
                else:
                    _append_rows(out_pbp, pbp_rows, PBP_COLS)
                    _append_rows(out_batters, batter_rows, BATTER_COLS)
                    _append_rows(out_pitchers, pitcher_rows, PITCHER_COLS)
                    logger.info(
                        f"{cid} — pbp={len(pbp_rows)}, "
                        f"batters={len(batter_rows)}, pitchers={len(pitcher_rows)}"
                    )

                if (i + 1) % 100 == 0:
                    _flush_no_data(pbp_path, div, year, no_data)
                    no_data = set()
                    logger.info(f"checkpoint: {i + 1}/{len(to_do)} games processed")

                if i + 1 < len(to_do):
                    time.sleep(base_delay)

            _flush_no_data(pbp_path, div, year, no_data)
            logger.info(f"{div} {year}: done")


if __name__ == "__main__":
    print("[start] scrapers.collect_game", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=str, default=["ncaa_1", "ncaa_2", "ncaa_3"])
    parser.add_argument("--indir", required=True, help="Directory containing schedule CSVs")
    parser.add_argument("--pbp_outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/pbp")
    parser.add_argument("--lineups_outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/lineups")
    parser.add_argument("--base_delay", type=float, default=0.2)
    args = parser.parse_args()

    scrape_games(
        year=args.year,
        divisions=args.divisions,
        schedules_dir=args.indir,
        pbp_outdir=args.pbp_outdir,
        lineups_outdir=args.lineups_outdir,
        base_delay=args.base_delay,
    )
