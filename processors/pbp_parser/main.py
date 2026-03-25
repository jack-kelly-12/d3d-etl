import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rfuzz_process

from processors.logging_utils import div_file_prefix, division_year_label, get_logger
from scrapers.constants import CURRENT_YEAR

from .columns import (
    bat_order,
    classify_batted_ball_type,
    classify_event_type,
    determine_batter_and_runners,
    flags,
    metadata,
    outs_after,
    outs_before,
    outs_on_play,
    runs_on_play,
    runs_rest_of_inn,
    runs_this_inn,
    score_before,
)

from .names import standardize_names

logger = get_logger(__name__)


def parse_pbp(
    pbp: pd.DataFrame,
    team_history: pd.DataFrame,
    year: int,
    batting_lineups: pd.DataFrame,
    pitching_lineups: pd.DataFrame,
    cube_fallback_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pbp_metadata = metadata(pbp)
    pbp_metadata = pbp_metadata.sort_values(["contest_id", "play_id"], kind="stable")

    pbp_with_teams = add_team_names(pbp_metadata, team_history, year)

    pbp_flags = flags(pbp_with_teams)

    pbp_outs = parse_pbp_outs(pbp_flags)
    pbp_runs = parse_pbp_runs(pbp_outs, year)
    pbp_bases = parse_pbp_base_state(pbp_runs)

    pbp_final = standardize_names(pbp_bases, batting_lineups, pitching_lineups, cube_fallback_df=cube_fallback_df)

    return pbp_final


def parse_pbp_runs(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if year >= CURRENT_YEAR:
        df["away_score_after"] = pd.to_numeric(df.get("away_score"), errors="coerce").fillna(0).astype(int)
        df["home_score_after"] = pd.to_numeric(df.get("home_score"), errors="coerce").fillna(0).astype(int)
        df["away_score_before"] = (
            df.groupby("contest_id")["away_score_after"].shift(fill_value=0).astype(int)
        )
        df["home_score_before"] = (
            df.groupby("contest_id")["home_score_after"].shift(fill_value=0).astype(int)
        )
        df["runs_on_play"] = np.where(
            df["half"] == "Top",
            (df["away_score_after"] - df["away_score_before"]).clip(lower=0),
            (df["home_score_after"] - df["home_score_before"]).clip(lower=0),
        ).astype(int)
    else:
        df["runs_on_play"] = runs_on_play(df["play_description"])
        df["away_score_before"] = score_before(
            df["game_end_fl"], df["runs_on_play"], df["half"], home_team=0
        )
        df["home_score_before"] = score_before(
            df["game_end_fl"], df["runs_on_play"], df["half"], home_team=1
        )
        df["home_score_after"] = df["home_score_before"] + np.where(
            df.half == "Bottom", df.runs_on_play, 0
        )
        df["away_score_after"] = df["away_score_before"] + np.where(
            df.half == "Top", df.runs_on_play, 0
        )

    df["runs_this_inn"] = runs_this_inn(df["inn_end_fl"], df["runs_on_play"])
    df["runs_roi"] = runs_rest_of_inn(df["inn_end_fl"], df["runs_on_play"], df["runs_this_inn"])
    return df


def parse_pbp_outs(df: pd.DataFrame) -> pd.DataFrame:
    df["outs_on_play"], df["outs_on_play_reason"] = outs_on_play(
        df["p1_text"], df["p2_text"], df["p3_text"], df["p4_text"]
    )
    df["outs_before"] = outs_before(df)
    df["outs_after"] = outs_after(df)
    return df


def parse_pbp_base_state(df: pd.DataFrame) -> pd.DataFrame:
    df = determine_batter_and_runners(df)
    df["bat_order"] = bat_order(df)
    df["event_type"] = classify_event_type(df)
    df["batted_ball_type"] = classify_batted_ball_type(df)

    return df


def add_team_names(pbp: pd.DataFrame, team_history: pd.DataFrame, year: int) -> pd.DataFrame:
    team_lookup = (
        team_history[team_history["year"] == year][["team_id", "team_name"]]
        .drop_duplicates()
        .assign(team_id=lambda d: d["team_id"].astype(str))
    )

    existing_away_team_name = pbp.get("away_team_name")
    existing_home_team_name = pbp.get("home_team_name")
    existing_bat_team_name = pbp.get("bat_team_name")
    existing_pitch_team_name = pbp.get("pitch_team_name")

    if existing_away_team_name is not None:
        pbp = pbp.rename(columns={"away_team_name": "_away_team_name_orig"})
    if existing_home_team_name is not None:
        pbp = pbp.rename(columns={"home_team_name": "_home_team_name_orig"})

    pbp = pbp.merge(
        team_lookup.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"}),
        on="away_team_id",
        how="left",
    )
    pbp = pbp.merge(
        team_lookup.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"}),
        on="home_team_id",
        how="left",
    )

    if existing_away_team_name is not None:
        pbp["away_team_name"] = pbp["away_team_name"].fillna(pbp["_away_team_name_orig"])
        pbp = pbp.drop(columns=["_away_team_name_orig"])
    if existing_home_team_name is not None:
        pbp["home_team_name"] = pbp["home_team_name"].fillna(pbp["_home_team_name_orig"])
        pbp = pbp.drop(columns=["_home_team_name_orig"])

    pbp["bat_team_id"] = np.where(pbp["half"] == "Top", pbp["away_team_id"], pbp["home_team_id"])
    pbp["pitch_team_id"] = np.where(pbp["half"] == "Top", pbp["home_team_id"], pbp["away_team_id"])

    bat_team_name_new = np.where(pbp["half"] == "Top", pbp["away_team_name"], pbp["home_team_name"])
    pitch_team_name_new = np.where(
        pbp["half"] == "Top", pbp["home_team_name"], pbp["away_team_name"]
    )

    pbp["bat_team_name"] = bat_team_name_new
    pbp["pitch_team_name"] = pitch_team_name_new

    if existing_bat_team_name is not None:
        pbp["bat_team_name"] = pbp["bat_team_name"].fillna(existing_bat_team_name)
    if existing_pitch_team_name is not None:
        pbp["pitch_team_name"] = pbp["pitch_team_name"].fillna(existing_pitch_team_name)

    pbp["bat_team_id"] = pbp["bat_team_id"].astype(str)
    pbp["pitch_team_id"] = pbp["pitch_team_id"].astype(str)

    return pbp



CubeByName = dict[tuple[int, int], list[tuple[str, str]]]
CubeByJersey = dict[tuple[int, int, int], str]


def _load_cube_players(data_dir: Path) -> tuple[CubeByName, CubeByJersey]:
    """Load cube stats into two lookups.

    by_name:   {(cube_college_id, year): [(player_name, player_id), ...]}
    by_jersey: {(cube_college_id, year, jersey): player_id}

    player_id is a string — raw integer (pre-hash) or 16-char hex (post-hash).
    Jersey lookup is preferred for pitchers; name fuzzy match is the fallback.
    """
    cube_stats_dir = data_dir / "cube_stats"
    if not cube_stats_dir.exists():
        return {}, {}

    frames = []
    for pattern in ("*_batting_*.csv", "*_pitching_*.csv"):
        for csv_path in sorted(cube_stats_dir.glob(pattern)):
            try:
                df = pd.read_csv(
                    csv_path,
                    usecols=["player_id", "player_name", "number", "team_id", "year"],
                    dtype={"player_id": str},
                )
                frames.append(df)
            except Exception:
                continue

    if not frames:
        return {}, {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["player_name", "team_id"])
    combined["player_id"] = combined["player_id"].astype(str).str.strip()
    combined = combined[~combined["player_id"].isin({"", "nan", "None", "--"})]
    combined = combined.drop_duplicates(subset=["player_id", "team_id", "year"])

    by_name: CubeByName = {}
    by_jersey: CubeByJersey = {}

    for _, row in combined.iterrows():
        team_year = (row["team_id"], int(row["year"]))
        name = str(row["player_name"]).strip()
        pid = row["player_id"]

        by_name.setdefault(team_year, []).append((name, pid))

        num = row.get("number")
        if pd.notna(num):
            m = re.match(r"\d+", str(num).strip())
            if m:
                jersey_key = (row["team_id"], int(row["year"]), int(m.group(0)))
                by_jersey[jersey_key] = pid

    return by_name, by_jersey


def _reconcile_player_ids(
    df: pd.DataFrame,
    cube_by_name: CubeByName,
    cube_by_jersey: CubeByJersey,
    year: int,
    threshold: int = 70,
) -> pd.DataFrame:
    if df.empty or "team_id" not in df.columns:
        return df

    df = df.copy()
    if "player_id" not in df.columns:
        df["player_id"] = pd.NA

    has_jersey = "number" in df.columns
    mask = df["player_id"].isna() & df["team_id"].notna()
    if not mask.any():
        return df

    for i in df[mask].index:
        row = df.loc[i]
        team_id = row["team_id"]

        if has_jersey and pd.notna(row.get("number")):
            jersey_key = (team_id, year, int(row["number"]))
            pid = cube_by_jersey.get(jersey_key)
            if pid is not None:
                df.at[i, "player_id"] = pid
                continue

        candidates = cube_by_name.get((team_id, year), [])
        if not candidates:
            continue

        player_name = str(row.get("player_name") or "").strip()
        if not player_name:
            continue

        candidate_names = [c[0] for c in candidates]
        match = rfuzz_process.extractOne(
            player_name, candidate_names, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        if match:
            idx = candidate_names.index(match[0])
            df.at[i, "player_id"] = candidates[idx][1]

    return df


def load_data(
    data_dir: Path, division: str, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = div_file_prefix(division)

    team_history_path = data_dir / "ncaa_team_history.csv"
    pbp_path = data_dir / f"pbp/{prefix}_pbp_{year}.csv"
    schedule_path = data_dir / f"schedules/{prefix}_schedules_{year}.csv"
    batting_path = data_dir / f"lineups/{prefix}_batting_lineups_{year}.csv"
    pitching_path = data_dir / f"lineups/{prefix}_pitching_lineups_{year}.csv"

    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP file not found: {pbp_path}")

    pbp = pd.read_csv(pbp_path)
    team_history = pd.read_csv(team_history_path) if team_history_path.exists() else pd.DataFrame()

    if team_history.empty:
        logger.warning("No team history found at %s", team_history_path)

    batting_lineups = pd.read_csv(batting_path) if batting_path.exists() else pd.DataFrame()
    pitching_lineups = pd.read_csv(pitching_path) if pitching_path.exists() else pd.DataFrame()

    if batting_lineups.empty:
        logger.warning("No batting lineups found at %s", batting_path)
    if pitching_lineups.empty:
        logger.warning("No pitching lineups found at %s", pitching_path)

    return pbp, team_history, batting_lineups, pitching_lineups


def _build_cube_fallback_df(
    cube_by_name: CubeByName,
    year: int,
) -> pd.DataFrame:
    rows = []
    for (team_id, yr), players in cube_by_name.items():
        if yr != year:
            continue
        for name, pid in players:
            rows.append({"team_id": team_id, "player_name": name, "player_id": pid})

    if not rows:
        return pd.DataFrame(columns=["team_id", "player_name", "player_id"])
    return pd.DataFrame(rows)


def main(data_dir: str, year: int, divisions: list[str]):
    data_dir = Path(data_dir)
    output_dir = data_dir / "pbp"
    output_dir.mkdir(parents=True, exist_ok=True)

    cube_by_name, cube_by_jersey = _load_cube_players(data_dir)
    cube_fallback_df = _build_cube_fallback_df(cube_by_name, year) if cube_by_name else None

    for division in divisions:
        try:
            pbp, team_history, batting_lineups, pitching_lineups = load_data(
                data_dir, division, year
            )
        except FileNotFoundError as e:
            logger.warning("%s, skipping %s", e, division_year_label(division, year))
            continue

        if cube_by_name or cube_by_jersey:
            batting_lineups = _reconcile_player_ids(
                batting_lineups, cube_by_name, cube_by_jersey, year
            )
            pitching_lineups = _reconcile_player_ids(
                pitching_lineups, cube_by_name, cube_by_jersey, year
            )

        parsed = parse_pbp(pbp, team_history, year, batting_lineups, pitching_lineups, cube_fallback_df=cube_fallback_df)
        if parsed.empty:
            logger.warning("No play by play processed for %s", division_year_label(division, year))
            continue

        output_path = output_dir / f"{division}_parsed_pbp_{year}.csv"
        parsed.to_csv(output_path, index=False)
        logger.info("Saved %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True, help="Root directory containing the data folders"
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--divisions", required=True, nargs="+", type=str)
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
