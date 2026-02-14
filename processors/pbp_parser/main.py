from pathlib import Path

import numpy as np
import pandas as pd

from processors.logging_utils import division_year_label, get_logger

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
    roster: pd.DataFrame,
) -> pd.DataFrame:
    pbp_metadata = metadata(pbp)
    pbp_metadata = pbp_metadata.sort_values(["contest_id", "play_id"], kind="stable")

    pbp_with_teams = add_team_names(pbp_metadata, team_history, year)

    pbp_flags = flags(pbp_with_teams)

    pbp_outs = parse_pbp_outs(pbp_flags)
    pbp_runs = parse_pbp_runs(pbp_outs)
    pbp_bases = parse_pbp_base_state(pbp_runs)

    pbp_final = standardize_names(pbp_bases, batting_lineups, pitching_lineups, roster)

    return pbp_final


def parse_pbp_runs(df: pd.DataFrame) -> pd.DataFrame:
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
    team_lookup = team_history[team_history["year"] == year][
        ["team_id", "team_name"]
    ].drop_duplicates()
    team_lookup["team_id"] = team_lookup["team_id"].astype("Int64")

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

    pbp["bat_team_id"] = pbp["bat_team_id"].astype("Int64")
    pbp["pitch_team_id"] = pbp["pitch_team_id"].astype("Int64")

    return pbp


def load_data(
    data_dir: Path, division: int, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    div_name = f"d{division}"

    team_history_path = data_dir / "ncaa_team_history.csv"
    pbp_path = data_dir / f"pbp/{div_name}_pbp_{year}.csv"
    batting_path = data_dir / f"lineups/{div_name}_batting_lineups_{year}.csv"
    pitching_path = data_dir / f"lineups/{div_name}_pitching_lineups_{year}.csv"
    roster_path = data_dir / f"rosters/{div_name}_rosters_{year}.csv"

    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP file not found: {pbp_path}")

    pbp = pd.read_csv(pbp_path)
    team_history = pd.read_csv(team_history_path) if team_history_path.exists() else pd.DataFrame()

    if team_history.empty:
        logger.warning("No team history found at %s", team_history_path)

    batting_lineups = pd.read_csv(batting_path) if batting_path.exists() else pd.DataFrame()
    pitching_lineups = pd.read_csv(pitching_path) if pitching_path.exists() else pd.DataFrame()
    roster = pd.read_csv(roster_path) if roster_path.exists() else pd.DataFrame()

    if batting_lineups.empty:
        logger.warning("No batting lineups found at %s", batting_path)
    if pitching_lineups.empty:
        logger.warning("No pitching lineups found at %s", pitching_path)
    if roster.empty:
        logger.warning("No roster found at %s", roster_path)

    return pbp, team_history, batting_lineups, pitching_lineups, roster


def main(data_dir: str, year: int, divisions: list[int]):
    data_dir = Path(data_dir)
    output_dir = data_dir / "pbp"
    output_dir.mkdir(parents=True, exist_ok=True)

    for division in divisions:
        div_name = f"d{division}"

        try:
            pbp, team_history, batting_lineups, pitching_lineups, roster = load_data(
                data_dir, division, year
            )
        except FileNotFoundError as e:
            logger.warning("%s, skipping %s", e, division_year_label(division, year))
            continue

        parsed = parse_pbp(pbp, team_history, year, batting_lineups, pitching_lineups, roster)
        if parsed.empty:
            logger.warning("No play by play processed for %s", division_year_label(division, year))
            continue

        output_path = output_dir / f"{div_name}_parsed_pbp_{year}.csv"
        parsed.to_csv(output_path, index=False)
        logger.info("Saved %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True, help="Root directory containing the data folders"
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--divisions", required=True, nargs="+", type=int)
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
