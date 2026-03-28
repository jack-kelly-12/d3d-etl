import gc
from pathlib import Path

import pandas as pd

from processors.logging_utils import division_year_label, get_logger

from .baserunning import calculate_baserunning_stats, calculate_team_baserunning
from .batted_ball import calculate_batted_ball_stats
from .common import filter_by_team_history, load_guts, load_linear_weights, load_pbp_with_hands
from .rolling import calculate_rolling_woba
from .situational import (
    analyze_batting_situations,
    analyze_batting_team_situations,
    analyze_pitching_situations,
    analyze_pitching_team_situations,
)
from .splits import (
    analyze_batting_splits,
    analyze_batting_team_splits,
    analyze_pitching_splits,
    analyze_pitching_team_splits,
)
from .value import analyze_value

logger = get_logger(__name__)

MIN_DATA_COLUMNS = {
    "batted_ball_batter": ("batted_balls", 1),
    "batted_ball_batting_team": ("batted_balls", 1),
    "batted_ball_pitcher": ("batted_balls", 1),
    "batted_ball_pitching_team": ("batted_balls", 1),
    "splits_batter": ("pa_overall", 1),
    "splits_pitcher": ("pa_overall", 1),
    "splits_batting_team": ("pa_overall", 1),
    "splits_pitching_team": ("pa_overall", 1),
    "situational_batter": ("pa_overall", 1),
    "situational_pitcher": ("pa_overall", 1),
    "situational_batting_team": ("pa_overall", 1),
    "situational_pitching_team": ("pa_overall", 1),
    "baserunning": ("games", 1),
    "baserunning_team": ("games", 1),
}


def drop_empty_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in MIN_DATA_COLUMNS:
        return df
    col, threshold = MIN_DATA_COLUMNS[name]
    if col in df.columns:
        df = df[pd.to_numeric(df[col], errors="coerce").fillna(0) >= threshold]
    return df


def run_analysis(data_dir: Path, year: int, division: str) -> dict[str, pd.DataFrame]:
    pbp_df = load_pbp_with_hands(data_dir, year, division)
    weights = load_linear_weights(data_dir, division, year)
    guts = load_guts(data_dir, division, year)

    results = {
        "situational_batter": analyze_batting_situations(pbp_df, weights),
        "situational_pitcher": analyze_pitching_situations(pbp_df, weights),
        "situational_batting_team": analyze_batting_team_situations(pbp_df, weights),
        "situational_pitching_team": analyze_pitching_team_situations(pbp_df, weights),
        "splits_batter": analyze_batting_splits(pbp_df, weights),
        "splits_pitcher": analyze_pitching_splits(pbp_df, weights),
        "splits_batting_team": analyze_batting_team_splits(pbp_df, weights),
        "splits_pitching_team": analyze_pitching_team_splits(pbp_df, weights),
        "batted_ball_batter": calculate_batted_ball_stats(pbp_df, type="batter"),
        "batted_ball_batting_team": calculate_batted_ball_stats(pbp_df, type="batter_team"),
        "batted_ball_pitcher": calculate_batted_ball_stats(pbp_df, type="pitcher"),
        "batted_ball_pitching_team": calculate_batted_ball_stats(pbp_df, type="pitcher_team"),
        "rolling_batter": calculate_rolling_woba(pbp_df, is_pitcher=False),
        "rolling_pitcher": calculate_rolling_woba(pbp_df, is_pitcher=True),
        "baserunning": calculate_baserunning_stats(pbp_df, guts),
        "baserunning_team": calculate_team_baserunning(pbp_df, guts),
    }

    value_results = analyze_value(pbp_df, data_dir, year, division)
    results.update(value_results)

    del pbp_df
    gc.collect()

    return results


def main(data_dir: str, year: int, divisions: list[str] = None):
    data_dir = Path(data_dir)
    if divisions is None:
        divisions = ['ncaa_1', 'ncaa_2', 'ncaa_3']

    leaderboards_dir = data_dir / "leaderboards"
    leaderboards_dir.mkdir(exist_ok=True)

    team_history_path = data_dir / "ncaa_team_history.csv"
    team_history = pd.read_csv(team_history_path) if team_history_path.exists() else pd.DataFrame()
    if team_history.empty:
        logger.warning("ncaa_team_history.csv not found or empty. Skipping team filtering.")

    output_files = {
        "situational_batter": [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "year",
            "division",
        ],
        "situational_pitcher": [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "year",
            "division",
        ],
        "situational_batting_team": ["team_id", "team_name", "year", "division"],
        "situational_pitching_team": ["team_id", "team_name", "year", "division"],
        "splits_batter": ["player_id", "player_name", "team_id", "team_name", "year", "division"],
        "splits_pitcher": ["player_id", "player_name", "team_id", "team_name", "year", "division"],
        "splits_batting_team": ["team_id", "team_name", "year", "division"],
        "splits_pitching_team": ["team_id", "team_name", "year", "division"],
        "batted_ball_batter": [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "year",
            "division",
        ],
        "batted_ball_batting_team": ["team_id", "team_name", "year", "division"],
        "batted_ball_pitcher": [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "year",
            "division",
        ],
        "batted_ball_pitching_team": ["team_id", "team_name", "year", "division"],
        "baserunning": ["player_id", "player_name", "team_id", "team_name", "year", "division"],
        "baserunning_team": ["team_id", "team_name", "year", "division"],
        "rolling_batter": ["player_id", "year", "division"],
        "rolling_pitcher": ["player_id", "year", "division"],
        "value_batter": ["player_id", "player_name", "team_id", "team_name", "year", "division"],
        "value_batting_team": ["team_id", "team_name", "year", "division"],
        "value_pitcher": ["player_id", "player_name", "team_id", "team_name", "year", "division"],
        "value_pitching_team": ["team_id", "team_name", "year", "division"],
    }

    for division in divisions:
        logger.info("Processing %s...", division_year_label(division, year))

        try:
            results = run_analysis(data_dir, year, division)

            for name, df in results.items():
                if df is None or df.empty:
                    continue

                df["year"] = int(year)
                df["division"] = division

                existing_file = leaderboards_dir / f"{name}.csv"
                data_list = []

                if existing_file.exists():
                    try:
                        existing_df = pd.read_csv(
                            existing_file,
                            dtype={"player_id": str, "batter_id": str, "pitcher_id": str},
                        )
                        existing_df = existing_df[
                            ~(
                                (existing_df["year"] == int(year))
                                & (existing_df["division"] == division)
                            )
                        ]
                        if not existing_df.empty:
                            data_list.append(existing_df)
                    except Exception as e:
                        logger.error("Error loading existing %s: %s", name, e)

                data_list.append(df)
                combined = pd.concat(data_list, ignore_index=True)

                combined = filter_by_team_history(combined, team_history)
                combined = drop_empty_rows(combined, name)

                dedup_cols = output_files[name]
                existing_cols = [c for c in dedup_cols if c in combined.columns]
                if existing_cols:
                    combined = combined.drop_duplicates(subset=existing_cols)

                if not team_history.empty and "team_id" in combined.columns:
                    combined["team_id"] = combined["team_id"].astype(str)
                    combined["division"] = combined["division"].astype(str)
                    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")

                    th_cols = [c for c in ["team_id", "division", "year", "conference", "team_name"] if c in team_history.columns]
                    team_info = team_history[th_cols].drop_duplicates().copy()
                    team_info["team_id"] = team_info["team_id"].astype(str)
                    team_info["division"] = team_info["division"].astype(str)
                    team_info["year"] = pd.to_numeric(team_info["year"], errors="coerce").astype("Int64")

                    overlap = set(combined["team_id"]) & set(team_info["team_id"])
                    if overlap:
                        merge_cols = ["team_id", "division", "year"]
                        pull_cols = [c for c in ["conference", "team_name"] if c in team_info.columns]
                        combined = combined.drop(columns=pull_cols, errors="ignore")
                        combined = combined.merge(team_info[merge_cols + pull_cols], on=merge_cols, how="left")

                combined.to_csv(existing_file, index=False)
                logger.info("Saved %s: %s records", name, len(combined))

                del combined, df
                if data_list:
                    del data_list

            del results
            gc.collect()

        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", division_year_label(division, year), e)
        except Exception as e:
            logger.exception("Error processing %s: %s", division_year_label(division, year), e)

    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--divisions", nargs="+", type=str, default=['ncaa_1', 'ncaa_2', 'ncaa_3'])
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
