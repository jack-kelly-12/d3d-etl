from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from processors.logging_utils import div_file_prefix, division_year_label, get_logger
from processors.war_calculation.batting import calculate_batting_war, calculate_team_batting
from processors.war_calculation.constants import (
    BAT_STAT_COLUMNS,
    PITCH_STAT_COLUMNS,
    REP_WP,
    batting_columns,
    pitching_columns,
)
from processors.war_calculation.pitching import calculate_pitching_war, calculate_team_pitching
from processors.war_calculation.sos_utils import normalize_division_war, sos_reward_punish

logger = get_logger(__name__)


@dataclass
class DivisionData:
    batting: pd.DataFrame
    pitching: pd.DataFrame
    pbp: pd.DataFrame
    guts: pd.DataFrame
    park_factors: pd.DataFrame
    rankings: pd.DataFrame


def load_stats(data_dir: Path, division: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = div_file_prefix(division)
    cube_bat = data_dir / f"cube_stats/{division}_batting_{year}.csv"
    cube_pit = data_dir / f"cube_stats/{division}_pitching_{year}.csv"

    if cube_bat.exists() and cube_pit.exists():
        batting = pd.read_csv(cube_bat, dtype={"player_id": str})
        pitching = pd.read_csv(cube_pit, dtype={"player_id": str})
    else:
        batting = pd.read_csv(data_dir / f"cube_stats/{prefix}_batting_{year}.csv")
        pitching = pd.read_csv(data_dir / f"cube_stats/{prefix}_pitching_{year}.csv")

        map_path = data_dir / "ncaa_to_cube_player_map.csv"
        if map_path.exists() and "cube_player_id" in batting.columns:
            pmap = pd.read_csv(map_path, dtype={"cube_player_id": "Int64", "player_id": "str"})
            pmap = pmap.dropna(subset=["cube_player_id", "player_id"])
            pmap = pmap[pmap["year"] == year][["cube_player_id", "player_id"]].drop_duplicates(subset=["cube_player_id"])
            batting["cube_player_id"] = batting["cube_player_id"].astype("Int64")
            pitching["cube_player_id"] = pitching["cube_player_id"].astype("Int64")
            batting = batting.merge(pmap, on="cube_player_id", how="left")
            pitching = pitching.merge(pmap, on="cube_player_id", how="left")

    player_info_path = data_dir / "cube_stats/cube_player_info.csv"
    if player_info_path.exists() and "pos" not in batting.columns:
        player_info = pd.read_csv(player_info_path, dtype={"player_id": str}, usecols=["player_id", "positions"])
        player_info["pos"] = player_info["positions"].str.split(",").str[0].str.strip()
        player_info = player_info[["player_id", "pos"]]
        batting = batting.merge(player_info, on="player_id", how="left")

    for col in BAT_STAT_COLUMNS:
        if col in batting.columns:
            batting[col] = batting[col].fillna(0)
    for col in PITCH_STAT_COLUMNS:
        if col in pitching.columns:
            pitching[col] = pitching[col].fillna(0)

    return batting, pitching


def load_pbp(data_dir: Path, division: str, year: int) -> pd.DataFrame:
    prefix = div_file_prefix(division)
    return pd.read_csv(
        data_dir / f"pbp/{prefix}_pbp_with_metrics_{year}.csv",
        dtype={"player_id": str, "pitcher_id": str, "batter_id": str},
        low_memory=False,
    )


def load_rankings(data_dir: Path, division: str, year: int) -> pd.DataFrame:
    prefix = div_file_prefix(division)
    rankings = pd.read_csv(data_dir / f"rankings/{prefix}_rankings_{year}.csv")
    rankings["year"] = year
    rankings["division"] = division

    record_parts = rankings["record"].str.split("-", expand=True)
    rankings["wins"] = record_parts[0].astype(int)
    rankings["losses"] = record_parts[1].astype(int)
    rankings["ties"] = record_parts[2].fillna(0).astype(int) if record_parts.shape[1] > 2 else 0
    rankings["games"] = rankings["wins"] + rankings["losses"] + rankings["ties"]

    return rankings


def load_division_data(data_dir: Path, division: str, year: int) -> DivisionData:
    prefix = div_file_prefix(division)
    batting, pitching = load_stats(data_dir, division, year)

    guts = pd.read_csv(data_dir / "guts/guts_constants.csv")
    guts = guts[(guts["division"] == division) & (guts["year"] == int(year))]

    if guts.empty:
        raise ValueError(f"No GUTS constants for {division} {year}")

    return DivisionData(
        batting=batting,
        pitching=pitching,
        pbp=load_pbp(data_dir, division, year),
        guts=guts,
        park_factors=pd.read_csv(data_dir / f"park_factors/{prefix}_park_factors.csv"),
        rankings=load_rankings(data_dir, division, year),
    )


def filter_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]]


def process_division(
    data: DivisionData, mappings: pd.DataFrame, division: str, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Really hacky way to pass total games to batting war function for replacement runs
    data.guts = data.guts.copy()
    data.guts["total_games"] = data.pitching['gs'].sum() / 2
    batting_war, team_batting_clutch = calculate_batting_war(
        data.batting, data.guts, data.park_factors, data.pbp, division, year
    )

    pitching_war, team_pitching_clutch = calculate_pitching_war(
        data.pitching,
        data.pbp,
        data.park_factors,
        data.guts,
        bat_war_total=batting_war["war"].sum(),
        year=year,
        division=division,
    )

    batting_war, pitching_war, missing = sos_reward_punish(
        batting_war,
        pitching_war,
        data.rankings,
        mappings,
        division,
        year,
        alpha=0.2,
        clip_sd=3,
        group_keys=("year", "division"),
        harder_if="higher",
    )
    if missing:
        logger.info("  SoS missing for %s teams", len(missing))

    batting_team = calculate_team_batting(
        batting_war, data.guts, data.park_factors, team_batting_clutch, division, year
    )
    pitching_team = calculate_team_pitching(
        pitching_war, data.pbp, data.park_factors, data.guts, team_pitching_clutch, division, year
    )

    batting_war, pitching_war = normalize_division_war(
        batting_war, pitching_war, data.rankings, division, year
    )

    return batting_war, pitching_war, batting_team, pitching_team


def save_war_files(
    war_dir: Path,
    division: str,
    year: int,
    batting_war: pd.DataFrame,
    pitching_war: pd.DataFrame,
    batting_team: pd.DataFrame,
    pitching_team: pd.DataFrame,
):
    prefix = div_file_prefix(division)
    files = {
        f"{prefix}_batting_war_{year}.csv": filter_columns(batting_war, batting_columns),
        f"{prefix}_pitching_war_{year}.csv": filter_columns(pitching_war, pitching_columns),
        f"{prefix}_batting_team_war_{year}.csv": filter_columns(batting_team, batting_columns),
        f"{prefix}_pitching_team_war_{year}.csv": filter_columns(
            pitching_team, pitching_columns
        ),
    }

    for filename, df in files.items():
        (war_dir / filename).write_text(df.to_csv(index=False))


def calculate_war(data_dir: Path, year: int, divisions: list[str] = None):
    if divisions is None:
        divisions = ['ncaa_1', 'ncaa_2', 'ncaa_3']

    war_dir = data_dir / "war"
    war_dir.mkdir(exist_ok=True)

    mappings = pd.read_csv(data_dir / "team_mappings.csv")

    for division in divisions:
        logger.info("Processing %s...", division_year_label(division, year))

        try:
            data = load_division_data(data_dir, division, year)
        except FileNotFoundError as e:
            logger.warning("  Skipping: %s", e)
            continue
        except ValueError as e:
            logger.warning("  Skipping: %s", e)
            continue

        batting_war, pitching_war, batting_team, pitching_team = process_division(
            data, mappings, division, year
        )

        save_war_files(
            war_dir, division, year, batting_war, pitching_war, batting_team, pitching_team
        )

        standings = data.rankings
        target = standings["wins"].sum() - REP_WP * standings["games"].sum()
        actual = batting_war["war"].sum() + pitching_war["war"].sum()
        logger.info("  Target WAR: %.1f | Actual: %.1f", target, actual)

    logger.info("Done!")


def main(data_dir: str, year: int, divisions: list[str] = None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    calculate_war(data_dir, year, divisions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=str, default=['ncaa_1', 'ncaa_2', 'ncaa_3'])
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
