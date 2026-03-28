from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from processors.logging_utils import div_file_prefix, division_year_label, get_logger
from processors.war_calculation.calculator import WARCalculator
from processors.war_calculation.constants import BAT_STAT_COLUMNS, PITCH_STAT_COLUMNS, REP_WP
from processors.war_calculation.models import (
    BattingWarSchema,
    PitchingWarSchema,
    WarResults,
)

logger = get_logger(__name__)


@dataclass
class DivisionData:
    batting: pd.DataFrame
    pitching: pd.DataFrame
    pbp: pd.DataFrame
    guts: pd.DataFrame
    park_factors: pd.DataFrame
    rankings: pd.DataFrame
    lineups: pd.DataFrame


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
        cube_ids = batting.get("cube_player_id")
        if map_path.exists() and cube_ids is not None:
            pmap = pd.read_csv(map_path, dtype={"cube_player_id": "Int64", "player_id": "str"})
            pmap = pmap.dropna(subset=["cube_player_id", "player_id"])
            pmap = pmap[pmap["year"] == year][["cube_player_id", "player_id"]].drop_duplicates(subset=["cube_player_id"])
            batting["cube_player_id"] = batting["cube_player_id"].astype("Int64")
            pitching["cube_player_id"] = pitching["cube_player_id"].astype("Int64")
            batting = batting.merge(pmap, on="cube_player_id", how="left")
            pitching = pitching.merge(pmap, on="cube_player_id", how="left")

    player_info_path = data_dir / "cube_stats/cube_player_info.csv"
    if player_info_path.exists():
        player_info = pd.read_csv(player_info_path, dtype=str, usecols=["player_id", "positions"])
        player_info["pos"] = player_info["positions"].str.split(",").str[0].str.strip()
        pos_map = player_info.set_index("player_id")["pos"]
        existing_pos = batting.get("pos", pd.Series("", index=batting.index))
        batting["pos"] = existing_pos.where(
            existing_pos.notna() & (existing_pos != ""),
            batting["player_id"].map(pos_map),
        )

    batting = batting.assign(**dict.fromkeys(set(BAT_STAT_COLUMNS) - set(batting.columns), 0))
    batting[BAT_STAT_COLUMNS] = batting[BAT_STAT_COLUMNS].fillna(0)

    pitching = pitching.assign(**dict.fromkeys(set(PITCH_STAT_COLUMNS) - set(pitching.columns), 0))
    pitching[PITCH_STAT_COLUMNS] = pitching[PITCH_STAT_COLUMNS].fillna(0)

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


def load_lineups(data_dir: Path, division: str, year: int) -> pd.DataFrame:
    prefix = div_file_prefix(division)
    path = data_dir / f"lineups/{prefix}_batting_lineups_{year}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"player_id": str})
    return pd.DataFrame()


def load_division_data(data_dir: Path, division: str, year: int) -> DivisionData:
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
        park_factors=pd.read_csv(data_dir / "park_factors/pf.csv"),
        rankings=load_rankings(data_dir, division, year),
        lineups=load_lineups(data_dir, division, year),
    )


def process_division(
    data: DivisionData, mappings: pd.DataFrame, division: str, year: int
) -> WarResults:
    calculator = WARCalculator(
        batting_df=data.batting,
        pitching_df=data.pitching,
        pbp_df=data.pbp,
        guts_df=data.guts,
        park_factors_df=data.park_factors,
        lineups_df=data.lineups,
        rankings_df=data.rankings,
        mappings_df=mappings,
        division=division,
        year=year,
    )
    return calculator.run()


def save_war_files(
    war_dir: Path,
    division: str,
    year: int,
    results: WarResults,
):
    prefix = div_file_prefix(division)
    files = {
        f"{prefix}_batting_war_{year}.csv": BattingWarSchema.select(results.batting),
        f"{prefix}_pitching_war_{year}.csv": PitchingWarSchema.select(results.pitching),
        f"{prefix}_batting_team_war_{year}.csv": BattingWarSchema.select(results.batting_team),
        f"{prefix}_pitching_team_war_{year}.csv": PitchingWarSchema.select(results.pitching_team),
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

        results = process_division(data, mappings, division, year)

        save_war_files(war_dir, division, year, results)

        standings = data.rankings
        target = standings["wins"].sum() - REP_WP * standings["games"].sum()
        actual = results.batting["war"].sum() + results.pitching["war"].sum()
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
