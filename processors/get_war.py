from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from war_calculator.batting import calculate_batting_war, calculate_team_batting
from war_calculator.constants import REP_WP, batting_columns, pitching_columns
from war_calculator.pitching import calculate_pitching_war, calculate_team_pitching
from war_calculator.sos_utils import normalize_division_war, sos_reward_punish


@dataclass
class DivisionData:
    batting: pd.DataFrame
    pitching: pd.DataFrame
    pbp: pd.DataFrame
    guts: pd.DataFrame
    park_factors: pd.DataFrame
    rankings: pd.DataFrame


def load_stats(data_dir: Path, division: int, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = pd.read_csv(data_dir / f'stats/d{division}_batting_{year}.csv')
    pitching = pd.read_csv(data_dir / f'stats/d{division}_pitching_{year}.csv')

    roster = pd.read_csv(
                data_dir / f'rosters/d{division}_rosters_{year}.csv',
        dtype={'player_id': str, 'ncaa_id': 'Int64'}
            )
    roster = roster[(roster['year'] == year) & (roster['division'] == division)]

    batting['ncaa_id'] = batting['ncaa_id'].astype('Int64')
    pitching['ncaa_id'] = pitching['ncaa_id'].astype('Int64')

    batting = batting.merge(roster[['ncaa_id', 'player_id']], on='ncaa_id', how='left')
    pitching = pitching.merge(roster[['ncaa_id', 'player_id']], on='ncaa_id', how='left')

    return batting, pitching


def load_pbp(data_dir: Path, division: int, year: int) -> pd.DataFrame:
    return pd.read_csv(
            data_dir / f'pbp/d{division}_pbp_with_metrics_{year}.csv',
        dtype={'player_id': str, 'pitcher_id': str, 'batter_id': str},
        low_memory=False
        )


def load_rankings(data_dir: Path, division: int, year: int) -> pd.DataFrame:
    rankings = pd.read_csv(data_dir / f'rankings/d{division}_rankings_{year}.csv')
    rankings['year'] = year
    rankings['division'] = division

    record_parts = rankings['record'].str.split('-', expand=True)
    rankings['wins'] = record_parts[0].astype(int)
    rankings['losses'] = record_parts[1].astype(int)
    rankings['ties'] = record_parts[2].fillna(0).astype(int) if record_parts.shape[1] > 2 else 0
    rankings['games'] = rankings['wins'] + rankings['losses'] + rankings['ties']

    return rankings


def load_division_data(data_dir: Path, division: int, year: int) -> DivisionData:
    batting, pitching = load_stats(data_dir, division, year)

    guts = pd.read_csv(data_dir / 'guts/guts_constants.csv')
    guts = guts[(guts['division'] == division) & (guts['year'] == int(year))]

    if guts.empty:
        raise ValueError(f"No GUTS constants for D{division} {year}")

    return DivisionData(
        batting=batting,
        pitching=pitching,
        pbp=load_pbp(data_dir, division, year),
        guts=guts,
        park_factors=pd.read_csv(data_dir / f'park_factors/d{division}_park_factors.csv'),
        rankings=load_rankings(data_dir, division, year),
    )


def filter_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]]


def process_division(
    data: DivisionData,
    mappings: pd.DataFrame,
    division: int,
    year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    batting_war, team_batting_clutch = calculate_batting_war(
        data.batting, data.guts, data.park_factors, data.pbp, division, year
    )

    pitching_war, team_pitching_clutch = calculate_pitching_war(
        data.pitching, data.pbp, data.park_factors,
        bat_war_total=batting_war['war'].sum(),
        year=year, division=division
    )

    batting_war, pitching_war, missing = sos_reward_punish(
        batting_war, pitching_war, data.rankings, mappings,
        division, year, alpha=0.2, clip_sd=3,
        group_keys=('year', 'division'), harder_if='higher'
    )
    if missing:
        print(f"  SoS missing for {len(missing)} teams")

    batting_team = calculate_team_batting(
        batting_war, data.guts, data.park_factors, team_batting_clutch, division, year
    )
    pitching_team = calculate_team_pitching(
        pitching_war, data.park_factors, team_pitching_clutch, division, year
    )

    batting_war, pitching_war = normalize_division_war(
        batting_war, pitching_war, data.rankings, division, year
    )

    return batting_war, pitching_war, batting_team, pitching_team


def save_war_files(
    war_dir: Path,
    division: int,
    year: int,
    batting_war: pd.DataFrame,
    pitching_war: pd.DataFrame,
    batting_team: pd.DataFrame,
    pitching_team: pd.DataFrame
):
    files = {
        f'd{division}_batting_war_{year}.csv': filter_columns(batting_war, batting_columns),
        f'd{division}_pitching_war_{year}.csv': filter_columns(pitching_war, pitching_columns),
        f'd{division}_batting_team_war_{year}.csv': filter_columns(batting_team, batting_columns),
        f'd{division}_pitching_team_war_{year}.csv': filter_columns(pitching_team, pitching_columns),
    }

    for filename, df in files.items():
        (war_dir / filename).write_text(df.to_csv(index=False))


def calculate_war(data_dir: Path, year: int, divisions: list[int] = None):
    if divisions is None:
        divisions = [1, 2, 3]

    war_dir = data_dir / 'war'
    war_dir.mkdir(exist_ok=True)

    mappings = pd.read_csv(data_dir / 'team_mappings.csv')

    for division in divisions:
        print(f"Processing D{division} {year}...")

        try:
            data = load_division_data(data_dir, division, year)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        batting_war, pitching_war, batting_team, pitching_team = process_division(
            data, mappings, division, year
        )

        save_war_files(
            war_dir, division, year,
            batting_war, pitching_war, batting_team, pitching_team
        )

        standings = data.rankings
        target = standings['wins'].sum() - REP_WP * standings['games'].sum()
        actual = batting_war['war'].sum() + pitching_war['war'].sum()
        print(f"  Target WAR: {target:.1f} | Actual: {actual:.1f}")

    print("Done!")


def main(data_dir: str, year: int, divisions: list[int] = None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    calculate_war(data_dir, year, divisions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3])
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
