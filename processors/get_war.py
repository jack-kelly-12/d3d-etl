from pathlib import Path

import pandas as pd

from war_utils.constants import batting_columns, pitching_columns, REP_WP
from war_utils.stats import (
    calculate_batting_war,
    calculate_pitching_war,
    calculate_batting_team_war,
    calculate_pitching_team_war,
)
from war_utils.sos_utils import (
    sos_reward_punish,
    normalize_division_war,
)

def get_data(year, data_dir, divisions=None):
    if divisions is None:
        divisions = [1, 2, 3]
    pitching, batting, pbp, rosters, guts, park_factors, rankings = {}, {}, {}, {}, {}, {}, {}

    for division in divisions:
        pitch_raw = pd.read_csv(data_dir / f'stats/d{division}_pitching_{year}.csv')
        bat_raw = pd.read_csv(data_dir / f'stats/d{division}_batting_{year}.csv')

        ros = (
            pd.read_csv(
                data_dir / f'rosters/d{division}_rosters_{year}.csv',
                dtype={'player_id': str, 'ncaa_id': str}
            )
            .query(f'year == {year}')
            .query(f'division == {division}')
        )

        bat_raw['ncaa_id'] = bat_raw['ncaa_id'].astype(str)
        pitch_raw['ncaa_id'] = pitch_raw['ncaa_id'].astype(str)

        bat = bat_raw.merge(ros[['ncaa_id', 'player_id']], on='ncaa_id', how='left')
        pitch = pitch_raw.merge(ros[['ncaa_id', 'player_id']], on='ncaa_id', how='left')

        batting[division] = bat
        pitching[division] = pitch
        rosters[division] = ros

        pbp[division] = pd.read_csv(
            data_dir / f'pbp/d{division}_parsed_pbp_new_{year}.csv',
            dtype={'player_id': str, 'pitcher_id': str}, low_memory=False
        )

        park_factors[division] = pd.read_csv(data_dir / f'park_factors/d{division}_park_factors.csv')

        g = pd.read_csv(data_dir / 'guts/guts_constants.csv')
        guts[division] = g[(g['division'] == division) & (g['year'] == int(year))]

        r = pd.read_csv(data_dir / f'rankings/d{division}_rankings_{year}.csv')
        r['year'] = year
        r['division'] = division
        r['wins'] = r['record'].str.split('-').str[0].astype(int)
        r['losses'] = r['record'].str.split('-').str[1].astype(int)
        r['ties'] = r['record'].str.split('-').str[2].fillna(0).astype(int)
        r['games'] = r['wins'] + r['losses'] + r['ties']
        rankings[division] = r

    mappings = pd.read_csv(data_dir / 'team_mappings.csv')
    return batting, pitching, pbp, guts, park_factors, rosters, rankings, mappings


def calculate_war(data_dir, year, divisions=None):
    if divisions is None:
        divisions = [1, 2, 3]
    batting, pitching, pbp, guts, park_factors, rosters, rankings, mappings = get_data(year, data_dir, divisions)

    war_dir = Path(data_dir) / 'war'
    war_dir.mkdir(exist_ok=True)

    for division in divisions:
        print(f"Processing Division {division}, Year {year}")

        batting_df = batting[division]
        pitching_df = pitching[division]
        pbp_df = pbp[division]
        guts_df = guts[division]
        park_factors_df = park_factors[division]
        rosters_df = rosters[division]
        rankings_df = rankings[division]

        batting_war, team_batting_clutch = calculate_batting_war(
            batting_df, guts_df, park_factors_df, pbp_df, division, year
        )
        pitching_war, team_pitching_clutch = calculate_pitching_war(
            pitching_df, pbp_df, park_factors_df, batting_war.war.sum(), year, division
        )

        batting_war, pitching_war, missing = sos_reward_punish(
            batting_war, pitching_war, rankings_df, mappings, division, year,
            alpha=0.2, clip_sd=3, group_keys=('year', 'division'), harder_if='higher'
        )
        if missing:
            print(f"[d{division} {year}] SoS missing -> filled with min SoS; unique teams affected: {len(missing)}")

        batting_team_war = calculate_batting_team_war(batting_war, guts_df, park_factors_df, team_batting_clutch, division, year)
        pitching_team_war = calculate_pitching_team_war(pitching_war, park_factors_df, team_pitching_clutch, division, year)

        batting_war, pitching_war = normalize_division_war(batting_war, pitching_war, rankings_df, division, year)

        batting_war = batting_war[[c for c in batting_columns if c in batting_war.columns]]
        pitching_war = pitching_war[[c for c in pitching_columns if c in pitching_war.columns]]
        batting_team_war = batting_team_war[[c for c in batting_columns if c in batting_team_war.columns]]
        pitching_team_war = pitching_team_war[[c for c in pitching_columns if c in pitching_team_war.columns]]

        (war_dir / f'd{division}_pitching_team_war_{year}.csv').write_bytes(pitching_team_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_batting_team_war_{year}.csv').write_bytes(batting_team_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_pitching_war_{year}.csv').write_bytes(pitching_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_batting_war_{year}.csv').write_bytes(batting_war.to_csv(index=False).encode('utf-8'))

        s = rankings_df[(rankings_df['division'] == division) & (rankings_df['year'] == year)]
        target_total = s['wins'].sum() - REP_WP * s['games'].sum()
        current_total = batting_war['war'].sum() + pitching_war['war'].sum()
        print(f"[d{division} {year}] Target WAR≈{target_total:.2f} | Actual WAR≈{current_total:.2f}")


def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    calculate_war(data_dir, year, divisions)
    print("Successfully processed all statistics!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Root directory containing the data folders')
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()
    main(args.data_dir, args.year, args.divisions)
