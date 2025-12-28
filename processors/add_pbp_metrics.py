import os
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def get_data(year, division, data_dir):
    data_dir = Path(data_dir)

    pbp_file = data_dir / 'pbp' / f'd{division}_parsed_pbp_{year}.csv'
    roster_file = data_dir / 'rosters' / f'd{division}_rosters_{year}.csv'
    le_file = data_dir / 'miscellaneous' / 'leverage_index.csv'
    we_file = data_dir / 'miscellaneous' / 'win_expectancy.csv'
    re_file = data_dir / 'miscellaneous' / f'd{division}_expected_runs_{year}.csv'
    lw_file = data_dir / 'miscellaneous' / f'd{division}_linear_weights_{year}.csv'

    required_files = {
        'play-by-play': pbp_file,
        'roster': roster_file,
        'leverage index': le_file,
        'win expectancy': we_file,
        'run expectancy': re_file,
        'linear weights': lw_file
    }

    for name, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")

    pbp_df = pd.read_csv(pbp_file, low_memory=False)
    roster = pd.read_csv(roster_file, dtype={'player_id': str}, low_memory=False)
    le = pd.read_csv(le_file, low_memory=False)
    we = pd.read_csv(we_file, low_memory=False).rename(columns={'Tie': '0'})
    re = pd.read_csv(re_file, low_memory=False)
    lw = pd.read_csv(lw_file, low_memory=False)

    pbp_df['play_id'] = pbp_df.groupby('contest_id').cumcount() + 1
    pbp_df = pbp_df.sort_values(['contest_id', 'play_id'])

    pbp_df['top_inning'] = pbp_df.top_inning.replace({0: 'Bottom', 1: 'Top'})
    pbp_df['description'] = np.where(pbp_df['away_text'].isna(),
                                     pbp_df['home_text'],
                                     pbp_df['away_text'])
    pbp_df['bat_team'] = np.where(pbp_df['top_inning'] == 'Top',
                                  pbp_df['away_team'],
                                  pbp_df['home_team'])
    pbp_df['pitch_team'] = np.where(pbp_df['top_inning'] == 'Top',
                                    pbp_df['home_team'],
                                    pbp_df['away_team'])

    return pbp_df.dropna(subset=['description']), le, we, re, roster, lw



def standardize_names(pbp_df, roster, threshold=30):
    def format_name(name):
        if pd.isna(name):
            return name
        if ',' in name:
            last, first = name.split(',', 1)
            return f"{first.strip()} {last.strip()}"
        return name

    roster = roster.copy()
    roster['player_name'] = roster['player_name'].apply(format_name)
    pbp_df['bat_name'] = pbp_df['bat_name'].apply(format_name)
    pbp_df['pitcher'] = pbp_df['pitcher'].apply(format_name)

    roster_lookup = {
        team: group.set_index('player_name')['player_id'].to_dict()
        for team, group in roster.groupby('team_name')
    }

    pbp_df['player'] = pbp_df.apply(extract_player_from_description, axis=1)
    pbp_df['player'] = pbp_df['player'].fillna(pbp_df['bat_name'])

    def match_players(df, lookup, player_col, team_col, id_col, standardized_col):
        results = []
        for team, name in zip(df[team_col], df[player_col], strict=True):
            if pd.isna(name) or pd.isna(team):
                results.append((None, None))
                continue
            team_dict = lookup.get(team, {})
            if not team_dict:
                results.append((None, None))
                continue
            team_players = list(team_dict.keys())
            matches = process.extractOne(name,
                                         team_players,
                                         scorer=fuzz.token_sort_ratio,
                                         score_cutoff=threshold)
            if matches:
                standardized_name = matches[0]
                player_id = team_dict[standardized_name]
                results.append((standardized_name, player_id))
            else:
                results.append((None, None))
        df[standardized_col], df[id_col] = zip(*results, strict=True)

    pbp_df = pbp_df.copy()
    match_players(pbp_df, roster_lookup, 'pitcher', 'pitch_team', 'pitcher_id', 'pitcher_standardized')
    match_players(pbp_df, roster_lookup, 'bat_name', 'bat_team', 'batter_id', 'batter_standardized')
    match_players(pbp_df, roster_lookup, 'player', 'bat_team', 'player_id', 'player_standardized')

    pbp_df['times_through_order'] = pbp_df.groupby(['contest_id', 'pitcher', 'bat_name', 'bat_order']).cumcount() + 1
    pbp_df['pitcher_standardized'] = pbp_df['pitcher_standardized'].fillna(pbp_df['pitcher'])
    pbp_df['pitcher_standardized'] = pbp_df['pitcher_standardized'].fillna('Starter')
    return pbp_df

def extract_player_from_description(row):
    desc = row['description']
    if pd.isna(desc):
        return None
    if pd.notna(row['bat_name']):
        return row['bat_name']
    action_words = ['stole', 'advanced', 'scored', 'picked off', 'out at', 'caught stealing']
    for action in action_words:
        if action in desc:
            return desc.split(action)[0].strip()
    return None

def process_pitchers(df):
    df = df.copy()

    pitcher_changes = df['description'].str.contains('to p for', na=False)

    df['pitcher'] = pd.Series(dtype='string')
    df.loc[pitcher_changes, 'pitcher'] = df.loc[pitcher_changes,
                                                'sub_in']

    grouped = df.groupby(['contest_id', 'pitch_team'])

    def process_group(group):
        change_idx = group.index[pitcher_changes[group.index]]

        if len(change_idx) > 0:
            for i in range(len(change_idx)):
                start = change_idx[i-1] if i > 0 else group.index[0]
                end = change_idx[i]

                mask = (group.index >= start) & (group.index < end)
                group.loc[mask, 'pitcher'] = str(group.loc[end, 'sub_out'])

        group['pitcher'] = group['pitcher'].ffill()
        return group

    result = grouped.apply(process_group)

    df = result.reset_index(drop=True)

    return df


def calculate_woba(df, lw):
    weights = lw.set_index('events')['normalized_weight'].to_dict()

    df = df.copy()

    df['woba'] = 0.0

    df.loc[df['event_cd'] == 14, 'woba'] = weights.get('walk', 0)  # BB
    df.loc[df['event_cd'] == 16, 'woba'] = weights.get(
        'hit_by_pitch', 0)  # HBP
    df.loc[df['event_cd'] == 20, 'woba'] = weights.get('single', 0)  # 1B
    df.loc[df['event_cd'] == 21, 'woba'] = weights.get('double', 0)  # 2B
    df.loc[df['event_cd'] == 22, 'woba'] = weights.get('triple', 0)  # 3B
    df.loc[df['event_cd'] == 23, 'woba'] = weights.get('home_run', 0)  # HR

    return df


def melt_run_expectancy(df):
    df = df.copy()
    df = df[['bases', 'erv_0', 'erv_1', 'erv_2']]
    melted = pd.melt(
        df,
        id_vars=['bases'],
        value_vars=['erv_0', 'erv_1', 'erv_2'],
        var_name='outs',
        value_name='run_expectancy'
    )
    melted['outs'] = melted['outs'].str.extract(r'(\d+)').astype(int)
    melted = melted[['bases', 'outs', 'run_expectancy']]
    return melted


base_state_map = {
    '_ _ _': 0,
    '1B _ _': 1,
    '_ 2B _': 2,
    '1B 2B _': 3,
    '_ _ 3B': 4,
    '1B _ 3B': 5,
    '_ 2B 3B': 6,
    '1B 2B 3B': 7,
    '1B & 2B': 3,
    '1B & 3B': 5,
    '1B only': 1,
    '2B & 3B': 6,
    '2B only': 2,
    '3B only': 4,
    'Empty': 0,
    'Loaded': 7
}


def merge_baseball_stats(df, leverage_melted, win_expectancy, re_melted):
    df_copy = df.copy()
    max_innings = df_copy.groupby('contest_id')['inning'].max()
    inning_adjustments = 9 - max_innings

    inning_adjustment_map = inning_adjustments[inning_adjustments > 0].to_dict(
    )

    df_copy['effective_inning'] = df_copy['inning'].clip(upper=9)
    for contest_id, adjustment in inning_adjustment_map.items():
        mask = df_copy['contest_id'] == contest_id
        df_copy.loc[mask, 'effective_inning'] = df_copy.loc[mask,
                                                            'effective_inning'] + adjustment

    merged_df = df_copy.merge(
        leverage_melted,
        left_on=[
            'score_diff_before',
            'outs_before',
            'base_cd_before',
            'effective_inning',
            'top_inning'
        ],
        right_on=[
            'score_diff',
            'outs',
            'runners',
            'inning',
            'top/bot'
        ],
        how='left'
    ).drop(columns=['outs', 'runners', 'inning_y', 'top/bot', 'score_diff']).rename(columns={'leverage_index': 'li', 'inning_x': 'inning'})

    merged_df = merged_df.merge(
        win_expectancy,
        left_on=[
            'score_diff_before',
            'outs_before',
            'base_cd_before',
            'effective_inning',
            'top_inning'
        ],
        right_on=[
            'score_diff',
            'outs',
            'runners',
            'inn',
            'top/bot'
        ],
        how='left'
    ).drop(columns=['outs', 'runners', 'inn', 'top/bot', 'score_diff']).rename(columns={'win_expectancy': 'home_win_exp_before'})

    merged_df = merged_df.merge(
        re_melted,
        left_on=[
            'outs_before',
            'base_cd_before'
        ],
        right_on=[
            'outs',
            'bases'
        ],
        how='left'
    ).drop(columns=['outs', 'bases']).rename(columns={'run_expectancy': 'run_expectancy_before'})

    merged_df.loc[abs(merged_df['score_diff_before']) >= 10, 'li'] = 0
    merged_df.loc[merged_df['score_diff_before']
                  >= 10, 'home_win_exp_before'] = 1
    merged_df.loc[merged_df['score_diff_before']
                  <= -10, 'home_win_exp_before'] = 0

    merged_df['outs_after_new'] = merged_df['outs_after'] % 3

    inning_transition = ((merged_df['inn_end_flag'] == 1) & (
        merged_df['outs_after'] >= 3))
    game_transition = (merged_df['game_end_flag'] == 1)

    merged_df['base_cd_after'] = np.where(
        inning_transition,
        0,
        merged_df['base_cd_after']
    )
    merged_df['top_inning_new'] = np.where(
        inning_transition,
        np.where(merged_df['top_inning'] == 'Top', 'Bottom', 'Top'),
        merged_df['top_inning']
    )
    merged_df['effective_inning_new'] = np.where(
        (inning_transition) & (merged_df['top_inning'] == 'Bottom'),
        merged_df['effective_inning'] + 1,
        merged_df['effective_inning']
    )
    merged_df['effective_inning_new'] = merged_df['effective_inning_new'].clip(
        upper=9)

    merged_df = merged_df.merge(
        re_melted,
        left_on=[
            'outs_after_new',
            'base_cd_after'
        ],
        right_on=[
            'outs',
            'bases'
        ],
        how='left'
    ).drop(columns=['outs', 'bases']).rename(columns={'run_expectancy': 'run_expectancy_after'})

    merged_df.loc[inning_transition, 'run_expectancy_after'] = 0
    merged_df.loc[game_transition, 'run_expectancy_after'] = 0

    merged_df = merged_df.merge(
        win_expectancy,
        left_on=[
            'score_diff_after',
            'outs_after_new',
            'base_cd_after',
            'effective_inning_new',
            'top_inning_new'
        ],
        right_on=[
            'score_diff',
            'outs',
            'runners',
            'inn',
            'top/bot'
        ],
        how='left'
    ).drop(columns=['outs', 'runners', 'inn', 'top/bot', 'score_diff']).rename(columns={'win_expectancy': 'home_win_exp_after'})

    merged_df.loc[merged_df['score_diff_after']
                  >= 10, 'home_win_exp_after'] = 1
    merged_df.loc[merged_df['score_diff_after']
                  <= -10, 'home_win_exp_after'] = 0
    merged_df.loc[abs(merged_df['score_diff_after']) >= 10, 'li'] = 0

    merged_df['game_end_flag'] = np.where(
        (merged_df['score_diff_after'] > 0) &
        (merged_df['effective_inning'] == 9) &
        (merged_df['top_inning'] == 'Bottom'),
        1,
        merged_df.game_end_flag
    )

    merged_df['game_end_flag'] = np.where(
        (merged_df['score_diff_after'] > 0) &
        (merged_df['effective_inning'] == 9) &
        (merged_df['top_inning'] == 'Top') &
        (merged_df['outs_after'] == 3),
        1,
        merged_df.game_end_flag
    )

    fix_all_game_ends(df)

    return merged_df


def fix_all_game_ends(df):
    first_ends = df[df['game_end_flag'] == 1].groupby('contest_id').first()

    rows_to_drop = []
    for idx, row in df.iterrows():
        contest_id = row['contest_id']
        if contest_id in first_ends.index:
            if idx > first_ends.loc[contest_id].name:
                rows_to_drop.append(idx)

    df.drop(rows_to_drop, inplace=True)


def calculate_dre_and_dwe(df):
    df = df.loc[:, ~df.columns.duplicated()]

    df['run_expectancy_after'] = df['run_expectancy_after'].copy()

    game_end_mask = df['game_end_flag'] == 1
    home_wins_mask = df['home_score_after'] > df['away_score_after']

    df.loc[game_end_mask & home_wins_mask, 'home_win_exp_after'] = 1
    df.loc[game_end_mask & ~home_wins_mask, 'home_win_exp_after'] = 0

    df['run_expectancy_delta'] = df['run_expectancy_after'] - \
        df['run_expectancy_before']
    df['delta_home_win_exp'] = df['home_win_exp_after'] - \
        df['home_win_exp_before']

    df = df[(df['delta_home_win_exp'] > -0.97) &
            (df['delta_home_win_exp'] < 0.97)]

    df['rea'] = df['run_expectancy_delta'] + df['runs_on_play']

    df['wpa'] = np.where(
        df['bat_team'] == df['home_team'],
        df['delta_home_win_exp'],
        -df['delta_home_win_exp']
    )
    df['wpa'] = df['wpa'].fillna(0)

    df['wpa/li'] = df['wpa'] / df['li'].replace(0, np.nan)
    df['wpa/li'] = df['wpa/li'].fillna(0)

    return df


def process_single_year(args):
    year, division, data_dir = args
    try:
        pbp_df, leverage_melted, win_expectancy, re, roster, lw = get_data(
            year, division, data_dir)

        pbp_df['top_inning'] = np.where(
            pbp_df.away_team == pbp_df.bat_team, 'Top', 'Bottom')
    except FileNotFoundError as e:
        print(f"Error loading data for {year}: {e}")
        return None

    pbp_processed = process_pitchers(pbp_df)
    pbp_processed = standardize_names(pbp_processed, roster)
    pbp_processed = calculate_woba(pbp_processed, lw)

    pbp_processed['base_cd_after'] = pbp_processed.groupby(
        'contest_id')['base_cd_before'].shift(-1)
    pbp_processed['base_cd_after'] = pbp_processed['base_cd_after'].fillna(0)

    pbp_processed['score_diff_before'] = pbp_processed['home_score_before'] - \
        pbp_processed['away_score_before']
    pbp_processed['score_diff_after'] = pbp_processed['home_score_after'] - \
        pbp_processed['away_score_after']

    pbp_processed = pbp_processed.dropna(subset=['top_inning'])
    pbp_processed['outs_before'] = pbp_processed['outs_after'] - \
        pbp_processed['outs_on_play']

    re_melted = melt_run_expectancy(re)

    leverage_melted['runners'] = pd.to_numeric(
        leverage_melted['runners'].replace(base_state_map))
    win_expectancy['runners'] = pd.to_numeric(
        win_expectancy['runners'].replace(base_state_map))
    re_melted['bases'] = pd.to_numeric(
        re_melted['bases'].replace(base_state_map))
    pbp_processed['base_cd_after'] = pd.to_numeric(pbp_processed['base_cd_after'].replace(
        base_state_map))

    merged_df = merge_baseball_stats(pbp_processed, leverage_melted, win_expectancy, re_melted).drop_duplicates(
        ['contest_id', 'inning', 'home_score_after', 'away_score_after', 'home_text', 'away_text']).dropna(subset=['description'])
    merged_df = merged_df[~((merged_df['top_inning_new'] == 'Bottom') & (merged_df['score_diff_after'] > 0) & (
        merged_df['effective_inning'] == 9) & (merged_df['game_end_flag'] == 0))]
    merged_df = calculate_dre_and_dwe(merged_df)

    output_path = os.path.join(
        data_dir, 'pbp', f'd{division}_parsed_pbp_new_{year}.csv')
    merged_df = merged_df.sort_values(['contest_id',
                                       'play_id'], ascending=True)
    columns = [
        'home_team', 'away_team', 'home_score', 'away_score', 'date',
        'inning', 'top_inning', 'contest_id', 'description',
        'home_win_exp_before', 'wpa', 'run_expectancy_delta', 'woba', 'home_win_exp_after',
        'player_id', 'pitcher_id', 'batter_id', 'li', 'home_score_after',
        'away_score_after', 'event_cd', 'times_through_order', 'base_cd_before', 'base_cd_after',
        'hit_type'
    ]

    merged_df.to_csv(output_path, index=False)

    final_df = merged_df[columns].copy()
    final_df['year'] = year
    final_df['division'] = division

    return final_df


def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    if divisions is None:
        divisions = [1, 2, 3]
    all_pbp_data = []

    for division in divisions:
        processed_data = process_single_year((year, division, data_dir))
        if processed_data is not None:
            all_pbp_data.append(processed_data)
            print(f"Successfully processed data for D{division} {year}")
        else:
            print(f"Failed to process data for D{division} {year}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
