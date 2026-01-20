from pathlib import Path

import numpy as np
import pandas as pd
from pbp_parser.constants import EventType


def load_data(data_dir: Path, division: int, year: int) -> dict:
    """Load all required data files."""
    paths = {
        'pbp': data_dir / 'pbp' / f'd{division}_parsed_pbp_{year}.csv',
        'leverage': data_dir / 'miscellaneous' / 'leverage_index.csv',
        'win_exp': data_dir / 'miscellaneous' / 'win_expectancy.csv',
        'run_exp': data_dir / 'miscellaneous' / f'd{division}_expected_runs_{year}.csv',
        'linear_weights': data_dir / 'miscellaneous' / f'd{division}_linear_weights_{year}.csv',
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    return {
        'pbp': pd.read_csv(paths['pbp'], low_memory=False),
        'leverage': pd.read_csv(paths['leverage']),
        'win_exp': pd.read_csv(paths['win_exp']).rename(columns={'Tie': '0'}),
        'run_exp': pd.read_csv(paths['run_exp']),
        'linear_weights': pd.read_csv(paths['linear_weights']),
    }

WOBA_EVENTS = {
    EventType.WALK: 'walk',
    EventType.INTENTIONAL_WALK: 'walk',
    EventType.HIT_BY_PITCH: 'hit_by_pitch',
    EventType.SINGLE: 'single',
    EventType.DOUBLE: 'double',
    EventType.TRIPLE: 'triple',
    EventType.HOME_RUN: 'home_run',
}

def add_woba(df: pd.DataFrame, lw: pd.DataFrame) -> pd.DataFrame:
    """Add wOBA column based on event type and linear weights."""
    weights = lw.set_index('events')['normalized_weight'].to_dict()

    df['woba'] = 0.0
    for event_type, weight_key in WOBA_EVENTS.items():
        mask = df['event_type'] == event_type.value
        df.loc[mask, 'woba'] = weights.get(weight_key, 0)

    return df

def build_re_lookup(re_df: pd.DataFrame) -> dict:
    """Build lookup dict: (bases, outs) -> run_expectancy."""
    lookup = {}
    for _, row in re_df.iterrows():
        bases = row['bases']
        for outs in range(3):
            lookup[(bases, outs)] = row[f'erv_{outs}']
    return lookup


def add_run_expectancy(df: pd.DataFrame, re_lookup: dict) -> pd.DataFrame:
    """Add run expectancy before and after each play."""
    df['re_before'] = df.apply(
        lambda r: re_lookup.get((r['bases_before'], int(r['outs_before'])), 0.0)
        if pd.notna(r['bases_before']) and pd.notna(r['outs_before']) else 0.0,
        axis=1
    )

    def get_re_after(row):
        if row.get('inn_end_fl') == 1:
            return 0.0
        bases = row.get('bases_after')
        outs = row.get('outs_after', 0) % 3
        if pd.isna(bases) or pd.isna(outs):
            return 0.0
        return re_lookup.get((bases, int(outs)), 0.0)

    df['re_after'] = df.apply(get_re_after, axis=1)
    df['re_delta'] = df['re_after'] - df['re_before']
    df['rea'] = df['re_delta'] + df['runs_on_play'].fillna(0)

    return df


def build_we_lookup(we_df: pd.DataFrame) -> dict:
    """Build lookup dict: (inning, half, bases, outs, score_diff) -> win_exp."""
    lookup = {}
    for _, row in we_df.iterrows():
        key = (int(row['inning']), row['half'], row['runners'], int(row['outs']), int(row['score_diff']))
        lookup[key] = row['win_expectancy']
    return lookup


def build_li_lookup(li_df: pd.DataFrame) -> dict:
    """Build lookup dict: (inning, half, bases, outs, score_diff) -> leverage_index."""
    lookup = {}
    for _, row in li_df.iterrows():
        key = (int(row['inning']), row['half'], row['runners'], int(row['outs']), int(row['score_diff']))
        lookup[key] = row['leverage_index']
    return lookup


def bases_to_runners(bases: str) -> str:
    if pd.isna(bases) or not isinstance(bases, str) or len(bases) != 3:
        return 'NNN'
    return ''.join('Y' if c.isdigit() else 'N' for c in bases)


def add_win_expectancy_and_leverage(
    df: pd.DataFrame,
    we_lookup: dict,
    li_lookup: dict
) -> pd.DataFrame:
    """Add win expectancy (before/after), leverage index, and WPA."""

    max_inn = df.groupby('contest_id')['inning'].transform('max')
    df['_eff_inn'] = (df['inning'] + (9 - max_inn).clip(lower=0)).clip(upper=9)

    df['score_diff_before'] = df['home_score_before'] - df['away_score_before']
    df['score_diff_after'] = df['home_score_after'] - df['away_score_after']

    def clamp_diff(x):
        return int(np.clip(x, -15, 15)) if pd.notna(x) else 0

    def get_we_before(row):
        diff = clamp_diff(row['score_diff_before'])
        if abs(diff) >= 15:
            return 1.0 if diff >= 15 else 0.0
        runners = bases_to_runners(row['bases_before'])
        key = (int(row['_eff_inn']), row['half'], runners, int(row['outs_before']), diff)
        return we_lookup.get(key)

    df['home_win_exp_before'] = df.apply(get_we_before, axis=1)

    def get_li(row):
        diff = clamp_diff(row['score_diff_before'])
        if abs(diff) >= 10:
            return 0.0
        runners = bases_to_runners(row['bases_before'])
        key = (int(row['_eff_inn']), row['half'], runners, int(row['outs_before']), diff)
        return li_lookup.get(key, 1.0)

    df['li'] = df.apply(get_li, axis=1)

    def get_we_after(row):
        diff = clamp_diff(row['score_diff_after'])

        if row.get('game_end_fl') == 1:
            return 1.0 if row['home_score_after'] > row['away_score_after'] else 0.0

        if abs(diff) >= 10:
            return 1.0 if diff >= 10 else 0.0

        eff_inn = int(row['_eff_inn'])
        half = row['half']
        runners = bases_to_runners(row['bases_after'])
        outs = int(row.get('outs_after', 0)) % 3

        if row.get('inn_end_fl') == 1:
            runners = 'NNN'
            outs = 0
            if half == 'Bottom':
                eff_inn = min(eff_inn + 1, 9)
            half = 'Top' if half == 'Bottom' else 'Bottom'

        key = (eff_inn, half, runners, outs, diff)
        return we_lookup.get(key)

    df['home_win_exp_after'] = df.apply(get_we_after, axis=1)

    df['delta_home_win_exp'] = df['home_win_exp_after'] - df['home_win_exp_before']

    is_home_batting = df['half'] == 'Bottom'
    df['wpa'] = np.where(is_home_batting, df['delta_home_win_exp'], -df['delta_home_win_exp'])
    df['wpa'] = df['wpa'].fillna(0)

    non_play_mask = (
        (df['sub_fl'] == 1) |
        (df['event_type'] == EventType.NO_PLAY.value) |
        (df['event_type'] == EventType.UNKNOWN.value)
    )
    df.loc[non_play_mask, 'wpa'] = 0.0
    df.loc[non_play_mask, 'delta_home_win_exp'] = 0.0

    df['wpa_li'] = np.where(df['li'] > 0, df['wpa'] / df['li'], 0)

    df.drop(columns=['_eff_inn'], inplace=True)

    return df

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['times_through_order'] = df.groupby(['contest_id', 'pitcher_name', 'batter_name', 'bat_order']).cumcount() + 1
    df['high_leverage_fl'] = df['li'] >= 2
    df['low_leverage_fl'] = df['li'] <= 0.85
    return df



def process_division(data_dir: Path, division: int, year: int) -> pd.DataFrame:
    """Process a single division's play-by-play data."""
    print(f"Processing D{division} {year}...")

    data = load_data(data_dir, division, year)
    df = data['pbp']

    re_lookup = build_re_lookup(data['run_exp'])
    we_lookup = build_we_lookup(data['win_exp'])
    li_lookup = build_li_lookup(data['leverage'])

    df = add_woba(df, data['linear_weights'])
    df = add_run_expectancy(df, re_lookup)
    df = add_win_expectancy_and_leverage(df, we_lookup, li_lookup)
    df = add_flags(df)

    df = df.sort_values(['contest_id', 'play_id'])

    df = df.drop(columns=['away_text', 'home_text', 'p1_text', 'p2_text', 'p3_text', 'p4_text'])

    output_path = data_dir / 'pbp' / f'd{division}_pbp_with_metrics_{year}.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


def main(data_dir: str, year: int, divisions: list[int]):
    """Process play-by-play data for specified divisions."""
    data_dir = Path(data_dir)

    for division in divisions:
        try:
            process_division(data_dir, division, year)
        except FileNotFoundError as e:
            print(f"Skipping D{division} {year}: {e}")
        except Exception as e:
            print(f"Error processing D{division} {year}: {e}")
            raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add advanced metrics to PBP data')
    parser.add_argument('--data_dir', required=True, help='Root data directory')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3])
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
