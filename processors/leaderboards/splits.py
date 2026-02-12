import pandas as pd

from .common import calculate_batting_metrics


def analyze_batting_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df['batter_name'].isna()].copy()

    splits = [
        ('vs_lhp', df[df['pitcher_hand'] == 'L']),
        ('vs_rhp', df[df['pitcher_hand'] == 'R']),
        ('overall', df)
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = (
            data.groupby(['batter_id', 'batter_name', 'bat_team_name', 'bat_team_id'])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped['split'] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=['batter_id', 'batter_name', 'bat_team_name', 'bat_team_id'],
        columns='split',
        values=['woba', 'ba', 'pa', 'rea', 'obp', 'slg']
    )
    pivot.columns = [f'{stat}_{split}' for stat, split in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={
        'batter_id': 'player_id',
        'batter_name': 'player_name',
        'bat_team_name': 'team_name',
        'bat_team_id': 'team_id'
    })


def analyze_pitching_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df['bat_order'].isna()].copy()

    splits = [
        ('vs_lhh', df[(df['batter_hand'] == 'L') |
                      ((df['pitcher_hand'] == 'R') & (df['batter_hand'] == 'S'))]),
        ('vs_rhh', df[(df['batter_hand'] == 'R') |
                      ((df['pitcher_hand'] == 'L') & (df['batter_hand'] == 'S'))]),
        ('overall', df)
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = (
            data.groupby(['pitcher_id', 'pitcher_name', 'pitch_team_name', 'pitch_team_id'])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped['split'] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=['pitcher_id', 'pitcher_name', 'pitch_team_name', 'pitch_team_id'],
        columns='split',
        values=['woba', 'ba', 'pa', 'rea', 'obp', 'slg']
    )
    pivot.columns = [f'{stat}_{split}' for stat, split in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={
        'pitcher_id': 'player_id',
        'pitcher_name': 'player_name',
        'pitch_team_name': 'team_name',
        'pitch_team_id': 'team_id'
    })


def analyze_batting_team_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df['batter_name'].isna()].copy()

    splits = [
        ('vs_lhp', df[df['pitcher_hand'] == 'L']),
        ('vs_rhp', df[df['pitcher_hand'] == 'R']),
        ('overall', df)
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = (
            data.groupby(['bat_team_id', 'bat_team_name'])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped['split'] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=['bat_team_id', 'bat_team_name'],
        columns='split',
        values=['woba', 'ba', 'pa', 'rea', 'obp', 'slg']
    )
    pivot.columns = [f'{stat}_{split}' for stat, split in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={
        'bat_team_name': 'team_name',
        'bat_team_id': 'team_id'
    })


def analyze_pitching_team_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df['bat_order'].isna()].copy()

    splits = [
        ('vs_lhh', df[(df['batter_hand'] == 'L') |
                      ((df['pitcher_hand'] == 'R') & (df['batter_hand'] == 'S'))]),
        ('vs_rhh', df[(df['batter_hand'] == 'R') |
                      ((df['pitcher_hand'] == 'L') & (df['batter_hand'] == 'S'))]),
        ('overall', df)
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = (
            data.groupby(['pitch_team_id', 'pitch_team_name'])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped['split'] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=['pitch_team_id', 'pitch_team_name'],
        columns='split',
        values=['woba', 'ba', 'pa', 'rea', 'obp', 'slg']
    )
    pivot.columns = [f'{stat}_{split}' for stat, split in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={
        'pitch_team_name': 'team_name',
        'pitch_team_id': 'team_id'
    })
