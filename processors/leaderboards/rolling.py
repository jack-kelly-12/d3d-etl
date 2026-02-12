import pandas as pd


def calculate_rolling_woba(df: pd.DataFrame, is_pitcher: bool = False, windows: list[int] = None) -> pd.DataFrame:
    if windows is None:
        windows = [25, 50, 100]

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    id_col = 'pitcher_id' if is_pitcher else 'batter_id'
    df = df.dropna(subset=[id_col, 'woba'])
    df = df.sort_values([id_col, 'date'])

    all_results = None

    for window in windows:
        results = []

        for player_id, group in df.groupby(id_col):
            if len(group) < 2 * window:
                continue

            group = group.copy()
            group['rolling_woba_now'] = group['woba'].rolling(window).mean()
            group['rolling_woba_then'] = group['rolling_woba_now'].shift(window)

            valid = group.dropna(subset=['rolling_woba_now', 'rolling_woba_then'])
            if valid.empty:
                continue

            latest = valid.iloc[-1]
            results.append({
                id_col: player_id,
                f'{window}_then': latest['rolling_woba_then'],
                f'{window}_now': latest['rolling_woba_now'],
                f'{window}_delta': latest['rolling_woba_now'] - latest['rolling_woba_then']
            })

        if not results:
            continue

        window_df = pd.DataFrame(results)

        if all_results is None:
            all_results = window_df
        else:
            merge_cols = [col for col in window_df.columns if str(window) in col]
            merge_cols.append(id_col)
            all_results = pd.merge(all_results, window_df[merge_cols], on=id_col, how='outer')

    if all_results is None:
        return pd.DataFrame()

    return all_results.rename(columns={
        'batter_id': 'player_id',
        'pitcher_id': 'player_id'
    })
