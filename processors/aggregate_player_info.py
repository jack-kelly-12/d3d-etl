from pathlib import Path

import pandas as pd


def is_valid(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, str) and val.strip() in ('', '-', 'nan', 'None'):
        return False
    return True


def first_valid(series: pd.Series):
    for val in series:
        if is_valid(val):
            return val
    return None


def standardize_hand(val) -> str:
    if not is_valid(val):
        return ''
    val = str(val).strip().upper()
    if val in ('R', 'RIGHT'):
        return 'R'
    if val in ('L', 'LEFT'):
        return 'L'
    if val in ('S', 'SWITCH', 'B', 'BOTH'):
        return 'S'
    return val


def load_rosters(data_dir: Path, divisions: list[int], years: list[int]) -> pd.DataFrame:
    dfs = []
    for division in divisions:
        for year in years:
            path = data_dir / f'rosters/d{division}_rosters_{year}.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path, dtype={'player_id': str}, low_memory=False)
            df['source_year'] = year
            df['source'] = 'roster'
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_stats(data_dir: Path, divisions: list[int], years: list[int]) -> pd.DataFrame:
    dfs = []
    for division in divisions:
        for year in years:
            for stat_type in ['batting', 'pitching']:
                path = data_dir / f'stats/d{division}_{stat_type}_{year}.csv'
                if not path.exists():
                    continue
                df = pd.read_csv(path, dtype={'player_id': str}, low_memory=False)
                df['source_year'] = year
                df['source'] = stat_type
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def aggregate_players(data_dir: Path, divisions: list[int], years: list[int]) -> pd.DataFrame:
    print("Loading rosters...")
    rosters = load_rosters(data_dir, divisions, years)

    print("Loading stats...")
    stats = load_stats(data_dir, divisions, years)

    all_data = pd.concat([rosters, stats], ignore_index=True) if not stats.empty else rosters

    if all_data.empty:
        return pd.DataFrame()

    all_data = all_data[all_data['player_id'].apply(is_valid)]
    all_data = all_data.sort_values('source_year', ascending=False)

    print(f"Aggregating {len(all_data)} records for {all_data['player_id'].nunique()} players...")

    fields = ['player_name', 'bats', 'throws', 'height', 'weight']
    fields = [f for f in fields if f in all_data.columns]

    result = all_data.groupby('player_id').agg(dict.fromkeys(fields, first_valid)).reset_index()

    result['bats'] = result['bats'].apply(standardize_hand)
    result['throws'] = result['throws'].apply(standardize_hand)

    for col in ['height', 'weight']:
        if col in result.columns:
            result[col] = result[col].fillna('')

    return result


def merge_to_rosters(data_dir: Path, player_info: pd.DataFrame,
                     divisions: list[int], years: list[int]):
    """Merge aggregated player info back to all roster files."""
    if player_info.empty:
        return

    fill_cols = ['bats', 'throws', 'height', 'weight']
    fill_cols = [c for c in fill_cols if c in player_info.columns]

    for division in divisions:
        for year in years:
            path = data_dir / f'rosters/d{division}_rosters_{year}.csv'
            if not path.exists():
                continue

            roster = pd.read_csv(path, dtype={'player_id': str}, low_memory=False)

            merged = roster.merge(
                player_info[['player_id'] + fill_cols],
                on='player_id',
                how='left',
                suffixes=('', '_agg')
            )

            for col in fill_cols:
                agg_col = f'{col}_agg'
                if agg_col in merged.columns:
                    merged[col] = merged[col].where(
                        merged[col].apply(is_valid),
                        merged[agg_col]
                    )
                    merged.drop(columns=[agg_col], inplace=True)

            merged.to_csv(path, index=False)
            print(f"Updated {path}")


def main(data_dir: str, divisions: list[int] = None, years: list[int] = None):
    data_dir = Path(data_dir)

    if divisions is None:
        divisions = [1, 2, 3]

    if years is None:
        roster_dir = data_dir / 'rosters'
        years = sorted({
            int(f.stem.split('_')[-1])
            for f in roster_dir.glob('d*_rosters_*.csv')
        }) if roster_dir.exists() else list(range(2021, 2026))

    print(f"Processing D{divisions} for {years}")

    result = aggregate_players(data_dir, divisions, years)

    if result.empty:
        print("No data found!")
        return

    output_path = data_dir / 'rosters' / 'player_information.csv'
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} players to {output_path}")

    print("Merging player info back to roster files...")
    merge_to_rosters(data_dir, result, divisions, years)

    print("Done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--years', nargs='+', type=int, default=None)
    args = parser.parse_args()

    main(args.data_dir, args.divisions, args.years)
