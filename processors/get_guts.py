from pathlib import Path

import pandas as pd

from processors.pbp_parser.constants import EventType


def safe_divide(num, denom, fill=0.0):
    """Safe division returning fill value when denominator is 0."""
    return num / denom if denom > 0 else fill


def ip_to_float(ip_str) -> float:
    """Convert baseball IP notation (e.g., '6.2') to decimal innings."""
    try:
        ip = str(ip_str)
        if '.' not in ip:
            return float(ip)
        whole, partial = ip.split('.')
        thirds = {0: 0, 1: 1/3, 2: 2/3}.get(int(partial), 0)
        return int(whole) + thirds
    except (ValueError, TypeError):
        return 0.0


def calculate_woba_constants(lw_df: pd.DataFrame, batting_df: pd.DataFrame) -> dict:
    """Calculate wOBA and linear weight constants."""
    lw = lw_df.set_index('events')['normalized_weight']

    weights = {
        'wbb': lw.get('walk', 0),
        'whbp': lw.get('hit_by_pitch', 0),
        'w1b': lw.get('single', 0),
        'w2b': lw.get('double', 0),
        'w3b': lw.get('triple', 0),
        'whr': lw.get('home_run', 0),
        'woba_scale': lw.get('woba_scale', 1.0),
    }

    # Calculate league wOBA
    singles = batting_df['h'] - batting_df['2b'] - batting_df['3b'] - batting_df['hr']

    woba_num = (
        batting_df['bb'].sum() * weights['wbb'] +
        batting_df['hbp'].sum() * weights['whbp'] +
        singles.sum() * weights['w1b'] +
        batting_df['2b'].sum() * weights['w2b'] +
        batting_df['3b'].sum() * weights['w3b'] +
        batting_df['hr'].sum() * weights['whr']
    )

    woba_denom = (
        batting_df['ab'].sum() +
        batting_df['bb'].sum() +
        batting_df['hbp'].sum() +
        batting_df['sf'].sum()
    )

    weights['woba'] = round(safe_divide(woba_num, woba_denom), 3)

    return weights


def calculate_baserunning_constants(pbp_df: pd.DataFrame) -> dict:
    """Calculate stolen base run values."""
    runs_out = safe_divide(
        pbp_df['runs_on_play'].sum(),
        pbp_df['outs_on_play'].sum()
    )

    run_sb = 0.2
    run_cs = -(2 * runs_out + 0.075)

    sb_events = len(pbp_df[pbp_df['event_type'] == EventType.STOLEN_BASE])
    cs_events = len(pbp_df[pbp_df['event_type'] == EventType.CAUGHT_STEALING])

    cs_rate = safe_divide(cs_events, sb_events + cs_events)

    return {
        'runs_sb': round(run_sb, 3),
        'runs_cs': round(run_cs, 3),
        'cs_rate': round(cs_rate, 3),
    }


def calculate_run_constants(pbp_df: pd.DataFrame) -> dict:
    """Calculate run environment constants."""
    # Plate appearances = rows with a batter
    pa_events = pbp_df[pbp_df['bat_order'].notna()]

    runs_pa = safe_divide(
        pbp_df['runs_on_play'].sum(),
        len(pa_events)
    )

    runs_out = safe_divide(
        pbp_df['runs_on_play'].sum(),
        pbp_df['outs_on_play'].sum()
    )

    runs_win = (pbp_df.groupby('contest_id')['runs_on_play'].sum().mean() / 2) * 1.5 + 3

    return {
        'runs_pa': round(runs_pa, 4),
        'runs_out': round(runs_out, 4),
        'runs_win': round(runs_win, 3),
    }


def calculate_fip_constant(pitching_df: pd.DataFrame) -> float:
    pitching_df = pitching_df.copy()
    pitching_df['ip_float'] = pitching_df['ip'].apply(ip_to_float)

    total_ip = pitching_df['ip_float'].sum()

    lg_era = (pitching_df['er'].sum() * 9) / total_ip

    fip_components = (
        13 * pitching_df['hr_a'].sum() +
        3 * (pitching_df['bb'].sum() + pitching_df['hbp'].sum()) -
        2 * pitching_df['so'].sum()
    ) / total_ip

    return round(lg_era - fip_components, 3)


def calculate_guts_constants(division: int, year: int, data_dir: Path) -> dict | None:
    """Calculate all GUTS constants for a division/year."""
    try:
        pbp_df = pd.read_csv(
            data_dir / f'pbp/d{division}_pbp_with_metrics_{year}.csv',
            low_memory=False
        )
        lw_df = pd.read_csv(
            data_dir / f'miscellaneous/d{division}_linear_weights_{year}.csv'
        )
        pitching_df = pd.read_csv(
            data_dir / f'stats/d{division}_pitching_{year}.csv'
        )
        batting_df = pd.read_csv(
            data_dir / f'stats/d{division}_batting_{year}.csv'
        )

        constants = {
            'year': year,
            'division': division,
            **calculate_woba_constants(lw_df, batting_df),
            **calculate_baserunning_constants(pbp_df),
            **calculate_run_constants(pbp_df),
            'cfip': calculate_fip_constant(pitching_df),
        }

        return constants

    except FileNotFoundError as e:
        print(f"Missing file for D{division} {year}: {e}")
        return None
    except Exception as e:
        print(f"Error calculating constants for D{division} {year}: {e}")
        return None


def main(data_dir: str, year: int, divisions: list[int] = None):
    """Calculate and save GUTS constants."""
    data_dir = Path(data_dir)
    guts_dir = data_dir / 'guts'
    guts_dir.mkdir(exist_ok=True)

    guts_file = guts_dir / 'guts_constants.csv'

    if divisions is None:
        divisions = [1, 2, 3]

    if guts_file.exists():
        existing = pd.read_csv(guts_file)
        existing = existing[existing['year'] != int(year)]
    else:
        existing = pd.DataFrame()

    new_constants = []
    for division in divisions:
        constants = calculate_guts_constants(division, year, data_dir)
        if constants:
            new_constants.append(constants)
            print(f"Calculated D{division} {year} constants")

    if not new_constants:
        print("No constants calculated")
        return

    new_df = pd.DataFrame(new_constants)
    guts = pd.concat([existing, new_df], ignore_index=True)
    guts = guts.sort_values(['division', 'year'], ascending=[True, False])

    guts.to_csv(guts_file, index=False)
    print(f"Saved {len(new_constants)} rows to {guts_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3])
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
