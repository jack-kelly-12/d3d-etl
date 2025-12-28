from pathlib import Path

import pandas as pd


def calculate_woba_constants(lw_df, batting_df):
    woba_scale = lw_df[lw_df['events'] ==
                       'woba_scale']['normalized_weight'].iloc[0]

    weights = {
        'wbb': lw_df[lw_df['events'] == 'walk']['normalized_weight'].iloc[0],
        'whbp': lw_df[lw_df['events'] == 'hit_by_pitch']['normalized_weight'].iloc[0],
        'w1b': lw_df[lw_df['events'] == 'single']['normalized_weight'].iloc[0],
        'w2b': lw_df[lw_df['events'] == 'double']['normalized_weight'].iloc[0],
        'w3b': lw_df[lw_df['events'] == 'triple']['normalized_weight'].iloc[0],
        'whr': lw_df[lw_df['events'] == 'home_run']['normalized_weight'].iloc[0]
    }

    woba_numerator = (
        batting_df['bb'].sum() * weights['wbb'] +
        batting_df['hbp'].sum() * weights['whbp'] +
        batting_df['1b'].sum() * weights['w1b'] +
        batting_df['2b'].sum() * weights['w2b'] +
        batting_df['3b'].sum() * weights['w3b'] +
        batting_df['hr'].sum() * weights['whr']
    )

    woba_denominator = batting_df['ab'].sum(
    ) + batting_df['bb'].sum() + batting_df['hbp'].sum() + batting_df['sf'].sum()
    woba = woba_numerator / woba_denominator if woba_denominator > 0 else 0

    return {'woba': round(woba, 3), 'woba_scale': woba_scale, **weights}


def calculate_baserunning_constants(pbp_df):
    runs_out = pbp_df['runs_on_play'].sum() / pbp_df['outs_on_play'].sum()
    run_sb = 0.2
    run_cs = -(2 * runs_out + 0.075)

    cs_attempts = len(pbp_df[pbp_df['event_cd'] == 6])
    sb_attempts = len(pbp_df[pbp_df['event_cd'] == 4])
    cs_rate = cs_attempts / \
        (cs_attempts + sb_attempts) if (cs_attempts + sb_attempts) > 0 else 0

    return {
        'runs_sb': run_sb,
        'runs_cs': run_cs,
        'cs_rate': round(cs_rate, 3)
    }


def calculate_run_constants(pbp_df):
    runs_pa = pbp_df['runs_on_play'].sum(
    ) / len(pbp_df[~pbp_df['bat_order'].isna()])
    runs_out = pbp_df['runs_on_play'].sum() / pbp_df['outs_on_play'].sum()
    runs_win = pbp_df.groupby('contest_id')['runs_on_play'].sum().mean()

    return {
        'runs_pa': runs_pa,
        'runs_out': runs_out,
        'runs_win': runs_win
    }

def ip_to_real(innings):
        ip_str = str(innings)
        if '.' in ip_str:
            whole_innings, partial = ip_str.split('.')
            whole_innings = int(whole_innings)
            partial = int(partial)
        else:
            return float(ip_str)

        if partial == 0:
            decimal_part = 0
        elif partial == 1:
            decimal_part = 1/3
        elif partial == 2:
            decimal_part = 2/3
        else:
            raise ValueError(
                "Invalid partial inning value. Should be 0, 1, or 2.")

        return whole_innings + decimal_part


def calculate_fip_constant(pitching_df):
    pitching_df['ip_float'] = pitching_df.ip.apply(ip_to_real)
    lgERA = (pitching_df['er'].sum() * 9) / pitching_df['ip_float'].sum()
    fip_components = (13 * pitching_df['hr_a'].sum() +
                      3 * (pitching_df['bb'].sum() + pitching_df['hbp'].sum()) -
                      2 * pitching_df['so'].sum()) / pitching_df['ip_float'].sum()
    return lgERA - fip_components


def calculate_guts_constants(division, year, output_path):
    try:
        pbp_df = pd.read_csv(
            output_path / f'pbp/d{division}_parsed_pbp_new_{year}.csv')
        lw_df = pd.read_csv(
            output_path / f'miscellaneous/d{division}_linear_weights_{year}.csv')
        pitching_df = pd.read_csv(
            output_path / f'stats/d{division}_pitching_{year}.csv')
        batting_df = pd.read_csv(
            output_path / f'stats/d{division}_batting_{year}.csv')

        batting_df['1b'] = batting_df['h'] - batting_df['2b'] - \
            batting_df['3b'] - batting_df['hr']

        constants = {
            'year': year,
            'division': division,
            **calculate_woba_constants(lw_df, batting_df),
            **calculate_baserunning_constants(pbp_df),
            **calculate_run_constants(pbp_df),
            'cfip': calculate_fip_constant(pitching_df)
        }

        return constants

    except Exception as e:
        print(f"Error calculating constants for D{division} {year}: {e}")
        return None


def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    guts_dir = data_dir / 'guts'
    guts_dir.mkdir(exist_ok=True)

    guts_file = guts_dir / 'guts_constants.csv'

    if divisions is None:
        divisions = [1, 2, 3]

    all_constants = []
    existing_guts = pd.read_csv(guts_file)
    existing_guts = existing_guts[(
        existing_guts['year'] != int(year))]

    for division in divisions:
        constants = calculate_guts_constants(division, year, data_dir)
        if constants:
            all_constants.append(constants)

    new_guts = pd.DataFrame(all_constants)

    guts = pd.concat([existing_guts, new_guts])

    guts = guts.sort_values(
        ['division', 'year'], ascending=[True, False])

    guts.to_csv(guts_file, index=False)
    print(f"Saved {len(new_guts)} rows of Guts constants")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
