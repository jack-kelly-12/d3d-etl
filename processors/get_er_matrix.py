from pathlib import Path

import numpy as np
import pandas as pd


def get_expected_runs_matrix_2(bases_before, outs, runs_rest_of_inn):
    er = pd.DataFrame({
        'bases_before': bases_before,
        'outs': outs,
        'runs_rest_of_inn': runs_rest_of_inn
    }).dropna()

    er = (er.groupby(['bases_before', 'outs'])
          .agg(
              erv=('runs_rest_of_inn', 'mean'),
              prob_score=('runs_rest_of_inn', lambda x: (x > 0).mean()),
              count=('runs_rest_of_inn', 'size')
    )
        .reset_index())

    er['erv'] = er['erv'].round(3)
    er['prob_score'] = er['prob_score'].round(3)

    er_matrix = np.zeros((8, 3))
    prob_matrix = np.zeros((8, 3))

    base_state_map = {'NNN': 0, 'YNN': 1, 'NYN': 2, 'YYN': 3,
                      'NNY': 4, 'YNY': 5, 'NYY': 6, 'YYY': 7}

    for _, row in er.iterrows():
        base_state = row['bases_before']
        base_idx = base_state_map.get(base_state, -1)
        out_idx = int(row['outs'])
        if 0 <= base_idx < 8 and 0 <= out_idx < 3:
            er_matrix[base_idx, out_idx] = row['erv']
            prob_matrix[base_idx, out_idx] = row['prob_score']

    er_matrix = pd.DataFrame(
        er_matrix,
        index=['NNN', 'YNN', 'NYN', 'YYN',
               'NNY', 'YNY', 'NYY', 'YYY'],
        columns=['0', '1', '2']
    )

    prob_matrix = pd.DataFrame(
        prob_matrix,
        index=['NNN', 'YNN', 'NYN', 'YYN',
               'NNY', 'YNY', 'NYY', 'YYY'],
        columns=['0', '1', '2']
    )

    return er_matrix, prob_matrix


def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    if divisions is None:
        divisions = [1, 2, 3]
    all_matrices = {}
    all_prob_matrices = {}

    misc_dir = data_dir / 'miscellaneous'
    misc_dir.mkdir(exist_ok=True)

    for division in divisions:
        try:
            pbp_file = data_dir / 'pbp' / \
                f'd{division}_parsed_pbp_{year}.csv'

            if not pbp_file.exists():
                print(f"Play by play file not found: {pbp_file}")
                continue

            pbp_df = pd.read_csv(pbp_file)
            print(f"Loaded {len(pbp_df)} rows for D{division} {year}")

            bases_before = pbp_df['bases_before']
            outs = pbp_df['outs_before']
            runs_rest_of_inn = pbp_df['runs_roi']

            matrix, prob_matrix = get_expected_runs_matrix_2(
                bases_before, outs, runs_rest_of_inn)
            all_matrices[f"D{division}_{year}"] = matrix
            all_prob_matrices[f"D{division}_{year}"] = prob_matrix
            print(f"Processed D{division} {year}")

        except Exception as e:
            print(f"Error processing D{division} {year}: {str(e)}")
            continue

    final_dfs = []
    for name, matrix in all_matrices.items():
        division = int(name[1])
        year_val = int(name.split('_')[1])
        prob_matrix = all_prob_matrices[name]

        df = pd.DataFrame({
            'division': division,
            'year': year_val,
            'bases': matrix.index,
            'erv_0': matrix['0'],
            'erv_1': matrix['1'],
            'erv_2': matrix['2'],
            'prob_0': prob_matrix['0'],
            'prob_1': prob_matrix['1'],
            'prob_2': prob_matrix['2']
        })
        final_dfs.append(df)

    if not final_dfs:
        print("No matrices were generated!")
        return None

    final_df = pd.concat(final_dfs, ignore_index=True)
    final_df = final_df.sort_values(['division', 'year', 'bases'])

    for division in divisions:
        output_file = misc_dir / f'd{division}_expected_runs_{year}.csv'
        division_df = final_df[final_df['division'] == division]
        if not division_df.empty:
            division_df.to_csv(
                output_file, index=False)
            print(f"Saved expected runs matrix for D{division} to {output_file}")


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
