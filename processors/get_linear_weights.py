from pathlib import Path

import numpy as np
import pandas as pd


def calculate_college_linear_weights(pbp_data: pd.DataFrame, re24_matrix: pd.DataFrame) -> pd.DataFrame:
    event_types = ["walk", "hit_by_pitch", "single", "double", "triple", "home_run", "out", "other"]
    event_counts = dict.fromkeys(event_types, 0)
    event_re24_sums = dict.fromkeys(event_types, 0.0)

    event_lookup = {
        "2": "out", "3": "out", "6": "out",
        "14": "walk", "16": "hit_by_pitch",
        "20": "single", "21": "double",
        "22": "triple", "23": "home_run"
    }

    base_state_map = {
        "0": "_ _ _",  # No runners
        "1": "1B _ _",  # Runner on first
        "2": "_ 2B _",  # Runner on second
        "3": "_ _ 3B",  # Runner on third
        "4": "1B 2B _",  # Runners on first and second
        "5": "1B _ 3B",  # Runners on first and third
        "6": "_ 2B 3B",  # Runners on second and third
        "7": "1B 2B 3B"  # Bases loaded
    }

    def get_re(base_cd, outs):
        base_states = base_cd.astype(str).map(base_state_map).fillna("___")
        base_idx = base_states.map(
            {v: i for i, v in enumerate(re24_matrix["bases"], start=0)}
        )

        out_cols = ["erv_0", "erv_1", "erv_2"]
        outs_idx = np.clip(outs.astype(int).values, 0, 2)

        values = []
        for i, row in enumerate(base_idx):
            if pd.isna(row) or pd.isna(outs_idx[i]):
                values.append(0)
            else:
                col = out_cols[outs_idx[i]]
                values.append(re24_matrix.loc[row, col])
        return np.array(values, dtype=float)

    re_start = get_re(pbp_data["base_cd_before"], pbp_data["outs_before"])
    re_end = np.append(get_re(pbp_data["base_cd_before"].iloc[1:], pbp_data["outs_before"].iloc[1:]), 0)

    re_end[pbp_data["inn_end_flag"] == 1] = 0

    runs_on_play = pd.to_numeric(pbp_data["runs_on_play"], errors="coerce").fillna(0).values
    re24 = re_end - re_start + runs_on_play

    events = pbp_data["event_cd"].astype(str).map(event_lookup).fillna("other")

    event_table = events.value_counts().reindex(event_types, fill_value=0)
    for e in event_types:
        event_counts[e] = event_table.get(e, 0)

    event_sums = {e: re24[events == e].sum() for e in event_types}
    for e in event_types:
        event_re24_sums[e] = event_sums.get(e, 0.0)

    df = pd.DataFrame({
        "events": event_types,
        "count": [event_counts[e] for e in event_types],
        "linear_weights_above_average": [
            event_re24_sums[e] / event_counts[e] if event_counts[e] > 0 else np.nan
            for e in event_types
        ]
    })

    df = df[df["events"] != "other"].copy()
    df["linear_weights_above_average"] = df["linear_weights_above_average"].round(3)
    out_val = df.loc[df["events"] == "out", "linear_weights_above_average"].values[0]
    df["linear_weights_above_outs"] = df["linear_weights_above_average"] - out_val
    df = df.sort_values("linear_weights_above_average", ascending=False)

    return df


def calculate_normalized_linear_weights(linear_weights: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    required = {"events", "linear_weights_above_outs", "count"}
    if not required.issubset(linear_weights.columns):
        raise ValueError(f"linear_weights must contain columns: {required}")

    lw = linear_weights.copy()
    if "woba_scale" in lw["events"].values:
        lw = lw[lw["events"] != "woba_scale"]

    total_value = (lw["linear_weights_above_outs"] * lw["count"]).sum()
    total_pa = lw["count"].sum()
    denominator = total_value / total_pa if total_pa > 0 else np.nan

    league_obp = (stats["h"].sum() + stats["bb"].sum() + stats["hbp"].sum()) / (
        stats["ab"].sum() + stats["bb"].sum() + stats["hbp"].sum() + stats["sf"].sum() + stats["sh"].sum()
    )

    woba_scale = league_obp / denominator if denominator != 0 else np.nan

    lw["normalized_weight"] = (lw["linear_weights_above_outs"] * woba_scale).round(3)
    woba_scale_row = pd.DataFrame([{
        "events": "woba_scale",
        "linear_weights_above_outs": np.nan,
        "count": np.nan,
        "normalized_weight": round(woba_scale, 3) if not pd.isna(woba_scale) else np.nan
    }])

    return pd.concat([lw, woba_scale_row], ignore_index=True)


def main(data_dir: str, year: int, divisions: list = None):
    if divisions is None:
        divisions = [1, 2, 3]
    for division in divisions:
        div_name = f"d{division}"
        print(f"Processing division {division}...")

        pbp_path = Path(data_dir) / f"pbp/{div_name}_parsed_pbp_{year}.csv"
        stats_path = Path(data_dir) / f"stats/{div_name}_batting_{year}.csv"
        re_path = Path(data_dir) / f"miscellaneous/{div_name}_expected_runs_{year}.csv"

        if not pbp_path.exists():
            print(f"⚠️ PBP file not found: {pbp_path}, skipping")
            continue

        pbp_data = pd.read_csv(pbp_path)
        re24_matrix = pd.read_csv(re_path)
        stats = pd.read_csv(stats_path)

        lw = calculate_college_linear_weights(pbp_data, re24_matrix)
        lw = calculate_normalized_linear_weights(lw, stats)

        output_path = Path(data_dir) / f"miscellaneous/{div_name}_linear_weights_{year}.csv"
        lw.to_csv(output_path, index=False)
        print(f"✅ Saved linear weights to {output_path}")

    print("🎉 Linear weights calculated successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()
    main(args.data_dir, args.year, args.divisions)
