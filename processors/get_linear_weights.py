from pathlib import Path

import numpy as np
import pandas as pd

from processors.pbp_parser.constants import EventType


def calculate_linear_weights(pbp_data: pd.DataFrame, re24_matrix: pd.DataFrame) -> pd.DataFrame:
    event_types = ["walk", "hit_by_pitch", "single", "double", "triple", "home_run", "out", "other"]
    event_counts = dict.fromkeys(event_types, 0)
    event_re24_sums = dict.fromkeys(event_types, 0.0)

    out_events = {
        EventType.GENERIC_OUT, EventType.STRIKEOUT, EventType.CAUGHT_STEALING,
        EventType.PICKOFF, EventType.FIELDERS_CHOICE, EventType.STRIKEOUT_PASSED_BALL,
        EventType.STRIKEOUT_WILD_PITCH
    }

    def map_event(event_val):
        try:
            ev = int(event_val)
        except (ValueError, TypeError):
            return "other"

        if ev in out_events or ev in {e.value for e in out_events}:
            return "out"
        if ev == EventType.WALK or ev == EventType.INTENTIONAL_WALK:
            return "walk"
        if ev == EventType.HIT_BY_PITCH:
            return "hit_by_pitch"
        if ev == EventType.SINGLE:
            return "single"
        if ev == EventType.DOUBLE:
            return "double"
        if ev == EventType.TRIPLE:
            return "triple"
        if ev == EventType.HOME_RUN:
            return "home_run"
        return "other"

    re_bases_to_idx = {b: i for i, b in enumerate(re24_matrix["bases"])}

    def get_re(bases, outs):
        out_cols = ["erv_0", "erv_1", "erv_2"]
        values = []
        for base_state, out_val in zip(bases, outs, strict=True):
            if pd.isna(base_state) or pd.isna(out_val):
                values.append(0.0)
                continue

            base_idx = re_bases_to_idx.get(base_state, -1)
            out_idx = int(out_val) if not pd.isna(out_val) else 0
            out_idx = np.clip(out_idx, 0, 2)

            if base_idx >= 0 and base_idx < len(re24_matrix):
                col = out_cols[out_idx]
                values.append(float(re24_matrix.iloc[base_idx][col]))
            else:
                values.append(0.0)
        return np.array(values, dtype=float)

    re_start = get_re(pbp_data["bases_before"], pbp_data["outs_before"])

    bases_after = pbp_data["bases_after"] if "bases_after" in pbp_data.columns else pbp_data["bases_before"].shift(-1)
    outs_after = pbp_data["outs_after"] if "outs_after" in pbp_data.columns else pbp_data["outs_before"].shift(-1)
    re_end = get_re(bases_after, outs_after)

    inn_end = pbp_data["inn_end_fl"].fillna(0).astype(int).values
    re_end[inn_end == 1] = 0

    runs_on_play = pd.to_numeric(pbp_data["runs_on_play"], errors="coerce").fillna(0).values
    re24 = re_end - re_start + runs_on_play

    events = pbp_data["event_type"].apply(map_event)

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
            print(f"Play by play file not found: {pbp_path}, skipping")
            continue

        pbp_data = pd.read_csv(pbp_path, low_memory=False)
        re24_matrix = pd.read_csv(re_path)
        stats = pd.read_csv(stats_path)

        lw = calculate_linear_weights(pbp_data, re24_matrix)
        lw = calculate_normalized_linear_weights(lw, stats)

        output_path = Path(data_dir) / f"miscellaneous/{div_name}_linear_weights_{year}.csv"
        lw.to_csv(output_path, index=False)
        print(f"Saved linear weights to {output_path}")

    print("Linear weights calculated successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()
    main(args.data_dir, args.year, args.divisions)
