import numpy as np
import pandas as pd

from processors.logging_utils import div_file_prefix
from processors.pbp_parser.constants import EventType


def load_linear_weights(data_path, division: str, year: int) -> dict:
    prefix = div_file_prefix(division)
    path = data_path / f"miscellaneous/{prefix}_linear_weights_{year}.csv"
    return pd.read_csv(path).set_index("events")["normalized_weight"].to_dict()


def load_guts(data_path, division: str, year: int) -> dict:
    path = data_path / "guts/guts_constants.csv"
    guts = pd.read_csv(path)
    row = guts[(guts["division"] == division) & (guts["year"] == year)]
    if row.empty:
        raise FileNotFoundError(f"No GUTS constants for {division} {year}")
    return row.iloc[0].to_dict()


def calculate_batting_metrics(group: pd.DataFrame, weights: dict) -> pd.Series:
    walks = (group["event_type"].eq(EventType.WALK.value)).sum()
    hbp = (group["event_type"].eq(EventType.HIT_BY_PITCH.value)).sum()
    singles = (group["event_type"].eq(EventType.SINGLE.value)).sum()
    doubles = (group["event_type"].eq(EventType.DOUBLE.value)).sum()
    triples = (group["event_type"].eq(EventType.TRIPLE.value)).sum()
    home_runs = (group["event_type"].eq(EventType.HOME_RUN.value)).sum()
    outs = (group["event_type"].eq(EventType.GENERIC_OUT.value)).sum() + (
        group["event_type"].eq(EventType.STRIKEOUT.value)
    ).sum()
    errors = (group["event_type"].eq(EventType.ERROR.value)).sum()

    sf = (group.get("sf_fl", pd.Series(0)) == 1).sum() if "sf_fl" in group.columns else 0
    hits = singles + doubles + triples + home_runs
    ab = hits + outs + errors
    pa = ab + walks + sf + hbp

    if pa == 0:
        return pd.Series(
            {"woba": np.nan, "ba": np.nan, "pa": 0, "rea": 0, "slg": np.nan, "obp": np.nan}
        )

    ba = hits / ab if ab > 0 else np.nan
    rea = group["rea"].sum() if "rea" in group.columns else 0

    woba_num = (
        weights.get("walk", 0) * walks
        + weights.get("hit_by_pitch", 0) * hbp
        + weights.get("single", 0) * singles
        + weights.get("double", 0) * doubles
        + weights.get("triple", 0) * triples
        + weights.get("home_run", 0) * home_runs
    )

    woba_denom = ab + walks + sf + hbp
    woba = woba_num / woba_denom if woba_denom > 0 else np.nan

    slg = (singles + 2 * doubles + 3 * triples + 4 * home_runs) / ab if ab > 0 else np.nan
    obp = (hits + walks + hbp) / (ab + walks + sf + hbp) if (ab + walks + sf + hbp) > 0 else np.nan

    return pd.Series({"woba": woba, "ba": ba, "pa": pa, "rea": rea, "slg": slg, "obp": obp})


_HAND_NORMALIZE = {
    "RIGHT": "R",
    "LEFT": "L",
    "SWITCH": "S",
    "BOTH": "B",
    "R": "R",
    "L": "L",
    "S": "S",
    "B": "B",
}


def add_handedness(
    pbp: pd.DataFrame,
    info: pd.DataFrame,
    pitcher_id_col: str = "pitcher_id",
    batter_id_col: str = "batter_id",
    info_id_col: str = "player_id",
    info_throws_col: str = "throws",
    info_bats_col: str = "bats",
    out_pitcher_hand: str = "pitcher_hand",
    out_batter_hand: str = "batter_hand",
) -> pd.DataFrame:
    throws_map = (
        info[[info_id_col, info_throws_col]]
        .dropna(subset=[info_id_col])
        .drop_duplicates(subset=[info_id_col])
        .set_index(info_id_col)[info_throws_col]
        .astype("string")
        .str.upper()
        .str.strip()
        .map(_HAND_NORMALIZE)
    )

    bats_map = (
        info[[info_id_col, info_bats_col]]
        .dropna(subset=[info_id_col])
        .drop_duplicates(subset=[info_id_col])
        .set_index(info_id_col)[info_bats_col]
        .astype("string")
        .str.upper()
        .str.strip()
        .map(_HAND_NORMALIZE)
    )

    pbp[out_pitcher_hand] = pbp[pitcher_id_col].map(throws_map)
    pbp[out_batter_hand] = pbp[batter_id_col].map(bats_map)

    return pbp


_TEAM_NAME_FALLBACKS = [
    ("bat_team_name", "bat_team_id"),
    ("pitch_team_name", "pitch_team_id"),
    ("away_team_name", "away_team_id"),
    ("home_team_name", "home_team_id"),
]


def load_pbp_with_hands(data_dir, year: int, division: str) -> pd.DataFrame:
    prefix = div_file_prefix(division)
    pbp_df = pd.read_csv(
        data_dir / f"pbp/{prefix}_pbp_with_metrics_{year}.csv",
        dtype={"player_id": str, "batter_id": str, "pitcher_id": str},
        low_memory=False,
    )

    # Old pbp files may have NaN team name columns (dtype float64) — fill from the ID column
    for name_col, id_col in _TEAM_NAME_FALLBACKS:
        if name_col in pbp_df.columns and id_col in pbp_df.columns:
            pbp_df[name_col] = pbp_df[name_col].astype(object)
            pbp_df[id_col] = pbp_df[id_col].astype(str)
            mask = pbp_df[name_col].isna() & pbp_df[id_col].notna()
            if mask.any():
                pbp_df.loc[mask, name_col] = pbp_df.loc[mask, id_col]

    player_info_path = data_dir / "cube_stats" / "cube_player_info.csv"
    if player_info_path.exists():
        player_info = pd.read_csv(
            player_info_path, dtype={"player_id": str}, usecols=["player_id", "bats", "throws"]
        )
        pbp_df = add_handedness(pbp_df, player_info)
    else:
        pbp_df["pitcher_hand"] = None
        pbp_df["batter_hand"] = None

    return pbp_df


_METRIC_COLS = ["woba", "ba", "pa", "rea", "slg", "obp"]


def apply_batting_metrics(data: pd.DataFrame, group_cols: list, weights: dict) -> pd.DataFrame:
    """Groupby aggregation using calculate_batting_metrics, robust across pandas versions.

    Returns a DataFrame with group_cols + metric cols always present, even when empty,
    so downstream pivot calls never fail on missing columns.
    """
    empty = pd.DataFrame(columns=group_cols + _METRIC_COLS)

    missing_cols = [c for c in group_cols if c not in data.columns]
    if missing_cols or data.empty:
        return empty

    rows = []
    for keys, group in data.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = calculate_batting_metrics(group, weights)
        row = dict(zip(group_cols, keys))
        row.update(metrics.to_dict())
        rows.append(row)

    if not rows:
        return empty
    return pd.DataFrame(rows)


def filter_by_team_history(df: pd.DataFrame, team_history: pd.DataFrame) -> pd.DataFrame:
    if df.empty or team_history.empty:
        return df

    if "team_id" not in df.columns or "division" not in df.columns:
        return df

    valid_teams = team_history[["team_id", "division"]].drop_duplicates().copy()
    valid_teams["team_id"] = valid_teams["team_id"].astype(str)
    valid_teams["division"] = valid_teams["division"].astype(str)

    df = df.copy()
    df["team_id"] = df["team_id"].astype(str)
    df["division"] = df["division"].astype(str)

    overlap = set(df["team_id"]) & set(valid_teams["team_id"])
    if not overlap:
        return df

    return df.merge(valid_teams, on=["team_id", "division"], how="inner")
