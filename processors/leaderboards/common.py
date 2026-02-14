import numpy as np
import pandas as pd

from processors.pbp_parser.constants import EventType


def load_linear_weights(data_path, division: int, year: int) -> dict:
    path = data_path / f"miscellaneous/d{division}_linear_weights_{year}.csv"
    return pd.read_csv(path).set_index("events")["normalized_weight"].to_dict()


def load_guts(data_path, division: int, year: int) -> dict:
    path = data_path / "guts/guts_constants.csv"
    return pd.read_csv(path).query(f"division == {division} and year == {year}").iloc[0].to_dict()


def calculate_batting_metrics(group: pd.DataFrame, weights: dict) -> pd.Series:
    walks = (group["event_type"] == EventType.WALK).sum()
    hbp = (group["event_type"] == EventType.HIT_BY_PITCH).sum()
    singles = (group["event_type"] == EventType.SINGLE).sum()
    doubles = (group["event_type"] == EventType.DOUBLE).sum()
    triples = (group["event_type"] == EventType.TRIPLE).sum()
    home_runs = (group["event_type"] == EventType.HOME_RUN).sum()
    outs = (group["event_type"] == EventType.GENERIC_OUT).sum() + (
        group["event_type"] == EventType.STRIKEOUT
    ).sum()
    errors = (group["event_type"] == EventType.ERROR).sum()

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
    )

    bats_map = (
        info[[info_id_col, info_bats_col]]
        .dropna(subset=[info_id_col])
        .drop_duplicates(subset=[info_id_col])
        .set_index(info_id_col)[info_bats_col]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    pbp[out_pitcher_hand] = pbp[pitcher_id_col].map(throws_map).fillna("unknown").astype(str)
    pbp[out_batter_hand] = pbp[batter_id_col].map(bats_map).fillna("unknown").astype(str)

    pbp[out_pitcher_hand] = pbp[out_pitcher_hand].where(pbp[out_pitcher_hand].isin(["R", "L"]))
    pbp[out_batter_hand] = pbp[out_batter_hand].where(
        pbp[out_batter_hand].isin(["R", "L", "S", "B"])
    )

    return pbp


def load_pbp_with_hands(data_dir, year: int, division: int) -> pd.DataFrame:
    pbp_df = pd.read_csv(
        data_dir / f"pbp/d{division}_pbp_with_metrics_{year}.csv",
        dtype={"player_id": str, "batter_id": str, "pitcher_id": str},
    )

    player_info_path = data_dir / "rosters/player_information.csv"
    if player_info_path.exists():
        player_info = pd.read_csv(player_info_path, dtype={"player_id": str})
        pbp_df = add_handedness(pbp_df, player_info)
    else:
        pbp_df["pitcher_hand"] = None
        pbp_df["batter_hand"] = None

    return pbp_df


def filter_by_team_history(df: pd.DataFrame, team_history: pd.DataFrame) -> pd.DataFrame:
    if df.empty or team_history.empty:
        return df

    if "team_id" not in df.columns or "division" not in df.columns:
        return df

    valid_teams = team_history[["team_id", "division"]].drop_duplicates()
    valid_teams["team_id"] = pd.to_numeric(valid_teams["team_id"], errors="coerce").astype("Int64")
    valid_teams["division"] = pd.to_numeric(valid_teams["division"], errors="coerce").astype(
        "Int64"
    )

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["division"] = pd.to_numeric(df["division"], errors="coerce").astype("Int64")

    df = df.merge(valid_teams, on=["team_id", "division"], how="inner")

    return df
