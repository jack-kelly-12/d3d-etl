import pandas as pd

from .common import apply_batting_metrics

_PIVOT_VALUES = ["woba", "ba", "pa", "rea", "obp", "slg"]


def _pivot_and_rename(
    combined: pd.DataFrame,
    index_cols: list,
    col_key: str,
    rename_map: dict,
) -> pd.DataFrame:
    if combined.empty or not all(v in combined.columns for v in _PIVOT_VALUES):
        return pd.DataFrame()

    pivot = combined.pivot(
        index=index_cols,
        columns=col_key,
        values=_PIVOT_VALUES,
    )
    pivot.columns = [f"{stat}_{split}" for stat, split in pivot.columns]
    return pivot.reset_index().rename(columns=rename_map)


def analyze_batting_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df["batter_name"].isna()].copy()
    index_cols = ["batter_id", "batter_name", "bat_team_name", "bat_team_id"]

    splits = [
        ("vs_lhp", df[df["pitcher_hand"] == "L"]),
        ("vs_rhp", df[df["pitcher_hand"] == "R"]),
        ("overall", df),
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["split"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "split",
        {
            "batter_id": "player_id",
            "batter_name": "player_name",
            "bat_team_name": "team_name",
            "bat_team_id": "team_id",
        },
    )


def analyze_pitching_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df["bat_order"].isna()].copy()
    index_cols = ["pitcher_id", "pitcher_name", "pitch_team_name", "pitch_team_id"]

    splits = [
        (
            "vs_lhh",
            df[
                (df["batter_hand"] == "L")
                | ((df["pitcher_hand"] == "R") & (df["batter_hand"] == "B"))
            ],
        ),
        (
            "vs_rhh",
            df[
                (df["batter_hand"] == "R")
                | ((df["pitcher_hand"] == "L") & (df["batter_hand"] == "B"))
            ],
        ),
        ("overall", df),
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["split"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "split",
        {
            "pitcher_id": "player_id",
            "pitcher_name": "player_name",
            "pitch_team_name": "team_name",
            "pitch_team_id": "team_id",
        },
    )


def analyze_batting_team_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df["batter_name"].isna()].copy()
    index_cols = ["bat_team_id", "bat_team_name"]

    splits = [
        ("vs_lhp", df[df["pitcher_hand"] == "L"]),
        ("vs_rhp", df[df["pitcher_hand"] == "R"]),
        ("overall", df),
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["split"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "split",
        {"bat_team_name": "team_name", "bat_team_id": "team_id"},
    )


def analyze_pitching_team_splits(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df[~df["bat_order"].isna()].copy()
    index_cols = ["pitch_team_id", "pitch_team_name"]

    splits = [
        (
            "vs_lhh",
            df[
                (df["batter_hand"] == "L")
                | ((df["pitcher_hand"] == "R") & (df["batter_hand"] == "B"))
            ],
        ),
        (
            "vs_rhh",
            df[
                (df["batter_hand"] == "R")
                | ((df["pitcher_hand"] == "L") & (df["batter_hand"] == "B"))
            ],
        ),
        ("overall", df),
    ]

    results = []
    for name, data in splits:
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["split"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "split",
        {"pitch_team_name": "team_name", "pitch_team_id": "team_id"},
    )
