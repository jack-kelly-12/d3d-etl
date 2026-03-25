import pandas as pd

from .common import apply_batting_metrics

_PIVOT_VALUES = ["woba", "ba", "pa", "rea", "obp", "slg"]


def add_situation_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["risp_fl"] = (~df["r2_name"].isna() | ~df["r3_name"].isna()).astype(int)
    df["runners_on_fl"] = (
        ~df["r1_name"].isna() | ~df["r2_name"].isna() | ~df["r3_name"].isna()
    ).astype(int)

    return df


def _build_situations(df: pd.DataFrame) -> list:
    return [
        ("risp", df[df["risp_fl"] == 1]),
        ("runners_on", df[df["runners_on_fl"] == 1]),
        ("high_leverage", df[df["high_leverage_fl"] == 1]),
        ("low_leverage", df[df["low_leverage_fl"] == 1]),
        ("overall", df),
    ]


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
    pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]
    return pivot.reset_index().rename(columns=rename_map)


def analyze_batting_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)
    index_cols = ["batter_id", "batter_name", "bat_team_name", "bat_team_id"]

    results = []
    for name, data in _build_situations(df):
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "situation",
        {
            "batter_id": "player_id",
            "batter_name": "player_name",
            "bat_team_name": "team_name",
            "bat_team_id": "team_id",
        },
    )


def analyze_pitching_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)
    index_cols = ["pitcher_id", "pitcher_name", "pitch_team_name", "pitch_team_id"]

    results = []
    for name, data in _build_situations(df):
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "situation",
        {
            "pitcher_id": "player_id",
            "pitcher_name": "player_name",
            "pitch_team_name": "team_name",
            "pitch_team_id": "team_id",
        },
    )


def analyze_batting_team_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)
    index_cols = ["bat_team_id", "bat_team_name"]

    results = []
    for name, data in _build_situations(df):
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "situation",
        {"bat_team_name": "team_name", "bat_team_id": "team_id"},
    )


def analyze_pitching_team_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)
    index_cols = ["pitch_team_id", "pitch_team_name"]

    results = []
    for name, data in _build_situations(df):
        if data.empty:
            continue
        grouped = apply_batting_metrics(data, index_cols, weights)
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return _pivot_and_rename(
        combined,
        index_cols,
        "situation",
        {"pitch_team_name": "team_name", "pitch_team_id": "team_id"},
    )
