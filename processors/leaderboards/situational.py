import pandas as pd

from .common import calculate_batting_metrics


def add_situation_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["risp_fl"] = (~df["r2_name"].isna() | ~df["r3_name"].isna()).astype(int)
    df["runners_on_fl"] = (
        ~df["r1_name"].isna() | ~df["r2_name"].isna() | ~df["r3_name"].isna()
    ).astype(int)

    return df


def analyze_batting_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)

    situations = [
        ("risp", df[df["risp_fl"] == 1]),
        ("runners_on", df[df["runners_on_fl"] == 1]),
        ("high_leverage", df[df["high_leverage_fl"] == 1]),
        ("low_leverage", df[df["low_leverage_fl"] == 1]),
        ("overall", df),
    ]

    results = []
    for name, data in situations:
        if data.empty:
            continue
        grouped = (
            data.groupby(["batter_id", "batter_name", "bat_team_name", "bat_team_id"])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=["batter_id", "batter_name", "bat_team_name", "bat_team_id"],
        columns="situation",
        values=["woba", "ba", "pa", "rea", "obp", "slg"],
    )
    pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]

    result = pivot.reset_index()
    return result.rename(
        columns={
            "batter_id": "player_id",
            "batter_name": "player_name",
            "bat_team_name": "team_name",
            "bat_team_id": "team_id",
        }
    )


def analyze_pitching_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)

    situations = [
        ("risp", df[df["risp_fl"] == 1]),
        ("runners_on", df[df["runners_on_fl"] == 1]),
        ("high_leverage", df[df["high_leverage_fl"] == 1]),
        ("low_leverage", df[df["low_leverage_fl"] == 1]),
        ("overall", df),
    ]

    results = []
    for name, data in situations:
        if data.empty:
            continue
        grouped = (
            data.groupby(["pitcher_id", "pitcher_name", "pitch_team_name", "pitch_team_id"])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=["pitcher_id", "pitcher_name", "pitch_team_name", "pitch_team_id"],
        columns="situation",
        values=["woba", "ba", "pa", "rea", "obp", "slg"],
    )
    pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]

    result = pivot.reset_index()
    return result.rename(
        columns={
            "pitcher_id": "player_id",
            "pitcher_name": "player_name",
            "pitch_team_name": "team_name",
            "pitch_team_id": "team_id",
        }
    )


def analyze_batting_team_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)

    situations = [
        ("risp", df[df["risp_fl"] == 1]),
        ("runners_on", df[df["runners_on_fl"] == 1]),
        ("high_leverage", df[df["high_leverage_fl"] == 1]),
        ("low_leverage", df[df["low_leverage_fl"] == 1]),
        ("overall", df),
    ]

    results = []
    for name, data in situations:
        if data.empty:
            continue
        grouped = (
            data.groupby(["bat_team_id", "bat_team_name"])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=["bat_team_id", "bat_team_name"],
        columns="situation",
        values=["woba", "ba", "pa", "rea", "obp", "slg"],
    )
    pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={"bat_team_name": "team_name", "bat_team_id": "team_id"})


def analyze_pitching_team_situations(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_situation_flags(df)

    situations = [
        ("risp", df[df["risp_fl"] == 1]),
        ("runners_on", df[df["runners_on_fl"] == 1]),
        ("high_leverage", df[df["high_leverage_fl"] == 1]),
        ("low_leverage", df[df["low_leverage_fl"] == 1]),
        ("overall", df),
    ]

    results = []
    for name, data in situations:
        if data.empty:
            continue
        grouped = (
            data.groupby(["pitch_team_id", "pitch_team_name"])
            .apply(lambda g: calculate_batting_metrics(g, weights), include_groups=False)
            .reset_index()
        )
        grouped["situation"] = name
        results.append(grouped)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    pivot = combined.pivot(
        index=["pitch_team_id", "pitch_team_name"],
        columns="situation",
        values=["woba", "ba", "pa", "rea", "obp", "slg"],
    )
    pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]

    result = pivot.reset_index()
    return result.rename(columns={"pitch_team_name": "team_name", "pitch_team_id": "team_id"})
