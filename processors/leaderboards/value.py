import numpy as np
import pandas as pd


def load_guts(data_dir, year: int, division: int) -> pd.Series:
    guts = pd.read_csv(data_dir / "guts/guts_constants.csv")
    row = guts[(guts["year"] == year) & (guts["division"] == division)]
    if row.empty:
        row = guts[(guts["division"] == division)].iloc[-1:]
    return row.iloc[0] if not row.empty else pd.Series({"runs_win": 13.0})


def calculate_batting_value(
    df: pd.DataFrame, runs_per_win: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = df[~df["batter_id"].isna()].copy()

    player_stats = (
        batting.groupby("batter_id")
        .agg(
            {
                "batter_name": "first",
                "bat_team_name": "first",
                "bat_team_id": "first",
                "wpa": "sum",
                "rea": "sum",
                "wpa_li": "sum",
                "li": "mean",
            }
        )
        .reset_index()
    )

    neg_wpa = batting[batting["wpa"] < 0].groupby("batter_id")["wpa"].sum()
    pos_wpa = batting[batting["wpa"] > 0].groupby("batter_id")["wpa"].sum()

    player_stats["neg_wpa"] = player_stats["batter_id"].map(neg_wpa).fillna(0)
    player_stats["pos_wpa"] = player_stats["batter_id"].map(pos_wpa).fillna(0)
    player_stats["rew"] = player_stats["rea"] / runs_per_win
    player_stats["clutch"] = np.where(
        player_stats["li"] > 0,
        (player_stats["wpa"] / player_stats["li"]) - player_stats["wpa_li"],
        np.nan,
    )

    player_stats = player_stats.rename(
        columns={
            "batter_id": "player_id",
            "batter_name": "player_name",
            "bat_team_name": "team_name",
            "bat_team_id": "team_id",
            "li": "pli",
        }
    )

    team_stats = (
        batting.groupby("bat_team_id")
        .agg({"bat_team_name": "first", "wpa": "sum", "rea": "sum", "wpa_li": "sum", "li": "mean"})
        .reset_index()
    )

    team_neg_wpa = batting[batting["wpa"] < 0].groupby("bat_team_id")["wpa"].sum()
    team_pos_wpa = batting[batting["wpa"] > 0].groupby("bat_team_id")["wpa"].sum()

    team_stats["neg_wpa"] = team_stats["bat_team_id"].map(team_neg_wpa).fillna(0)
    team_stats["pos_wpa"] = team_stats["bat_team_id"].map(team_pos_wpa).fillna(0)
    team_stats["rew"] = team_stats["rea"] / runs_per_win
    team_stats["clutch"] = np.where(
        team_stats["li"] > 0, (team_stats["wpa"] / team_stats["li"]) - team_stats["wpa_li"], np.nan
    )

    team_stats = team_stats.rename(
        columns={"bat_team_id": "team_id", "bat_team_name": "team_name", "li": "pli"}
    )

    player_cols = [
        "player_id",
        "player_name",
        "team_name",
        "team_id",
        "wpa",
        "neg_wpa",
        "pos_wpa",
        "rea",
        "rew",
        "pli",
        "wpa_li",
        "clutch",
    ]
    team_cols = [
        "team_id",
        "team_name",
        "wpa",
        "neg_wpa",
        "pos_wpa",
        "rea",
        "rew",
        "pli",
        "wpa_li",
        "clutch",
    ]

    return player_stats[player_cols], team_stats[team_cols]


def calculate_pitching_value(
    df: pd.DataFrame, runs_per_win: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pitching = df[~df["pitcher_id"].isna()].copy()

    pitching["pwpa"] = -pitching["wpa"]
    pitching["prea"] = -pitching["rea"]
    pitching["pwpa_li"] = np.where(pitching["li"] > 0, pitching["pwpa"] / pitching["li"], 0)

    player_stats = (
        pitching.groupby("pitcher_id")
        .agg(
            {
                "pitcher_name": "first",
                "pitch_team_name": "first",
                "pitch_team_id": "first",
                "pwpa": "sum",
                "prea": "sum",
                "pwpa_li": "sum",
                "li": "mean",
            }
        )
        .reset_index()
    )

    neg_wpa = pitching[pitching["pwpa"] < 0].groupby("pitcher_id")["pwpa"].sum()
    pos_wpa = pitching[pitching["pwpa"] > 0].groupby("pitcher_id")["pwpa"].sum()

    player_stats["neg_wpa"] = player_stats["pitcher_id"].map(neg_wpa).fillna(0)
    player_stats["pos_wpa"] = player_stats["pitcher_id"].map(pos_wpa).fillna(0)
    player_stats["rew"] = player_stats["prea"] / runs_per_win
    player_stats["clutch"] = np.where(
        player_stats["li"] > 0,
        (player_stats["pwpa"] / player_stats["li"]) - player_stats["pwpa_li"],
        np.nan,
    )

    player_stats = player_stats.rename(
        columns={
            "pitcher_id": "player_id",
            "pitcher_name": "player_name",
            "pitch_team_name": "team_name",
            "pitch_team_id": "team_id",
            "pwpa": "wpa",
            "prea": "rea",
            "pwpa_li": "wpa_li",
            "li": "pli",
        }
    )

    team_stats = (
        pitching.groupby("pitch_team_id")
        .agg(
            {
                "pitch_team_name": "first",
                "pwpa": "sum",
                "prea": "sum",
                "pwpa_li": "sum",
                "li": "mean",
            }
        )
        .reset_index()
    )

    team_neg_wpa = pitching[pitching["pwpa"] < 0].groupby("pitch_team_id")["pwpa"].sum()
    team_pos_wpa = pitching[pitching["pwpa"] > 0].groupby("pitch_team_id")["pwpa"].sum()

    team_stats["neg_wpa"] = team_stats["pitch_team_id"].map(team_neg_wpa).fillna(0)
    team_stats["pos_wpa"] = team_stats["pitch_team_id"].map(team_pos_wpa).fillna(0)
    team_stats["rew"] = team_stats["prea"] / runs_per_win

    pitching_changes = (
        pitching[(pitching["sub_fl"] == 1) & (pitching["sub_pos"] == "p")]
        .groupby("pitch_team_id")
        .size()
    )
    team_stats["pitching_changes"] = (
        team_stats["pitch_team_id"].map(pitching_changes).fillna(0).astype(int)
    )

    team_stats["clutch"] = np.where(
        team_stats["li"] > 0,
        (team_stats["pwpa"] / team_stats["li"]) - team_stats["pwpa_li"],
        np.nan,
    )

    team_stats = team_stats.rename(
        columns={
            "pitch_team_id": "team_id",
            "pitch_team_name": "team_name",
            "pwpa": "wpa",
            "prea": "rea",
            "pwpa_li": "wpa_li",
            "li": "pli",
        }
    )

    player_cols = [
        "player_id",
        "player_name",
        "team_name",
        "team_id",
        "wpa",
        "neg_wpa",
        "pos_wpa",
        "rea",
        "rew",
        "pli",
        "wpa_li",
        "clutch",
    ]
    team_cols = [
        "team_id",
        "team_name",
        "wpa",
        "neg_wpa",
        "pos_wpa",
        "rea",
        "rew",
        "pli",
        "wpa_li",
        "clutch",
        "pitching_changes",
    ]

    return player_stats[player_cols], team_stats[team_cols]


def analyze_value(df: pd.DataFrame, data_dir, year: int, division: int) -> dict[str, pd.DataFrame]:
    guts = load_guts(data_dir, year, division)
    runs_per_win = guts.get("runs_win", 13.0)

    batting_player, batting_team = calculate_batting_value(df, runs_per_win)
    pitching_player, pitching_team = calculate_pitching_value(df, runs_per_win)

    return {
        "value_batter": batting_player,
        "value_batting_team": batting_team,
        "value_pitcher": pitching_player,
        "value_pitching_team": pitching_team,
    }
