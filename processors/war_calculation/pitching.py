import numpy as np
import pandas as pd

from processors.pbp_parser.constants import BattedBallType
from processors.pbp_parser.regexes import RX_FLIED_OUT, RX_GROUNDED_OUT

from .common import safe_divide


def era(er, ip):
    return safe_divide(er * 9, ip)


def k9(so, ip):
    return safe_divide(so * 9, ip)


def bb9(bb, ip):
    return safe_divide(bb * 9, ip)


def h9(h, ip):
    return safe_divide(h * 9, ip)


def hr9(hr, ip):
    return safe_divide(hr * 9, ip)


def ra9(r, ip):
    return safe_divide(r * 9, ip)


def whip(bb, h, ip):
    return safe_divide(bb + h, ip)


def k_pct(so, bf):
    return safe_divide(so, bf) * 100


def bb_pct(bb, bf):
    return safe_divide(bb, bf) * 100


def k_minus_bb_pct(k_pct_val, bb_pct_val):
    return k_pct_val - bb_pct_val


def hr_div_fb(hr, fb):
    return safe_divide(hr, fb) * 100


def inherited_runners_scored_pct(inh_run_score, inh_run):
    return safe_divide(inh_run_score, inh_run) * 100


def era_plus(player_era, lg_era, pf):
    return 100 * (2 - (player_era / lg_era) * (100 / pf))


def pitching_babip(h, hr, ab, k, sfa):
    return safe_divide(h - hr, ab - hr - k + sfa)


def pitching_ba(h, ab):
    return safe_divide(h, ab)


def pitching_obp(h, bb, hbp, ab, sfa):
    return safe_divide(h + bb + hbp, ab + bb + hbp + sfa)


def dynamic_rpw(ip_per_game, conf_ra9, pra9):
    return (((18 - ip_per_game) * conf_ra9 + ip_per_game * pra9) / 18 + 2) * 1.5


def replacement_level(gs, app):
    gs_rate = safe_divide(gs, app)
    return 0.03 * (1 - gs_rate) + 0.12 * gs_rate


def pitching_war(raap9, drpw, replacement, ip):
    wpgaa = safe_divide(raap9, drpw)
    wpgar = wpgaa + replacement
    return wpgar * (ip / 9)

def leverage_adjustment(war_val, gmli, app, gs):
    relief_pct = np.where(app > 0, (app - gs) / app, 0)
    multiplier = relief_pct * (1 + gmli) / 2 + (1 - relief_pct)
    return war_val * multiplier

def calculate_pitcher_batted_balls(pbp_df: pd.DataFrame) -> pd.DataFrame:
    valid = pbp_df["pitcher_id"].notna() & (pbp_df["pitcher_id"] != "")
    df = pbp_df[valid]

    fo_mask = df["play_description"].str.contains(RX_FLIED_OUT, na=False)
    go_mask = df["play_description"].str.contains(RX_GROUNDED_OUT, na=False)
    fb_mask = df["batted_ball_type"] == BattedBallType.FLY_BALL

    stats = pd.DataFrame(
        {
            "fo": df[fo_mask].groupby("pitcher_id").size(),
            "go": df[go_mask].groupby("pitcher_id").size(),
            "fb": df[fb_mask].groupby("pitcher_id").size(),
        }
    ).fillna(0).reset_index().rename(columns={"pitcher_id": "player_id"})

    return stats

def calculate_pitcher_clutch_stats(pbp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_df.copy()

    pitcher_stats = (
        df.groupby("pitcher_id")
        .agg(
            rea=("rea", "sum"),
            wpa=("wpa", "sum"),
            wpa_li=("wpa_li", "sum"),
            li=("li", "mean"),
        )
        .reset_index()
    )
    pitcher_stats["prea"] = -pitcher_stats["rea"]
    pitcher_stats["pwpa"] = -pitcher_stats["wpa"]
    pitcher_stats["pwpa_li"] = -pitcher_stats["wpa_li"]

    pitcher_stats["clutch"] = np.where(
        pitcher_stats["li"] > 0,
        (pitcher_stats["pwpa"] / pitcher_stats["li"]) - pitcher_stats["pwpa_li"],
        np.nan,
    )

    team_stats = (
        df.groupby("pitch_team_id")
        .agg(
            rea=("rea", "sum"),
            wpa=("wpa", "sum"),
            wpa_li=("wpa_li", "sum"),
            li=("li", "mean"),
        )
        .reset_index()
    )
    team_stats["prea"] = -team_stats["rea"]
    team_stats["pwpa"] = -team_stats["wpa"]
    team_stats["pwpa_li"] = -team_stats["wpa_li"]

    team_stats["clutch"] = np.where(
        team_stats["li"] > 0,
        (team_stats["pwpa"] / team_stats["li"]) - team_stats["pwpa_li"],
        np.nan,
    )

    return pitcher_stats, team_stats


def calculate_gmli(pbp_df: pd.DataFrame) -> pd.DataFrame:
    df = pbp_df[pbp_df["pitcher_id"].notna()].copy()

    df = df.sort_values(["pitcher_id", "contest_id", "play_id"])
    df["li"] = df.groupby(["pitcher_id", "contest_id"])["li"].shift(-1)
    first_app = df.groupby(["pitcher_id", "contest_id"]).first().reset_index()
    relievers = first_app[first_app["inning"] > 1]
    if relievers.empty:
        return pd.DataFrame(columns=["pitcher_id", "gmli"])

    result = relievers.groupby("pitcher_id").agg({"li": "mean"}).reset_index()
    return result.rename(columns={"li": "gmli"})


def add_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ip = df["ip_float"]
    outs = ip * 3

    df["bf"] = outs + df["h"] + df["bb"] + df["hbp"]

    df["ra9"] = ra9(df["r"], ip)
    df["k9"] = k9(df["so"], ip)
    df["h9"] = h9(df["h"], ip)
    df["bb9"] = bb9(df["bb"], ip)
    df["hr9"] = hr9(df["hr_a"], ip)
    df["whip"] = whip(df["bb"], df["h"], ip)

    df["k_pct"] = k_pct(df["so"], df["bf"])
    df["bb_pct"] = bb_pct(df["bb"], df["bf"])
    df["k_minus_bb_pct"] = k_minus_bb_pct(df["k_pct"], df["bb_pct"])

    df["hr_div_fb"] = hr_div_fb(df["hr_a"], df["fb"])

    sfa = df["sfa"].fillna(0)
    ab = df["bf"] - df["bb"] - df["hbp"] - sfa
    df["ba_against"] = pitching_ba(df["h"], ab)
    df["obp_against"] = pitching_obp(df["h"], df["bb"], df["hbp"], ab, sfa)
    df["babip_against"] = pitching_babip(df["h"], df["hr_a"], ab, df["so"], sfa)

    return df


def add_era_plus(df: pd.DataFrame, valid_mask: pd.Series, lg_era_val: float) -> pd.DataFrame:
    df = df.copy()
    df["era_plus"] = np.nan
    if valid_mask.any() and lg_era_val > 0:
        df.loc[valid_mask, "era_plus"] = era_plus(
            df.loc[valid_mask, "era"], lg_era_val, df.loc[valid_mask, "pf"]
        )
    return df
