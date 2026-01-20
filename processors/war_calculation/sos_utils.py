import numpy as np
import pandas as pd


def norm_team(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", "and").replace(".", "").replace("  ", " ")
    return s

def build_team_to_sos(rankings_df, mappings):
    rk = rankings_df.copy()

    for c in ("massey_team",):
        rk[c] = rk[c].astype(str).map(norm_team)

    mappings = mappings.copy()
    for c in ("ncaa_team", "massey_team"):
        if c not in mappings.columns:
            raise ValueError("team_mappings.csv must include columns: ncaa_team, massey_team")
        mappings[c] = mappings[c].astype(str).map(norm_team)

    out = (
        mappings[["ncaa_team", "massey_team"]].dropna()
        .merge(rk[["massey_team", "sos_val"]].dropna().drop_duplicates("massey_team"),
               on="massey_team", how="left")
    )
    return out[["ncaa_team", "sos_val"]]

def sos_reward_punish(
    batting_war, pitching_war, rankings_df, ntm, division, year,
    alpha=0.06, group_keys=('year', 'division'), clip_sd=2.5, harder_if='auto'
):
    t2s = build_team_to_sos(rankings_df, ntm)

    for df_ in (batting_war, pitching_war):
        df_["team_name_norm"] = df_["team_name"].astype(str).map(norm_team)

    b = batting_war.merge(t2s, left_on="team_name_norm", right_on="ncaa_team", how="left").drop(columns=["ncaa_team"])
    p = pitching_war.merge(t2s, left_on="team_name_norm", right_on="ncaa_team", how="left").drop(columns=["ncaa_team"])

    min_sos = pd.to_numeric(rankings_df["sos_val"], errors="coerce").min()
    for df_ in (b, p):
        df_["sos_val"] = pd.to_numeric(df_.get("sos_val"), errors="coerce")
        df_["sos_val"] = df_["sos_val"].fillna(min_sos)
        df_["year"] = year
        df_["division"] = division

    missing_teams = sorted(set(b.loc[b["sos_val"].isna(), "team_name"]) | set(p.loc[p["sos_val"].isna(), "team_name"]))

    bp = pd.concat([b.assign(component="batting"), p.assign(component="pitching")], ignore_index=True)

    sign = 1.0 if harder_if == 'higher' else -1.0
    grp = bp.groupby(list(group_keys))["sos_val"]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)

    bp["diff_z"] = sign * (bp["sos_val"] - mu) / sd
    if clip_sd is not None:
        bp["diff_z"] = bp["diff_z"].clip(-clip_sd, clip_sd)

    bp["sos_adj_war"] = bp["war"] * (1 + alpha * bp["diff_z"] * np.sign(bp["war"]).replace(0, 1.0))

    def _rescale(g):
        raw = g["war"].sum()
        adj = g["sos_adj_war"].sum()
        s = 1.0 if adj == 0 else raw / max(adj, 1e-12)
        g["sos_adj_war"] *= s
        return g

    component_col = bp["component"].copy()
    bp = bp.groupby(["component"] + list(group_keys), group_keys=False).apply(_rescale, include_groups=False)
    if "component" not in bp.columns:
        bp["component"] = component_col

    b_out = bp[bp["component"] == "batting"].drop(columns=["component"])
    p_out = bp[bp["component"] == "pitching"].drop(columns=["component"])
    return b_out, p_out, missing_teams

def normalize_division_war(bat_df, pitch_df, standings_df, division, year, pitcher_share=0.40):
    s = standings_df[(standings_df['division'] == division) & (standings_df['year'] == year)]
    total_wins = s['wins'].sum()
    total_games = s['games'].sum()
    rep_wp = 0.294
    target_total = total_wins - rep_wp * total_games

    bat_total = bat_df['war'].sum()
    pitch_total = pitch_df['war'].sum()

    target_bat = target_total * (1 - pitcher_share)
    target_pitch = target_total * pitcher_share

    sb = 1.0 if bat_total == 0 else target_bat / max(bat_total, 1e-12)
    sp = 1.0 if pitch_total == 0 else target_pitch / max(pitch_total, 1e-12)

    for col in ("war", "sos_adj_war"):
        if col not in bat_df.columns or col not in pitch_df.columns:
            raise ValueError(f"{col} missing before division normalization")
        bat_df[col] *= sb
        pitch_df[col] *= sp

    bat_df['year'] = year
    bat_df['division'] = division

    pitch_df['year'] = year
    pitch_df['division'] = division

    return bat_df, pitch_df
