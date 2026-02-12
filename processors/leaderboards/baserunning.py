import re

import numpy as np
import pandas as pd

from processors.pbp_parser.constants import EventType

_SCORE_RE = re.compile(
    r"(?:\bscored\b|\bscores\b|\badvanced to home\b|\badvances to home\b|\bsteals home\b|\bstole home\b)",
    flags=re.IGNORECASE,
)
_DOUBLE_SCORED_SCORED_RE = re.compile(r"scored,\s*scored", flags=re.IGNORECASE)
_OUT_AT_RE = re.compile(r"\bout at\b|\bthrown out\b", flags=re.IGNORECASE)


def safe_divide(n, d):
    return n / d if d else 0.0


def add_runner_dests(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["contest_id", "play_id"]).copy()
    g = df.groupby("contest_id", sort=False)
    r1n = g["r1_id"].shift(-1)
    r2n = g["r2_id"].shift(-1)
    r3n = g["r3_id"].shift(-1)

    def dest(r, a, b, c):
        if pd.isna(r):
            return pd.NA
        if not pd.isna(a) and r == a:
            return 1
        if not pd.isna(b) and r == b:
            return 2
        if not pd.isna(c) and r == c:
            return 3
        return 0

    df["r1_dest"] = [dest(r, a, b, c) for r, a, b, c in zip(df["r1_id"], r1n, r2n, r3n)]
    df["r2_dest"] = [dest(r, a, b, c) for r, a, b, c in zip(df["r2_id"], r1n, r2n, r3n)]
    return df


def _score_count(desc: pd.Series) -> pd.Series:
    s = desc.fillna("").astype(str)
    n = s.str.count(_SCORE_RE) - s.str.count(_DOUBLE_SCORED_SCORED_RE)
    return n.clip(lower=0).astype("Int64")


def _out_at_flags(desc: pd.Series) -> pd.Series:
    return desc.fillna("").astype(str).str.contains(_OUT_AT_RE, na=False)


def calculate_player_steal_stats(df: pd.DataFrame) -> pd.DataFrame:
    et = pd.to_numeric(df["event_type"], errors="coerce")
    is_sb = et.eq(EventType.STOLEN_BASE.value)
    is_cs = et.eq(EventType.CAUGHT_STEALING.value)
    is_att = is_sb | is_cs

    r1 = df["r1_id"].notna()
    r2 = df["r2_id"].notna()

    parts = []
    if r1.any():
        parts.append(
            pd.DataFrame(
                {
                    "player_id": df.loc[r1, "r1_id"].values,
                    "contest_id": df.loc[r1, "contest_id"].values,
                    "opp_2b": (~is_att.loc[r1]).astype(int).values,
                    "att_2b": is_att.loc[r1].astype(int).values,
                    "sb_2b": is_sb.loc[r1].astype(int).values,
                    "cs_2b": is_cs.loc[r1].astype(int).values,
                    "opp_3b": 0,
                    "att_3b": 0,
                    "sb_3b": 0,
                    "cs_3b": 0,
                }
            )
        )
    if r2.any():
        parts.append(
            pd.DataFrame(
                {
                    "player_id": df.loc[r2, "r2_id"].values,
                    "contest_id": df.loc[r2, "contest_id"].values,
                    "opp_3b": (~is_att.loc[r2]).astype(int).values,
                    "att_3b": is_att.loc[r2].astype(int).values,
                    "sb_3b": is_sb.loc[r2].astype(int).values,
                    "cs_3b": is_cs.loc[r2].astype(int).values,
                    "opp_2b": 0,
                    "att_2b": 0,
                    "sb_2b": 0,
                    "cs_2b": 0,
                }
            )
        )

    if not parts:
        return pd.DataFrame(
            columns=[
                "player_id",
                "games",
                "opp_2b",
                "att_2b",
                "sb_2b",
                "cs_2b",
                "opp_3b",
                "att_3b",
                "sb_3b",
                "cs_3b",
            ]
        )

    c = pd.concat(parts, ignore_index=True)
    out = (
        c.groupby("player_id", dropna=False)
        .agg(
            games=("contest_id", "nunique"),
            opp_2b=("opp_2b", "sum"),
            opp_3b=("opp_3b", "sum"),
            att_2b=("att_2b", "sum"),
            att_3b=("att_3b", "sum"),
            sb_2b=("sb_2b", "sum"),
            sb_3b=("sb_3b", "sum"),
            cs_2b=("cs_2b", "sum"),
            cs_3b=("cs_3b", "sum"),
        )
        .reset_index()
    )
    return add_steal_rates(out)


def calculate_team_steal_stats(df: pd.DataFrame) -> pd.DataFrame:
    et = pd.to_numeric(df["event_type"], errors="coerce")
    is_sb = et.eq(EventType.STOLEN_BASE.value)
    is_cs = et.eq(EventType.CAUGHT_STEALING.value)
    is_att = is_sb | is_cs

    r1 = df["r1_id"].notna()
    r2 = df["r2_id"].notna()

    d = pd.DataFrame(
        {
            "team_id": df["bat_team_id"].values,
            "contest_id": df["contest_id"].values,
            "opp_2b": (r1 & ~is_att).astype(int).values,
            "opp_3b": (r2 & ~is_att).astype(int).values,
            "att_2b": (is_att & r1).astype(int).values,
            "att_3b": (is_att & r2).astype(int).values,
            "sb_2b": (is_sb & r1).astype(int).values,
            "sb_3b": (is_sb & r2).astype(int).values,
            "cs_2b": (is_cs & r1).astype(int).values,
            "cs_3b": (is_cs & r2).astype(int).values,
        }
    )

    out = (
        d.groupby("team_id", dropna=False)
        .agg(
            games=("contest_id", "nunique"),
            opp_2b=("opp_2b", "sum"),
            opp_3b=("opp_3b", "sum"),
            att_2b=("att_2b", "sum"),
            att_3b=("att_3b", "sum"),
            sb_2b=("sb_2b", "sum"),
            sb_3b=("sb_3b", "sum"),
            cs_2b=("cs_2b", "sum"),
            cs_3b=("cs_3b", "sum"),
        )
        .reset_index()
    )
    return add_steal_rates(out)


def add_steal_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sb"] = df["sb_2b"] + df["sb_3b"]
    df["cs"] = df["cs_2b"] + df["cs_3b"]
    df["sb_att"] = df["att_2b"] + df["att_3b"]

    df["sb_2b_pct"] = np.where(df["att_2b"] > 0, df["sb_2b"] / df["att_2b"], 0.0)
    df["sb_3b_pct"] = np.where(df["att_3b"] > 0, df["sb_3b"] / df["att_3b"], 0.0)
    df["sb_pct"] = np.where(df["sb_att"] > 0, df["sb"] / df["sb_att"], 0.0)

    df["att_2b_per_game"] = np.where(df["games"] > 0, df["att_2b"] / df["games"], 0.0)
    df["att_3b_per_game"] = np.where(df["games"] > 0, df["att_3b"] / df["games"], 0.0)
    df["sb_att_per_game"] = np.where(df["games"] > 0, df["sb_att"] / df["games"], 0.0)
    return df


def calculate_wgdp(df: pd.DataFrame, group_col: str, id_col: str) -> pd.DataFrame:
    ob = pd.to_numeric(df["outs_before"], errors="coerce")
    opps = df[df["r1_id"].notna() & ob.lt(2)].copy()
    is_gdp = opps["play_description"].fillna("").astype(str).str.contains("double play", case=False, na=False)

    opp_counts = opps.groupby(group_col).size()
    gdp_counts = opps[is_gdp].groupby(group_col).size()

    out = (
        pd.DataFrame({"gdp_opps": opp_counts, "gdp": gdp_counts.reindex(opp_counts.index, fill_value=0)})
        .reset_index()
        .rename(columns={group_col: id_col})
    )

    lg_rate = safe_divide(out["gdp"].sum(), out["gdp_opps"].sum())
    out["wgdp"] = (out["gdp_opps"] * lg_rate - out["gdp"]) * 0.5
    return out[[id_col, "gdp_opps", "gdp", "wgdp"]]


def calculate_webt(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = add_runner_dests(df)
    et = pd.to_numeric(df["event_type"], errors="coerce")
    is_single = et.eq(EventType.SINGLE.value)
    is_double = et.eq(EventType.DOUBLE.value)

    has_r1 = df["r1_id"].notna()
    has_r2 = df["r2_id"].notna()

    c_13 = has_r1 & is_single
    c_2h = has_r2 & is_single
    c_1h = has_r1 & is_double

    scored_n = _score_count(df["play_description"])
    out_at = _out_at_flags(df["play_description"])

    p2_is_r1 = has_r1
    p2_is_r2 = (~has_r1) & has_r2

    r1_scored = p2_is_r1 & scored_n.gt(0)
    r2_scored = p2_is_r2 & scored_n.gt(0)

    r1_out = p2_is_r1 & out_at
    r2_out = p2_is_r2 & out_at

    taken_13 = c_13 & df["r1_dest"].eq(3)
    out_13 = c_13 & r1_out
    hold_13 = c_13 & ~taken_13 & ~out_13

    taken_2h = c_2h & r2_scored
    out_2h = c_2h & r2_out
    hold_2h = c_2h & ~taken_2h & ~out_2h

    taken_1h = c_1h & r1_scored
    out_1h = c_1h & r1_out
    hold_1h = c_1h & ~taken_1h & ~out_1h

    def pack(mask, who, col):
        t = df.loc[mask, [who]].rename(columns={who: "runner_id"})
        t[col] = 1
        return t

    r = (
        pd.concat(
            [
                pack(c_13, "r1_id", "opp_13"),
                pack(taken_13, "r1_id", "taken_13"),
                pack(out_13, "r1_id", "out_13"),
                pack(hold_13, "r1_id", "hold_13"),
                pack(c_2h, "r2_id", "opp_2h"),
                pack(taken_2h, "r2_id", "taken_2h"),
                pack(out_2h, "r2_id", "out_2h"),
                pack(hold_2h, "r2_id", "hold_2h"),
                pack(c_1h, "r1_id", "opp_1h"),
                pack(taken_1h, "r1_id", "taken_1h"),
                pack(out_1h, "r1_id", "out_1h"),
                pack(hold_1h, "r1_id", "hold_1h"),
            ],
            ignore_index=True,
        )
        .groupby("runner_id", dropna=False)
        .sum(numeric_only=True)
        .reset_index()
    )

    for c in [
        "opp_13",
        "taken_13",
        "out_13",
        "hold_13",
        "opp_2h",
        "taken_2h",
        "out_2h",
        "hold_2h",
        "opp_1h",
        "taken_1h",
        "out_1h",
        "hold_1h",
    ]:
        if c not in r.columns:
            r[c] = 0
        r[c] = r[c].fillna(0).astype(int)

    lg = r.sum(numeric_only=True)
    lg_rates = {}
    for t in ["13", "2h", "1h"]:
        opp = float(lg.get(f"opp_{t}", 0.0))
        taken = float(lg.get(f"taken_{t}", 0.0))
        outc = float(lg.get(f"out_{t}", 0.0))
        lg_rates[t] = {
            "taken": 0.0 if opp == 0 else taken / opp,
            "out": 0.0 if opp == 0 else outc / opp,
        }

    runs_out = float(weights["runs_out"])
    webt = np.zeros(len(r), dtype=float)
    for t in ["13", "2h", "1h"]:
        opp = r[f"opp_{t}"].to_numpy(dtype=float)
        taken = r[f"taken_{t}"].to_numpy(dtype=float)
        outc = r[f"out_{t}"].to_numpy(dtype=float)
        webt += (taken - lg_rates[t]["taken"] * opp) + (outc - lg_rates[t]["out"] * opp) * (-runs_out)

    r["webt"] = webt
    r["ebt_opps"] = r["opp_13"] + r["opp_2h"] + r["opp_1h"]
    r["ebt_taken"] = r["taken_13"] + r["taken_2h"] + r["taken_1h"]
    r["ebt_out"] = r["out_13"] + r["out_2h"] + r["out_1h"]
    r["ebt_hold"] = r["hold_13"] + r["hold_2h"] + r["hold_1h"]

    return r[
        [
            "runner_id",
            "webt",
            "opp_13",
            "taken_13",
            "out_13",
            "hold_13",
            "opp_2h",
            "taken_2h",
            "out_2h",
            "hold_2h",
            "opp_1h",
            "taken_1h",
            "out_1h",
            "hold_1h",
            "ebt_opps",
            "ebt_taken",
            "ebt_out",
            "ebt_hold",
        ]
    ].rename(columns={"runner_id": "player_id"})


def calculate_wsb(steal_df: pd.DataFrame, weights: dict) -> pd.Series:
    run_sb = float(weights["runs_sb"])
    run_cs = float(weights["runs_cs"])

    lg_sb = float(steal_df["sb"].sum())
    lg_cs = float(steal_df["cs"].sum())
    lg_opps = float(steal_df["sb_att"].sum())

    lgwSB = safe_divide(lg_sb * run_sb + lg_cs * run_cs, lg_opps)
    opps = steal_df["sb_att"].clip(lower=0)
    return (steal_df["sb"] * run_sb) + (steal_df["cs"] * run_cs) - (lgwSB * opps)


def calculate_baserunning_stats(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    steal = calculate_player_steal_stats(df)

    name_map = (
        pd.concat(
            [
                df.loc[df["r1_id"].notna(), ["r1_id", "r1_name", "bat_team_id", "bat_team_name"]].rename(
                    columns={"r1_id": "player_id", "r1_name": "player_name"}
                ),
                df.loc[df["r2_id"].notna(), ["r2_id", "r2_name", "bat_team_id", "bat_team_name"]].rename(
                    columns={"r2_id": "player_id", "r2_name": "player_name"}
                ),
            ],
            ignore_index=True,
        )
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"bat_team_id": "team_id", "bat_team_name": "team_name"})
    )

    wgdp = calculate_wgdp(df, "batter_id", "player_id")
    webt = calculate_webt(df, weights)

    out = steal.merge(name_map, on="player_id", how="left")
    out = out.merge(wgdp, on="player_id", how="left")
    out = out.merge(webt, on="player_id", how="left")

    out["wgdp"] = out["wgdp"].fillna(0.0)
    out["webt"] = out["webt"].fillna(0.0)
    out["wsb"] = calculate_wsb(out, weights)
    out["baserunning"] = out["wsb"] + out["wgdp"] + out["webt"]

    out["gdp_opps"] = out["gdp_opps"].fillna(0).astype(int)
    out["gdp"] = out["gdp"].fillna(0).astype(int)

    for c in [
        "opp_13",
        "taken_13",
        "out_13",
        "hold_13",
        "opp_2h",
        "taken_2h",
        "out_2h",
        "hold_2h",
        "opp_1h",
        "taken_1h",
        "out_1h",
        "hold_1h",
        "ebt_opps",
        "ebt_taken",
        "ebt_out",
        "ebt_hold",
    ]:
        out[c] = out[c].fillna(0).astype(int)

    return out[
        [
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "games",
            "sb",
            "cs",
            "sb_att",
            "sb_pct",
            "sb_att_per_game",
            "sb_2b",
            "cs_2b",
            "att_2b",
            "sb_2b_pct",
            "att_2b_per_game",
            "sb_3b",
            "cs_3b",
            "att_3b",
            "sb_3b_pct",
            "att_3b_per_game",
            "wsb",
            "gdp_opps",
            "gdp",
            "wgdp",
            "opp_13",
            "taken_13",
            "out_13",
            "hold_13",
            "opp_2h",
            "taken_2h",
            "out_2h",
            "hold_2h",
            "opp_1h",
            "taken_1h",
            "out_1h",
            "hold_1h",
            "ebt_opps",
            "ebt_taken",
            "ebt_out",
            "ebt_hold",
            "webt",
            "baserunning",
        ]
    ]


def calculate_team_baserunning(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    steal = calculate_team_steal_stats(df)

    name_map = (
        df.groupby("bat_team_id", dropna=False)
        .agg(team_name=("bat_team_name", "first"))
        .reset_index()
        .rename(columns={"bat_team_id": "team_id"})
    )

    wgdp = calculate_wgdp(df, "bat_team_id", "team_id")

    webt_player = calculate_webt(df, weights)
    runner_to_team = pd.concat(
        [
            df.loc[df["r1_id"].notna(), ["r1_id", "bat_team_id"]].rename(columns={"r1_id": "player_id"}),
            df.loc[df["r2_id"].notna(), ["r2_id", "bat_team_id"]].rename(
                columns={"r2_id": "player_id"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    webt_team = (
        webt_player.merge(runner_to_team.rename(columns={"bat_team_id": "team_id"}), on="player_id", how="left")
        .groupby("team_id", dropna=False)
        .agg(
            webt=("webt", "sum"),
            opp_13=("opp_13", "sum"),
            taken_13=("taken_13", "sum"),
            out_13=("out_13", "sum"),
            hold_13=("hold_13", "sum"),
            opp_2h=("opp_2h", "sum"),
            taken_2h=("taken_2h", "sum"),
            out_2h=("out_2h", "sum"),
            hold_2h=("hold_2h", "sum"),
            opp_1h=("opp_1h", "sum"),
            taken_1h=("taken_1h", "sum"),
            out_1h=("out_1h", "sum"),
            hold_1h=("hold_1h", "sum"),
            ebt_opps=("ebt_opps", "sum"),
            ebt_taken=("ebt_taken", "sum"),
            ebt_out=("ebt_out", "sum"),
            ebt_hold=("ebt_hold", "sum"),
        )
        .reset_index()
    )

    out = steal.merge(name_map, on="team_id", how="left")
    out = out.merge(wgdp, on="team_id", how="left")
    out = out.merge(webt_team, on="team_id", how="left")

    out["wgdp"] = out["wgdp"].fillna(0.0)
    out["webt"] = out["webt"].fillna(0.0)
    out["wsb"] = calculate_wsb(out, weights)
    out["baserunning"] = out["wsb"] + out["wgdp"] + out["webt"]

    out["gdp_opps"] = out["gdp_opps"].fillna(0).astype(int)
    out["gdp"] = out["gdp"].fillna(0).astype(int)

    for c in [
        "opp_13",
        "taken_13",
        "out_13",
        "hold_13",
        "opp_2h",
        "taken_2h",
        "out_2h",
        "hold_2h",
        "opp_1h",
        "taken_1h",
        "out_1h",
        "hold_1h",
        "ebt_opps",
        "ebt_taken",
        "ebt_out",
        "ebt_hold",
    ]:
        out[c] = out[c].fillna(0).astype(int)

    return out[
        [
            "team_id",
            "team_name",
            "games",
            "sb",
            "cs",
            "sb_att",
            "sb_pct",
            "sb_att_per_game",
            "sb_2b",
            "cs_2b",
            "att_2b",
            "sb_2b_pct",
            "att_2b_per_game",
            "sb_3b",
            "cs_3b",
            "att_3b",
            "sb_3b_pct",
            "att_3b_per_game",
            "wsb",
            "gdp_opps",
            "gdp",
            "wgdp",
            "opp_13",
            "taken_13",
            "out_13",
            "hold_13",
            "opp_2h",
            "taken_2h",
            "out_2h",
            "hold_2h",
            "opp_1h",
            "taken_1h",
            "out_1h",
            "hold_1h",
            "ebt_opps",
            "ebt_taken",
            "ebt_out",
            "ebt_hold",
            "webt",
            "baserunning",
        ]
    ]
