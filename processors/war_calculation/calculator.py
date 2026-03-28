from __future__ import annotations

import numpy as np
import pandas as pd

from processors.logging_utils import get_logger

from .batting import (
    add_batting_stats,
    add_linear_weights,
    batting_runs,
    calculate_bfh,
    calculate_position_adjustments,
    calculate_webt,
    calculate_wgdp,
    fallback_position_adjustment,
    get_batter_clutch_stats,
    replacement_runs,
)
from .common import aggregate_team, fill_missing, float_to_ip, ip_to_float, safe_divide
from .constants import BATTING_SUM_COLS, PITCHING_SUM_COLS
from .models import (
    BattingInputSchema,
    BattingWarSchema,
    GutsConstants,
    PitchingInputSchema,
    PitchingWarSchema,
    WarResults,
)
from .pitching import (
    add_era_plus,
    add_pitching_stats,
    calculate_gmli,
    calculate_pitcher_batted_balls,
    dynamic_rpw,
    era,
    calculate_pitcher_clutch_stats,
    pitching_war,
    ra9,
    reliever_leverage_adjustment,
    replacement_level,
)
from .sos_utils import normalize_division_war, sos_reward_punish

logger = get_logger(__name__)


class WARCalculator:
    """Encapsulates the full WAR calculation pipeline for a single division-year.

    All shared data (guts constants, park factors, play-by-play, lineups) are
    stored on the instance so individual pipeline steps don't need to pass
    DataFrames around.  Call ``run()`` to execute the full pipeline and get
    back a ``WarResults`` with the four output DataFrames.
    """

    def __init__(
        self,
        batting_df: pd.DataFrame,
        pitching_df: pd.DataFrame,
        pbp_df: pd.DataFrame,
        guts_df: pd.DataFrame,
        park_factors_df: pd.DataFrame,
        lineups_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        mappings_df: pd.DataFrame,
        division: str,
        year: int,
    ):
        self.division = division
        self.year = year
        self.guts = GutsConstants.from_dataframe(guts_df)

        self._park_factors = park_factors_df
        self._pbp = pbp_df
        self._lineups = lineups_df
        self._rankings = rankings_df
        self._mappings = mappings_df
        self._batting_raw = batting_df
        self._pitching_raw = pitching_df

        self._pf_by_id = park_factors_df.set_index("team_id")["pf"].to_dict()

        self._total_games = pitching_df["gs"].sum() / 2

    def run(self) -> WarResults:
        bat_war, bat_team_clutch = self._batting_war()
        pitch_war, pitch_team_clutch = self._pitching_war(
            bat_war["war"].sum() if not bat_war.empty else 0.0
        )

        bat_war, pitch_war, missing = sos_reward_punish(
            bat_war,
            pitch_war,
            self._rankings,
            self._mappings,
            self.division,
            self.year,
            alpha=0.2,
            clip_sd=3,
            group_keys=("year", "division"),
            harder_if="higher",
        )

        bat_team = self._batting_team(bat_war, bat_team_clutch)
        pitch_team = self._pitching_team(pitch_war, pitch_team_clutch)

        bat_war, pitch_war = normalize_division_war(
            bat_war, pitch_war, self._rankings, self.division, self.year
        )

        if missing:
            logger.info("  SoS missing for %s teams", len(missing))

        BattingWarSchema.ensure_columns(bat_war)
        PitchingWarSchema.ensure_columns(pitch_war)
        BattingWarSchema.ensure_columns(bat_team)
        PitchingWarSchema.ensure_columns(pitch_team)

        return WarResults(
            batting=BattingWarSchema.select(bat_war),
            pitching=PitchingWarSchema.select(pitch_war),
            batting_team=BattingWarSchema.select(bat_team),
            pitching_team=PitchingWarSchema.select(pitch_team),
            sos_missing=missing,
        )

    def _batting_war(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._batting_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = self._batting_raw.copy()
        BattingInputSchema.ensure_columns(df)

        if "b_t" in df.columns:
            df[["bats", "throws"]] = df["b_t"].str.split("/", n=1, expand=True)

        df["pos"] = df["pos"].apply(
            lambda x: "" if pd.isna(x) else str(x).split("/")[0].upper()
        )
        df = df[df["ab"] > 0].copy()
        df["gp"] = pd.to_numeric(df["gp"], errors="coerce").fillna(0).astype(int)

        df["pf"] = df["team_id"].map(self._pf_by_id).fillna(100)

        df = add_batting_stats(df)
        df = add_linear_weights(df, self.guts)

        # PBP-derived stats ---------------------------------------------------
        gdp_stats = calculate_wgdp(self._pbp)
        df = df.drop(columns=["gdp"])
        df = df.merge(gdp_stats, left_on="player_id", right_index=True, how="left")
        df = fill_missing(df, ["wgdp", "gdp_opps", "gdp"])

        bfh_stats = calculate_bfh(self._pbp)
        df = df.merge(bfh_stats, left_on="player_id", right_index=True, how="left")
        df = fill_missing(df, ["bfh"])

        webt_stats = calculate_webt(self._pbp, self.guts.runs_out).rename(
            columns={"runner_id": "player_id"}
        )
        df = df.merge(webt_stats, on="player_id", how="left")
        df = fill_missing(df, ["webt", "ebt_opps", "ebt"])
        df["baserunning"] = df["wsb"] + df["wgdp"] + df["webt"]

        player_clutch, team_clutch = get_batter_clutch_stats(self._pbp)
        df = df.merge(
            player_clutch[["batter_id", "rea", "wpa", "wpa_li", "clutch"]],
            left_on="player_id",
            right_on="batter_id",
            how="left",
        )

        lg_rpa = safe_divide(df["r"].sum(), df["pa"].sum())
        conf_rpa = (
            df.groupby("conference")["r"].transform("sum")
            / df.groupby("conference")["pa"].transform("sum")
        ).fillna(lg_rpa)

        df["batting"] = batting_runs(df["wraa"], df["pa"], df["pf"], lg_rpa, conf_rpa)

        pos_adj = calculate_position_adjustments(self._lineups, self.division)
        if not pos_adj.empty:
            df = df.merge(pos_adj, on="player_id", how="left")
            has_lineup = df["adjustment"].notna()
            fallback = df.apply(
                lambda r: fallback_position_adjustment(r["pos"], r["gp"], self.division),
                axis=1,
            )
            df["adjustment"] = df["adjustment"].where(has_lineup, fallback)
        else:
            df["adjustment"] = df.apply(
                lambda r: fallback_position_adjustment(r["pos"], r["gp"], self.division),
                axis=1,
            )

        team_count = max(len(df["team_name"].unique()), 1)
        df["replacement_level_runs"] = replacement_runs(
            df["pa"], df["pa"].sum(), team_count, self._total_games, self.guts.runs_win
        )

        for conf in df["conference"].unique():
            mask = df["conference"] == conf
            lg_total = (
                df.loc[mask, "batting"].sum()
                + df.loc[mask, "wsb"].sum()
                + df.loc[mask, "adjustment"].sum()
            )
            lg_pa = df.loc[mask, "pa"].sum()
            df.loc[mask, "league_adjustment"] = (
                (-lg_total / lg_pa if lg_pa > 0 else 0) * df.loc[mask, "pa"]
            )

        df["war"] = (
            df["batting"]
            + df["replacement_level_runs"]
            + df["baserunning"]
            + df["adjustment"]
            + df["league_adjustment"]
        ) / self.guts.runs_win

        df["year"] = self.year
        df["division"] = self.division
        numeric = df.select_dtypes(include="number")
        df[numeric.columns] = numeric.where(np.isfinite(numeric))

        return df.dropna(subset=["war"]), team_clutch

    def _pitching_war(self, bat_war_total: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._pitching_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = self._pitching_raw.copy()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

        PitchingInputSchema.ensure_columns(df)

        df['throws'] = df['throws'].fillna('')

        df = df[(df["app"] > 0) & df["era"].notna()].copy()
        df["ip_float"] = df["ip"].apply(ip_to_float)

        df["pf"] = df["team_id"].map(self._pf_by_id).fillna(100)

        df = df.drop(columns=["fo", "go"], errors="ignore")
        batted_balls = calculate_pitcher_batted_balls(self._pbp)
        df = df.merge(batted_balls, on="player_id", how="left")
        df = fill_missing(df, ["fo", "go", "fb"])

        df = add_pitching_stats(df)
        valid_mask = df["ip_float"] > 0

        lg_era_val = (
            era(df.loc[valid_mask, "er"].sum(), df.loc[valid_mask, "ip_float"].sum())
            if valid_mask.any()
            else 0.0
        )
        df = add_era_plus(df, valid_mask, lg_era_val)

        conf_ra9_map: dict[str, float] = {}
        for conf in df["conference"].unique():
            conf_df = df[(df["conference"] == conf) & valid_mask]
            if not conf_df.empty and conf_df["ip_float"].sum() > 0:
                conf_ra9_map[conf] = ra9(conf_df["r"].sum(), conf_df["ip_float"].sum())

        df["conf_ra9"] = df["conference"].map(conf_ra9_map)
        df["park_adj_ra9"] = np.where(valid_mask, df["ra9"] / (df["pf"] / 100), np.nan)
        df["raap9"] = np.where(valid_mask, df["conf_ra9"] - df["park_adj_ra9"], 0)
        df["ip_per_g"] = safe_divide(df["ip_float"], df["app"])
        df["drpw"] = np.where(
            valid_mask,
            dynamic_rpw(df["ip_per_g"], df["conf_ra9"], df["park_adj_ra9"]),
            0,
        )
        df["replacement_level"] = replacement_level(df["gs"], df["app"])
        df["war"] = np.where(
            valid_mask,
            pitching_war(df["raap9"], df["drpw"], df["replacement_level"], df["ip_float"]),
            0,
        )

        gmli_df = calculate_gmli(self._pbp)
        df = df.merge(
            gmli_df,
            left_on="player_id",
            right_on="pitcher_id",
            how="left",
            suffixes=("", "_gmli"),
        )
        df["gmli"] = df["gmli"].fillna(0)
        df = df.drop(columns=[c for c in df.columns if c.endswith("_gmli")])

        valid_mask = df["ip_float"] > 0
        reliever_mask = (df["gs"] < 3) & valid_mask
        reliever_adj = reliever_leverage_adjustment(df["war"], df["gmli"])
        df["war"] = np.where(reliever_mask.fillna(False), reliever_adj, df["war"])

        target_war = (bat_war_total * 0.43) / 0.57
        current_war = df["war"].sum()
        ip_sum = df.loc[valid_mask, "ip_float"].sum()
        if ip_sum > 0:
            war_adj = (target_war - current_war) / ip_sum
            df.loc[valid_mask, "war"] += war_adj * df.loc[valid_mask, "ip_float"]

        pitcher_clutch, team_clutch = calculate_pitcher_clutch_stats(self._pbp)
        df = df.merge(
            pitcher_clutch[["pitcher_id", "prea", "pwpa", "pwpa_li", "clutch"]],
            left_on="player_id",
            right_on="pitcher_id",
            how="left",
            suffixes=("", "_clutch"),
        )

        numeric = df.select_dtypes(include="number")
        df[numeric.columns] = numeric.where(np.isfinite(numeric))
        df["year"] = self.year
        df["division"] = self.division

        return df.dropna(subset=["war"]), team_clutch

    def _batting_team(
        self, batting_war: pd.DataFrame, team_clutch: pd.DataFrame
    ) -> pd.DataFrame:
        if batting_war.empty:
            return pd.DataFrame()

        team_df = aggregate_team(batting_war, BATTING_SUM_COLS)
        team_df = fill_missing(team_df, BATTING_SUM_COLS)

        team_df["pf"] = team_df["team_id"].map(self._pf_by_id).fillna(100)

        team_df = add_batting_stats(team_df)
        team_df = add_linear_weights(team_df, self.guts)

        team_df = team_df.merge(
            team_clutch[["bat_team_id", "rea", "wpa", "wpa_li", "clutch"]],
            left_on="team_id",
            right_on="bat_team_id",
            how="left",
        )

        team_df["year"] = self.year
        team_df["division"] = self.division
        return team_df

    def _pitching_team(
        self, pitching_war: pd.DataFrame, team_clutch: pd.DataFrame
    ) -> pd.DataFrame:
        if pitching_war.empty:
            return pd.DataFrame()

        team_df = aggregate_team(pitching_war, PITCHING_SUM_COLS)
        team_df = fill_missing(team_df, PITCHING_SUM_COLS)

        team_df["pf"] = team_df["team_id"].map(self._pf_by_id).fillna(100)
        team_df["ip"] = team_df["ip_float"].apply(float_to_ip)

        valid = team_df["ip_float"] > 0
        team_df["era"] = np.where(valid, era(team_df["er"], team_df["ip_float"]), np.nan)
        team_df = add_pitching_stats(team_df)

        team_df = team_df.merge(
            team_clutch[["pitch_team_id", "prea", "pwpa", "pwpa_li", "clutch"]],
            left_on="team_id",
            right_on="pitch_team_id",
            how="left",
        )

        team_df["year"] = self.year
        team_df["division"] = self.division
        return team_df
