from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

position_adjustments = {
    'SS': 1.85, 'C': 3.09, '2B': 0.62, '3B': 0.62, 'UT': 0.62,
    'CF': 0.62, 'INF': 0.62, 'LF': -1.85, 'RF': -1.85, '1B': -3.09,
    'DH': -3.09, 'OF': 0.25, 'PH': -0.74, 'PR': -0.74, 'P': 0.62,
    'RP': 0.62, 'SP': 0.62, '': 0
}

batting_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'b_t', 'team_name', 'conference', 'gp',
    'bb', 'cs', 'gs', 'hbp', 'ibb', 'k', 'rbi', 'sf', 'ab',
    'pa', 'h', '2b', '3b', 'hr', 'r', 'sb', 'ops_plus', 'picked',
    'sac', 'ba', 'slg_pct', 'ob_pct', 'iso', 'woba', 'k%', 'bb%',
    'sb%', 'wrc_plus', 'wrc', 'r/pa',
    'wsb', 'wgdp', 'wteb', 'ebt', 'opportunities', 'outs_ob',
    'gdp_opps', 'gdp', 'clutch', 'wpa', 'rea', 'wpa/li',
    'batting', 'baserunning', 'adjustment', 'war', 'sos_adj_war'
]

pitching_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'b_t', 'team_name', 'conference', 'app',
    'gs', 'era', 'ip', 'w', 'l', 'sv', 'ip_float', 'h', 'r', 'er',
    'bb', 'so', 'hr_a', '2b_a', '3b_a', 'hbp', 'bf', 'fo', 'go', 'pitches',
    'gmli', 'k9', 'bb9', 'hr9', 'ra9', 'h9', 'ir_a%', 'k%', 'bb%', 'k-bb%', 'hr/fb', 'fip',
    'xfip', 'era+', 'inh_run', 'inh_run_score',
    'clutch', 'pwpa', 'prea', 'pwpa/li', 'war', 'sos_adj_war'
]

def norm_team(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s).lower().strip()
    s = s.replace("&", "and").replace(".", "").replace("  ", " ")
    return s

def _best_name_match(query, choices, *, cutoff=80, scorer=fuzz.token_sort_ratio):
    if not query or not choices:
        return None
    res = process.extractOne(query, choices, scorer=scorer, score_cutoff=cutoff)
    return res[0] if res else None

@lru_cache(maxsize=4096)
def _team_roster_names(team_dict_keys_tuple):
    return list(team_dict_keys_tuple)

def build_team_to_sos(rankings_df, mappings):
    rk = rankings_df.copy()
    # Normalize ranking team column into 'massey_team'
    if "massey_team" not in rk.columns:
        if "team" in rk.columns:
            rk = rk.rename(columns={"team": "massey_team"})
        elif "school" in rk.columns:
            rk = rk.rename(columns={"school": "massey_team"})
        else:
            raise ValueError("Rankings must include a team column (team|school|massey_team)")

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

def sos_reward_punish_players(
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

    # Keep track of truly unmapped after fill (debug only); most cases will have fill applied
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

def calculate_wgdp(pbp_df):
    gdp_opps = pbp_df[(pbp_df['r1_name'] != '') & (pbp_df['outs_before'].astype(int) < 2)].copy()
    gdp_events = gdp_opps[gdp_opps['description'].str.contains('double play', case=False, na=False)]

    gdp_stats = pd.DataFrame({
        'gdp_opps': gdp_opps.groupby('batter_id').size(),
        'gdp': gdp_events.groupby('batter_id').size()
    }).fillna(0)

    lg_gdp_rate = gdp_stats['gdp'].sum() / max(gdp_stats['gdp_opps'].sum(), 1e-12)
    gdp_run_value = -0.5
    gdp_stats['wgdp'] = ((gdp_stats['gdp_opps'] * lg_gdp_rate - gdp_stats['gdp']) * gdp_run_value)

    return gdp_stats

def calculate_extra_bases(df, roster, weights):
    """
    Faster EBT + outs_on_bases using cached fuzzy matching.
    Keeps column names: ebt, outs_ob, opportunities, success_rate, wteb
    """
    extra_bases, opportunities, outs_on_bases = {}, {}, {}

    batting_lookup = {
        team: group.set_index('player_name')['player_id'].to_dict()
        for team, group in roster.groupby('team_name')
    }

    def next_play_row(idx, plays):
        try:
            pos = plays.index.get_loc(idx)
            if isinstance(pos, slice):
                pos = pos.start
            npos = pos + 1
            if npos < len(plays.index):
                return plays.iloc[npos]
        except Exception:
            pass
        return None

    def match_name(name, team_dict):
        if not pd.notna(name) or not team_dict:
            return None
        names_list = _team_roster_names(tuple(team_dict.keys()))
        return _best_name_match(name, names_list, cutoff=50)

    # Singles
    singles = df[df['event_cd'] == 20]
    for idx, play in singles.iterrows():
        nxt = next_play_row(idx, df)
        team_dict = batting_lookup.get(play.bat_team, {})
        
        if pd.notna(play.r1_name) and team_dict:
            s1 = match_name(play.r1_name, team_dict)
            if s1:
                opportunities[s1] = opportunities.get(s1, 0) + 1
                if nxt is not None:
                    next_r1 = match_name(nxt.get('r1_name'), team_dict) if pd.notna(nxt.get('r1_name')) else None
                    next_r2 = match_name(nxt.get('r2_name'), team_dict) if pd.notna(nxt.get('r2_name')) else None
                    next_r3 = match_name(nxt.get('r3_name'), team_dict) if pd.notna(nxt.get('r3_name')) else None
                    next_names = {next_r1, next_r2, next_r3} - {None}
                    if play.outs_on_play > 0 and s1 not in next_names:
                        outs_on_bases[s1] = outs_on_bases.get(s1, 0) + 1
                    elif s1 == next_r3:
                        extra_bases[s1] = extra_bases.get(s1, 0) + 1
        
        if pd.notna(play.r2_name) and team_dict:
            s2 = match_name(play.r2_name, team_dict)
            if s2:
                opportunities[s2] = opportunities.get(s2, 0) + 1
                if nxt is not None:
                    next_r1 = match_name(nxt.get('r1_name'), team_dict) if pd.notna(nxt.get('r1_name')) else None
                    next_r2 = match_name(nxt.get('r2_name'), team_dict) if pd.notna(nxt.get('r2_name')) else None
                    next_r3 = match_name(nxt.get('r3_name'), team_dict) if pd.notna(nxt.get('r3_name')) else None
                    next_names = {next_r1, next_r2, next_r3} - {None}
                    if play.outs_on_play > 0 and s2 not in next_names:
                        outs_on_bases[s2] = outs_on_bases.get(s2, 0) + 1
                    elif (s2 not in next_names) and (play.runs_on_play > 0):
                        extra_bases[s2] = extra_bases.get(s2, 0) + 1

    # Doubles
    doubles = df[df['event_cd'] == 21]
    for idx, play in doubles.iterrows():
        nxt = next_play_row(idx, df)
        team_dict = batting_lookup.get(play.bat_team, {})
        
        if pd.notna(play.r1_name) and team_dict:
            s1 = match_name(play.r1_name, team_dict)
            if s1:
                opportunities[s1] = opportunities.get(s1, 0) + 1
                if nxt is not None:
                    next_r1 = match_name(nxt.get('r1_name'), team_dict) if pd.notna(nxt.get('r1_name')) else None
                    next_r2 = match_name(nxt.get('r2_name'), team_dict) if pd.notna(nxt.get('r2_name')) else None
                    next_r3 = match_name(nxt.get('r3_name'), team_dict) if pd.notna(nxt.get('r3_name')) else None
                    next_names = {next_r1, next_r2, next_r3} - {None}
                    if play.outs_on_play > 0 and s1 not in next_names:
                        outs_on_bases[s1] = outs_on_bases.get(s1, 0) + 1
                    elif (s1 not in next_names) and (play.runs_on_play > 0):
                        extra_bases[s1] = extra_bases.get(s1, 0) + 1

    results = pd.DataFrame({
        'ebt': pd.Series(extra_bases, dtype='float'),
        'outs_ob': pd.Series(outs_on_bases, dtype='float'),
        'opportunities': pd.Series(opportunities, dtype='float')
    }).fillna(0.0)

    if results.empty or results['opportunities'].sum() == 0:
        results['success_rate'] = 0.0
        results['wteb'] = 0.0
        return results.sort_values('ebt', ascending=False)

    results['success_rate'] = (results['ebt'] / results['opportunities']).replace([np.inf, -np.inf], 0.0).round(3)

    lg_teb_rate = results['ebt'].sum() / max(results['opportunities'].sum(), 1e-12)
    lg_out_rate = results['outs_ob'].sum() / max(results['opportunities'].sum(), 1e-12)

    run_extra_base = 0.3
    runs_per_out = weights['runs_out']
    run_out = -1 * (2 * runs_per_out + 0.075)

    results['wteb'] = (
        (results['ebt'] * run_extra_base) +
        (results['outs_ob'] * run_out) -
        (results['opportunities'] * (lg_teb_rate * run_extra_base + lg_out_rate * run_out))
    )

    return results.sort_values('ebt', ascending=False)

def get_clutch_stats(pbp_df):
    pbp_df = pbp_df.copy()

    player_stats = (pbp_df.groupby(['player_id']).agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa/li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('wpa/li', ascending=False)
    .reset_index(drop=True))

    player_stats['clutch'] = np.where(
        player_stats['li'] == 0,
        np.nan,
        (player_stats['wpa'] / player_stats['li']) - player_stats['wpa/li']
    )

    team_stats = (pbp_df.groupby('bat_team').agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa/li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('wpa/li', ascending=False)
    .reset_index(drop=True))

    team_stats['clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['wpa'] / team_stats['li']) - team_stats['wpa/li']
    )

    return player_stats, team_stats

def get_pitcher_clutch_stats(pbp_df):
    pbp_df = pbp_df.copy()
    pbp_df['prea'] = (-pbp_df['run_expectancy_delta'] - pbp_df['runs_on_play'])
    pbp_df['pwpa'] = np.where(
        pbp_df['pitch_team'] == pbp_df['home_team'],
        pbp_df['delta_home_win_exp'],
        -pbp_df['delta_home_win_exp']
    )
    pbp_df['pwpa/li'] = pbp_df['pwpa'].div(pbp_df['li'].replace(0, float('nan')))

    pitcher_stats = (pbp_df.groupby(['pitcher_id']).agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa/li': 'sum', 'li': 'mean', 'pitcher_standardized': 'first',
    })
    .reset_index()
    .sort_values('pwpa/li', ascending=False)
    .reset_index(drop=True))

    pitcher_stats = pitcher_stats[pitcher_stats['pitcher_standardized'] != "Starter"]
    pitcher_stats['clutch'] = np.where(
        pitcher_stats['li'] == 0,
        np.nan,
        (pitcher_stats['pwpa'] / pitcher_stats['li']) - pitcher_stats['pwpa/li']
    )

    team_stats = (pbp_df.groupby(['pitch_team']).agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa/li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('pwpa/li', ascending=False)
    .reset_index(drop=True))

    team_stats['clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['pwpa'] / team_stats['li']) - team_stats['pwpa/li']
    )

    return pitcher_stats, team_stats


def calculate_batting_war(batting_df, guts_df, park_factors_df, pbp_df, rosters_df, division, year):
    if batting_df.empty:
        return batting_df, pd.DataFrame()

    weights = guts_df.iloc[0]

    df = batting_df.copy()
    df['pos'] = df['pos'].apply(lambda x: '' if pd.isna(x) else str(x).split('/')[0].upper())

    gdp_stats = calculate_wgdp(pbp_df)
    teb_stats = calculate_extra_bases(pbp_df, rosters_df, weights)

    df = df.merge(gdp_stats, left_on='player_id', right_index=True, how='left')
    df = df.merge(teb_stats, left_on='player_name', right_index=True, how='left')

    df[['wgdp', 'gdp_opps', 'gdp']] = df[['wgdp', 'gdp_opps', 'gdp']].fillna(0)

    fill_cols = ['hr','r','gp','gs','2b','3b','h','cs','bb','k','sb','ibb','rbi','picked','sh','ab','hbp','sf']
    df[fill_cols] = df[fill_cols].fillna(0)

    df['gp'] = pd.to_numeric(df['gp'], errors='coerce').fillna(0).astype(int)
    df['gs'] = pd.to_numeric(df['gs'], errors='coerce').fillna(0).astype(int)
    df = df[df['ab'] > 0]

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])
    df['1b'] = df['h'] - df['hr'] - df['3b'] - df['2b']
    df['pa'] = df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf']

    # Rates
    df['sb%'] = (df['sb'] / (df['sb'] + df['cs'])).replace([np.inf,-np.inf],0)*100
    df['bb%'] = (df['bb'] / df['pa']).replace([np.inf,-np.inf],0)*100
    df['k%']  = (df['k']  / df['pa']).replace([np.inf,-np.inf],0)*100
    df['ba']  = df['h'] / df['ab']
    df['slg_pct'] = (df['1b'] + 2*df['2b'] + 3*df['3b'] + 4*df['hr']) / df['ab']

    df['ob_pct']  = (df['h'] + df['bb'] + df['hbp'] + df['ibb']) / (df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf'])
    df['iso'] = df['slg_pct'] - df['ba']
    tb = df['slg_pct'] * df['ab']
    df['runs_created'] = tb * (df['h'] + df['bb']) / (df['ab'] + df['bb'])
    
    lg_obp = (df['h'].sum() + df['bb'].sum() + df['hbp'].sum()) / (df['ab'].sum() + df['bb'].sum() + df['hbp'].sum() + df['sf'].sum())
    lg_slg = (df['1b'].sum() + 2*df['2b'].sum() + 3*df['3b'].sum() + 4*df['hr'].sum()) / df['ab'].sum()
    df['ops_plus'] = 100 * (df['ob_pct'] / lg_obp + df['slg_pct'] / lg_slg - 1)
    df['r/pa'] = df['runs_created'] / df['pa']

    num = (weights['wbb']*df['bb'] + weights['whbp']*df['hbp'] + weights['w1b']*df['1b'] +
           weights['w2b']*df['2b'] + weights['w3b']*df['3b'] + weights['whr']*df['hr'])
    den = df['ab'] + df['bb'] - df['ibb'] + df['sf'] + df['hbp']
    df['woba'] = num / den

    player_stats, team_stats = get_clutch_stats(pbp_df)
    df = df.merge(player_stats.drop(columns=['year','division'], errors='ignore'),
              on='player_id', how='left')


    df['pf'] = df['pf'].fillna(100)
    pf = df['pf'] / 100

    league_woba = weights['woba']
    league_rpa  = (df['r'].sum() / df['pa'].sum())
    pa = df['pa']
    woba = df['woba']
    woba_scale = weights['woba_scale']
    rpw = weights['runs_win']
    runs_per_out = weights['runs_out']
    runCS = -1 * (2 * runs_per_out + 0.075)
    runSB = 0.2

    # Conference run environment
    conf_wrc = {}
    for conf in df['conference'].unique():
        conf_df = df[df['conference'] == conf]
        if len(conf_df) > 0:
            conf_wrc[conf] = conf_df['r'].sum() / max(conf_df['pa'].sum(), 1e-12)

    def calculate_batting_runs(row):
        pf_local = row['pf'] / 100
        conf_rpa = conf_wrc.get(row['conference'], league_rpa)
        return (row['wraa'] +
                (league_rpa - (pf_local * league_rpa)) * row['pa'] +
                (league_rpa - conf_rpa) * row['pa'])

    df['wrc']  = (((woba - league_woba) / woba_scale) + league_rpa) * pa
    df['wraa'] = ((woba - league_woba) / woba_scale) * pa
    df['batting_runs'] = df.apply(calculate_batting_runs, axis=1)

    lgwSB = ((df['sb'].sum() * runSB + df['cs'].sum() * runCS) /
             max((df['1b'].sum() + df['bb'].sum() + df['hbp'].sum() - df['ibb'].sum()), 1e-12))
    df['wsb'] = (df['sb'] * runSB + df['cs'] * runCS - lgwSB * (df['1b'] + df['bb'] + df['hbp'] - df['ibb']))

    # Replacement allocation
    team_count = max(len(df['team_name'].unique()), 1)
    games_played = (df['gs'].sum() / 9) / team_count
    replacement_constant = ((team_count / 2) * games_played - (team_count * games_played * 0.294))
    # Scale replacement to players by PA share
    df['replacement_level_runs'] = (replacement_constant * rpw) * (df['pa'] / max(pa.sum(), 1e-12))

    df['baserunning'] = (df['wsb'].fillna(0) + df['wgdp'].fillna(0) + df['wteb'].fillna(0))

    base_adj = df['pos'].map(position_adjustments).fillna(0)
    df['adjustment'] = base_adj * (df['gp'] / (40 if division == 3 else 50))

    conf_adjustments = {}
    for conf in df['conference'].unique():
        conf_df = df[df['conference'] == conf]
        if len(conf_df) > 0:
            lg_batting = conf_df['batting_runs'].sum()
            lg_baserun = conf_df['wsb'].sum()
            lg_pos = conf_df['adjustment'].sum()
            lg_pa = conf_df['pa'].sum()
            if lg_pa > 0:
                conf_adjustments[conf] = (-1 * (lg_batting + lg_baserun + lg_pos) / lg_pa)

    df['league_adjustment'] = df.apply(lambda x: conf_adjustments.get(x['conference'], 0) * x['pa'], axis=1)

    df['war'] = ((df['batting_runs'] + df['replacement_level_runs'] +
                  df['baserunning'] + df['adjustment'] + df['league_adjustment']) / rpw)

    wraa_pa = df['wraa'] / df['pa']
    league_wrcpa = (df['wrc'].sum() / max(df['pa'].sum(), 1e-12))
    df['wrc_plus'] = (((wraa_pa + league_rpa) + (league_rpa - pf * league_rpa)) / league_wrcpa) * 100

    df = df.rename(columns={'sh': 'sac', 'batting_runs': 'batting'})
    df['year'] = year
    df['division'] = division
    df[['player_name', 'team_name', 'conference', 'class']] = df[['player_name','team_name','conference','class']].fillna('-')
    df = df.fillna(0)

    batting_pre_cols = [c for c in batting_columns if c in df.columns and c != "sos_adj_war"]
    df = df[batting_pre_cols]

    return df.dropna(subset=['war']), team_stats

def calculate_pitching_war(pitching_df, pbp_df, park_factors_df, bat_war_total, year, division):
    if pitching_df.empty:
        return pitching_df, pd.DataFrame()

    df = pitching_df[pitching_df['app'] > 0].copy()
    df = df[df['era'].notna()]

    fill_cols = ['hr_a','fo','ip','bb','so','sv','gs','hbp','bf','h','r']
    df[fill_cols] = df[fill_cols].fillna(0)

    def safe_per_nine(numerator, ip):
        return np.where(ip > 0, (numerator / ip) * 9, 0)

    def safe_percentage(numerator, denominator):
        return np.where(denominator > 0, (numerator / denominator) * 100, 0)

    def ip_to_real(innings):
        ip_str = str(innings)
        if '.' in ip_str:
            whole_innings, partial = ip_str.split('.')
            whole_innings = int(whole_innings); partial = int(partial)
        else:
            return float(ip_str)
        if partial == 0: decimal_part = 0
        elif partial == 1: decimal_part = 1/3
        elif partial == 2: decimal_part = 2/3
        else: raise ValueError("Invalid partial inning value. Should be 0, 1, or 2.")
        return whole_innings + decimal_part

    df['ip_float'] = df.ip.apply(ip_to_real)

    df['ra9'] = safe_per_nine(df['r'], df['ip_float'])
    df['k9'] = safe_per_nine(df['so'], df['ip_float'])
    df['h9'] = safe_per_nine(df['h'], df['ip_float'])
    df['bb9'] = safe_per_nine(df['bb'], df['ip_float'])
    df['hr9'] = safe_per_nine(df['hr_a'], df['ip_float'])

    df['bb%'] = safe_percentage(df['bb'], df['bf'])
    df['k%']  = safe_percentage(df['so'], df['bf'])
    df['k-bb%'] = df['k%'] - df['bb%']

    fb_total = df['hr_a'] + df['fo']
    df['hr/fb'] = safe_percentage(df['hr_a'], fb_total)
    df['ir_a%'] = safe_percentage(df['inh_run_score'], df['inh_run'])

    valid_ip_mask = df['ip'] > 0
    df.loc[~valid_ip_mask, ['fip','xfip']] = np.nan

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])

    lg_era = (df.loc[valid_ip_mask, 'er'].sum() / max(df.loc[valid_ip_mask, 'ip_float'].sum(), 1e-12)) * 9
    df.loc[valid_ip_mask, 'era+'] = 100 * (2 - (df.loc[valid_ip_mask, 'era'] / lg_era) * (1 / (df.loc[valid_ip_mask, 'pf'] / 100)))
    df.loc[~valid_ip_mask, 'era+'] = np.nan

    fip_components = ((13 * df['hr_a'].sum() + 3 * (df['bb'].sum() + df['hbp'].sum()) - 2 * df['so'].sum())
                      / max(df['ip_float'].sum(), 1e-12))
    f_constant = lg_era - fip_components
    fip  = f_constant + ((13 * df['hr_a'] + 3 * (df['bb'] + df['hbp']) - 2 * df['so']) / df['ip_float'])
    lg_hr_fb_rate = df['hr_a'].sum() / max((df['hr_a'].sum() + df['fo'].sum()), 1e-12)
    xfip = f_constant + ((13 * ((df['fo'] + df['hr_a']) * lg_hr_fb_rate) + 3 * (df['bb'] + df['hbp']) - 2 * df['so']) / df['ip_float'])

    df['fip'] = fip
    df['xfip'] = xfip

    # gmLI for relievers    
    first_li_per_game = (
        pbp_df.sort_values(['pitcher_id', 'contest_id', 'play_id'])
        .groupby(['pitcher_id', 'contest_id'])
        .apply(lambda x: pd.Series({
            'li': x['li'].iloc[0] if x['inning'].iloc[0] != 1 else np.nan,
            'inning': x['inning'].iloc[0]
        }), include_groups=False)
        .reset_index()
    )
    gmli = (first_li_per_game.groupby('pitcher_id').agg({'li': 'mean'})
            .reset_index().rename(columns={'li': 'gmli'}))
    df = df.merge(gmli, how='left', left_on='player_id', right_on='pitcher_id')
    df['gmli'] = df['gmli'].fillna(0.0)
    
    def calculate_if_fip_constant(group_df):
        group_df = group_df[group_df['ip'] > 0]
        if len(group_df) == 0:
            return np.nan
        lg_ip = group_df['ip_float'].sum()
        lg_hr = group_df['hr_a'].sum()
        lg_bb = group_df['bb'].sum()
        lg_hbp = group_df['hbp'].sum()
        lg_k  = group_df['so'].sum()
        lg_era = (group_df['er'].sum() / max(lg_ip,1e-12)) * 9
        numerator = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_k))
        return lg_era - (numerator / max(lg_ip,1e-12))

    def calculate_player_if_fip(row, constant):
        if row['ip'] == 0:
            return np.nan
        numerator = ((13 * row['hr_a']) + (3 * (row['bb'] + row['hbp'])) - (2 * row['so']))
        return (numerator / row['ip_float']) + constant

    valid_ip_mask = df['ip'] > 0
    if_consts = (df.loc[valid_ip_mask]
                 .groupby('conference', group_keys=False)
                 .apply(calculate_if_fip_constant, include_groups=False)
                 .reset_index())
    if_consts.columns = ['conference', 'if_fip_constant']
    df = df.merge(if_consts, on='conference', how='left')

    df['iffip'] = df.apply(lambda row: calculate_player_if_fip(row, row['if_fip_constant']), axis=1)

    valid_df = df[valid_ip_mask]
    if len(valid_df) > 0:
        lgRA9 = (valid_df['r'].sum() / max(valid_df['ip_float'].sum(), 1e-12)) * 9
        lgERA = (valid_df['er'].sum() / max(valid_df['ip_float'].sum(), 1e-12)) * 9
        adjustment = lgRA9 - lgERA
    else:
        adjustment = 0.0

    df['fipr9'] = np.where(valid_ip_mask, df['iffip'] + adjustment, np.nan)
    df['pf'] = df['pf'].fillna(100)
    df['pfipr9'] = np.where(valid_ip_mask, df['fipr9'] / (df['pf'] / 100), np.nan)

    def calculate_league_adjustments(group_df):
        valid_group = group_df[group_df['ip'] > 0]
        if len(valid_group) == 0:
            return np.nan
        lg_ip = valid_group['ip_float'].sum()
        lg_hr = valid_group['hr_a'].sum()
        lg_bb = valid_group['bb'].sum()
        lg_hbp = valid_group['hbp'].sum()
        lg_k  = valid_group['so'].sum()
        lg_ifFIP = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_k)) / max(lg_ip,1e-12) + valid_group['if_fip_constant'].iloc[0]
        lgRA9 = (valid_group['r'].sum() / max(lg_ip,1e-12)) * 9
        lgERA = (valid_group['er'].sum() / max(lg_ip,1e-12)) * 9
        adjustment = lgRA9 - lgERA
        return lg_ifFIP + adjustment

    league_adjustments = (
        df.loc[valid_ip_mask]
        .groupby('conference', group_keys=False)
        .apply(calculate_league_adjustments, include_groups=False)
        .reset_index()
    )
    league_adjustments.columns = ['conference', 'conf_fipr9']
    df = df.merge(league_adjustments, on='conference', how='left')

    df['raap9'] = np.where(valid_ip_mask, df['conf_fipr9'] - df['pfipr9'], 0)
    df['ip/g']  = np.where(df['app'] > 0, df['ip_float'] / df['app'], 0)

    df['drpw'] = np.where(
        valid_ip_mask,
        (((18 - df['ip/g']) * df['conf_fipr9'] + df['ip/g'] * df['pfipr9']) / 18 + 2) * 1.5,
        0
    )
    df['wpgaa'] = np.where(valid_ip_mask, df['raap9'] / df['drpw'], 0)

    df['gs/g'] = np.where(df['app'] > 0, df['gs'] / df['app'], 0)
    df['replacement_level'] = 0.03 * (1 - df['gs/g']) + 0.12 * df['gs/g']
    df['wpgar'] = np.where(valid_ip_mask, df['wpgaa'] + df['replacement_level'], 0)
    df['war']   = np.where(valid_ip_mask, df['wpgar'] * (df['ip_float'] / 9), 0)

    # Relief leverage bump
    relief_mask = df['gs'] < 3
    df.loc[relief_mask & valid_ip_mask, 'war'] *= (1 + df.loc[relief_mask & valid_ip_mask, 'gmli']) / 2

    total_pitching_war = df['war'].sum()
    target_pitching_war = (bat_war_total * 0.43) / 0.57  # ~43% share pitching
    ip_sum = max(df.loc[valid_ip_mask, 'ip_float'].sum(), 1e-12)
    war_adjustment = (target_pitching_war - total_pitching_war) / ip_sum
    df.loc[valid_ip_mask, 'war'] += war_adjustment * df.loc[valid_ip_mask, 'ip_float']

    df = df.replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0)
    df[['player_name', 'team_name', 'conference', 'class']] = df[['player_name', 'team_name', 'conference', 'class']].fillna('-')

    pitcher_stats, team_stats = get_pitcher_clutch_stats(pbp_df)
    df = df.merge(pitcher_stats.drop(columns=['year','division'], errors='ignore'),
              left_on='player_id', right_on='pitcher_id', how='left')


    pitching_pre_cols = [c for c in pitching_columns if c in df.columns and c != "sos_adj_war"]
    df = df[pitching_pre_cols]

    return df.dropna(subset=['war']), team_stats

def calculate_pitching_team_war(pitching_df, park_factors_df, team_clutch):
    df = pitching_df.copy()
    numeric_cols = [
        'app','ip_float','l','sv','gs','h','w','2b_a','3b_a','inh_run','inh_run_score',
        'hr_a','r','er','war','fo','bb','hbp','so','bf','pitches'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    agg_dict = dict.fromkeys(numeric_cols, 'sum')
    agg_dict.update({'conference': 'first', 'year': 'first', 'division': 'first'})
    df = df.groupby('team_name').agg(agg_dict).reset_index()

    def convert_to_baseball_ip(decimal_ip):
        whole_innings = int(decimal_ip)
        frac = decimal_ip - whole_innings
        if frac < 0.001: partial = 0
        elif abs(frac - 1/3) < 0.001: partial = 1
        elif abs(frac - 2/3) < 0.001: partial = 2
        else:
            if frac < 1/6: partial = 0
            elif frac < 0.5: partial = 1
            elif frac < 5/6: partial = 2
            else:
                whole_innings += 1
                partial = 0
        return float(f"{whole_innings}.{partial}")

    df['ip'] = df.ip_float.apply(convert_to_baseball_ip)
    df['era'] = (df['er'] * 9) / df['ip_float']
    df['ir_a%'] = ((df['inh_run_score'] / df['inh_run']) * 100).fillna(0)
    df['ra9'] = (df['r'] / df['ip_float']) * 9
    df['k9']  = (df['so'] / df['ip_float']) * 9
    df['h9']  = (df['h'] / df['ip_float']) * 9
    df['bb9'] = (df['bb'] / df['ip_float']) * 9
    df['hr9'] = (df['hr_a'] / df['ip_float']) * 9
    df['bb%'] = (df['bb'] / df['bf']) * 100
    df['k%']  = (df['so'] / df['bf']) * 100
    df['k-bb%'] = df['k%'] - df['bb%']
    df['hr/fb'] = (df['hr_a'] / (df['hr_a'] + df['fo'])).replace([np.inf,-np.inf], np.nan) * 100

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])
    df['era+'] = 100 * (2 - (df.era / ((df.er.sum() / max(df.ip.sum(),1e-12)) * 9)) * (1 / (df.pf / 100)))

    df = df.merge(team_clutch, left_on='team_name', right_on='pitch_team', how='left')
    return df

def calculate_batting_team_war(batting_df, guts_df, park_factors_df, team_clutch):
    df = batting_df.copy()
    df = (df.groupby('team_name').agg({
        'gp': 'max', 'conference': 'first', 'ab': 'sum', 'bb': 'sum', 'ibb': 'sum', 'sf': 'sum',
        'hbp': 'sum', 'pa': 'sum', 'h': 'sum', '2b': 'sum', '3b': 'sum', 'hr': 'sum', 'r': 'sum',
        'sb': 'sum', 'picked': 'sum', 'sac': 'sum', 'wrc': 'sum', 'batting': 'sum',
        'baserunning': 'sum', 'adjustment': 'sum', 'war': 'sum',
        'k': 'sum', 'cs': 'sum', 'rbi': 'sum', 'gs': 'max', 'year': 'first', 'division': 'first',
        'sos_adj_war': 'sum'
    }).reset_index())

    weights = guts_df.iloc[0]

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])
    df['1b'] = df['h'] - df['hr'] - df['3b'] - df['2b']
    df['pa'] = df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf']
    df['sb%'] = (df['sb'] / (df['sb'] + df['cs'])).replace([np.inf,-np.inf],0) * 100
    df['bb%'] = (df['bb'] / df['pa']).replace([np.inf,-np.inf],0) * 100
    df['k%']  = (df['k'] / df['pa']).replace([np.inf,-np.inf],0) * 100
    df['ba']  = df['h'] / df['ab']
    df['slg_pct'] = (df['1b'] + 2*df['2b'] + 3*df['3b'] + 4*df['hr']) / df['ab']
    df['ob_pct']  = (df['h'] + df['bb'] + df['hbp'] + df['ibb']) / (df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf'])
    df['iso'] = df['slg_pct'] - df['ba']

    lg_obp = (df['h'].sum() + df['bb'].sum() + df['hbp'].sum()) / (df['ab'].sum() + df['bb'].sum() + df['hbp'].sum() + df['sf'].sum())
    lg_slg = (df['1b'].sum() + 2*df['2b'].sum() + 3*df['3b'].sum() + 4*df['hr'].sum()) / df['ab'].sum()
    df['ops_plus'] = 100 * (df['ob_pct'] / lg_obp + df['slg_pct'] / lg_slg - 1)
    # TB x (H + BB) / (AB + BB)
    tb = df['slg_pct'] * df['ab']
    df['runs_created'] = tb * (df['h'] + df['bb']) / (df['ab'] + df['bb'])
    
    df['r/pa'] = df['runs_created'] / df['pa']

    num = (weights['wbb']*df['bb'] + weights['whbp']*df['hbp'] + weights['w1b']*df['1b'] +
           weights['w2b']*df['2b'] + weights['w3b']*df['3b'] + weights['whr']*df['hr'])
    den = df['ab'] + df['bb'] - df['ibb'] + df['sf'] + df['hbp']
    df['woba'] = num / den

    df = df.merge(team_clutch, left_on='team_name', right_on='bat_team', how='left')
    return df

def get_data(year, data_dir, divisions=None):
    if divisions is None:
        divisions = [1, 2, 3]
    pitching, batting, pbp, rosters, guts, park_factors, rankings = {}, {}, {}, {}, {}, {}, {}

    for division in divisions:
        pitch_raw = pd.read_csv(data_dir / f'stats/d{division}_pitching_{year}.csv')
        bat_raw   = pd.read_csv(data_dir / f'stats/d{division}_batting_{year}.csv')

        ros = (
            pd.read_csv(
                data_dir / f'rosters/d{division}_rosters_{year}.csv',
                dtype={'player_id': str, 'ncaa_id': str}
            )
            .query(f'year == {year}')
            .query(f'division == {division}')
        )

        bat_raw['ncaa_id']   = bat_raw['ncaa_id'].astype(str)
        pitch_raw['ncaa_id'] = pitch_raw['ncaa_id'].astype(str)

        bat   = bat_raw.merge(ros[['ncaa_id', 'player_id']], on='ncaa_id', how='left')
        pitch = pitch_raw.merge(ros[['ncaa_id', 'player_id']], on='ncaa_id', how='left')

        batting[division]  = bat
        pitching[division] = pitch
        rosters[division]  = ros

        pbp[division] = pd.read_csv(
            data_dir / f'pbp/d{division}_parsed_pbp_new_{year}.csv',
            dtype={'player_id': str, 'pitcher_id': str}, low_memory=False
        )

        park_factors[division] = pd.read_csv(data_dir / f'park_factors/d{division}_park_factors.csv')

        g = pd.read_csv(data_dir / 'guts/guts_constants.csv')
        guts[division] = g[(g['division'] == division) & (g['year'] == int(year))]

        r = pd.read_csv(data_dir / f'rankings/d{division}_rankings_{year}.csv')
        r['year']      = year
        r['division']  = division
        r['wins']      = r['record'].str.split('-').str[0].astype(int)
        r['losses']    = r['record'].str.split('-').str[1].astype(int)
        r['ties']      = r['record'].str.split('-').str[2].fillna(0).astype(int)
        r['games']     = r['wins'] + r['losses'] + r['ties']
        rankings[division] = r

    mappings = pd.read_csv(data_dir / 'team_mappings.csv')
    return batting, pitching, pbp, guts, park_factors, rosters, rankings, mappings

def calculate_war(data_dir, year, divisions=None):
    if divisions is None:
        divisions = [1, 2, 3]
    batting, pitching, pbp, guts, park_factors, rosters, rankings, mappings = get_data(year, data_dir, divisions)

    war_dir = Path(data_dir) / 'war'
    war_dir.mkdir(exist_ok=True)

    for division in divisions:
        print(f"Processing Division {division}, Year {year}")

        batting_df       = batting[division]
        pitching_df      = pitching[division]
        pbp_df           = pbp[division]
        guts_df          = guts[division]
        park_factors_df  = park_factors[division]
        rosters_df       = rosters[division]
        rankings_df      = rankings[division]

        # Player WAR
        batting_war, team_batting_clutch   = calculate_batting_war(batting_df, guts_df, park_factors_df, pbp_df, rosters_df, division, year)
        pitching_war, team_pitching_clutch = calculate_pitching_war(pitching_df, pbp_df, park_factors_df, batting_war.war.sum(), year, division)

        batting_war, pitching_war, missing = sos_reward_punish_players(
            batting_war, pitching_war, rankings_df, mappings, division, year,
            alpha=0.2, clip_sd=3, group_keys=('year', 'division'), harder_if='higher'
        )
        if missing:
            print(f"[d{division} {year}] SoS missing -> filled with min SoS; unique teams affected: {len(missing)}")

        # Normalize totals vs standings (WAR -> wins above replacement share)
        batting_war, pitching_war = normalize_division_war(batting_war, pitching_war, rankings_df, division, year)

        batting_war = batting_war[[c for c in batting_columns if c in batting_war.columns]]
        pitching_war = pitching_war[[c for c in pitching_columns if c in pitching_war.columns]]

        # Team summaries
        batting_team_war = calculate_batting_team_war(batting_war, guts_df, park_factors_df, team_batting_clutch)
        pitching_team_war = calculate_pitching_team_war(pitching_war, park_factors_df, team_pitching_clutch)

        batting_team_war = batting_team_war[[c for c in batting_columns if c in batting_team_war.columns]]
        pitching_team_war = pitching_team_war[[c for c in pitching_columns if c in pitching_team_war.columns]]

        # Output
        (war_dir / f'd{division}_pitching_team_war_{year}.csv').write_bytes(pitching_team_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_batting_team_war_{year}.csv').write_bytes(batting_team_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_pitching_war_{year}.csv').write_bytes(pitching_war.to_csv(index=False).encode('utf-8'))
        (war_dir / f'd{division}_batting_war_{year}.csv').write_bytes(batting_war.to_csv(index=False).encode('utf-8'))

        # Quick sanity log
        rep_wp = 0.294
        s = rankings_df[(rankings_df['division']==division) & (rankings_df['year']==year)]
        target_total = s['wins'].sum() - rep_wp*s['games'].sum()
        current_total = batting_war['war'].sum() + pitching_war['war'].sum()
        print(f"[d{division} {year}] Target WAR≈{target_total:.2f} | Actual WAR≈{current_total:.2f}")

def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    calculate_war(data_dir, year, divisions)
    print("Successfully processed all statistics!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Root directory containing the data folders')
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()
    main(args.data_dir, args.year, args.divisions)
