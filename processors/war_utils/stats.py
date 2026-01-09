import pandas as pd
import numpy as np
from war_utils.constants import (
    position_adjustments, batting_columns, pitching_columns,
    batting_agg_dict, pitching_agg_dict
)

ID_COLUMNS = ['player_id', 'pitcher_id', 'batter_id', 'ncaa_id', 'team_id', 'org_id', 'contest_id']

def normalize_id_columns(df: pd.DataFrame, id_cols: list = None) -> pd.DataFrame:
    if id_cols is None:
        id_cols = [c for c in ID_COLUMNS if c in df.columns]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'0': '-', '0.0': '-', 'nan': '-', 'None': '-', '': '-'})
    return df

def build_ncaa_id_to_player_id_map(roster_df):
    roster_df = roster_df.copy()
    roster_df['ncaa_id'] = roster_df['ncaa_id'].astype(str).replace({'0': '-', '0.0': '-', 'nan': '-', 'None': '-', '': '-'})
    return roster_df.set_index('ncaa_id')['player_id'].to_dict()

def safe_per_nine(numerator, ip):
    return np.where(ip > 0, (numerator / ip) * 9, 0)

def safe_percentage(numerator, denominator):
    return np.where(denominator > 0, (numerator / denominator) * 100, 0)

def ip_to_real(innings):
    ip_str = str(innings)
    if '.' in ip_str:
        whole_innings, partial = ip_str.split('.')
        whole_innings = int(whole_innings)
        partial = int(partial)
    else:
        return float(ip_str)
    if partial == 0:
        decimal_part = 0
    elif partial == 1:
        decimal_part = 1/3
    elif partial == 2:
        decimal_part = 2/3
    else:
        raise ValueError("Invalid partial inning value. Should be 0, 1, or 2.")
    return whole_innings + decimal_part

def convert_to_baseball_ip(decimal_ip):
    whole_innings = int(decimal_ip)
    frac = decimal_ip - whole_innings
    if frac < 0.001:
        partial = 0
    elif abs(frac - 1/3) < 0.001:
        partial = 1
    elif abs(frac - 2/3) < 0.001:
        partial = 2
    else:
        if frac < 1/6:
            partial = 0
        elif frac < 0.5:
            partial = 1
        elif frac < 5/6:
            partial = 2
        else:
            whole_innings += 1
            partial = 0
    return float(f"{whole_innings}.{partial}")

def calculate_sb_pct(df: pd.DataFrame) -> pd.Series:
    return (df['sb'] / (df['sb'] + df['cs'])).replace([np.inf, -np.inf], 0) * 100

def calculate_bb_pct(df: pd.DataFrame) -> pd.Series:
    return (df['bb'] / df['pa']).replace([np.inf, -np.inf], 0) * 100

def calculate_k_pct(df: pd.DataFrame) -> pd.Series:
    return (df['k'] / df['pa']).replace([np.inf, -np.inf], 0) * 100

def calculate_ba(df: pd.DataFrame) -> pd.Series:
    return df['h'] / df['ab']

def calculate_slg_pct(df: pd.DataFrame) -> pd.Series:
    return (df['1b'] + 2*df['2b'] + 3*df['3b'] + 4*df['hr']) / df['ab']

def calculate_babip(df: pd.DataFrame) -> pd.Series:
    return (df['h'] - df['hr']) / (df['ab'] - df['hr'] - df['k'] + df['sf'])

def calculate_ob_pct(df: pd.DataFrame) -> pd.Series:
    return (df['h'] + df['bb'] + df['hbp'] + df['ibb']) / (df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf'])

def calculate_iso(df: pd.DataFrame) -> pd.Series:
    return df['slg_pct'] - df['ba']

def calculate_runs_created(df: pd.DataFrame) -> pd.Series:
    tb = df['slg_pct'] * df['ab']
    return tb * (df['h'] + df['bb']) / (df['ab'] + df['bb'])

def calculate_ops_plus(df: pd.DataFrame) -> pd.Series:
    lg_obp = (df['h'].sum() + df['bb'].sum() + df['hbp'].sum()) / (df['ab'].sum() + df['bb'].sum() + df['hbp'].sum() + df['sf'].sum())
    lg_slg = (df['1b'].sum() + 2*df['2b'].sum() + 3*df['3b'].sum() + 4*df['hr'].sum()) / df['ab'].sum()
    return 100 * (df['ob_pct'] / lg_obp + df['slg_pct'] / lg_slg - 1)

def calculate_rc_per_pa(df: pd.DataFrame) -> pd.Series:
    return df['runs_created'] / df['pa']

def calculate_woba_from_components(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    num = (weights['wbb']*df['bb'] + weights['whbp']*df['hbp'] + weights['w1b']*df['1b'] +
           weights['w2b']*df['2b'] + weights['w3b']*df['3b'] + weights['whr']*df['hr'])
    den = df['ab'] + df['bb'] - df['ibb'] + df['sf'] + df['hbp']
    return num / den

def calculate_wrc_from_woba(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    woba = df['woba']
    woba_scale = weights['woba_scale']
    league_woba = weights['woba']
    league_rpa = (df['r'].sum() / max(df['pa'].sum(), 1e-12))
    pa = df['pa']
    return (((woba - league_woba) / woba_scale) + league_rpa) * pa

def calculate_wraa_from_woba(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    woba = df['woba']
    woba_scale = weights['woba_scale']
    league_woba = weights['woba']
    pa = df['pa']
    return ((woba - league_woba) / woba_scale) * pa

def calculate_wsb(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    runSB = 0.2
    runs_per_out = weights['runs_out']
    runCS = -1 * (2 * runs_per_out + 0.075)
    lgwSB = ((df['sb'].sum() * runSB + df['cs'].sum() * runCS) /
             max((df['1b'].sum() + df['bb'].sum() + df['hbp'].sum() - df['ibb'].sum()), 1e-12))
    return (df['sb'] * runSB + df['cs'] * runCS - lgwSB * (df['1b'] + df['bb'] + df['hbp'] - df['ibb']))

def calculate_wrc_plus(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    wraa_pa = df['wraa'] / df['pa']
    league_rpa = (df['r'].sum() / max(df['pa'].sum(), 1e-12))
    league_wrcpa = (df['wrc'].sum() / max(df['pa'].sum(), 1e-12))
    pf = df['pf'] / 100
    return (((wraa_pa + league_rpa) + (league_rpa - pf * league_rpa)) / league_wrcpa) * 100

def conference_wrc(df: pd.DataFrame) -> dict:
    conf_wrc = {}
    for conf in df['conference'].unique():
        conf_df = df[df['conference'] == conf]
        if len(conf_df) > 0:
            conf_wrc[conf] = conf_df['r'].sum() / max(conf_df['pa'].sum(), 1e-12)
    return conf_wrc

def calculate_batting_runs(row, league_rpa, conf_wrc):
    pf_local = row['pf'] / 100
    conf_rpa = conf_wrc.get(row['conference'], league_rpa)
    return (row['wraa'] +
            (league_rpa - (pf_local * league_rpa)) * row['pa'] +
            (league_rpa - conf_rpa) * row['pa'])

def calculate_batting_replacement_level(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    team_count = max(len(df['team_name'].unique()), 1)
    games_played = (df['gs'].sum() / 9) / team_count
    rpw = weights['runs_win']
    replacement_constant = ((team_count / 2) * games_played - (team_count * games_played * 0.294))
    return (replacement_constant * rpw) * (df['pa'] / max(df['pa'].sum(), 1e-12))

def calculate_baserunning(df: pd.DataFrame) -> pd.Series:
    return (df['wsb'].fillna(0) + df['wgdp'].fillna(0) + df['wteb'].fillna(0))

def calculate_adjustment(row: pd.Series) -> float:
    pos = row.get('pos', '')
    base_adj = position_adjustments.get(pos, 0)
    division = row.get('division', 3)
    return base_adj * (row['gp'] / (40 if division == 3 else 50))

def calculate_league_adjustments(df: pd.DataFrame) -> pd.Series:
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
    return df.apply(lambda x: conf_adjustments.get(x['conference'], 0) * x['pa'], axis=1)

def calculate_wgdp(pbp_df):
    gdp_opps = pbp_df[(pbp_df['r1_id'].notna()) & (pbp_df['r1_id'] != '') & (pbp_df['outs_before'].astype(int) < 2)].copy()
    gdp_events = gdp_opps[gdp_opps['description'].str.contains('double play', case=False, na=False)]
    
    gdp_opps = gdp_opps[gdp_opps['batter_id'].notna() & (gdp_opps['batter_id'] != '')]
    gdp_events = gdp_events[gdp_events['batter_id'].notna() & (gdp_events['batter_id'] != '')]
    
    gdp_stats = pd.DataFrame({
        'gdp_opps': gdp_opps.groupby('batter_id').size(),
        'gdp': gdp_events.groupby('batter_id').size()
    }).fillna(0)
    
    lg_gdp_rate = gdp_stats['gdp'].sum() / max(gdp_stats['gdp_opps'].sum(), 1e-12)
    gdp_run_value = 0.5
    gdp_stats['wgdp'] = (gdp_stats['gdp_opps'] * lg_gdp_rate - gdp_stats['gdp']) * gdp_run_value
    return gdp_stats

def calculate_baserunning_components(df, weights):
    extra_bases, ebt_opps, outs_on_bases = {}, {}, {}

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

    def valid_id(val):
        if pd.notna(val):
            s = str(val).strip()
            return s if s and s not in ['', 'nan', 'None'] else None
        return None

    singles = df[df['event_cd'] == 20]
    for idx, play in singles.iterrows():
        nxt = next_play_row(idx, df)

        s1 = valid_id(play.get('r1_id'))
        if s1:
            ebt_opps[s1] = ebt_opps.get(s1, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s1 not in next_ids:
                    outs_on_bases[s1] = outs_on_bases.get(s1, 0) + 1
                elif s1 == valid_id(nxt.get('r3_id')):
                    extra_bases[s1] = extra_bases.get(s1, 0) + 1

        s2 = valid_id(play.get('r2_id'))
        if s2:
            ebt_opps[s2] = ebt_opps.get(s2, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s2 not in next_ids:
                    outs_on_bases[s2] = outs_on_bases.get(s2, 0) + 1
                elif (s2 not in next_ids) and (play.runs_on_play > 0):
                    extra_bases[s2] = extra_bases.get(s2, 0) + 1

        s3 = valid_id(play.get('r3_id'))
        if s3:
            ebt_opps[s3] = ebt_opps.get(s3, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s3 not in next_ids:
                    outs_on_bases[s3] = outs_on_bases.get(s3, 0) + 1
                elif (s3 not in next_ids) and (play.runs_on_play > 0):
                    extra_bases[s3] = extra_bases.get(s3, 0) + 1

    doubles = df[df['event_cd'] == 21]
    for idx, play in doubles.iterrows():
        nxt = next_play_row(idx, df)

        s1 = valid_id(play.get('r1_id'))
        if s1:
            ebt_opps[s1] = ebt_opps.get(s1, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s1 not in next_ids:
                    outs_on_bases[s1] = outs_on_bases.get(s1, 0) + 1
                elif (s1 not in next_ids) and (play.runs_on_play > 0):
                    extra_bases[s1] = extra_bases.get(s1, 0) + 1

        s2 = valid_id(play.get('r2_id'))
        if s2:
            ebt_opps[s2] = ebt_opps.get(s2, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s2 not in next_ids:
                    outs_on_bases[s2] = outs_on_bases.get(s2, 0) + 1
                elif (s2 not in next_ids) and (play.runs_on_play > 0):
                    extra_bases[s2] = extra_bases.get(s2, 0) + 1

        s3 = valid_id(play.get('r3_id'))
        if s3:
            ebt_opps[s3] = ebt_opps.get(s3, 0) + 1
            if nxt is not None:
                next_ids = {valid_id(nxt.get('r1_id')), valid_id(nxt.get('r2_id')), valid_id(nxt.get('r3_id'))} - {None}
                if play.outs_on_play > 0 and s3 not in next_ids:
                    outs_on_bases[s3] = outs_on_bases.get(s3, 0) + 1
                elif (s3 not in next_ids) and (play.runs_on_play > 0):
                    extra_bases[s3] = extra_bases.get(s3, 0) + 1

    results = pd.DataFrame({
        'ebt': pd.Series(extra_bases, dtype='float'),
        'outs_ob': pd.Series(outs_on_bases, dtype='float'),
        'ebt_opps': pd.Series(ebt_opps, dtype='float')
    }).fillna(0.0)

    if results.empty or results['ebt_opps'].sum() == 0:
        results['success_rate'] = 0.0
        results['wteb'] = 0.0
        return results.sort_values('ebt', ascending=False)

    results['success_rate'] = (results['ebt'] / results['ebt_opps']).replace([np.inf, -np.inf], 0.0).round(3)

    lg_teb_rate = results['ebt'].sum() / max(results['ebt_opps'].sum(), 1e-12)
    lg_out_rate = results['outs_ob'].sum() / max(results['ebt_opps'].sum(), 1e-12)

    run_extra_base = 0.3
    runs_per_out = weights['runs_out']
    run_out = -1 * (2 * runs_per_out + 0.075)

    results['wteb'] = (
        (results['ebt'] * run_extra_base) +
        (results['outs_ob'] * run_out) -
        (results['ebt_opps'] * (lg_teb_rate * run_extra_base + lg_out_rate * run_out))
    )

    return results.sort_values('ebt', ascending=False)

def calculate_pitching_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['ra9'] = safe_per_nine(df['r'], df['ip_float'])
    df['k9'] = safe_per_nine(df['so'], df['ip_float'])
    df['h9'] = safe_per_nine(df['h'], df['ip_float'])
    df['bb9'] = safe_per_nine(df['bb'], df['ip_float'])
    df['hr9'] = safe_per_nine(df['hr_a'], df['ip_float'])
    df['bb_pct'] = safe_percentage(df['bb'], df['bf'])
    df['k_pct'] = safe_percentage(df['so'], df['bf'])
    df['k_minus_bb_pct'] = df['k_pct'] - df['bb_pct']
    fb_total = df['hr_a'] + df['fo']
    df['hr_div_fb'] = safe_percentage(df['hr_a'], fb_total)
    df['ir_a_pct'] = safe_percentage(df['inh_run_score'], df['inh_run'])
    return df

def calculate_era_plus(df: pd.DataFrame, valid_ip_mask: pd.Series) -> pd.Series:
    lg_era = (df.loc[valid_ip_mask, 'er'].sum() / max(df.loc[valid_ip_mask, 'ip_float'].sum(), 1e-12)) * 9
    era_plus = pd.Series(np.nan, index=df.index)
    era_plus.loc[valid_ip_mask] = 100 * (2 - (df.loc[valid_ip_mask, 'era'] / lg_era) * (1 / (df.loc[valid_ip_mask, 'pf'] / 100)))
    return era_plus

def calculate_fip(df: pd.DataFrame, lg_era: float) -> pd.Series:
    fip_components = ((13 * df['hr_a'].sum() + 3 * (df['bb'].sum() + df['hbp'].sum()) - 2 * df['so'].sum())
                      / max(df['ip_float'].sum(), 1e-12))
    f_constant = lg_era - fip_components
    return f_constant + ((13 * df['hr_a'] + 3 * (df['bb'] + df['hbp']) - 2 * df['so']) / df['ip_float'])

def calculate_xfip(df: pd.DataFrame, f_constant: float) -> pd.Series:
    lg_hr_fb_rate = df['hr_a'].sum() / max((df['hr_a'].sum() + df['fo'].sum()), 1e-12)
    return f_constant + ((13 * ((df['fo'] + df['hr_a']) * lg_hr_fb_rate) + 3 * (df['bb'] + df['hbp']) - 2 * df['so']) / df['ip_float'])

def calculate_gmli(pbp_df: pd.DataFrame) -> pd.DataFrame:
    first_li_per_game = (
        pbp_df.sort_values(['pitcher_id', 'contest_id', 'play_id'])
        .groupby(['pitcher_id', 'contest_id'])
        .apply(lambda x: pd.Series({
            'li': x['li'].iloc[0] if x['inning'].iloc[0] != 1 else np.nan,
            'inning': x['inning'].iloc[0]
        }), include_groups=False)
        .reset_index()
    )
    return (first_li_per_game.groupby('pitcher_id').agg({'li': 'mean'})
            .reset_index().rename(columns={'li': 'gmli'}))

def calculate_if_fip_constant(group_df):
    group_df = group_df[group_df['ip'] > 0]
    if len(group_df) == 0:
        return np.nan
    lg_ip = group_df['ip_float'].sum()
    lg_hr = group_df['hr_a'].sum()
    lg_bb = group_df['bb'].sum()
    lg_hbp = group_df['hbp'].sum()
    lg_k = group_df['so'].sum()
    lg_era = (group_df['er'].sum() / max(lg_ip, 1e-12)) * 9
    numerator = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_k))
    return lg_era - (numerator / max(lg_ip, 1e-12))

def calculate_player_if_fip(row, constant):
    if row['ip'] == 0:
        return np.nan
    numerator = ((13 * row['hr_a']) + (3 * (row['bb'] + row['hbp'])) - (2 * row['so']))
    return (numerator / row['ip_float']) + constant

def calculate_pitching_conf_fipr9(group_df):
    valid_group = group_df[group_df['ip'] > 0]
    if len(valid_group) == 0:
        return np.nan
    lg_ip = valid_group['ip_float'].sum()
    lg_hr = valid_group['hr_a'].sum()
    lg_bb = valid_group['bb'].sum()
    lg_hbp = valid_group['hbp'].sum()
    lg_k = valid_group['so'].sum()
    lg_ifFIP = ((13 * lg_hr) + (3 * (lg_bb + lg_hbp)) - (2 * lg_k)) / max(lg_ip, 1e-12) + valid_group['if_fip_constant'].iloc[0]
    lgRA9 = (valid_group['r'].sum() / max(lg_ip, 1e-12)) * 9
    lgERA = (valid_group['er'].sum() / max(lg_ip, 1e-12)) * 9
    adjustment = lgRA9 - lgERA
    return lg_ifFIP + adjustment

def calculate_pitching_war_components(df: pd.DataFrame, valid_ip_mask: pd.Series) -> pd.DataFrame:
    df['raap9'] = np.where(valid_ip_mask, df['conf_fipr9'] - df['pfipr9'], 0)
    df['ip/g'] = np.where(df['app'] > 0, df['ip_float'] / df['app'], 0)
    df['drpw'] = np.where(
        valid_ip_mask,
        (((18 - df['ip/g']) * df['conf_fipr9'] + df['ip/g'] * df['pfipr9']) / 18 + 2) * 1.5,
        0
    )
    df['wpgaa'] = np.where(valid_ip_mask, df['raap9'] / df['drpw'], 0)
    df['gs/g'] = np.where(df['app'] > 0, df['gs'] / df['app'], 0)
    df['replacement_level'] = 0.03 * (1 - df['gs/g']) + 0.12 * df['gs/g']
    df['wpgar'] = np.where(valid_ip_mask, df['wpgaa'] + df['replacement_level'], 0)
    df['war'] = np.where(valid_ip_mask, df['wpgar'] * (df['ip_float'] / 9), 0)
    return df

def apply_relief_leverage_bump(df: pd.DataFrame, valid_ip_mask: pd.Series) -> pd.DataFrame:
    relief_mask = df['gs'] < 3
    df.loc[relief_mask & valid_ip_mask, 'war'] *= (1 + df.loc[relief_mask & valid_ip_mask, 'gmli']) / 2
    return df

def normalize_pitching_war_to_batting(df: pd.DataFrame, bat_war_total: float, valid_ip_mask: pd.Series) -> pd.DataFrame:
    total_pitching_war = df['war'].sum()
    target_pitching_war = (bat_war_total * 0.43) / 0.57
    ip_sum = max(df.loc[valid_ip_mask, 'ip_float'].sum(), 1e-12)
    war_adjustment = (target_pitching_war - total_pitching_war) / ip_sum
    df.loc[valid_ip_mask, 'war'] += war_adjustment * df.loc[valid_ip_mask, 'ip_float']
    return df

def get_clutch_stats(pbp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp_df = pbp_df.copy()
    player_stats = (pbp_df.groupby(['player_id']).agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa_li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('wpa_li', ascending=False)
    .reset_index(drop=True))

    player_stats['clutch'] = np.where(
        player_stats['li'] == 0,
        np.nan,
        (player_stats['wpa'] / player_stats['li']) - player_stats['wpa_li']
    )

    team_stats = (pbp_df.groupby('bat_team').agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa_li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('wpa_li', ascending=False)
    .reset_index(drop=True))

    team_stats['clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['wpa'] / team_stats['li']) - team_stats['wpa_li']
    )

    return player_stats, team_stats

def get_pitcher_clutch_stats(pbp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pbp_df = pbp_df.copy()
    pbp_df['prea'] = (-pbp_df['run_expectancy_delta'] - pbp_df['runs_on_play'])
    pbp_df['pwpa'] = np.where(
        pbp_df['pitch_team'] == pbp_df['home_team'],
        pbp_df['delta_home_win_exp'],
        -pbp_df['delta_home_win_exp']
    )
    pbp_df['pwpa_li'] = pbp_df['pwpa'].div(pbp_df['li'].replace(0, float('nan')))

    pitcher_stats = (pbp_df.groupby(['pitcher_id']).agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa_li': 'sum', 'li': 'mean', 'pitcher_name': 'first',
    })
    .reset_index()
    .sort_values('pwpa_li', ascending=False)
    .reset_index(drop=True))

    pitcher_stats = pitcher_stats[pitcher_stats['pitcher_name'] != "Starter"]
    pitcher_stats['clutch'] = np.where(
        pitcher_stats['li'] == 0,
        np.nan,
        (pitcher_stats['pwpa'] / pitcher_stats['li']) - pitcher_stats['pwpa_li']
    )

    team_stats = (pbp_df.groupby(['pitch_team']).agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa_li': 'sum', 'li': 'mean'
    })
    .reset_index()
    .sort_values('pwpa_li', ascending=False)
    .reset_index(drop=True))

    team_stats['clutch'] = np.where(
        team_stats['li'] == 0,
        np.nan,
        (team_stats['pwpa'] / team_stats['li']) - team_stats['pwpa_li']
    )

    return pitcher_stats, team_stats

def calculate_batting_war(batting_df, guts_df, park_factors_df, pbp_df, division, year, group_by='player_id'):
    if batting_df.empty:
        return (batting_df, pd.DataFrame()) if group_by == 'player_id' else batting_df

    weights = guts_df.iloc[0]
    df = normalize_id_columns(batting_df.copy())

    if 'b_t' in df.columns:
        df[['bats', 'throws']] = df['b_t'].str.split('/', n=1, expand=True)
        df['bats'] = df['bats'].fillna('-')
        df['throws'] = df['throws'].fillna('-')

    if group_by == 'player_id':
        df['pos'] = df['pos'].apply(lambda x: '' if pd.isna(x) else str(x).split('/')[0].upper())
        gdp_stats = calculate_wgdp(pbp_df)
        teb_stats = calculate_baserunning_components(pbp_df, weights)
        df = df.merge(gdp_stats, left_on='player_id', right_index=True, how='left')
        df = df.merge(teb_stats, left_on='player_id', right_index=True, how='left')
        df[['wgdp', 'gdp_opps', 'gdp']] = df[['wgdp', 'gdp_opps', 'gdp']].fillna(0)
        player_stats, team_stats = get_clutch_stats(pbp_df)
        df = df.merge(player_stats.drop(columns=['year', 'division'], errors='ignore'),
                      on='player_id', how='left')
    else:
        numeric_cols = [k for k, v in batting_agg_dict.items() if v == 'sum' and k in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        agg = {k: v for k, v in batting_agg_dict.items() if k in df.columns and k != 'batting'}
        df = df.groupby(group_by).agg(agg).reset_index()
        team_stats = pd.DataFrame()

    df['gp'] = pd.to_numeric(df['gp'], errors='coerce').fillna(0).astype(int)
    df['gs'] = pd.to_numeric(df['gs'], errors='coerce').fillna(0).astype(int)
    df = df[df['ab'] > 0]

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])
    df['pf'] = df['pf'].fillna(100)
    df['1b'] = df['h'] - df['hr'] - df['3b'] - df['2b']
    df['pa'] = df['ab'] + df['bb'] + df['ibb'] + df['hbp'] + df['sf']

    df['sb_pct'] = calculate_sb_pct(df)
    df['bb_pct'] = calculate_bb_pct(df)
    df['k_pct'] = calculate_k_pct(df)
    df['ba'] = calculate_ba(df)
    df['slg_pct'] = calculate_slg_pct(df)
    df['babip'] = calculate_babip(df)
    df['ob_pct'] = calculate_ob_pct(df)
    df['iso'] = calculate_iso(df)
    df['runs_created'] = calculate_runs_created(df)
    df['ops_plus'] = calculate_ops_plus(df)
    df['rc_per_pa'] = calculate_rc_per_pa(df)

    df['woba'] = calculate_woba_from_components(df, weights)

    league_rpa = (df['r'].sum() / max(df['pa'].sum(), 1e-12))
    conf_wrc = conference_wrc(df)

    df['wrc'] = calculate_wrc_from_woba(df, weights)
    df['wraa'] = calculate_wraa_from_woba(df, weights)
    df['batting_runs'] = df.apply(calculate_batting_runs, axis=1, args=(league_rpa, conf_wrc))

    df['wsb'] = calculate_wsb(df, weights)

    rpw = weights['runs_win']
    if group_by == 'player_id':
        df['replacement_level_runs'] = calculate_batting_replacement_level(df, weights)
        df['baserunning'] = calculate_baserunning(df)
        df['division'] = division
        df['adjustment'] = df.apply(calculate_adjustment, axis=1)
    else:
        df['replacement_level_runs'] = 0
        df['baserunning'] = df.get('baserunning', 0)
        df['adjustment'] = df.get('adjustment', 0)

    df['league_adjustment'] = calculate_league_adjustments(df)

    df['war'] = ((df['batting_runs'] + df['replacement_level_runs'] +
                  df['baserunning'] + df['adjustment'] + df['league_adjustment']) / rpw)

    df['wrc_plus'] = calculate_wrc_plus(df, weights)

    df = df.rename(columns={'sh': 'sac', 'batting_runs': 'batting'})
    if year is not None:
        df['year'] = year
    if division is not None:
        df['division'] = division
    for col in ['player_name', 'team_name', 'conference', 'class']:
        if col in df.columns:
            df[col] = df[col].fillna('-')
    df = df.fillna(0)

    output_cols = [c for c in batting_columns if c in df.columns and c != "sos_adj_war"]
    df = df[output_cols]

    if group_by == 'player_id':
        return df.dropna(subset=['war']), team_stats
    return df

def calculate_pitching_war(pitching_df, pbp_df, park_factors_df, bat_war_total, year, division, group_by='player_id'):
    if pitching_df.empty:
        return (pitching_df, pd.DataFrame()) if group_by == 'player_id' else pitching_df

    pitching_df = normalize_id_columns(pitching_df.copy())

    if 'b_t' in pitching_df.columns:
        pitching_df[['bats', 'throws']] = pitching_df['b_t'].str.split('/', n=1, expand=True)
        pitching_df['bats'] = pitching_df['bats'].fillna('-')
        pitching_df['throws'] = pitching_df['throws'].fillna('-')

    if group_by == 'player_id':
        df = pitching_df[pitching_df['app'] > 0].copy()
        df = df[df['era'].notna()]
    else:
        numeric_cols = [k for k, v in pitching_agg_dict.items() if v == 'sum' and k in pitching_df.columns]
        for col in numeric_cols:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors="coerce").fillna(0)
        agg = {k: v for k, v in pitching_agg_dict.items() if k in pitching_df.columns}
        df = pitching_df.groupby(group_by).agg(agg).reset_index()

    if group_by == 'player_id':
        df['ip_float'] = df['ip'].apply(ip_to_real)
    else:
        df['ip'] = df['ip_float'].apply(convert_to_baseball_ip)
        df['era'] = np.where(df['ip_float'] > 0, (df['er'] * 9) / df['ip_float'], 0)

    df = calculate_pitching_rate_stats(df)

    valid_ip_mask = df['ip_float'] > 0
    df.loc[~valid_ip_mask, ['fip', 'xfip']] = np.nan

    df['pf'] = df['team_name'].map(park_factors_df.set_index('team_name')['pf'])
    df['pf'] = df['pf'].fillna(100)

    df['era+'] = calculate_era_plus(df, valid_ip_mask)

    lg_era = (df.loc[valid_ip_mask, 'er'].sum() / max(df.loc[valid_ip_mask, 'ip_float'].sum(), 1e-12)) * 9
    fip_components = ((13 * df['hr_a'].sum() + 3 * (df['bb'].sum() + df['hbp'].sum()) - 2 * df['so'].sum())
                      / max(df['ip_float'].sum(), 1e-12))
    f_constant = lg_era - fip_components

    df['fip'] = calculate_fip(df, lg_era)
    df['xfip'] = calculate_xfip(df, f_constant)

    if group_by == 'player_id':
        gmli = calculate_gmli(pbp_df)
        df = df.merge(gmli, how='left', left_on='player_id', right_on='pitcher_id')
        df['gmli'] = df['gmli'].fillna(0.0)
        
        valid_ip_mask = df['ip_float'] > 0

        if_consts = (df[valid_ip_mask]
                     .groupby('conference', group_keys=False)
                     .apply(calculate_if_fip_constant, include_groups=False)
                     .reset_index())
        if_consts.columns = ['conference', 'if_fip_constant']
        df = df.merge(if_consts, on='conference', how='left')
        
        valid_ip_mask = df['ip_float'] > 0

        df['iffip'] = df.apply(lambda row: calculate_player_if_fip(row, row['if_fip_constant']), axis=1)

        valid_df = df[valid_ip_mask]
        if len(valid_df) > 0:
            lgRA9 = (valid_df['r'].sum() / max(valid_df['ip_float'].sum(), 1e-12)) * 9
            lgERA = (valid_df['er'].sum() / max(valid_df['ip_float'].sum(), 1e-12)) * 9
            adjustment = lgRA9 - lgERA
        else:
            adjustment = 0.0

        df['fipr9'] = np.where(valid_ip_mask, df['iffip'] + adjustment, np.nan)
        df['pfipr9'] = np.where(valid_ip_mask, df['fipr9'] / (df['pf'] / 100), np.nan)

        league_adjustments = (
            df[valid_ip_mask]
            .groupby('conference', group_keys=False)
            .apply(calculate_pitching_conf_fipr9, include_groups=False)
            .reset_index()
        )
        league_adjustments.columns = ['conference', 'conf_fipr9']
        df = df.merge(league_adjustments, on='conference', how='left')
        
        valid_ip_mask = df['ip_float'] > 0

        df = calculate_pitching_war_components(df, valid_ip_mask)
        df = apply_relief_leverage_bump(df, valid_ip_mask)
        df = normalize_pitching_war_to_batting(df, bat_war_total, valid_ip_mask)

        pitcher_stats, team_stats = get_pitcher_clutch_stats(pbp_df)
        df = df.merge(pitcher_stats.drop(columns=['year', 'division'], errors='ignore'),
                      left_on='player_id', right_on='pitcher_id', how='left')
    else:
        team_stats = pd.DataFrame()

    df = df.replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0)
    for col in ['player_name', 'team_name', 'conference', 'class']:
        if col in df.columns:
            df[col] = df[col].fillna('-')

    output_cols = [c for c in pitching_columns if c in df.columns and c != "sos_adj_war"]
    if year is not None:
        df['year'] = year
    if division is not None:
        df['division'] = division
    df = df[output_cols]

    if group_by == 'player_id':
        return df.dropna(subset=['war']), team_stats
    return df

def calculate_batting_team_war(batting_df, guts_df, park_factors_df, team_clutch, division, year):
    df = batting_df.copy()
    
    numeric_cols = [k for k, v in batting_agg_dict.items() if v == 'sum' and k in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    agg_dict = {k: v for k, v in batting_agg_dict.items() if k in df.columns}
    team_agg = df.groupby('team_name').agg(agg_dict).reset_index()
    
    weights = guts_df.iloc[0]
    team_agg['pf'] = team_agg['team_name'].map(park_factors_df.set_index('team_name')['pf']).fillna(100)
    team_agg['1b'] = team_agg['h'] - team_agg['hr'] - team_agg['3b'] - team_agg['2b']
    if 'pa' not in team_agg.columns or team_agg['pa'].sum() == 0:
        team_agg['pa'] = team_agg['ab'] + team_agg['bb'] + team_agg.get('ibb', 0) + team_agg['hbp'] + team_agg['sf']
    
    team_agg['sb_pct'] = calculate_sb_pct(team_agg)
    team_agg['bb_pct'] = calculate_bb_pct(team_agg)
    team_agg['k_pct'] = calculate_k_pct(team_agg)
    team_agg['ba'] = calculate_ba(team_agg)
    team_agg['slg_pct'] = calculate_slg_pct(team_agg)
    team_agg['ob_pct'] = calculate_ob_pct(team_agg)
    team_agg['babip'] = calculate_babip(team_agg)
    team_agg['iso'] = calculate_iso(team_agg)
    team_agg['runs_created'] = calculate_runs_created(team_agg)
    team_agg['ops_plus'] = calculate_ops_plus(team_agg)
    team_agg['rc_per_pa'] = calculate_rc_per_pa(team_agg)
    team_agg['woba'] = calculate_woba_from_components(team_agg, weights)
    team_agg['wraa'] = calculate_wraa_from_woba(team_agg, weights)
    team_agg['wrc_plus'] = calculate_wrc_plus(team_agg, weights)
    
    team_agg['year'] = year
    team_agg['division'] = division
    team_agg = team_agg.merge(team_clutch, left_on='team_name', right_on='bat_team', how='left')
    
    return team_agg

def calculate_pitching_team_war(pitching_df, park_factors_df, team_clutch, division, year):
    df = pitching_df.copy()
    
    numeric_cols = [k for k, v in pitching_agg_dict.items() if v == 'sum' and k in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    agg_dict = {k: v for k, v in pitching_agg_dict.items() if k in df.columns}
    team_agg = df.groupby('team_name').agg(agg_dict).reset_index()
    
    team_agg['pf'] = team_agg['team_name'].map(park_factors_df.set_index('team_name')['pf']).fillna(100)
    team_agg['ip'] = team_agg['ip_float'].apply(convert_to_baseball_ip)
    
    valid_ip = team_agg['ip_float'] > 0
    team_agg['era'] = np.where(valid_ip, (team_agg['er'] * 9) / team_agg['ip_float'], 0)
    team_agg['k9'] = np.where(valid_ip, (team_agg['so'] * 9) / team_agg['ip_float'], 0)
    team_agg['bb9'] = np.where(valid_ip, (team_agg['bb'] * 9) / team_agg['ip_float'], 0)
    team_agg['hr9'] = np.where(valid_ip, (team_agg['hr_a'] * 9) / team_agg['ip_float'], 0)
    team_agg['h9'] = np.where(valid_ip, (team_agg['h'] * 9) / team_agg['ip_float'], 0)
    team_agg['ra9'] = np.where(valid_ip, (team_agg['r'] * 9) / team_agg['ip_float'], 0)
    
    team_agg['k_pct'] = np.where(team_agg['bf'] > 0, team_agg['so'] / team_agg['bf'], 0)
    team_agg['bb_pct'] = np.where(team_agg['bf'] > 0, team_agg['bb'] / team_agg['bf'], 0)
    team_agg['k_minus_bb_pct'] = team_agg['k_pct'] - team_agg['bb_pct']
    team_agg['hr_div_fb'] = np.where(team_agg['fo'] > 0, team_agg['hr_a'] / team_agg['fo'], 0)
    team_agg['ir_a_pct'] = np.where(team_agg['inh_run'] > 0, team_agg['inh_run_score'] / team_agg['inh_run'], 0)
    
    lg_era = team_agg.loc[valid_ip, 'era'].mean() if valid_ip.any() else 4.50
    team_agg['era+'] = np.where(valid_ip & (team_agg['era'] > 0), (lg_era / team_agg['era']) * 100, 0)
    
    fip_components = ((13 * team_agg['hr_a'].sum() + 3 * (team_agg['bb'].sum() + team_agg['hbp'].sum()) - 2 * team_agg['so'].sum())
                      / max(team_agg['ip_float'].sum(), 1))
    f_constant = lg_era - fip_components
    team_agg['fip'] = calculate_fip(team_agg, f_constant)
    team_agg['xfip'] = calculate_xfip(team_agg, f_constant)
    
    team_agg['year'] = year
    team_agg['division'] = division
    team_agg = team_agg.merge(team_clutch, left_on='team_name', right_on='pitch_team', how='left')
    
    return team_agg
