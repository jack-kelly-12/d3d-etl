import numpy as np
import pandas as pd
from war_calculator.common import aggregate_team, fill_missing, normalize_id_columns, safe_divide
from war_calculator.constants import BATTING_SUM_COLS, batting_columns, position_adjustments


def singles(h, doubles, triples, hr):
    return h - hr - triples - doubles


def plate_appearances(ab, bb, ibb, hbp, sf):
    return ab + bb + ibb + hbp + sf

def walks_per_strikeout(bb, k):
    return safe_divide(bb, k)

def total_bases(singles, doubles, triples, hr):
    return singles + 2*doubles + 3*triples + 4*hr

def batting_average(h, ab):
    return safe_divide(h, ab)

def on_base_pct(h, bb, hbp, ibb, ab, sf):
    return safe_divide(h + bb + hbp + ibb, ab + bb + ibb + hbp + sf)

def slugging_pct(tb, ab):
    return safe_divide(tb, ab)


def isolated_power(slg, ba):
    return slg - ba


def babip(h, hr, ab, k, sf):
    return safe_divide(h - hr, ab - hr - k + sf)


def walk_pct(bb, pa):
    return safe_divide(bb, pa) * 100


def strikeout_pct(k, pa):
    return safe_divide(k, pa) * 100


def stolen_base_pct(sb, cs):
    return safe_divide(sb, sb + cs) * 100


def runs_created(tb, h, bb, ab):
    return safe_divide(tb * (h + bb), ab + bb)


def rc_per_pa(rc, pa):
    return safe_divide(rc, pa)


def ops_plus(obp, slg, lg_obp, lg_slg):
    return 100 * (safe_divide(obp, lg_obp) + safe_divide(slg, lg_slg) - 1)


# =============================================================================
# Linear Weights
# =============================================================================

def woba(bb, hbp, singles, doubles, triples, hr, ab, ibb, sf, weights):
    num = (
        weights['wbb'] * bb +
        weights['whbp'] * hbp +
        weights['w1b'] * singles +
        weights['w2b'] * doubles +
        weights['w3b'] * triples +
        weights['whr'] * hr
    )
    denom = ab + bb - ibb + sf + hbp
    return safe_divide(num, denom)


def wrc(woba_val, lg_woba, woba_scale, lg_rpa, pa):
    return ((woba_val - lg_woba) / woba_scale + lg_rpa) * pa


def wraa(woba_val, lg_woba, woba_scale, pa):
    return ((woba_val - lg_woba) / woba_scale) * pa


def wrc_plus(wraa_val, pa, lg_rpa, lg_wrcpa, pf):
    wraa_pa = safe_divide(wraa_val, pa)
    pf_adj = pf / 100
    return safe_divide((wraa_pa + lg_rpa) + (lg_rpa - pf_adj * lg_rpa), lg_wrcpa) * 100


def wsb(sb, cs, runs_out, lg_sb, lg_cs, lg_opps):
    run_sb = 0.2
    run_cs = -(2 * runs_out + 0.075)
    lg_wsb = safe_divide(lg_sb * run_sb + lg_cs * run_cs, lg_opps)
    opps = sb + cs
    return sb * run_sb + cs * run_cs - lg_wsb * opps

def batting_runs(wraa_val, pa, pf, lg_rpa, conf_rpa):
    pf_adj = pf / 100
    return wraa_val + (lg_rpa - pf_adj * lg_rpa) * pa + (lg_rpa - conf_rpa) * pa


def position_adjustment(pos, gp, division):
    games_per_season = 40 if division == 3 else 50
    base = position_adjustments.get(str(pos).upper(), 0)
    return base * (gp / games_per_season)


def replacement_runs(pa, total_pa, team_count, total_gs, rpw):
    games_played = (total_gs / 9) / team_count
    rep_constant = (team_count / 2) * games_played - team_count * games_played * 0.294
    return (rep_constant * rpw) * safe_divide(pa, total_pa)


def league_adjustment(batting_runs_val, wsb_val, adj, pa, lg_total, lg_pa):
    conf_adj = -lg_total / lg_pa if lg_pa > 0 else 0
    return conf_adj * pa


def batting_war(batting_runs_val, replacement_val, baserunning_val, adjustment_val, league_adj_val, rpw):
    return (batting_runs_val + replacement_val + baserunning_val + adjustment_val + league_adj_val) / rpw

def get_batter_clutch_stats(pbp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_df.copy()
    df['bat_team_id'] = pd.to_numeric(df['bat_team_id'], errors='coerce').astype('Int64')

    player_stats = df.groupby('batter_id').agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa_li': 'sum', 'li': 'mean'
    }).reset_index()

    player_stats['clutch'] = np.where(
        player_stats['li'] > 0,
        (player_stats['wpa'] / player_stats['li']) - player_stats['wpa_li'],
        np.nan
    )

    team_stats = df.groupby('bat_team_id').agg({
        'rea': 'sum', 'wpa': 'sum', 'wpa_li': 'sum', 'li': 'mean'
    }).reset_index()

    team_stats['clutch'] = np.where(
        team_stats['li'] > 0,
        (team_stats['wpa'] / team_stats['li']) - team_stats['wpa_li'],
        np.nan
    )

    return player_stats, team_stats

def calculate_wgdp(pbp_df: pd.DataFrame) -> pd.DataFrame:
    gdp_opps = pbp_df[
        (pbp_df['r1_id'].notna()) &
        (pbp_df['r1_id'] != '') &
        (pbp_df['outs_before'].astype(int) < 2)
    ].copy()

    gdp_events = gdp_opps[
        gdp_opps['play_description'].str.contains('double play', case=False, na=False)
    ]

    valid = gdp_opps['batter_id'].notna() & (gdp_opps['batter_id'] != '')
    gdp_opps = gdp_opps[valid]
    gdp_events = gdp_events[gdp_events['batter_id'].notna() & (gdp_events['batter_id'] != '')]

    stats = pd.DataFrame({
        'gdp_opps': gdp_opps.groupby('batter_id').size(),
        'gdp': gdp_events.groupby('batter_id').size()
    }).fillna(0)

    lg_rate = safe_divide(stats['gdp'].sum(), stats['gdp_opps'].sum())
    stats['wgdp'] = (stats['gdp_opps'] * lg_rate - stats['gdp']) * 0.5

    return stats

def add_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['1b'] = singles(df['h'], df['2b'], df['3b'], df['hr'])
    df['pa'] = plate_appearances(df['ab'], df['bb'], df['ibb'], df['hbp'], df['sf'])
    df['tb'] = total_bases(df['1b'], df['2b'], df['3b'], df['hr'])

    df['ba'] = batting_average(df['h'], df['ab'])
    df['ob_pct'] = on_base_pct(df['h'], df['bb'], df['hbp'], df['ibb'], df['ab'], df['sf'])
    df['slg_pct'] = slugging_pct(df['tb'], df['ab'])
    df['iso'] = isolated_power(df['slg_pct'], df['ba'])
    df['babip'] = babip(df['h'], df['hr'], df['ab'], df['k'], df['sf'])
    df['bb_pct'] = walk_pct(df['bb'], df['pa'])
    df['k_pct'] = strikeout_pct(df['k'], df['pa'])
    df['sb_pct'] = stolen_base_pct(df['sb'], df['cs'])
    df['runs_created'] = runs_created(df['tb'], df['h'], df['bb'], df['ab'])
    df['rc_per_pa'] = rc_per_pa(df['runs_created'], df['pa'])
    df['bb_per_k'] = walks_per_strikeout(df['bb'], df['k'])

    lg_obp = safe_divide(
        df['h'].sum() + df['bb'].sum() + df['hbp'].sum(),
        df['ab'].sum() + df['bb'].sum() + df['hbp'].sum() + df['sf'].sum()
    )
    lg_slg = safe_divide(
        df['tb'].sum(),
        df['ab'].sum(),
    )
    df['ops_plus'] = ops_plus(df['ob_pct'], df['slg_pct'], lg_obp, lg_slg)

    return df


def add_linear_weights(df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    df = df.copy()

    df['woba'] = woba(
        df['bb'], df['hbp'], df['1b'], df['2b'], df['3b'], df['hr'],
        df['ab'], df['ibb'], df['sf'], weights
    )

    lg_rpa = safe_divide(df['r'].sum(), df['pa'].sum())
    df['wrc'] = wrc(df['woba'], weights['woba'], weights['woba_scale'], lg_rpa, df['pa'])
    df['wraa'] = wraa(df['woba'], weights['woba'], weights['woba_scale'], df['pa'])

    lg_wrcpa = safe_divide(df['wrc'].sum(), df['pa'].sum())
    df['wrc_plus'] = wrc_plus(df['wraa'], df['pa'], lg_rpa, lg_wrcpa, df['pf'])

    runs_out = weights['runs_out']
    lg_sb = df['sb'].sum()
    lg_cs = df['cs'].sum()
    lg_opps = df['1b'].sum() + df['bb'].sum() + df['hbp'].sum() - df['ibb'].sum()
    df['wsb'] = wsb(df['sb'], df['cs'], runs_out, lg_sb, lg_cs, lg_opps)

    return df


def calculate_batting_war(
    batting_df: pd.DataFrame,
    guts_df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    division: int,
    year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batting_df.empty:
        return batting_df, pd.DataFrame()

    weights = guts_df.iloc[0]
    df = normalize_id_columns(batting_df.copy())

    if 'b_t' in df.columns:
        df[['bats', 'throws']] = df['b_t'].str.split('/', n=1, expand=True)
        df['bats'] = df['bats'].fillna('-')
        df['throws'] = df['throws'].fillna('-')

    df['pos'] = df['pos'].apply(lambda x: '' if pd.isna(x) else str(x).split('/')[0].upper())
    df = df[df['ab'] > 0].copy()
    df['gp'] = pd.to_numeric(df['gp'], errors='coerce').fillna(0).astype(int)
    df['gs'] = pd.to_numeric(df['gs'], errors='coerce').fillna(0).astype(int)

    pf_map = park_factors_df.set_index('team_name')['pf'].to_dict()
    df['pf'] = df['team_name'].map(pf_map).fillna(100)

    df = add_batting_stats(df)
    df = add_linear_weights(df, weights)

    gdp_stats = calculate_wgdp(pbp_df)
    df = df.merge(gdp_stats, left_on='player_id', right_index=True, how='left')
    df = fill_missing(df, ['wgdp', 'gdp_opps', 'gdp'])
    df['baserunning'] = df['wsb'] + df['wgdp']

    player_clutch, team_clutch = get_batter_clutch_stats(pbp_df)
    df = df.merge(
        player_clutch[['batter_id', 'rea', 'wpa', 'wpa_li', 'clutch']],
        left_on='player_id', right_on='batter_id', how='left'
    )

    lg_rpa = safe_divide(df['r'].sum(), df['pa'].sum())
    conf_rpa = df.groupby('conference')['r'].transform('sum') / df.groupby('conference')['pa'].transform('sum')
    conf_rpa = conf_rpa.fillna(lg_rpa)

    df['batting'] = batting_runs(df['wraa'], df['pa'], df['pf'], lg_rpa, conf_rpa)
    df['adjustment'] = df.apply(lambda r: position_adjustment(r['pos'], r['gp'], division), axis=1)

    team_count = max(len(df['team_name'].unique()), 1)
    df['replacement_level_runs'] = replacement_runs(df['pa'], df['pa'].sum(), team_count, df['gs'].sum(), weights['runs_win'])

    for conf in df['conference'].unique():
        mask = df['conference'] == conf
        lg_total = df.loc[mask, 'batting'].sum() + df.loc[mask, 'wsb'].sum() + df.loc[mask, 'adjustment'].sum()
        lg_pa = df.loc[mask, 'pa'].sum()
        df.loc[mask, 'league_adjustment'] = (-lg_total / lg_pa if lg_pa > 0 else 0) * df.loc[mask, 'pa']

    df['war'] = batting_war(
        df['batting'], df['replacement_level_runs'], df['baserunning'],
        df['adjustment'], df['league_adjustment'], weights['runs_win']
    )

    df['year'] = year
    df['division'] = division
    df = df.fillna(0)

    output_cols = [c for c in batting_columns if c in df.columns and c != 'sos_adj_war']
    return df[output_cols].dropna(subset=['war']), team_clutch


def calculate_team_batting(
    player_df: pd.DataFrame,
    guts_df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    team_clutch: pd.DataFrame,
    division: int,
    year: int
) -> pd.DataFrame:
    if player_df.empty:
        return pd.DataFrame()

    weights = guts_df.iloc[0]

    team_df = aggregate_team(player_df, BATTING_SUM_COLS)
    team_df = fill_missing(team_df, BATTING_SUM_COLS)

    pf_map = park_factors_df.set_index('team_name')['pf'].to_dict()
    team_df['pf'] = team_df['team_name'].map(pf_map).fillna(100)

    team_df = add_batting_stats(team_df)
    team_df = add_linear_weights(team_df, weights)

    team_df['team_id'] = pd.to_numeric(team_df['team_id'], errors='coerce').astype('Int64')
    team_df = team_df.merge(
        team_clutch[['bat_team_id', 'rea', 'wpa', 'wpa_li', 'clutch']],
        left_on='team_id', right_on='bat_team_id', how='left'
    )

    team_df['year'] = year
    team_df['division'] = division

    return team_df
