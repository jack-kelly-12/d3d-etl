import numpy as np
import pandas as pd
from war_calculator.common import (
    aggregate_team,
    fill_missing,
    float_to_ip,
    ip_to_float,
    normalize_id_columns,
    safe_divide,
)
from war_calculator.constants import PITCHING_SUM_COLS, pitching_columns

# =============================================================================
# Rate Stats (per 9 innings)
# =============================================================================

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


def hr_per_fb(hr, fo):
    return safe_divide(hr, hr + fo) * 100


def inherited_runners_scored_pct(inh_run_score, inh_run):
    return safe_divide(inh_run_score, inh_run) * 100


def fip_constant(hr_sum, bb_sum, hbp_sum, so_sum, ip_sum, lg_era):
    fip_components = (13 * hr_sum + 3 * (bb_sum + hbp_sum) - 2 * so_sum) / ip_sum
    return lg_era - fip_components


def fip(hr, bb, hbp, so, ip, constant):
    return constant + safe_divide(13 * hr + 3 * (bb + hbp) - 2 * so, ip, fill=np.nan)


def xfip(fo, hr, bb, hbp, so, ip, constant, lg_hr_fb_rate):
    expected_hr = (fo + hr) * lg_hr_fb_rate
    return constant + safe_divide(13 * expected_hr + 3 * (bb + hbp) - 2 * so, ip, fill=np.nan)


def era_plus(player_era, lg_era, pf):
    return 100 * (2 - (player_era / lg_era) * (100 / pf))

def dynamic_rpw(ip_per_game, conf_fipr9, pfipr9):
    return (((18 - ip_per_game) * conf_fipr9 + ip_per_game * pfipr9) / 18 + 2) * 1.5


def replacement_level(gs, app):
    gs_rate = safe_divide(gs, app)
    return 0.03 * (1 - gs_rate) + 0.12 * gs_rate


def pitching_war(raap9, drpw, replacement, ip):
    wpgaa = safe_divide(raap9, drpw)
    wpgar = wpgaa + replacement
    return wpgar * (ip / 9)


def reliever_leverage_adjustment(war_val, gmli):
    return war_val * (1 + gmli) / 2

def get_pitcher_clutch_stats(pbp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_df.copy()
    df['pitch_team_id'] = pd.to_numeric(df['pitch_team_id'], errors='coerce').astype('Int64')

    df['prea'] = -df['rea']
    df['pwpa'] = -df['wpa']
    df['pwpa_li'] = safe_divide(df['pwpa'], df['li'])

    pitcher_stats = df.groupby('pitcher_id').agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa_li': 'sum', 'li': 'mean', 'pitcher_name': 'first'
    }).reset_index()

    pitcher_stats['clutch'] = np.where(
        pitcher_stats['li'] > 0,
        (pitcher_stats['pwpa'] / pitcher_stats['li']) - pitcher_stats['pwpa_li'],
        np.nan
    )

    team_stats = df.groupby('pitch_team_id').agg({
        'prea': 'sum', 'pwpa': 'sum', 'pwpa_li': 'sum', 'li': 'mean'
    }).reset_index()

    team_stats['clutch'] = np.where(
        team_stats['li'] > 0,
        (team_stats['pwpa'] / team_stats['li']) - team_stats['pwpa_li'],
        np.nan
    )

    return pitcher_stats, team_stats


def calculate_gmli(pbp_df: pd.DataFrame) -> pd.DataFrame:
    first_app = (
        pbp_df.sort_values(['pitcher_id', 'contest_id', 'play_id'])
        .groupby(['pitcher_id', 'contest_id'])
        .first()
        .reset_index()
    )
    relievers = first_app[first_app['inning'] != 1]
    return relievers.groupby('pitcher_id').agg({'li': 'mean'}).reset_index().rename(columns={'li': 'gmli'})


def add_pitching_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ip = df['ip_float']

    df['ra9'] = ra9(df['r'], ip)
    df['k9'] = k9(df['so'], ip)
    df['h9'] = h9(df['h'], ip)
    df['bb9'] = bb9(df['bb'], ip)
    df['hr9'] = hr9(df['hr_a'], ip)
    df['whip'] = whip(df['bb'], df['h'], ip)

    df['k_pct'] = k_pct(df['so'], df['bf'])
    df['bb_pct'] = bb_pct(df['bb'], df['bf'])
    df['k_minus_bb_pct'] = k_minus_bb_pct(df['k_pct'], df['bb_pct'])
    df['hr_div_fb'] = hr_per_fb(df['hr_a'], df['fo'])
    df['ir_a_pct'] = inherited_runners_scored_pct(df['inh_run_score'], df['inh_run'])

    return df


def add_fip_stats(df: pd.DataFrame, valid_mask: pd.Series) -> pd.DataFrame:
    df = df.copy()

    ip_sum = df.loc[valid_mask, 'ip_float'].sum()
    if ip_sum == 0:
        df['fip'] = np.nan
        df['xfip'] = np.nan
        df['era+'] = np.nan
        return df

    lg_era_val = era(df.loc[valid_mask, 'er'].sum(), ip_sum)
    constant = fip_constant(
        df['hr_a'].sum(), df['bb'].sum(), df['hbp'].sum(),
        df['so'].sum(), ip_sum, lg_era_val
    )

    df['fip'] = fip(df['hr_a'], df['bb'], df['hbp'], df['so'], df['ip_float'], constant)

    lg_hr_fb = safe_divide(df['hr_a'].sum(), df['hr_a'].sum() + df['fo'].sum(), fill=0.10)
    df['xfip'] = xfip(df['fo'], df['hr_a'], df['bb'], df['hbp'], df['so'], df['ip_float'], constant, lg_hr_fb)

    df['era+'] = np.nan
    df.loc[valid_mask, 'era+'] = era_plus(df.loc[valid_mask, 'era'], lg_era_val, df.loc[valid_mask, 'pf'])

    return df


def calculate_pitching_war(
    pitching_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    bat_war_total: float,
    year: int,
    division: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pitching_df.empty:
        return pitching_df, pd.DataFrame()

    df = normalize_id_columns(pitching_df.copy())

    if 'b_t' in df.columns:
        df[['bats', 'throws']] = df['b_t'].str.split('/', n=1, expand=True)
        df['bats'] = df['bats'].fillna('-')
        df['throws'] = df['throws'].fillna('-')

    df = df[(df['app'] > 0) & df['era'].notna()].copy()
    df['ip_float'] = df['ip'].apply(ip_to_float)
    valid_mask = df['ip_float'] > 0

    pf_map = park_factors_df.set_index('team_name')['pf'].to_dict()
    df['pf'] = df['team_name'].map(pf_map).fillna(100)

    df = add_pitching_stats(df)
    df = add_fip_stats(df, valid_mask)

    gmli_df = calculate_gmli(pbp_df)
    df = df.merge(gmli_df, left_on='player_id', right_on='pitcher_id', how='left')
    df['gmli'] = df['gmli'].fillna(0)

    valid_mask = df['ip_float'] > 0
    valid_df = df[valid_mask]

    if len(valid_df) > 0:
        lg_ra9 = ra9(valid_df['r'].sum(), valid_df['ip_float'].sum())
        lg_era_val = era(valid_df['er'].sum(), valid_df['ip_float'].sum())
        adjustment = lg_ra9 - lg_era_val
    else:
        adjustment = 0.0

    conf_fip = {}
    for conf in df['conference'].unique():
        conf_df = df[(df['conference'] == conf) & valid_mask]
        if len(conf_df) > 0 and conf_df['ip_float'].sum() > 0:
            conf_lg_era = era(conf_df['er'].sum(), conf_df['ip_float'].sum())
            conf_const = fip_constant(
                conf_df['hr_a'].sum(), conf_df['bb'].sum(), conf_df['hbp'].sum(),
                conf_df['so'].sum(), conf_df['ip_float'].sum(), conf_lg_era
            )
            conf_fip_val = fip(
                conf_df['hr_a'].sum(), conf_df['bb'].sum(), conf_df['hbp'].sum(),
                conf_df['so'].sum(), conf_df['ip_float'].sum(), conf_const
            )
            conf_fip[conf] = conf_fip_val + adjustment

    df['conf_fipr9'] = df['conference'].map(conf_fip)
    df['pfipr9'] = np.where(valid_mask, (df['fip'] + adjustment) / (df['pf'] / 100), np.nan)
    df['raap9'] = np.where(valid_mask, df['conf_fipr9'] - df['pfipr9'], 0)
    df['ip_per_g'] = safe_divide(df['ip_float'], df['app'])
    df['drpw'] = np.where(valid_mask, dynamic_rpw(df['ip_per_g'], df['conf_fipr9'], df['pfipr9']), 0)
    df['replacement_level'] = replacement_level(df['gs'], df['app'])
    df['war'] = np.where(valid_mask, pitching_war(df['raap9'], df['drpw'], df['replacement_level'], df['ip_float']), 0)

    reliever_mask = (df['gs'] < 3) & valid_mask
    df.loc[reliever_mask, 'war'] = reliever_leverage_adjustment(df.loc[reliever_mask, 'war'], df.loc[reliever_mask, 'gmli'])

    target_war = (bat_war_total * 0.43) / 0.57
    current_war = df['war'].sum()
    ip_sum = df.loc[valid_mask, 'ip_float'].sum()
    if ip_sum > 0:
        war_adj = (target_war - current_war) / ip_sum
        df.loc[valid_mask, 'war'] += war_adj * df.loc[valid_mask, 'ip_float']

    pitcher_clutch, team_clutch = get_pitcher_clutch_stats(pbp_df)
    df = df.merge(
        pitcher_clutch[['pitcher_id', 'prea', 'pwpa', 'pwpa_li', 'clutch']],
        left_on='player_id', right_on='pitcher_id', how='left', suffixes=('', '_clutch')
    )

    df = df.replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0)
    df['year'] = year
    df['division'] = division

    for col in ['player_name', 'team_name', 'conference', 'class']:
        if col in df.columns:
            df[col] = df[col].fillna('-')

    output_cols = [c for c in pitching_columns if c in df.columns and c != 'sos_adj_war']
    return df[output_cols].dropna(subset=['war']), team_clutch


def calculate_team_pitching(
    player_df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    team_clutch: pd.DataFrame,
    division: int,
    year: int
) -> pd.DataFrame:
    if player_df.empty:
        return pd.DataFrame()

    team_df = aggregate_team(player_df, PITCHING_SUM_COLS)
    team_df = fill_missing(team_df, PITCHING_SUM_COLS)

    pf_map = park_factors_df.set_index('team_name')['pf'].to_dict()
    team_df['pf'] = team_df['team_name'].map(pf_map).fillna(100)

    team_df['ip'] = team_df['ip_float'].apply(float_to_ip)

    valid = team_df['ip_float'] > 0
    team_df['era'] = np.where(valid, era(team_df['er'], team_df['ip_float']), 0)
    team_df = add_pitching_stats(team_df)
    team_df = add_fip_stats(team_df, valid)

    team_df['team_id'] = pd.to_numeric(team_df['team_id'], errors='coerce').astype('Int64')
    team_df = team_df.merge(
        team_clutch[['pitch_team_id', 'prea', 'pwpa', 'pwpa_li', 'clutch']],
        left_on='team_id', right_on='pitch_team_id', how='left'
    )

    team_df['year'] = year
    team_df['division'] = division

    return team_df
