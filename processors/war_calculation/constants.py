position_adjustments = {
    'SS': 1.85, 'C': 3.09, '2B': 0.62, '3B': 0.62, 'UT': 0.62,
    'CF': 0.62, 'INF': 0.62, 'LF': -1.85, 'RF': -1.85, '1B': -3.09,
    'DH': -3.09, 'OF': 0.25, 'PH': -0.74, 'PR': -0.74, 'P': 0.62,
    'RP': 0.62, 'SP': 0.62, '': 0
}

batting_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'bats', 'throws',
    'team_name', 'team_id', 'conference', 'gp', 'gs',
    'ab', 'pa', 'h', '1b', '2b', '3b', 'hr', 'r', 'rbi',
    'bb', 'ibb', 'hbp', 'k', 'sf', 'sac', 'sb', 'cs', 'picked',
    'ba', 'ob_pct', 'slg_pct', 'iso', 'babip', 'ops_plus',
    'bb_pct', 'k_pct', 'bb_per_k', 'sb_pct', 'runs_created', 'rc_per_pa',
    'woba', 'wrc', 'wraa', 'wrc_plus', 'wsb',
    'wgdp', 'gdp', 'gdp_opps',
    'baserunning', 'batting', 'adjustment', 'league_adjustment',
    'rea', 'wpa', 'wpa_li', 'clutch',
    'war', 'sos_adj_war'
]

pitching_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'bats', 'throws',
    'team_name', 'team_id', 'conference', 'app', 'gs', 'w', 'l', 'sv',
    'ip', 'ip_float', 'era', 'h', 'r', 'er', 'bb', 'so', 'hbp', 'bf',
    'hr_a', '2b_a', '3b_a', 'fo', 'go', 'pitches',
    'inh_run', 'inh_run_score',
    'ra9', 'k9', 'bb9', 'h9', 'hr9', 'whip',
    'k_pct', 'bb_pct', 'k_minus_bb_pct', 'hr_div_fb', 'ir_a_pct',
    'fip', 'xfip', 'era_plus', 'gmli',
    'prea', 'pwpa', 'pwpa_li', 'clutch',
    'war', 'sos_adj_war'
]

BATTING_SUM_COLS = [
    'ab', 'pa', 'h', '1b', '2b', '3b', 'hr', 'r', 'rbi',
    'bb', 'ibb', 'hbp', 'k', 'sf', 'sac', 'sb', 'cs', 'picked',
    'wrc', 'wsb', 'wgdp', 'gdp', 'gdp_opps',
    'baserunning', 'batting', 'adjustment',
    'war', 'sos_adj_war'
]

PITCHING_SUM_COLS = [
    'app', 'gs', 'w', 'l', 'sv',
    'ip_float', 'h', 'r', 'er', 'bb', 'so', 'hbp', 'bf',
    'hr_a', '2b_a', '3b_a', 'fo', 'go', 'pitches',
    'inh_run', 'inh_run_score',
    'war', 'sos_adj_war'
]

REP_WP = 0.294
