position_adjustments = {
    'SS': 1.85, 'C': 3.09, '2B': 0.62, '3B': 0.62, 'UT': 0.62,
    'CF': 0.62, 'INF': 0.62, 'LF': -1.85, 'RF': -1.85, '1B': -3.09,
    'DH': -3.09, 'OF': 0.25, 'PH': -0.74, 'PR': -0.74, 'P': 0.62,
    'RP': 0.62, 'SP': 0.62, '': 0
}

batting_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'bats', 'throws', 'team_name', 'team_id', 'conference', 'gp',
    'bb', 'cs', 'gs', 'hbp', 'ibb', 'k', 'rbi', 'sf', 'ab',
    'pa', 'h', '2b', '3b', 'hr', 'r', 'sb', 'ops_plus', 'picked',
    'sac', 'ba', 'slg_pct', 'ob_pct', 'iso', 'woba', 'k_pct', 'bb_pct',
    'sb_pct', 'wrc_plus', 'wrc', 'rc_per_pa',
    'wsb', 'wgdp', 'wteb', 'ebt', 'ebt_opps', 'outs_ob',
    'gdp_opps', 'gdp', 'clutch', 'wpa', 'rea', 'wpa_li', 'babip',
    'batting', 'baserunning', 'adjustment', 'war', 'sos_adj_war'
]

pitching_columns = [
    'player_name', 'division', 'year', 'class', 'player_id', 'bats', 'throws', 'team_name', 'team_id', 'conference', 'app',
    'gs', 'era', 'ip', 'w', 'l', 'sv', 'ip_float', 'h', 'r', 'er',
    'bb', 'so', 'hr_a', '2b_a', '3b_a', 'hbp', 'bf', 'fo', 'go', 'pitches',
    'gmli', 'k9', 'bb9', 'hr9', 'ra9', 'h9', 'ir_a_pct', 'k_pct', 'bb_pct', 'k_minus_bb_pct', 'hr_div_fb', 'fip',
    'xfip', 'era+', 'inh_run', 'inh_run_score',
    'clutch', 'pwpa', 'prea', 'pwpa_li', 'war', 'sos_adj_war'
]

batting_agg_dict = {
    'conference': 'first', 'year': 'first', 'division': 'first', 'team_id': 'first',
    'gp': 'max', 'gs': 'max',
    'ab': 'sum', 'pa': 'sum', 'h': 'sum', '2b': 'sum', '3b': 'sum', 'hr': 'sum', 'r': 'sum',
    'rbi': 'sum', 'bb': 'sum', 'ibb': 'sum', 'hbp': 'sum', 'sf': 'sum', 'sac': 'sum',
    'k': 'sum', 'sb': 'sum', 'cs': 'sum', 'picked': 'sum',
    'gdp': 'sum', 'gdp_opps': 'sum',
    'ebt': 'sum', 'ebt_opps': 'sum', 'outs_ob': 'sum',
    'wrc': 'sum', 'wsb': 'sum', 'wgdp': 'sum', 'wteb': 'sum',
    'batting': 'sum', 'baserunning': 'sum', 'adjustment': 'sum',
    'wpa': 'sum', 'rea': 'sum', 'wpa_li': 'sum',
    'war': 'sum', 'sos_adj_war': 'sum',
}

pitching_agg_dict = {
    'conference': 'first', 'year': 'first', 'division': 'first', 'team_id': 'first',
    'app': 'sum', 'gs': 'sum', 'w': 'sum', 'l': 'sum', 'sv': 'sum',
    'ip_float': 'sum', 'h': 'sum', 'r': 'sum', 'er': 'sum',
    'bb': 'sum', 'so': 'sum', 'hbp': 'sum', 'bf': 'sum',
    'hr_a': 'sum', '2b_a': 'sum', '3b_a': 'sum',
    'fo': 'sum', 'go': 'sum', 'pitches': 'sum',
    'inh_run': 'sum', 'inh_run_score': 'sum',
    'pwpa': 'sum', 'prea': 'sum', 'pwpa_li': 'sum',
    'war': 'sum', 'sos_adj_war': 'sum',
}

batting_fill_cols = ['hr', 'r', 'gp', 'gs', '2b', '3b', 'h', 'cs', 'bb', 'k', 'sb', 'ibb', 'rbi', 'picked', 'sh', 'ab', 'hbp', 'sf']
pitching_fill_cols = ['hr_a', 'fo', 'ip', 'bb', 'so', 'sv', 'gs', 'hbp', 'bf', 'h', 'r']

REP_WP = 0.294
