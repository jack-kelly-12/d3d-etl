from pathlib import Path

import numpy as np
import pandas as pd


class BaseballAnalytics:
    EVENT_CODES = {
        'WALK': 14,
        'HBP': 16,
        'SINGLE': 20,
        'DOUBLE': 21,
        'TRIPLE': 22,
        'HOME_RUN': 23,
        'OUT': [2, 3, 19],
        'ERROR': 18
    }

    def __init__(self, pbp_df, year, division, data_path):
        self.original_df = pbp_df.copy()
        self.year = year
        self.situations = None
        self.weights = pd.read_csv(data_path /
                                   f'miscellaneous/d{division}_linear_weights_{year}.csv').set_index(
            'events')['normalized_weight'].to_dict()
        self.division = division
        self.right_pattern = r'to right|to 1b|rf line|to rf|right side|by 1b|by first base|to first base|1b line|by rf|by right field'
        self.left_pattern = r'to left|to 3b|lf line|left side|to lf|by 3b|by third base|to third base|down the 3b line|by lf|by left field'

        self.fly_pattern = r'fly|flied|homered|tripled to (?:left|right|cf|center)|doubled to (?:right|rf)'
        self.lined_pattern = r'tripled (?:to second base|,)|singled to (?:center|cf)|doubled down the (?:lf|rf) line|lined|doubled|singled to (?:left|right|rf|lf)'
        self.popped_pattern = r'fouled (?:into|out)|popped'
        self.ground_pattern = (
            r'tripled to (?:catcher|first base)|'
            r'tripled(?:,\s*(?:scored|out))|'
            r'singled to catcher|'
            r'singled(?:\s*(?:\([^)]+\))?\s*(?:,\s*|\s*;\s*|\s*3a\s*|\s*:\s*|\s+up\s+the\s+middle))|'
            r'hit into (?:double|triple) play|'
            r'reached (?:first )?on (?:an?)|'
            r'fielder\'s choice|fielding error|'
            r'(?:singled|tripled) through the (?:left|right) side|'
            r'error by (?:1b|2b|ss|3b|first|second|short|third)|'
            r'ground|'
            r'down the (?:1b|rf|3b|lf) line|'
            r'singled to (?:p|3b|1b|2b|ss|third|first|second|short)'
        )

    def prepare_situations(self):
        self.situations = self.original_df.copy()

        self.situations['risp_fl'] = (
            ~self.situations['r2_name'].isna() |
            ~self.situations['r3_name'].isna()
        ).astype(int)

        self.situations['ro_fl'] = (
            ~self.situations['r1_name'].isna() |
            ~self.situations['r2_name'].isna() |
            ~self.situations['r3_name'].isna()
        ).astype(int)

        self.situations['li_hi_fl'] = (self.situations['li'] >= 2).astype(int)
        self.situations['li_lo_fl'] = (
            self.situations['li'] <= 0.85).astype(int)

    def prepare_batted_ball(self):
        self.df = self.original_df.copy()
        self.df = self.df.dropna(subset=['batter_id',])

        FIELD_PATTERNS = {
            "to_lf": [r'to left', r'to lf', r'left field', r'lf line', r'by lf'],
            "to_cf": [r'to center', r'to cf', r'center field', r'by cf', r'to left center', 'to right center'],
            "to_rf": [r'to right', r'to rf', r'right field', r'rf line', r'by rf'],
            "to_lf_hr": [r'homered to left', r'homered to lf', r'homers to lf', r'homers to left'],
            "to_cf_hr": [r'homered to center', r'homered to cf', r'homers to cf', r'homers to center'],
            "to_rf_hr": [r'homered to right', r'homered to rf', r'homers to rf', r'homers to right'],
            "to_3b": [r'to 3b', r'to third', r'third base', r'3b line', r'by 3b', r'3b to 2b'],
            "to_ss": [r'ss to 2b', r'to ss', r'to short', r'shortstop', r'by ss'],
            "up_middle": [r'up the middle', r'to pitcher', r'to p', r'to c', r'by p', r'by c', r'to pitcher', r'to catcher'],
            "to_2b": [r'2b to ss', r'to 2b', r'to second', r'second base', r'by 2b'],
            "to_1b": [r'to 1b', r'to first', r'first base', r'1b line', r'by 1b', r'1b to ss', r'1b to p', r'1b to 2b'],
        }

        pull_oppo_patterns = {
            'right': FIELD_PATTERNS['to_rf'] + FIELD_PATTERNS['to_rf_hr'] + FIELD_PATTERNS['to_1b'] + FIELD_PATTERNS['to_2b'],
            'left': FIELD_PATTERNS['to_lf'] + FIELD_PATTERNS['to_lf_hr'] + FIELD_PATTERNS['to_3b'] + FIELD_PATTERNS['to_ss'],
            'middle': FIELD_PATTERNS['to_cf'] + FIELD_PATTERNS['to_cf_hr'] + FIELD_PATTERNS['up_middle']
        }

        hit_type_patterns = {
            'ground': [
                'ground',
                'hit into double play',
                'hit into triple play',
                "fielder's choice",
                'singled to catcher',
                'reached on error by p',
                'reached on error by c',
                'reached on error by 1b',
                'reached on error by 2b',
                'reached on error by ss',
                'reached on error by 3b',
                'singled to p',
                'singled to 3b',
                'singled to 1b',
                'singled to 2b',
                'singled to ss',
                'down the 1b line',
                'down the 3b line',
                'down the rf line',
                'down the lf line',
                'through the left side',
                'through the right side'
            ],
            'fly': [
                'fly',
                'flied',
                'homered',
                'to rf',
                'to lf',
                'to cf',
            ],
            'lined': [
                'lined',
                'doubled down',
                'doubled to'
            ],
            'popped': [
                'popped',
                'fouled out',
                'fouled into'
            ]
        }

        self.df['processed_desc'] = self.df['description'].str.lower(
        ).str.split('3a').str[0]
        descriptions = self.df['processed_desc']
        batter_hand = self.df['batter_hand']
        pitcher_hand = self.df['pitcher_hand']

        self.df['is_pull'] = False
        self.df['is_oppo'] = False
        self.df['is_middle'] = False

        for direction, pattern_list in pull_oppo_patterns.items():
            mask = descriptions.str.contains(
                '|'.join(pattern_list), regex=True, na=False, case=False)

            if direction == 'right':
                pull_condition = mask & ((batter_hand == 'L') | (
                    (batter_hand == 'S') & (pitcher_hand == 'R')))
                oppo_condition = mask & ((batter_hand == 'R') | (
                    (batter_hand == 'S') & (pitcher_hand == 'L')))

                self.df.loc[pull_condition, 'is_pull'] = True
                self.df.loc[oppo_condition, 'is_oppo'] = True

            elif direction == 'left':
                pull_condition = mask & ((batter_hand == 'R') | (
                    (batter_hand == 'S') & (pitcher_hand == 'L')))
                oppo_condition = mask & ((batter_hand == 'L') | (
                    (batter_hand == 'S') & (pitcher_hand == 'R')))

                self.df.loc[pull_condition, 'is_pull'] = True
                self.df.loc[oppo_condition, 'is_oppo'] = True

            elif direction == 'middle':
                self.df.loc[mask, 'is_middle'] = True

        self.df['is_ground'] = False
        self.df['is_fly'] = False
        self.df['is_lined'] = False
        self.df['is_popped'] = False

        for hit_type, pattern_list in hit_type_patterns.items():
            mask = descriptions.str.contains(
                '|'.join(pattern_list), regex=True, na=False, case=False)

            if hit_type == 'ground':
                self.df.loc[mask, 'is_ground'] = True
            elif hit_type == 'fly':
                self.df.loc[mask, 'is_fly'] = True
            elif hit_type == 'lined':
                self.df.loc[mask, 'is_lined'] = True
            elif hit_type == 'popped':
                self.df.loc[mask, 'is_popped'] = True

        self.df['hit_type_sum'] = (
            self.df['is_ground'].astype(int) +
            self.df['is_fly'].astype(int) +
            self.df['is_lined'].astype(int) +
            self.df['is_popped'].astype(int)
        )

        hit_type_priority = ['is_ground', 'is_lined', 'is_fly', 'is_popped']

        multiple_hit_types = self.df['hit_type_sum'] > 1
        if multiple_hit_types.any():
            for hit_type in hit_type_priority:
                mask = multiple_hit_types & self.df[hit_type]
                if mask.any():
                    for ht in hit_type_priority:
                        self.df.loc[mask, ht] = False
                    self.df.loc[mask, hit_type] = True

        self.df.drop(['processed_desc'], axis=1, inplace=True)

    def get_rolling_leaderboard(self, is_pitcher=False):
        pbp = self.original_df.copy()

        pbp['date'] = pd.to_datetime(pbp['date'])
        if is_pitcher:
            pbp = pbp.dropna(subset=['pitcher_id', 'woba'])
            pbp = pbp.sort_values(['pitcher_id', 'date'])
        else:
            pbp = pbp.dropna(subset=['batter_id', 'woba'])
            pbp = pbp.sort_values(['batter_id', 'date'])

        all_results = None

        for window in [25, 50, 100]:
            results = []
            if is_pitcher:
                for pitcher, group in pbp.groupby('pitcher_id'):
                    if len(group) < 2 * window:
                        continue
                    group = group.copy()
                    group['rolling_woba_now'] = group['woba'].rolling(
                        window).mean()
                    group['rolling_woba_then'] = group['rolling_woba_now'].shift(
                        window)
                    valid = group.dropna(
                        subset=['rolling_woba_now', 'rolling_woba_then'])

                    if len(valid) > 0:
                        latest = valid.iloc[-1]
                        results.append({
                            'pitcher_id': pitcher,
                            f'{window}_then': latest['rolling_woba_then'],
                            f'{window}_now': latest['rolling_woba_now'],
                            f'{window}_delta': latest['rolling_woba_now'] - latest['rolling_woba_then']
                        })
            else:
                for batter, group in pbp.groupby('batter_id'):
                    if len(group) < 2 * window:
                        continue
                    group = group.copy()
                    group['rolling_woba_now'] = group['woba'].rolling(
                        window).mean()
                    group['rolling_woba_then'] = group['rolling_woba_now'].shift(
                        window)
                    valid = group.dropna(
                        subset=['rolling_woba_now', 'rolling_woba_then'])

                    if len(valid) > 0:
                        latest = valid.iloc[-1]
                        results.append({
                            'batter_id': batter,
                            f'{window}_then': latest['rolling_woba_then'],
                            f'{window}_now': latest['rolling_woba_now'],
                            f'{window}_delta': latest['rolling_woba_now'] - latest['rolling_woba_then']
                        })

            window_df = pd.DataFrame(results)

            if all_results is None:
                all_results = window_df
            else:
                window_columns = [
                    col for col in window_df.columns if str(window) in col]
                if is_pitcher:
                    window_columns.append('pitcher_id')
                    all_results = pd.merge(
                        all_results, window_df[window_columns], on='pitcher_id', how='outer')
                else:
                    window_columns.append('batter_id')
                    all_results = pd.merge(
                        all_results, window_df[window_columns], on='batter_id', how='outer')

        if all_results is None:
            return pd.DataFrame()

        return all_results

    def calculate_metrics(self, group):
        if self.weights is None:
            self.load_weights()

        events = {
            event: (group['event_cd'] == code).sum()
            if not isinstance(code, list)
            else group['event_cd'].isin(code).sum()
            for event, code in self.EVENT_CODES.items()
        }

        sf = (group['sf_fl'] == 1).sum()
        ab = (events['SINGLE'] + events['DOUBLE'] + events['TRIPLE'] +
              events['HOME_RUN'] + events['OUT'] + events['ERROR'])
        pa = ab + events['WALK'] + sf + events['HBP']

        if pa == 0:
            return pd.Series({
                'woba': np.nan,
                'ba': np.nan,
                'pa': 0,
                're24': 0,
                'slg_pct': np.nan,
                'ob_pct': np.nan
            })

        hits = (events['SINGLE'] + events['DOUBLE'] +
                events['TRIPLE'] + events['HOME_RUN'])
        ba = hits / ab if ab > 0 else np.nan

        rea = group['rea'].sum()

        woba_numerator = (
            self.weights.get('walk', 0) * events['WALK'] +
            self.weights.get('hit_by_pitch', 0) * events['HBP'] +
            self.weights.get('single', 0) * events['SINGLE'] +
            self.weights.get('double', 0) * events['DOUBLE'] +
            self.weights.get('triple', 0) * events['TRIPLE'] +
            self.weights.get('home_run', 0) * events['HOME_RUN']
        )

        woba = woba_numerator / \
            (ab + events['WALK'] + sf + events['HBP'])
        slg = (events['SINGLE'] + 2 * events['DOUBLE'] + 3 *
               events['TRIPLE'] + 4 * events['HOME_RUN']) / ab if ab > 0 else np.nan
        obp = (hits + events['WALK'] + sf + events['HBP']) / \
            (ab + events['WALK'] + sf + events['HBP'])

        return pd.Series({
            'woba': woba,
            'ba': ba,
            'pa': pa,
            're24': rea,
            'slg_pct': slg,
            'ob_pct': obp
        })

    def analyze_situations(self):
        if self.situations is None:
            self.prepare_situations()

        situations_list = [
            ('risp', self.situations[self.situations['risp_fl'] == 1]),
            ('runners_on', self.situations[self.situations['ro_fl'] == 1]),
            ('high_leverage',
             self.situations[self.situations['li_hi_fl'] == 1]),
            ('low_leverage',
             self.situations[self.situations['li_lo_fl'] == 1]),
            ('overall', self.situations)
        ]

        results = []
        for name, data in situations_list:
            grouped = (data.groupby(['batter_id', 'batter_standardized', 'bat_team'])
                       .apply(self.calculate_metrics, include_groups=False)
                       .reset_index())
            grouped['situation'] = name
            results.append(grouped)

        return pd.concat(results, axis=0).reset_index(drop=True)

    def analyze_situations_pitcher(self):
        if self.situations is None:
            self.prepare_situations()

        situations_list = [
            ('risp', self.situations[self.situations['risp_fl'] == 1]),
            ('runners_on', self.situations[self.situations['ro_fl'] == 1]),
            ('high_leverage',
             self.situations[self.situations['li_hi_fl'] == 1]),
            ('low_leverage',
             self.situations[self.situations['li_lo_fl'] == 1]),
            ('overall', self.situations)
        ]

        results = []
        for name, data in situations_list:
            grouped = (data.groupby(['pitcher_id', 'pitcher_standardized', 'pitch_team'])
                       .apply(self.calculate_metrics, include_groups=False)
                       .reset_index())
            grouped['situation'] = name
            results.append(grouped)

        return pd.concat(results, axis=0).reset_index(drop=True)

    def get_pivot_results(self):
        final_df = self.analyze_situations()

        pivot = final_df.pivot(
            index=['batter_id', 'batter_standardized', 'bat_team'],
            columns='situation',
            values=['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
        )

        pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]
        return pivot.reset_index()

    def get_pivot_results_pitcher(self):
        final_df = self.analyze_situations_pitcher()

        pivot = final_df.pivot(
            index=['pitcher_id', 'pitcher_standardized', 'pitch_team'],
            columns='situation',
            values=['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
        )

        pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]
        return pivot.reset_index()

    def calc_batted_ball_stats(self):
        self.prepare_batted_ball()
        stats = self.df.groupby('batter_id').agg({
            'batter_standardized': 'first',
            'bat_team': 'first',
            'batter_hand': 'first',
            'description': 'count',
            'is_pull': lambda x: (x).sum(),
            'is_oppo': lambda x: (x).sum(),
            'is_middle': lambda x: (x == True).sum(),
            'is_ground': lambda x: (x).sum(),
            'is_fly': lambda x: (x).sum(),
            'is_lined': lambda x: (x).sum(),
            'is_popped': lambda x: (x).sum(),
        })

        total_bb = stats['is_ground'] + stats['is_fly'] + \
            stats['is_lined'] + stats['is_popped']
        total_dir = stats['is_pull'] + stats['is_oppo'] + stats['is_middle']

        stats['pull_pct'] = (stats['is_pull'] / total_dir) * 100
        stats['oppo_pct'] = (stats['is_oppo'] / total_dir) * 100
        stats['middle_pct'] = (stats['is_middle'] / total_dir) * 100
        stats['gb_pct'] = (stats['is_ground'] / total_bb) * 100
        stats['fb_pct'] = (stats['is_fly'] / total_bb) * 100
        stats['ld_pct'] = (stats['is_lined'] / total_bb) * 100
        stats['pop_pct'] = (stats['is_popped'] / total_bb) * 100

        pull_air = self.df[((self.df['is_fly']) | (self.df['is_lined'])) & self.df['is_pull']].groupby(
            'batter_id').size()
        oppo_gb = self.df[(self.df['is_ground']) & self.df['is_oppo']].groupby(
            'batter_id').size()
        stats['pull_air_pct'] = (pull_air / total_dir * 100).fillna(0)
        stats['oppo_gb_pct'] = (oppo_gb / total_dir * 100).fillna(0)
        stats = stats.reset_index()

        stats.loc[stats['batter_hand'].isna(), ['pull_pct', 'oppo_pct',
                                                'pull_air_pct', 'oppo_gb_pct', 'middle_pct']] = np.nan

        return stats[[
            'batter_standardized', 'bat_team', 'batter_id', 'batter_hand', 'description',
            'pull_pct', 'middle_pct', 'oppo_pct',
            'gb_pct', 'fb_pct', 'ld_pct', 'pop_pct',
            'pull_air_pct', 'oppo_gb_pct'
        ]].rename(columns={'description': 'count'}).sort_values('count', ascending=False)

    def analyze_splits(self):
        splits_list = [
            ('vs_lhp',
             self.original_df[(self.original_df['pitcher_hand'] == 'L') &
                              (~self.original_df['bat_order'].isna())]),
            ('vs_rhp',
             self.original_df[(self.original_df['pitcher_hand'] == 'R') &
                              (~self.original_df['bat_order'].isna())]),
            ('overall',
             self.original_df[~self.original_df['bat_order'].isna()]),
        ]

        results = []
        for name, data in splits_list:
            if not data.empty:
                grouped = (data.groupby(['batter_id', 'batter_standardized', 'bat_team'])
                           .apply(self.calculate_metrics, include_groups=False)
                           .reset_index())
                grouped['split'] = name
                results.append(grouped)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, axis=0).reset_index(drop=True)

    def get_splits_results(self):
        final_df = self.analyze_splits()

        if final_df.empty:
            cols = ['batter_id', 'batter_standardized', 'bat_team']
            cols.extend([f"{stat}_vs {hand}" for stat in ['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
                        for hand in ['lhp', 'rhp']])
            return pd.DataFrame(columns=cols)

        pivot = final_df.pivot(
            index=['batter_id', 'batter_standardized', 'bat_team'],
            columns='split',
            values=['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
        )

        pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]
        return pivot.reset_index()

    def analyze_splits_pitcher(self):
        splits_list = [
            ('vs_lhh',
             self.original_df[((self.original_df['batter_hand'] == 'L') |
                              ((self.original_df['pitcher_hand'] == 'R') &
                               (self.original_df['batter_hand'] == 'S'))) &
                              (~self.original_df['bat_order'].isna())]),
            ('vs_rhh',
             self.original_df[((self.original_df['batter_hand'] == 'R') |
                              ((self.original_df['pitcher_hand'] == 'L') &
                               (self.original_df['batter_hand'] == 'S'))) &
                              (~self.original_df['bat_order'].isna())]),
            ('overall',
             self.original_df[~self.original_df['bat_order'].isna()]),
        ]

        results = []
        for name, data in splits_list:
            if not data.empty:
                grouped = (data.groupby(['pitcher_id', 'pitcher_standardized', 'pitch_team'])
                           .apply(self.calculate_metrics, include_groups=False)
                           .reset_index())
                grouped['Split'] = name
                results.append(grouped)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, axis=0).reset_index(drop=True)

    def get_splits_results_pitcher(self):
        final_df = self.analyze_splits_pitcher()

        if final_df.empty:
            cols = ['pitcher_id', 'pitcher_standardized', 'pitch_team']
            cols.extend([f"{stat}_vs_{hand}" for stat in ['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
                        for hand in ['lhp', 'rhp']])
            return pd.DataFrame(columns=cols)

        pivot = final_df.pivot(
            index=['pitcher_id', 'pitcher_standardized', 'pitch_team'],
            columns='Split',
            values=['woba', 'ba', 'pa', 're24', 'ob_pct', 'slg_pct']
        )

        pivot.columns = [f"{stat}_{sit}" for stat, sit in pivot.columns]
        return pivot.reset_index()


def run_analysis(pbp_df, year, division, data_path):
    try:
        analytics = BaseballAnalytics(pbp_df, year, division, data_path)
        situational = analytics.get_pivot_results()
        batted_ball = analytics.calc_batted_ball_stats()
        splits = analytics.get_splits_results()
        splits_pitcher = analytics.get_splits_results_pitcher()
        situational_pitcher = analytics.get_pivot_results_pitcher()
        rolling = analytics.get_rolling_leaderboard()
        rolling_pitcher = analytics.get_rolling_leaderboard(is_pitcher=True)
        return situational, batted_ball, splits, splits_pitcher, situational_pitcher, rolling, rolling_pitcher
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        return None, None, None, None, None, None, None


def get_data(year, division, data_dir):
    pbp_df = pd.read_csv(
        data_dir / f'pbp/d{division}_parsed_pbp_new_{year}.csv', dtype={'player_id': str, 'batter_id': str, 'pitcher_id': str})
    bat_war = pd.read_csv(
        data_dir / f'war/d{division}_batting_war_{year}.csv', dtype={'player_id': str}).rename(columns={'war': 'bwar'})
    rosters = pd.read_csv(
        data_dir / f'rosters/d{division}_rosters_{year}.csv', dtype={'player_id': str})
    pitch_war = pd.read_csv(
        data_dir / f'war/d{division}_pitching_war_{year}.csv', dtype={'player_id': str})

    bat_war['b_t'] = bat_war['b_t'].replace('0', np.nan).astype(str)
    pitch_war['b_t'] = pitch_war['b_t'].replace('0', np.nan).astype(str)

    roster_b_map = rosters.set_index('player_id')['bats'].to_dict()
    roster_p_map = rosters.set_index('player_id')['throws'].to_dict()
    batting_map = bat_war.set_index(
        'player_id')['b_t'].str.split('/').str[0].to_dict()
    pitching_map = pitch_war.set_index(
        'player_id')['b_t'].str.split('/').str[1].to_dict()

    combined_b_map = {id: batting_map.get(id) or roster_b_map.get(id)
                      for id in set(roster_b_map) | set(batting_map)}
    combined_p_map = {id: pitching_map.get(id) or roster_p_map.get(id)
                      for id in set(roster_p_map) | set(pitching_map)}

    combined_b_map = {k: standardize_hand(v)
                      for k, v in combined_b_map.items()}
    combined_p_map = {k: standardize_hand(v)
                      for k, v in combined_p_map.items()}

    pbp_df['batter_hand'] = pbp_df['batter_id'].map(combined_b_map)
    pbp_df['pitcher_hand'] = pbp_df['pitcher_id'].map(combined_p_map)

    return pbp_df, bat_war


def standardize_hand(x):
    if pd.isna(x) or x == '0':
        return np.nan
    x = str(x).upper()
    if x in ['L', 'LEFT']:
        return 'L'
    elif x in ['R', 'RIGHT']:
        return 'R'
    elif x in ['S', 'SWITCH', 'B']:
        return 'S'
    return np.nan


def main(data_dir, year, divisions=None):
    data_dir = Path(data_dir)
    if divisions is None:
        divisions = [1, 2, 3]
    leaderboards_dir = data_dir / 'leaderboards'
    leaderboards_dir.mkdir(exist_ok=True)

    all_situational = []
    all_baserunning = []
    all_batted_ball = []
    all_splits = []
    all_splits_pitcher = []
    all_situational_pitcher = []
    all_rolling = []
    all_rolling_pitcher = []

    data_sets = {
        'situational': (all_situational, ['batter_id', 'batter_standardized', 'bat_team', 'year', 'division']),
        'baserunning': (all_baserunning, ['player_id', 'Team', 'year', 'division']),
        'batted_ball': (all_batted_ball, ['batter_id', 'batter_standardized', 'bat_team', 'year', 'division']),
        'splits': (all_splits, ['batter_id', 'batter_standardized', 'bat_team', 'year', 'division']),
        'splits_pitcher': (all_splits_pitcher, ['pitcher_id', 'pitcher_standardized', 'pitch_team', 'year', 'division']),
        'situational_pitcher': (all_situational_pitcher, ['pitcher_id', 'pitcher_standardized', 'pitch_team', 'year', 'division']),
        'rolling': (all_rolling, ['batter_id', 'year', 'division']),
        'rolling_pitcher': (all_rolling_pitcher, ['pitcher_id', 'year', 'division'])
    }

    for name, (data_list, _) in data_sets.items():
        existing_file = leaderboards_dir / f'{name}.csv'
        if existing_file.exists():
            try:
                existing_df = pd.read_csv(existing_file)
                if 'year' in existing_df.columns:
                    existing_df = existing_df[existing_df['year'] != int(year)]
                if not existing_df.empty:
                    data_list.append(existing_df)
                    print(
                        f"Loaded existing {name} data with {len(existing_df)} records")
            except Exception as e:
                print(f"Error loading existing {name} data: {e}")

    for division in divisions:
        print(f'Processing data for {year} D{division}')
        try:
            pbp_df, bat_war = get_data(year, division, data_dir)
            situational, batted_ball, splits, splits_pitcher, situational_pitcher, rolling, rolling_pitcher = run_analysis(
                pbp_df, year, division, data_dir)

            if all(result is not None and not (isinstance(result, pd.DataFrame) and result.empty) for result in [situational, batted_ball, splits, splits_pitcher, situational_pitcher, rolling]):
                for df, lst in [(situational, all_situational),
                                (batted_ball, all_batted_ball),
                                (splits, all_splits),
                                (splits_pitcher, all_splits_pitcher),
                                (situational_pitcher, all_situational_pitcher),
                                (rolling, all_rolling), (rolling_pitcher, all_rolling_pitcher)]:

                    df['year'] = int(year)
                    df['division'] = division
                    lst.append(df)

                baserun = bat_war[['player_name', 'player_id', 'team_name', 'conference', 'sb%', 'wsb',
                                  'wgdp', 'wteb', 'baserunning', 'ebt', 'outs_ob', 'opportunities',
                                   'cs', 'sb', 'picked']].sort_values('baserunning')
                baserun['year'] = int(year)
                baserun['division'] = division
                all_baserunning.append(baserun)

        except Exception as e:
            print(f"Error processing {year} D{division}: {e}")
            continue

    for name, (data_list, dedup_cols) in data_sets.items():
        if data_list:
            try:
                combined_df = pd.concat(data_list, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=dedup_cols)
                output_file = leaderboards_dir / f'{name}.csv'
                combined_df.to_csv(output_file, index=False)
                print(
                    f"Saved {name} data with {len(combined_df)} records to {output_file}")
            except Exception as e:
                print(f"Error saving {name} data: {e}")

    print("Processing complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--year', required=True, type=int)
    parser.add_argument('--divisions', nargs='+', type=int, default=[1, 2, 3],
                        help='Divisions to process (default: 1 2 3)')
    args = parser.parse_args()

    main(args.data_dir, args.year, args.divisions)
