import numpy as np
import pandas as pd

from processors.pbp_parser.constants import BattedBallType, EventType

FIELD_PATTERNS = {
    'to_lf': r'to left|to lf|left field|lf line|by lf',
    'to_cf': r'to center|to cf|center field|by cf|to left center|to right center',
    'to_rf': r'to right|to rf|right field|rf line|by rf',
    'to_3b': r'to 3b|to third|third base|3b line|by 3b|3b to',
    'to_ss': r'ss to|to ss|to short|shortstop|by ss',
    'up_middle': r'up the middle|to pitcher|to p|to c|by p|by c|to catcher',
    'to_2b': r'2b to|to 2b|to second|second base|by 2b',
    'to_1b': r'to 1b|to first|first base|1b line|by 1b|1b to',
}

def add_batted_ball_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["batter_id"])

    desc = df["play_description"].fillna("").astype(str).str.lower()

    batter_hand = df.get("batter_hand", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
    pitcher_hand = df.get("pitcher_hand", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)

    is_lr = batter_hand.isin(["L", "R"])
    is_switch = batter_hand.isin(["S", "B"])
    has_pitch = pitcher_hand.isin(["L", "R"])

    right_pattern = "|".join([FIELD_PATTERNS["to_rf"], FIELD_PATTERNS["to_1b"], FIELD_PATTERNS["to_2b"]])
    left_pattern = "|".join([FIELD_PATTERNS["to_lf"], FIELD_PATTERNS["to_3b"], FIELD_PATTERNS["to_ss"]])
    middle_pattern = "|".join([FIELD_PATTERNS["to_cf"], FIELD_PATTERNS["up_middle"]])

    to_right = desc.str.contains(right_pattern, regex=True, na=False)
    to_left = desc.str.contains(left_pattern, regex=True, na=False)
    to_middle = desc.str.contains(middle_pattern, regex=True, na=False)

    df["is_pull"] = np.nan
    df["is_oppo"] = np.nan
    df["is_middle"] = np.nan

    pull_lr = (to_right & (batter_hand == "L")) | (to_left & (batter_hand == "R"))
    oppo_lr = (to_right & (batter_hand == "R")) | (to_left & (batter_hand == "L"))

    pull_sw = has_pitch & (
        (to_right & (pitcher_hand == "R")) | (to_left & (pitcher_hand == "L"))
    )
    oppo_sw = has_pitch & (
        (to_right & (pitcher_hand == "L")) | (to_left & (pitcher_hand == "R"))
    )

    df.loc[is_lr & pull_lr, "is_pull"] = 1.0
    df.loc[is_lr & oppo_lr, "is_oppo"] = 1.0

    df.loc[is_switch & pull_sw, "is_pull"] = 1.0
    df.loc[is_switch & oppo_sw, "is_oppo"] = 1.0

    df.loc[to_middle, "is_middle"] = 1.0

    bbt = df["batted_ball_type"]
    df["is_gb"] = bbt == BattedBallType.GROUND_BALL.value
    df["is_fb"] = bbt == BattedBallType.FLY_BALL.value
    df["is_ld"] = bbt == BattedBallType.LINE_DRIVE.value
    df["is_pu"] = bbt == BattedBallType.POP_UP.value

    et = pd.to_numeric(df["event_type"], errors="coerce")
    df["is_hr"] = et == EventType.HOME_RUN.value

    return df

def calculate_batted_ball_stats(df: pd.DataFrame, type='batter') -> pd.DataFrame:
    df = add_batted_ball_flags(df)

    if type == 'batter':
        group_col = 'batter_id'
        name_col = 'batter_name'
        team_col = 'bat_team_name'
        team_id_col = 'bat_team_id'
        hand_col = 'batter_hand'
    elif type == 'batter_team':
        group_col = 'bat_team_id'
        name_col = 'bat_team_name'
    elif type == 'pitcher':
        group_col = 'pitcher_id'
        name_col = 'pitcher_name'
        team_col = 'pitch_team_name'
        team_id_col = 'pitch_team_id'
        hand_col = 'pitcher_hand'
    elif type == 'pitcher_team':
        group_col = 'pitch_team_id'
        name_col = 'pitch_team_name'

    agg_dict = {
        name_col: 'first',
        'play_description': 'count',
        'is_pull': 'sum',
        'is_oppo': 'sum',
        'is_middle': 'sum',
        'is_gb': 'sum',
        'is_fb': 'sum',
        'is_ld': 'sum',
        'is_pu': 'sum',
        'is_hr': 'sum',
    }

    if type == 'batter' or type == 'pitcher':
        agg_dict[hand_col] = 'first'
        agg_dict[team_col] = 'first'
        agg_dict[team_id_col] = 'first'

    stats = df.groupby(group_col).agg(agg_dict).rename(columns={'play_description': 'batted_balls'})

    total_bb = stats['is_gb'] + stats['is_fb'] + stats['is_ld'] + stats['is_pu']
    total_dir = stats['is_pull'] + stats['is_oppo'] + stats['is_middle']

    stats['pull_pct'] = np.where(total_dir > 0, stats['is_pull'] / total_dir * 100, np.nan)
    stats['oppo_pct'] = np.where(total_dir > 0, stats['is_oppo'] / total_dir * 100, np.nan)
    stats['middle_pct'] = np.where(total_dir > 0, stats['is_middle'] / total_dir * 100, np.nan)

    stats['gb_pct'] = np.where(total_bb > 0, stats['is_gb'] / total_bb * 100, np.nan)
    stats['fb_pct'] = np.where(total_bb > 0, stats['is_fb'] / total_bb * 100, np.nan)
    stats['ld_pct'] = np.where(total_bb > 0, stats['is_ld'] / total_bb * 100, np.nan)
    stats['pu_pct'] = np.where(total_bb > 0, stats['is_pu'] / total_bb * 100, np.nan)

    stats['fb_per_gb'] = np.where(stats['is_gb'] > 0, stats['is_fb'] / stats['is_gb'], np.nan)
    stats['hr_per_fb'] = np.where(stats['is_fb'] > 0, stats['is_hr'] / stats['is_fb'], np.nan)

    pull_air = df[(df['is_fb'] | df['is_ld']) & df['is_pull']].groupby(group_col).size()
    oppo_gb = df[df['is_gb'] & df['is_oppo']].groupby(group_col).size()

    stats['pull_air_pct'] = np.where(total_dir > 0, pull_air.reindex(stats.index, fill_value=0) / total_dir * 100, np.nan)
    stats['oppo_gb_pct'] = np.where(total_dir > 0, oppo_gb.reindex(stats.index, fill_value=0) / total_dir * 100, np.nan)

    stats = stats.reset_index()

    stats = stats.rename(columns={
        'batter_id': 'player_id',
        'pitcher_id': 'player_id',
        'batter_name': 'player_name',
        'pitcher_name': 'player_name',
        'bat_team_id': 'team_id',
        'pitch_team_id': 'team_id',
        'bat_team_name': 'team_name',
        'pitch_team_name': 'team_name',
        'batter_hand': 'hand',
        'pitcher_hand': 'hand',
    })

    return stats.sort_values('batted_balls', ascending=False)
