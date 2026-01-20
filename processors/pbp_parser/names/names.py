import numpy as np
import pandas as pd

from .helpers import (
    build_name_lookup,
    format_name,
    match_name,
)


def prepare_lineups(batting_lineups: pd.DataFrame, pitching_lineups: pd.DataFrame,
                    roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster_id_map = roster.set_index('ncaa_id')['player_id'].to_dict()

    batting = batting_lineups.copy()
    batting['player_id'] = batting['ncaa_id'].map(roster_id_map)
    batting['player_name'] = batting['player_name'].apply(format_name)

    batting['team_id'] = batting.groupby('contest_id')['team_id'].transform(
        lambda x: x.ffill().bfill()
    )
    batting['team_id'] = batting['team_id'].astype('Int64')

    pitching = pitching_lineups.copy()
    pitching['player_id'] = pitching['ncaa_id'].map(roster_id_map)
    pitching['player_name'] = pitching['player_name'].apply(format_name)

    pitching['team_id'] = pitching.groupby('contest_id')['team_id'].transform(
        lambda x: x.ffill().bfill()
    )
    pitching['team_id'] = pitching['team_id'].astype('Int64')

    if 'pitch_order' not in pitching.columns:
        pitching['pitch_order'] = pitching.groupby(['contest_id', 'team_id']).cumcount()

    return batting, pitching


def fill_pitcher_names(pbp_df: pd.DataFrame, pitching_lineups: pd.DataFrame) -> pd.DataFrame:
    df = pbp_df.copy()

    pitching = pitching_lineups.sort_values(['contest_id', 'team_id', 'pitch_order'])

    pitcher_queue = {}
    for (contest_id, team_id), group in pitching.groupby(['contest_id', 'team_id']):
        pitcher_queue[(contest_id, team_id)] = list(zip(
            group['player_name'].tolist(),
            group['player_id'].tolist(),
            strict=True
        ))

    pitcher_index = {}

    pitcher_names = np.empty(len(df), dtype=object)
    pitcher_ids = np.empty(len(df), dtype=object)

    current_pitcher = {}
    current_pitcher_id = {}

    for i, row in df.iterrows():
        contest_id = row['contest_id']
        pitch_team_id = row.get('pitch_team_id')

        if pd.isna(pitch_team_id):
            pitcher_names[i] = current_pitcher.get((contest_id, None), "")
            pitcher_ids[i] = current_pitcher_id.get((contest_id, None), None)
            continue

        key = (contest_id, pitch_team_id)

        is_pitcher_sub = row.get('sub_pos') == 'p' and row.get('sub_fl') == 1

        if is_pitcher_sub or key not in current_pitcher:
            if key not in pitcher_index:
                pitcher_index[key] = 0
            else:
                pitcher_index[key] += 1

            queue = pitcher_queue.get(key, [])
            idx = pitcher_index[key]

            if idx < len(queue):
                current_pitcher[key] = queue[idx][0]
                current_pitcher_id[key] = queue[idx][1]
            else:
                sub_in = row.get('sub_in', '')
                current_pitcher[key] = sub_in if is_pitcher_sub and sub_in else current_pitcher.get(key, "")
                current_pitcher_id[key] = current_pitcher_id.get(key, None)

        pitcher_names[i] = current_pitcher.get(key, "")
        pitcher_ids[i] = current_pitcher_id.get(key, None)

    df['pitcher_name'] = pitcher_names
    df['pitcher_id'] = pitcher_ids

    return df


def build_game_lineup_lookup(batting_lineups: pd.DataFrame) -> dict[tuple, dict]:
    game_lookups = {}

    for (contest_id, team_id), group in batting_lineups.groupby(['contest_id', 'team_id']):
        if pd.isna(team_id):
            continue

        lookup = {}
        for _, row in group.iterrows():
            name = row['player_name']
            player_id = row['player_id']
            ncaa_id = row['ncaa_id']

            if pd.isna(name):
                continue

            name_lower = name.strip().lower()
            lookup[name_lower] = (name, player_id, ncaa_id)

            from .helpers import generate_name_variations, parse_name_parts
            first, last, num = parse_name_parts(name)
            for var in generate_name_variations(first, last, num):
                var_key = var.strip().lower()
                if var_key not in lookup:
                    lookup[var_key] = (name, player_id, ncaa_id)

        game_lookups[(contest_id, team_id)] = lookup

    return game_lookups


def match_player_in_game(name: str, contest_id, team_id, game_lookups: dict,
                         full_lookup: dict, threshold: int = 70) -> tuple[str, str | None]:
    if pd.isna(name) or not name:
        return "", None

    original_name = name.strip()

    if pd.isna(team_id):
        return original_name, None

    game_key = (contest_id, team_id)
    game_lookup = game_lookups.get(game_key, {})

    name_lower = original_name.lower()
    if name_lower in game_lookup:
        canonical, player_id, _ = game_lookup[name_lower]
        return canonical, player_id

    from .helpers import generate_name_variations, normalize_name, parse_name_parts

    name_norm = normalize_name(name)
    if name_norm in game_lookup:
        canonical, player_id, _ = game_lookup[name_norm]
        return canonical, player_id

    first, last, number = parse_name_parts(name)
    for var in generate_name_variations(first, last, number):
        var_key = var.strip().lower()
        if var_key in game_lookup:
            canonical, player_id, _ = game_lookup[var_key]
            return canonical, player_id

    if game_lookup:
        from rapidfuzz import fuzz, process
        all_variations = list(game_lookup.keys())
        match = process.extractOne(
            name_lower,
            all_variations,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )
        if match:
            canonical, player_id, _ = game_lookup[match[0]]
            return canonical, player_id

    matched_name, matched_id = match_name(name, team_id, full_lookup, threshold)
    if matched_name:
        return matched_name, matched_id

    return original_name, None


def match_players_with_lineups(df: pd.DataFrame, player_col: str, team_col: str,
                                contest_col: str, id_col: str, name_col: str,
                                game_lookups: dict, full_lookup: dict,
                                threshold: int = 70):
    if player_col not in df.columns:
        return

    names_out = []
    ids_out = []

    for name, contest_id, team in zip(df[player_col], df[contest_col], df[team_col], strict=True):
        matched_name, player_id = match_player_in_game(
            name, contest_id, team, game_lookups, full_lookup, threshold
        )
        names_out.append(matched_name)
        ids_out.append(player_id)

    df[name_col] = names_out
    df[id_col] = ids_out


def standardize_names(pbp_df: pd.DataFrame, batting_lineups: pd.DataFrame,
                      pitching_lineups: pd.DataFrame, roster: pd.DataFrame,
                      threshold: int = 70) -> pd.DataFrame:
    batting, pitching = prepare_lineups(batting_lineups, pitching_lineups, roster)

    roster = roster.copy()
    roster['player_name'] = roster['player_name'].apply(format_name)
    roster['team_id'] = roster['team_id'].astype('Int64')
    full_lookup = build_name_lookup(roster, team_col='team_id')

    game_lookups = build_game_lineup_lookup(batting)

    pbp_df = pbp_df.copy()

    for col in ['batter_name', 'r1_name', 'r2_name', 'r3_name', 'player_of_interest']:
        if col in pbp_df.columns:
            pbp_df[col] = pbp_df[col].apply(format_name)

    pbp_df = fill_pitcher_names(pbp_df, pitching)

    match_players_with_lineups(
        pbp_df, 'batter_name', 'bat_team_id', 'contest_id',
        'batter_id', 'batter_name', game_lookups, full_lookup, threshold
    )
    match_players_with_lineups(
        pbp_df, 'r1_name', 'bat_team_id', 'contest_id',
        'r1_id', 'r1_name', game_lookups, full_lookup, threshold
    )
    match_players_with_lineups(
        pbp_df, 'r2_name', 'bat_team_id', 'contest_id',
        'r2_id', 'r2_name', game_lookups, full_lookup, threshold
    )
    match_players_with_lineups(
        pbp_df, 'r3_name', 'bat_team_id', 'contest_id',
        'r3_id', 'r3_name', game_lookups, full_lookup, threshold
    )
    match_players_with_lineups(
        pbp_df, 'player_of_interest', 'bat_team_id', 'contest_id',
        'player_id', 'player_name', game_lookups, full_lookup, threshold
    )

    return pbp_df
