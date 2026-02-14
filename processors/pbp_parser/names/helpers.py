import re

import pandas as pd


def format_name(name: str | None) -> str:
    if pd.isna(name):
        return name
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name.strip()


def normalize_name(name: str | None) -> str:
    if pd.isna(name) or not name:
        return ""
    name = re.sub(r"[^\w\s]", "", name.lower())
    return " ".join(name.split())


def parse_name_parts(name: str) -> tuple[str, str, str | None]:
    if pd.isna(name) or not name:
        return "", "", None

    name = name.strip()

    num_match = re.match(r"^#?(\d+)\s+(.+)$", name)
    number = num_match.group(1) if num_match else None
    if num_match:
        name = num_match.group(2)

    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first = parts[1].strip() if len(parts) > 1 else ""
    else:
        parts = name.split()
        if len(parts) == 0:
            return "", "", number
        elif len(parts) == 1:
            word = parts[0]
            if re.match(r"^[A-Z]\.$", word) or len(word) <= 2:
                return word.rstrip("."), "", number
            return "", word, number
        else:
            first = parts[0]
            last = " ".join(parts[1:])

    first = first.rstrip(".")
    last = last.rstrip(".")

    return first, last, number


def generate_name_variations(first: str, last: str, number: str | None = None) -> list[str]:
    variations = []

    if not first and not last:
        return variations

    fn = first.strip()
    ln = last.strip()
    f_init = fn[0] if fn else ""
    l_init = ln[0] if ln else ""
    f_norm = normalize_name(fn)
    l_norm = normalize_name(ln)

    if fn and ln:
        variations.append(f"{fn} {ln}")
        variations.append(f"{fn.lower()} {ln.lower()}")
        variations.append(f"{ln}, {fn}")
        variations.append(f"{ln.lower()}, {fn.lower()}")

    if f_init and ln:
        variations.append(f"{f_init}. {ln}")
        variations.append(f"{f_init} {ln}")
        variations.append(f"{f_init.lower()}. {ln.lower()}")
        variations.append(f"{f_init.lower()} {ln.lower()}")

    if fn and l_init:
        variations.append(f"{fn} {l_init}.")
        variations.append(f"{fn} {l_init}")
        variations.append(f"{fn.lower()} {l_init.lower()}.")

    if f_init and l_init:
        variations.append(f"{f_init}. {l_init}.")
        variations.append(f"{f_init}.{l_init}.")
        variations.append(f"{f_init}{l_init}")

    if ln:
        variations.append(ln)
        variations.append(ln.lower())

    if fn and len(fn) >= 3 and ln and len(ln) >= 3:
        variations.append(f"{fn[:3]} {ln[:3]}")
        variations.append(f"{fn[:3].lower()} {ln[:3].lower()}")

    if number:
        if ln:
            variations.append(f"#{number} {ln}")
            variations.append(f"{number} {ln}")
        variations.append(f"#{number}")
        variations.append(number)

    if f_norm and l_norm:
        variations.append(f"{f_norm} {l_norm}")
    elif l_norm:
        variations.append(l_norm)

    return list(dict.fromkeys(variations))


def build_name_lookup(
    roster_df: pd.DataFrame,
    team_col: str = "team_name",
    name_col: str = "player_name",
    id_col: str = "player_id",
    number_col: str = "number",
) -> dict[str, dict[str, tuple[str, str]]]:
    lookup = {}

    has_number = number_col in roster_df.columns

    for _, row in roster_df.iterrows():
        team = row[team_col]
        canonical_name = row[name_col]
        player_id = row[id_col]
        number = str(row[number_col]) if has_number and pd.notna(row.get(number_col)) else None

        if pd.isna(team) or pd.isna(canonical_name) or pd.isna(player_id):
            continue

        if team not in lookup:
            lookup[team] = {}

        formatted = format_name(canonical_name)
        first, last, parsed_num = parse_name_parts(formatted)

        num_to_use = number or parsed_num

        variations = generate_name_variations(first, last, num_to_use)
        variations.append(canonical_name)
        variations.append(formatted)
        if "player_name_norm" in roster_df.columns and pd.notna(row.get("player_name_norm")):
            variations.append(row["player_name_norm"])

        for var in variations:
            if var and var.strip():
                var_key = var.strip().lower()
                if var_key not in lookup[team]:
                    lookup[team][var_key] = (canonical_name, player_id)

    return lookup


def match_name(
    name: str, team: str, lookup: dict, threshold: int = 70
) -> tuple[str | None, str | None]:
    from rapidfuzz import fuzz, process

    if pd.isna(name) or pd.isna(team) or not name or not team:
        return None, None

    team_lookup = lookup.get(team)
    if not team_lookup:
        return None, None

    name_lower = name.strip().lower()

    if name_lower in team_lookup:
        return team_lookup[name_lower]

    name_norm = normalize_name(name)
    if name_norm in team_lookup:
        return team_lookup[name_norm]

    first, last, number = parse_name_parts(name)
    test_variations = generate_name_variations(first, last, number)

    for var in test_variations:
        var_key = var.strip().lower()
        if var_key in team_lookup:
            return team_lookup[var_key]

    all_variations = list(team_lookup.keys())

    match = process.extractOne(
        name_lower, all_variations, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
    )

    if match:
        return team_lookup[match[0]]

    if last:
        match = process.extractOne(
            last.lower(), all_variations, scorer=fuzz.partial_ratio, score_cutoff=85
        )
        if match:
            return team_lookup[match[0]]

    return None, None
