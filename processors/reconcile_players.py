import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from rapidfuzz import fuzz, process

from processors.logging_utils import div_file_prefix, get_logger

logger = get_logger(__name__)

HEADSHOT_COLS = ["player_id", "img_url", "b_t", "hometown", "high_school", "height", "weight", "pos"]


def _s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def norm(x) -> str:
    return _s(x).strip()


def clean_name(name: str) -> str:
    s = unicodedata.normalize("NFKC", _s(name))
    s = re.sub(r"^\s*(?:no\.?|number)?\s*\d{1,3}\s*[-–—.:]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*\d{1,3}\s+", "", s)
    s = re.sub(r"^\s*\d{1,3}(?=[A-Za-z])", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def clean_high_school(x: str) -> str:
    s = norm(x)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.split(
        r"\b(?:previous|prev|last)\s*school\s*:?|\b(?:previous|prev)\s*sch(?:ool)?\s*:?|\blast\s*sch(?:ool)?\s*:?",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    s = re.split(r"\bprev\.?\s*school\b\s*[:\-–—]", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.split(r"\bprevious\s*school\b\s*[:\-–—]", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.split(r"\blast\s*school\b\s*[:\-–—]", s, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"\s+", " ", s).strip(" \t\r\n-–—:;|,")


def normalize_name(name: str) -> str:
    s = _s(name)
    s = re.sub(r"^\s*\d{1,3}\s*", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def get_base_url(url: str) -> str:
    parsed = urlparse(_s(url))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_b_t(bt: str) -> str:
    s = unicodedata.normalize("NFKC", _s(bt)).upper().strip()
    if not s:
        return ""
    s = s.replace("\\", "/").replace("|", "/")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^([LRSH])/?([LRSH])$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    letters = re.findall(r"[LRSH]", s)
    if len(letters) >= 2:
        return f"{letters[0]}/{letters[1]}"
    if len(letters) == 1:
        return f"{letters[0]}/{letters[0]}"
    return ""


_POS_MAP = {
    "CATCHER": "C",
    "C": "C",
    "FIRST BASE": "1B",
    "1B": "1B",
    "SECOND BASE": "2B",
    "2B": "2B",
    "THIRD BASE": "3B",
    "3B": "3B",
    "SHORTSTOP": "SS",
    "SS": "SS",
    "LEFT FIELD": "LF",
    "LF": "LF",
    "CENTER FIELD": "CF",
    "CF": "CF",
    "RIGHT FIELD": "RF",
    "RF": "RF",
    "OUTFIELD": "OF",
    "OF": "OF",
    "INFIELD": "INF",
    "IF": "INF",
    "INF": "INF",
    "PITCHER": "P",
    "RHP": "P",
    "LHP": "P",
    "P": "P",
    "DESIGNATED HITTER": "DH",
    "DH": "DH",
    "UTILITY": "UT",
    "UTIL": "UT",
    "UT": "UT",
}


def standardize_pos(pos: str) -> str:
    s = unicodedata.normalize("NFKC", _s(pos)).upper().strip()
    if not s:
        return ""
    s = re.sub(r"[.\s]+", " ", s).strip()
    parts = re.split(r"[/,;]| OR ", s)
    out = list(dict.fromkeys(
        _POS_MAP.get(p, _POS_MAP.get(p.replace(" ", ""), p))
        for p in (p.strip() for p in parts)
        if p.strip()
    ))
    out = [o for o in out if o]
    if not out:
        return ""
    return out[0] if len(out) == 1 else "/".join(out[:3])


def load_cube_player_info(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "cube_stats" / "cube_player_info.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "cube_player_id"])
    return pd.read_csv(
        path,
        dtype={"cube_player_id": str, "player_id": str},
        usecols=["player_id", "cube_player_id", "bats", "throws", "height", "weight", "high_school"],
    )


def enrich_rosters_with_cube_info(rosters: pd.DataFrame, cube_info: pd.DataFrame) -> pd.DataFrame:
    if cube_info.empty:
        return rosters

    merged = rosters.merge(cube_info, on="player_id", how="left", suffixes=("", "_cube"))

    for col in ["bats", "throws", "height", "weight", "high_school"]:
        cube_col = f"{col}_cube"
        merged[col] = merged[col].map(norm)
        merged[cube_col] = merged[cube_col].map(norm)
        merged[col] = merged[col].where(merged[col] != "", merged[cube_col])

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_cube")], errors="ignore")

    derived = (
        merged["bats"].map(lambda x: _s(x).upper()[:1]) + "/" + merged["throws"].map(lambda x: _s(x).upper()[:1])
    ).map(normalize_b_t)
    merged["b_t"] = merged["b_t"].map(norm)
    merged["b_t"] = merged["b_t"].where(merged["b_t"] != "", derived)

    return merged


def load_rosters(data_dir: Path, year: int, division: str) -> pd.DataFrame:
    prefix = div_file_prefix(division)
    rosters = pd.read_csv(data_dir / "rosters" / f"{prefix}_rosters_{year}.csv")
    rosters["player_name_norm"] = rosters["player_name"].map(normalize_name)
    rosters["b_t"] = (
        rosters["bats"].map(lambda x: _s(x).upper()[:1]) + "/" + rosters["throws"].map(lambda x: _s(x).upper()[:1])
    ).map(normalize_b_t)
    rosters["pos"] = rosters["position"].map(standardize_pos)
    rosters["img_url"] = rosters["img_url"].map(norm)
    rosters["hometown"] = rosters["hometown"].map(norm)
    rosters["high_school"] = rosters["high_school"].map(clean_high_school)
    rosters["height"] = rosters["height"].map(norm)
    rosters["weight"] = rosters["weight"].map(norm)
    return rosters


def load_headshots(data_dir: Path, year: int) -> pd.DataFrame:
    hfiles = list((data_dir / "headshots").glob(f"*{year}*.csv"))
    if not hfiles:
        return pd.DataFrame(
            columns=["team", "year", "roster_url", "name", "number", "position",
                     "height", "weight", "class", "b_t", "hometown", "highschool",
                     "previous_school", "img_url"]
        )

    tr = pd.concat(
        [pd.read_csv(f).drop(columns=["Unnamed: 0"], errors="ignore") for f in hfiles],
        ignore_index=True,
    )

    map_path = data_dir / "team_mappings.csv"
    if map_path.exists():
        mappings = pd.read_csv(map_path)
        tr = tr.merge(
            mappings[["school_name_official", "ncaa_team_name"]],
            left_on="team",
            right_on="school_name_official",
            how="left",
        )
        tr["team"] = tr["ncaa_team_name"].fillna(tr["team"])
        tr = tr.drop(columns=["school_name_official", "ncaa_team_name"], errors="ignore")

    tr["name_clean"] = tr["name"].map(clean_name)
    tr["tr_name_norm"] = tr["name_clean"].map(normalize_name)
    tr["img_url"] = tr["img_url"].map(norm)
    tr["b_t"] = tr["b_t"].map(normalize_b_t)
    tr["pos"] = tr["position"].map(standardize_pos)
    tr["hometown"] = tr["hometown"].map(norm)
    tr["high_school"] = tr["highschool"].map(clean_high_school)
    tr["height"] = tr["height"].map(norm)
    tr["weight"] = tr["weight"].map(norm)
    return tr


def build_headshot_matches(rosters: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=HEADSHOT_COLS)
    if rosters.empty or tr.empty:
        return empty

    tr = tr[tr["img_url"] != ""].copy()
    if tr.empty:
        return empty

    ros_players = rosters[["player_id", "player_name_norm", "number", "team_name"]].copy()
    ros_players["number"] = ros_players["number"].astype(str).str.strip()

    matches = []
    for team, group in tr.groupby("team", dropna=False):
        team = _s(team).strip()
        ros_sub = ros_players[ros_players["team_name"].astype(str).str.strip() == team]
        if ros_sub.empty:
            continue

        for _, tr_row in group.iterrows():
            tname = _s(tr_row["tr_name_norm"]).strip()
            img_url = _s(tr_row["img_url"]).strip()
            if not tname or not img_url:
                continue

            base = get_base_url(tr_row["roster_url"])
            if img_url.startswith("/") and base:
                img_url = base + img_url

            tnum = _s(tr_row["number"]).strip()
            direct = ros_sub[ros_sub["player_name_norm"] == tname]
            if tnum and len(direct) > 1:
                direct = direct[direct["number"] == tnum]

            if direct.empty:
                cand = process.extractOne(tname, ros_sub["player_name_norm"], scorer=fuzz.WRatio)
                if not cand or cand[1] < 90:
                    continue
                direct = ros_sub[ros_sub["player_name_norm"] == cand[0]]
                if tnum and len(direct) > 1:
                    direct = direct[direct["number"] == tnum]

            if direct.empty:
                continue

            r = direct.iloc[0]
            matches.append((
                r["player_id"], img_url, tr_row["b_t"], tr_row["hometown"],
                tr_row["high_school"], tr_row["height"], tr_row["weight"], tr_row["pos"],
            ))

    if not matches:
        return empty

    return pd.DataFrame(matches, columns=HEADSHOT_COLS).drop_duplicates(subset=["player_id"], keep="first")


def enrich_rosters_with_headshots(rosters: pd.DataFrame, hs: pd.DataFrame) -> pd.DataFrame:
    merged = rosters.merge(hs, on="player_id", how="left", suffixes=("", "_hs"))

    for col in ["img_url", "b_t", "hometown", "high_school", "height", "weight", "pos"]:
        hs_col = f"{col}_hs"
        merged[col] = merged[col].map(norm)
        merged[hs_col] = merged[hs_col].map(norm)
        merged[col] = merged[col].where(merged[col] != "", merged[hs_col])

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_hs")], errors="ignore")
    merged["b_t"] = merged["b_t"].map(normalize_b_t)
    merged["pos"] = merged["pos"].map(standardize_pos)
    merged["img_url"] = merged["img_url"].map(norm)
    return merged


def assign_headshots_to_cube_player_info(data_dir: Path) -> None:
    cube_info_path = data_dir / "cube_stats" / "cube_player_info.csv"
    team_map_path = data_dir / "team_mappings.csv"
    ncaa_history_path = data_dir / "ncaa_team_history.csv"

    if not cube_info_path.exists() or not team_map_path.exists():
        logger.warning("missing cube_player_info or team_mappings, skipping headshot assignment")
        return

    logger.info("Loading cube_player_info...")
    cube_info = pd.read_csv(cube_info_path, dtype={"player_id": str}, low_memory=False)
    cube_info = cube_info.dropna(subset=["player_id"])
    cube_info["player_id"] = cube_info["player_id"].astype(str).str.strip()
    cube_info = cube_info[cube_info["player_id"] != ""]
    if "img_url" not in cube_info.columns:
        cube_info["img_url"] = ""

    logger.info("Loading team mappings...")
    tm = pd.read_csv(team_map_path, usecols=["org_id", "cube_college_id", "school_name", "ncaa_team_name"])
    tm = tm.dropna(subset=["org_id", "cube_college_id"])

    org_to_cube: dict[int, int] = {
        int(row.org_id): int(row.cube_college_id)
        for row in tm.itertuples(index=False)
    }

    name_to_org: dict[str, int] = {}
    for row in tm.itertuples(index=False):
        org_id = int(row.org_id)
        for col in ("school_name", "ncaa_team_name"):
            v = getattr(row, col, None)
            if pd.notna(v):
                name_to_org[str(v).strip()] = org_id

    logger.info("Loading ncaa_team_history for year-specific team_id -> org_id mapping...")
    if ncaa_history_path.exists():
        ncaa_history = pd.read_csv(ncaa_history_path, usecols=["team_id", "org_id", "year"])
        ncaa_history = ncaa_history.dropna(subset=["team_id", "org_id", "year"])
        ncaa_history["team_id"] = pd.to_numeric(ncaa_history["team_id"], errors="coerce").astype("Int64")
        ncaa_history["org_id"] = pd.to_numeric(ncaa_history["org_id"], errors="coerce").astype("Int64")
        ncaa_history["year"] = pd.to_numeric(ncaa_history["year"], errors="coerce").astype("Int64")
    else:
        ncaa_history = pd.DataFrame(columns=["team_id", "org_id", "year"])

    logger.info("Loading cube_stats for player-team-year data...")
    stats_frames = []
    for path in sorted((data_dir / "cube_stats").glob("*_batting_*.csv")):
        try:
            df = pd.read_csv(
                path,
                usecols=["player_id", "player_name", "team_id", "year"],
                dtype={"player_id": str},
            )
            stats_frames.append(df)
        except Exception:
            pass

    for path in sorted((data_dir / "cube_stats").glob("*_pitching_*.csv")):
        try:
            df = pd.read_csv(
                path,
                usecols=["player_id", "player_name", "team_id", "year"],
                dtype={"player_id": str},
            )
            stats_frames.append(df)
        except Exception:
            pass

    if not stats_frames:
        logger.warning("no cube_stats files found, skipping headshot assignment")
        return

    stats = pd.concat(stats_frames, ignore_index=True)
    # Ensure player_id is present - this is the primary key, never null
    stats = stats.dropna(subset=["player_id"])
    stats["player_id"] = stats["player_id"].astype(str).str.strip()
    stats = stats[stats["player_id"] != ""]
    # Drop rows missing other required fields
    stats = stats.dropna(subset=["player_name", "team_id", "year"])
    stats["name_norm"] = stats["player_name"].map(normalize_name)
    stats["team_id"] = pd.to_numeric(stats["team_id"], errors="coerce").astype("Int64")
    stats["year"] = pd.to_numeric(stats["year"], errors="coerce").astype("Int64")
    stats = stats.dropna(subset=["team_id", "year"])
    stats = stats.drop_duplicates(subset=["player_id", "team_id", "year"])[
        ["player_id", "name_norm", "team_id", "year"]
    ]

    logger.info("Loading headshots (processing years 2021-2026 in order)...")
    hs_frames = []
    for year in range(2021, 2027):
        path = data_dir / "headshots" / f"team_headshots_{year}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, usecols=["team", "year", "name", "img_url", "roster_url"])
                hs_frames.append(df)
            except Exception:
                pass

    if not hs_frames:
        logger.warning("no headshot files found")
        return

    hs = pd.concat(hs_frames, ignore_index=True)
    hs = hs[hs["img_url"].notna() & (hs["img_url"].astype(str).str.strip() != "")]
    hs["year"] = pd.to_numeric(hs["year"], errors="coerce").astype("Int64")
    hs = hs.dropna(subset=["team", "name", "year"])
    hs["name_clean"] = hs["name"].map(clean_name)
    hs["name_norm"] = hs["name_clean"].map(normalize_name)
    hs["img_url"] = hs["img_url"].map(norm)

    hs["org_id"] = hs["team"].astype(str).str.strip().map(lambda t: name_to_org.get(t))
    hs = hs.dropna(subset=["org_id"])
    hs["org_id"] = hs["org_id"].astype("Int64")
    hs["cube_college_id"] = hs["org_id"].map(lambda oid: org_to_cube.get(int(oid) if pd.notna(oid) else None))
    hs = hs.dropna(subset=["cube_college_id"])
    hs["cube_college_id"] = hs["cube_college_id"].astype("Int64")

    logger.info("Matching headshots to players...")
    matched = []

    for _, hs_row in hs.iterrows():
        cube_team_id = int(hs_row["cube_college_id"])
        year = int(hs_row["year"])
        name_norm = hs_row["name_norm"]
        img_url = hs_row["img_url"]
        if img_url.startswith("/"):
            base = get_base_url(_s(hs_row.get("roster_url", "")))
            if base:
                img_url = base + img_url

        candidates = stats[
            (stats["team_id"] == cube_team_id) &
            (stats["year"] == year) &
            (stats["name_norm"] == name_norm)
        ]

        if not candidates.empty:
            player_id = candidates.iloc[0]["player_id"]
            matched.append({
                "player_id": player_id,
                "img_url": img_url,
                "year": year,
                "match_type": "direct",
            })
            continue

        candidates = stats[
            (stats["team_id"] == cube_team_id) &
            (stats["year"] == year)
        ]

        if not candidates.empty:
            candidate_names = candidates["name_norm"].tolist()
            match = process.extractOne(name_norm, candidate_names, scorer=fuzz.WRatio, score_cutoff=90)
            if match:
                matched_name = match[0]
                matched_row = candidates[candidates["name_norm"] == matched_name].iloc[0]
                matched.append({
                    "player_id": matched_row["player_id"],
                    "img_url": img_url,
                    "year": year,
                    "match_type": "fuzzy",
                })
                continue

        candidates = stats[stats["team_id"] == cube_team_id]
        if not candidates.empty:
            candidate_names = candidates["name_norm"].tolist()
            match = process.extractOne(name_norm, candidate_names, scorer=fuzz.WRatio, score_cutoff=90)
            if match:
                matched_name = match[0]
                matched_row = candidates[candidates["name_norm"] == matched_name].iloc[0]
                matched.append({
                    "player_id": matched_row["player_id"],
                    "img_url": img_url,
                    "year": year,
                    "match_type": "fuzzy_fallback",
                })

    if not matched:
        logger.warning("no headshots matched to players")
        return

    all_matches = pd.DataFrame(matched)
    all_matches = all_matches.sort_values("year", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    match_counts = all_matches["match_type"].value_counts().to_dict()
    player_img = dict(zip(all_matches["player_id"], all_matches["img_url"]))

    # Update img_url using player_id (never null) - preserve existing if no match found
    existing_img = cube_info["img_url"].map(norm)
    cube_info["img_url"] = cube_info["player_id"].map(player_img)
    # Fill missing with existing img_url, then normalize
    cube_info["img_url"] = cube_info["img_url"].fillna(existing_img)
    cube_info["img_url"] = cube_info["img_url"].map(norm)

    cube_info.to_csv(cube_info_path, index=False)
    logger.info("assigned %d headshot urls to cube_player_info (direct: %d, fuzzy: %d, fuzzy_fallback: %d)",
                len(player_img),
                match_counts.get("direct", 0),
                match_counts.get("fuzzy", 0),
                match_counts.get("fuzzy_fallback", 0))


def add_missing_players_to_cube_info(data_dir: Path) -> None:
    """Upsert cube_player_info from cube_stats.

    - Adds stub rows (player_id + proper_name, rest null) for any player_id in cube_stats
      that is missing from cube_player_info entirely.
    - Fills proper_name for existing rows where it is currently blank.
    """
    cube_info_path = data_dir / "cube_stats" / "cube_player_info.csv"
    if not cube_info_path.exists():
        logger.warning("cube_player_info.csv not found, skipping")
        return

    cube_info = pd.read_csv(cube_info_path, dtype={"player_id": str, "cube_player_id": str}, low_memory=False)
    cube_info["player_id"] = cube_info["player_id"].astype(str).str.strip()

    # Collect player_id + player_name from all cube_stats files
    frames = []
    for pattern in ("*_batting_*.csv", "*_pitching_*.csv"):
        for path in sorted((data_dir / "cube_stats").glob(pattern)):
            try:
                header = pd.read_csv(path, nrows=0).columns.tolist()
                cols = [c for c in ["player_id", "player_name"] if c in header]
                if "player_id" not in cols:
                    continue
                df = pd.read_csv(path, usecols=cols, dtype={"player_id": str})
                frames.append(df)
            except Exception:
                pass

    if not frames:
        logger.info("no cube_stats files found, skipping")
        return

    all_players = pd.concat(frames, ignore_index=True)
    all_players["player_id"] = all_players["player_id"].astype(str).str.strip()
    all_players = all_players.dropna(subset=["player_id"])
    all_players = all_players[all_players["player_id"] != ""]
    # Keep a non-empty name where possible
    if "player_name" in all_players.columns:
        all_players["player_name"] = all_players["player_name"].map(norm)
        all_players = all_players.sort_values("player_name", na_position="last")
    all_players = all_players.drop_duplicates(subset=["player_id"], keep="first")

    existing_ids = set(cube_info["player_id"])
    name_map = all_players.set_index("player_id")["player_name"].to_dict() if "player_name" in all_players.columns else {}

    # Fill blank player_name for existing rows (prefer cube_stats name)
    if "player_name" in cube_info.columns and name_map:
        blank = cube_info["player_name"].isna() | (cube_info["player_name"].astype(str).str.strip() == "")
        cube_info.loc[blank, "player_name"] = cube_info.loc[blank, "player_id"].map(name_map)
        filled = int(blank.sum())
        if filled:
            logger.info("filled player_name for %d existing rows", filled)

    # Add stub rows for completely missing player_ids
    missing = all_players[~all_players["player_id"].isin(existing_ids)]
    added = 0
    if not missing.empty:
        existing_cols = cube_info.columns.tolist()
        stub: dict = {"player_id": missing["player_id"].values}
        if "player_name" in missing.columns and "player_name" in existing_cols:
            stub["player_name"] = missing["player_name"].values
        stubs = pd.DataFrame(stub).reindex(columns=existing_cols)
        cube_info = pd.concat([cube_info, stubs], ignore_index=True)
        added = len(missing)
        logger.info("added %d missing players", added)

    before = len(cube_info)
    cube_info = cube_info.drop_duplicates(subset=["player_id"], keep="first")
    dupes = before - len(cube_info)
    if dupes:
        logger.warning("dropped %d duplicate player_id rows", dupes)

    cube_info.to_csv(cube_info_path, index=False)
    logger.info("cube_player_info saved: %d total rows", len(cube_info))


def main(data_dir: str, year: int = None):
    """Reconcile players: add missing stubs, then assign headshots.

    If year is provided, it's ignored - this processes all headshots (2021-2026).
    """
    data_dir = Path(data_dir)
    add_missing_players_to_cube_info(data_dir)
    assign_headshots_to_cube_player_info(data_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assign headshots to cube_player_info using team mappings and fuzzy name matching")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--year", type=int, default=None, help="Ignored - processes all years 2021-2026")
    args = parser.parse_args()
    main(args.data_dir, args.year)
