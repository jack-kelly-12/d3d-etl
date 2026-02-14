import glob
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from rapidfuzz import fuzz, process

from processors.logging_utils import division_year_label, get_logger

logger = get_logger(__name__)


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

    s = re.sub(r"\s+", " ", s).strip(" \t\r\n-–—:;|,")
    return s


def normalize_name(name: str) -> str:
    s = _s(name)
    s = re.sub(r"^\s*\d{1,3}\s*", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


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
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(_POS_MAP.get(p, _POS_MAP.get(p.replace(" ", ""), p)))

    out = [o for o in out if o]
    out = list(dict.fromkeys(out))
    if not out:
        return ""
    if len(out) == 1:
        return out[0]
    return "/".join(out[:3])


def load_rosters(data_dir: Path, year: int, division: int) -> pd.DataFrame:
    roster_path = data_dir / "rosters" / f"d{division}_rosters_{year}.csv"
    rosters = pd.read_csv(roster_path)

    rosters["player_name_norm"] = rosters["player_name"].map(normalize_name)
    rosters["b_t"] = (
        rosters["bats"].map(lambda x: _s(x).upper()[:1])
        + "/"
        + rosters["throws"].map(lambda x: _s(x).upper()[:1])
    )
    rosters["b_t"] = rosters["b_t"].map(normalize_b_t)
    rosters["pos"] = rosters["position"].map(standardize_pos)

    rosters["img_url"] = rosters["img_url"].map(norm)
    rosters["hometown"] = rosters["hometown"].map(norm)
    rosters["high_school"] = rosters["high_school"].map(clean_high_school)
    rosters["height"] = rosters["height"].map(norm)
    rosters["weight"] = rosters["weight"].map(norm)

    return rosters


def load_headshots(data_dir: Path, year: int) -> pd.DataFrame:
    hfiles = glob.glob(str(data_dir / "headshots" / f"*{year}*.csv"))
    if not hfiles:
        return pd.DataFrame(
            columns=[
                "team",
                "year",
                "roster_url",
                "name",
                "number",
                "position",
                "height",
                "weight",
                "class",
                "b_t",
                "hometown",
                "highschool",
                "previous_school",
                "img_url",
            ]
        )

    hs = []
    for f in hfiles:
        df = pd.read_csv(f).drop(columns=["Unnamed: 0"], errors="ignore")
        hs.append(df)

    tr = pd.concat(hs, ignore_index=True)

    map_path = data_dir / "team_mappings.csv"
    if map_path.exists():
        mappings = pd.read_csv(map_path)
        if {"school_name_official", "ncaa_team"}.issubset(mappings.columns):
            tr = tr.merge(
                mappings[["school_name_official", "ncaa_team"]],
                left_on="team",
                right_on="school_name_official",
                how="left",
            )
            tr["team"] = tr["ncaa_team"].fillna(tr["team"])
            tr = tr.drop(columns=["school_name_official", "ncaa_team"], errors="ignore")

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
    if rosters.empty or tr.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "img_url",
                "b_t",
                "hometown",
                "high_school",
                "height",
                "weight",
                "pos",
            ]
        )

    tr = tr[tr["img_url"]].copy()
    if tr.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "img_url",
                "b_t",
                "hometown",
                "high_school",
                "height",
                "weight",
                "pos",
            ]
        )

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
            if not tname:
                continue

            img_url = _s(tr_row["img_url"]).strip()
            if not img_url:
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
            matches.append(
                (
                    r["player_id"],
                    img_url,
                    tr_row["b_t"],
                    tr_row["hometown"],
                    tr_row["high_school"],
                    tr_row["height"],
                    tr_row["weight"],
                    tr_row["pos"],
                )
            )

    out = pd.DataFrame(
        matches,
        columns=[
            "player_id",
            "img_url",
            "b_t",
            "hometown",
            "high_school",
            "height",
            "weight",
            "pos",
        ],
    )
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return out


def enrich_rosters_with_headshots(rosters: pd.DataFrame, hs: pd.DataFrame) -> pd.DataFrame:
    merged = rosters.merge(hs, on="player_id", how="left", suffixes=("", "_hs"))

    def take(roster_col: str) -> None:
        hs_col = f"{roster_col}_hs"
        merged[roster_col] = merged[roster_col].map(norm)
        merged[hs_col] = merged[hs_col].map(norm)
        merged[roster_col] = merged[roster_col].where(~merged[roster_col], merged[hs_col])

    for col in ["img_url", "b_t", "hometown", "high_school", "height", "weight", "pos"]:
        take(col)

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_hs")], errors="ignore")
    merged["b_t"] = merged["b_t"].map(normalize_b_t)
    merged["pos"] = merged["pos"].map(standardize_pos)
    merged["img_url"] = merged["img_url"].map(norm)

    return merged


def main(data_dir: str, year: int):
    data_dir = Path(data_dir)
    headshots = load_headshots(data_dir, year)

    for division in (1, 2, 3):
        roster_path = data_dir / "rosters" / f"d{division}_rosters_{year}.csv"
        if not roster_path.exists():
            logger.warning("No roster file for %s, skipping", division_year_label(division, year))
            continue

        rosters = load_rosters(data_dir, year, division)

        teams = set(rosters["team_name"].astype(str).str.strip().unique().tolist())
        tr_subset = headshots[headshots["team"].astype(str).str.strip().isin(teams)].copy()

        hs_matches = build_headshot_matches(rosters, tr_subset)
        enriched = enrich_rosters_with_headshots(rosters, hs_matches)

        outpath = data_dir / "rosters" / f"d{division}_rosters_{year}.csv"
        enriched.to_csv(outpath, index=False)
        logger.info("wrote %s", outpath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", required=True, type=int)
    args = parser.parse_args()
    main(args.data_dir, args.year)
