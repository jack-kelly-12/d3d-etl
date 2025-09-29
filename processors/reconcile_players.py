import pandas as pd
import glob
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse
from rapidfuzz import process, fuzz

def clean_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    s = re.sub(r"^(?:[^\w]*|No\.?|Number)?\s*\d{1,3}[-–—.: ]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = re.sub(r"^\d+", "", str(name)).strip()
    name = re.sub(r"[^\w\s]", "", name)
    return name.lower()

def get_base_url(url: str) -> str:
    parsed = urlparse(url if isinstance(url, str) else "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"

def normalize_b_t(bt: str) -> str:
    if pd.isna(bt) or not str(bt).strip():
        return ""
    s = unicodedata.normalize("NFKC", str(bt)).upper().strip()
    s = re.sub(r"[^LRSH/]", "", s)
    parts = re.split(r"[^\w]+|/", s)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return f"{parts[0][0]}/{parts[1][0]}"
    if len(parts) == 1:
        return f"{parts[0][0]}/{parts[0][0]}"
    return ""

_POS_MAP = {
    "CATCHER":"C","C":"C",
    "FIRST BASE":"1B","1B":"1B",
    "SECOND BASE":"2B","2B":"2B",
    "THIRD BASE":"3B","3B":"3B",
    "SHORTSTOP":"SS","SS":"SS",
    "LEFT FIELD":"LF","LF":"LF",
    "CENTER FIELD":"CF","CF":"CF",
    "RIGHT FIELD":"RF","RF":"RF",
    "OUTFIELD":"OF","OF":"OF",
    "INFIELD":"INF","IF":"INF","INF":"INF",
    "PITCHER":"P","RHP":"P","LHP":"P","P":"P",
    "DESIGNATED HITTER":"DH","DH":"DH",
    "UTILITY":"UT","UTIL":"UT","UT":"UT"
}
def standardize_pos(pos: str) -> str:
    if pd.isna(pos) or not str(pos).strip():
        return ""
    s = unicodedata.normalize("NFKC", str(pos)).upper()
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
    name_series = rosters["player_name"] if "player_name" in rosters.columns else pd.Series([""] * len(rosters))
    rosters["player_name_norm"] = name_series.map(normalize_name)
    if "number" not in rosters.columns and "jersey" in rosters.columns:
        rosters["number"] = rosters["jersey"]
    if "b_t" not in rosters.columns and ("bats" in rosters.columns or "throws" in rosters.columns):
        bats = rosters.get("bats", pd.Series([pd.NA] * len(rosters))).astype(str).str.upper().str[:1].replace({"N": pd.NA, "0": pd.NA})
        throws = rosters.get("throws", pd.Series([pd.NA] * len(rosters))).astype(str).str.upper().str[:1].replace({"N": pd.NA, "0": pd.NA})
        rosters["b_t"] = (bats.fillna("") + "/" + throws.fillna("")).str.strip("/")
    if "pos" in rosters.columns:
        rosters["pos"] = rosters["pos"].map(standardize_pos)
    return rosters

def load_headshots(data_dir: Path, year: int) -> pd.DataFrame:
    hfiles = glob.glob(str(data_dir / "headshots" / f"*{year}*.csv"))
    if not hfiles:
        return pd.DataFrame(columns=["team_name","name","number","img_url","roster_url","height","weight","hometown","b_t","pos","season"])
    hs = []
    for f in hfiles:
        df = pd.read_csv(f)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df["season"] = year
        hs.append(df)
    tr = pd.concat(hs, ignore_index=True)
    try:
        mappings = pd.read_csv(data_dir / "team_mappings.csv")
        if "school_name_official" in mappings.columns and "ncaa_team" in mappings.columns:
            tr_team_key = None
            for c in ("school_name_official", "team", "team_name", "school_name"):
                if c in tr.columns:
                    tr_team_key = c
                    break
            if tr_team_key is not None:
                tr = tr.merge(
                    mappings[["school_name_official", "ncaa_team"]],
                    left_on=tr_team_key,
                    right_on="school_name_official",
                    how="left"
                )
                tr["team_name"] = tr["ncaa_team"]
                tr = tr.drop(columns=[col for col in ["school_name_official", "ncaa_team"] if col in tr.columns])
        else:
            if "team" in tr.columns and "team_name" not in tr.columns:
                tr["team_name"] = tr["team"]
    except Exception:
        if "team" in tr.columns and "team_name" not in tr.columns:
            tr["team_name"] = tr["team"]
    if "name" in tr.columns:
        name_series = tr["name"]
    elif "player" in tr.columns:
        name_series = tr["player"]
    else:
        name_series = pd.Series([""] * len(tr))
    tr["name_clean"] = name_series.map(clean_name)
    tr["tr_name_norm"] = tr["name_clean"].map(normalize_name)
    if "pos" in tr.columns:
        tr["pos"] = tr["pos"].map(standardize_pos)
    if "b_t" in tr.columns:
        tr["b_t"] = tr["b_t"].map(normalize_b_t)
    return tr

def build_headshot_matches(rosters: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    matches = []
    if "team_name" not in tr.columns:
        return pd.DataFrame(columns=["player_id","img_url","b_t","hometown","high_school","height","weight","pos"])
    for team, group in tr.groupby("team_name"):
        ros_sub = rosters[rosters.get("team_name", "") == team].copy()
        if ros_sub.empty:
            continue
        base_cols = ["player_id","player_name_norm"]
        if "number" in ros_sub.columns:
            base_cols.append("number")
        ros_players = ros_sub[base_cols].copy()
        for _, tr_row in group.iterrows():
            tname = tr_row.get("tr_name_norm", "")
            tnum = str(tr_row.get("number", "")).strip()
            img_url = tr_row.get("img_url", "")
            base = get_base_url(tr_row.get("roster_url", ""))
            if isinstance(img_url, str) and img_url.startswith("/") and base:
                img_url = base + img_url
            direct = ros_players[ros_players["player_name_norm"] == tname]
            if "number" in ros_players.columns and tnum:
                if len(direct) > 1:
                    direct = direct[direct["number"].astype(str) == tnum]
            if not direct.empty:
                r = direct.iloc[0]
                matches.append((r.player_id, img_url, tr_row.get("b_t", pd.NA), tr_row.get("hometown", pd.NA), None, tr_row.get("height", pd.NA), tr_row.get("weight", pd.NA), tr_row.get("pos", pd.NA)))
                continue
            cand = process.extractOne(tname, ros_players["player_name_norm"], scorer=fuzz.WRatio)
            if cand and cand[1] >= 90:
                rsub = ros_players[ros_players["player_name_norm"] == cand[0]]
                if "number" in ros_players.columns and tnum and len(rsub) > 1:
                    rsub = rsub[rsub["number"].astype(str) == tnum]
                if not rsub.empty:
                    r = rsub.iloc[0]
                    matches.append((r.player_id, img_url, tr_row.get("b_t", pd.NA), tr_row.get("hometown", pd.NA), None, tr_row.get("height", pd.NA), tr_row.get("weight", pd.NA), tr_row.get("pos", pd.NA)))
    if not matches:
        return pd.DataFrame(columns=["player_id","img_url","b_t","hometown","high_school","height","weight","pos"])
    out = pd.DataFrame(matches, columns=["player_id","img_url","b_t","hometown","high_school","height","weight","pos"])
    out = out.drop_duplicates(subset=["player_id"], keep="first")
    return out

def enrich_rosters_with_headshots(rosters: pd.DataFrame, hs: pd.DataFrame) -> pd.DataFrame:
    merged = rosters.merge(hs, on="player_id", how="left", suffixes=("", "_hs"))
    if "img_url" not in merged.columns and "img" in merged.columns:
        merged.rename(columns={"img": "img_url"}, inplace=True)

    def fill_preferring_base_then_hs(target: str):
        if target not in merged:
            merged[target] = pd.NA
        hs_col = f"{target}_hs"
        src = None
        if hs_col in merged:
            src = hs_col
        elif target in merged and target not in rosters.columns:
            src = target
        if src is not None:
            merged[target] = merged[target].where(
                merged[target].notna() & (merged[target].astype(str).str.strip() != ""),
                merged[src]
            )

    for col in ["img_url", "b_t", "hometown", "high_school", "height", "weight", "pos"]:
        fill_preferring_base_then_hs(col)

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_hs")], errors="ignore")
    if "b_t" in merged.columns:
        merged["b_t"] = merged["b_t"].map(normalize_b_t)
    if "pos" in merged.columns:
        merged["pos"] = merged["pos"].map(standardize_pos)
    if "img_url" in merged.columns:
        merged["img_url"] = merged["img_url"].fillna("")
    return merged

def main(data_dir: str, year: int):
    data_dir = Path(data_dir)
    headshots = load_headshots(data_dir, year)

    outputs = []
    for division in [1, 2, 3]:
        try:
            rosters = load_rosters(data_dir, year, division)
        except FileNotFoundError:
            print(f"No roster file for d{division} {year}, skipping")
            continue
        tr_subset = headshots
        if not headshots.empty:
            tr_subset = headshots.copy()
            img_col = "img_url" if "img_url" in tr_subset.columns else None
            if img_col is not None:
                tr_subset = tr_subset[tr_subset[img_col].astype(str).str.strip() != ""]
    
            if "team_name" in tr_subset.columns and "team_name" in rosters.columns:
                teams = rosters["team_name"].dropna().astype(str).str.strip().unique().tolist()
                tr_subset = tr_subset[tr_subset["team_name"].astype(str).str.strip().isin(teams)].copy()

        hs_matches = build_headshot_matches(rosters, tr_subset)
        enriched = enrich_rosters_with_headshots(rosters, hs_matches)
        outpath = data_dir / "rosters" / f"d{division}_rosters_{year}.csv"
        enriched.to_csv(outpath, index=False)
        print(f"wrote {outpath}")
        outputs.append(outpath)

    return outputs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--year", required=True)
    args = parser.parse_args()
    main(args.data_dir, args.year)