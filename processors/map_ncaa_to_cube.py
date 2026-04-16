"""
Map lineup player names to player_ids using cube stats.

Matching cascade (per team+year):
  1. Jersey number (when available)
  2. Exact full name
  3. Last name (unique within team)
  4. Jersey number as name
  5. First initial + last name (unique within team)
  6. Fuzzy token_sort_ratio >= threshold
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rfuzz_process

from processors.logging_utils import div_file_prefix, get_logger

logger = get_logger(__name__)

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
DIVISIONS = ["ncaa_1", "ncaa_2", "ncaa_3"]
FUZZY_THRESHOLD = 70

_JERSEY_LEADING = re.compile(
    r"^\s*(?:no\.?|#|number\s*)?\s*(\d{1,3})\s*([-–—.:]|\s+)\s*(.+)$",
    re.IGNORECASE,
)


def _parse_jersey_from_name(raw: str) -> tuple[int | None, str]:
    """If *raw* starts with a jersey-style prefix, return (number, remainder for name match)."""
    s = str(raw).strip()
    if not s:
        return None, ""
    if re.fullmatch(r"\d{1,3}", s):
        return int(s), ""
    m = _JERSEY_LEADING.match(s)
    if m:
        return int(m.group(1)), m.group(3).strip()
    return None, s


def _load_cube_lookup(data_dir: Path, divisions: list[str], years: list[int]) -> pd.DataFrame:
    """Load cube stats for the given divisions/years, normalizing team_id to ncaa_slug."""
    cube_dir = data_dir / "cube_stats"
    if not cube_dir.exists():
        return pd.DataFrame()

    frames = []
    for div in divisions:
        for year in years:
            for kind in ("batting", "pitching"):
                p = cube_dir / f"{div}_{kind}_{year}.csv"
                if not p.exists():
                    continue
                try:
                    df = pd.read_csv(
                        p,
                        usecols=lambda c: c in {"player_name", "player_id", "number", "team_id", "year"},
                        dtype={"player_id": str, "team_id": str},
                    )
                    frames.append(df)
                except Exception:
                    continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["player_name", "player_id", "team_id"])
    combined = combined[~combined["player_id"].isin({"", "nan", "None"})]
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")

    numeric_mask = combined["team_id"].str.match(r"^\d+$", na=False)
    if numeric_mask.any():
        tm_path = data_dir / "team_mappings.csv"
        if tm_path.exists():
            tm = pd.read_csv(tm_path, usecols=["cube_college_id", "ncaa_slug"], dtype=str)
            tm["cube_college_id"] = tm["cube_college_id"].str.strip()
            combined.loc[numeric_mask, "team_id"] = (
                combined.loc[numeric_mask, "team_id"]
                .map(dict(zip(tm["cube_college_id"], tm["ncaa_slug"])))
                .fillna(combined.loc[numeric_mask, "team_id"])
            )

    combined = combined.drop_duplicates(subset=["player_id", "team_id", "year"])
    return combined


def _build_team_year_index(cube: pd.DataFrame):
    """Return (by_name, by_last, by_initlast, by_number) dicts keyed by (team_id, year)."""
    by_name: dict[tuple, dict[str, str]] = {}   # {(team, yr): {full_name: pid}}
    by_last: dict[tuple, dict[str, list]] = {}   # {(team, yr): {last: [pid, ...]}}
    by_initlast: dict[tuple, dict[str, list]] = {}  # {(team, yr): {"J Smith": [pid, ...]}}
    by_number: dict[tuple, dict[int, str]] = {}  # {(team, yr): {number: pid}}

    for row in cube.itertuples(index=False):
        key = (row.team_id, int(row.year))
        name = str(row.player_name).strip()
        pid = str(row.player_id)

        by_name.setdefault(key, {})[name] = pid

        parts = name.split()
        if parts:
            last = parts[-1].lower()
            by_last.setdefault(key, {}).setdefault(last, []).append(pid)
            if len(parts) >= 2:
                initlast = f"{parts[0][0].upper()} {parts[-1]}"
                by_initlast.setdefault(key, {}).setdefault(initlast.lower(), []).append(pid)

        num = getattr(row, "number", None)
        if num is not None and pd.notna(num):
            m = re.match(r"\d+", str(num).strip())
            if m:
                by_number.setdefault(key, {})[int(m.group(0))] = pid

    return by_name, by_last, by_initlast, by_number


def _resolve(
    player_name: str,
    number,
    key: tuple,
    by_name: dict,
    by_last: dict,
    by_initlast: dict,
    by_number: dict,
    threshold: int = FUZZY_THRESHOLD,
) -> str | None:
    name = str(player_name).strip()

    # 1. Jersey number
    if pd.notna(number) and key in by_number:
        m = re.match(r"\d+", str(number).strip())
        if m:
            pid = by_number[key].get(int(m.group(0)))
            if pid:
                return pid

    candidates = by_name.get(key, {})
    candidate_names = list(candidates.keys())

    # 2. Exact
    if name in candidates:
        return candidates[name]

    # 3. Last name (unique)
    last = name.split()[-1].lower() if name else ""
    if last and key in by_last:
        pids = by_last[key].get(last, [])
        if len(pids) == 1:
            return pids[0]

    # 4. First initial + last (unique)
    parts = name.split()
    if len(parts) >= 2 and key in by_initlast:
        initlast = f"{parts[0][0].upper()} {parts[-1]}".lower()
        pids = by_initlast[key].get(initlast, [])
        if len(pids) == 1:
            return pids[0]

    # 5. Fuzzy
    if candidate_names:
        match = rfuzz_process.extractOne(
            name, candidate_names, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        if match:
            return candidates[match[0]]

    return None


def enrich_lineups(data_dir: Path, divisions: list[str], years: list[int]) -> None:
    cube = _load_cube_lookup(data_dir, divisions, years)
    if cube.empty:
        logger.warning("No cube stats found — lineups will not be enriched")
        return

    by_name, by_last, by_initlast, by_number = _build_team_year_index(cube)

    lineups_dir = data_dir / "lineups"
    for division in divisions:
        prefix = div_file_prefix(division)
        for year in years:
            for kind in ("batting", "pitching"):
                path = lineups_dir / f"{prefix}_{kind}_lineups_{year}.csv"
                if not path.exists():
                    continue

                df = pd.read_csv(path, dtype={"player_id": str})
                if "player_id" not in df.columns:
                    df["player_id"] = pd.NA

                number_added = "number" not in df.columns
                if number_added:
                    df["number"] = pd.NA
                has_number = True

                mask = df["player_id"].isna() | df["player_id"].isin({"", "nan", "None"})
                total = mask.sum()
                if total == 0:
                    if number_added:
                        df.to_csv(path, index=False)
                        logger.info("%s %s %s %d: added number column", division, kind, year, year)
                    else:
                        logger.info("%s %s %s %d: all player_ids present", division, kind, year, 0)
                    continue

                resolved = 0
                for i in df[mask].index:
                    row = df.loc[i]
                    team_id = row.get("team_id")
                    if pd.isna(team_id):
                        continue
                    key = (str(team_id), year)
                    number = row.get("number") if has_number else pd.NA
                    pid = _resolve(
                        row["player_name"], number, key,
                        by_name, by_last, by_initlast, by_number,
                    )
                    if pid:
                        df.at[i, "player_id"] = pid
                        resolved += 1

                df.to_csv(path, index=False)
                logger.info(
                    "%s %s %s %d: resolved %d/%d player_ids",
                    division, kind, year, year, resolved, total,
                )


def main(data_dir: str, divisions: list[str], years: list[int]) -> None:
    enrich_lineups(Path(data_dir), divisions, years)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--divisions", nargs="+", default=DIVISIONS)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()
    main(args.data_dir, args.divisions, args.years)
