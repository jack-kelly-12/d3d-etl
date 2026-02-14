from pathlib import Path

import pandas as pd

from processors.logging_utils import get_logger

SOURCE_PRIORITY = {"roster": 0, "batting_war": 1, "pitching_war": 2}
logger = get_logger(__name__)


def is_valid(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip().lower()


def clean_str(val) -> str:
    if not is_valid(val):
        return ""
    return str(val).strip()


def standardize_hand(val) -> str:
    v = clean_str(val).upper()
    if not v:
        return ""
    if v in {"R", "RIGHT"}:
        return "R"
    if v in {"L", "LEFT"}:
        return "L"
    if v in {"S", "SWITCH", "B", "BOTH"}:
        return "S"
    return ""


def first_valid(series: pd.Series) -> str:
    for val in series:
        if is_valid(val):
            return clean_str(val)
    return ""


def roster_paths(
    data_dir: Path, divisions: list[int], years: list[int]
) -> list[tuple[int, int, Path]]:
    paths = []
    for division in divisions:
        for year in years:
            path = data_dir / f"rosters/d{division}_rosters_{year}.csv"
            if path.exists():
                paths.append((division, year, path))
    return paths


def war_paths(
    data_dir: Path, divisions: list[int], years: list[int], war_type: str
) -> list[tuple[int, int, Path]]:
    paths = []
    for division in divisions:
        for year in years:
            path = data_dir / f"war/d{division}_{war_type}_war_{year}.csv"
            if path.exists():
                paths.append((division, year, path))
    return paths


def load_rows(path: Path, source: str, division: int, year: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"player_id": str}, low_memory=False)
    out = pd.DataFrame(index=df.index)
    out["player_id"] = df["player_id"] if "player_id" in df.columns else ""
    out["player_name"] = df["player_name"] if "player_name" in df.columns else ""
    out["team_name"] = df["team_name"] if "team_name" in df.columns else ""
    out["bats"] = df["bats"] if "bats" in df.columns else ""
    out["throws"] = df["throws"] if "throws" in df.columns else ""
    out["division"] = division
    out["year"] = year
    out["source"] = source
    out["source_priority"] = SOURCE_PRIORITY[source]

    out["player_id"] = out["player_id"].map(clean_str)
    out["player_name"] = out["player_name"].map(clean_str)
    out["team_name"] = out["team_name"].map(clean_str)
    out["bats"] = out["bats"].map(standardize_hand)
    out["throws"] = out["throws"].map(standardize_hand)
    return out


def load_sources(
    data_dir: Path, divisions: list[int], years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roster_rows = []
    batting_war_rows = []
    pitching_war_rows = []

    for division, year, path in roster_paths(data_dir, divisions, years):
        roster_rows.append(load_rows(path, "roster", division, year))
    for division, year, path in war_paths(data_dir, divisions, years, "batting"):
        batting_war_rows.append(load_rows(path, "batting_war", division, year))
    for division, year, path in war_paths(data_dir, divisions, years, "pitching"):
        pitching_war_rows.append(load_rows(path, "pitching_war", division, year))

    rosters = pd.concat(roster_rows, ignore_index=True) if roster_rows else pd.DataFrame()
    batting_war = (
        pd.concat(batting_war_rows, ignore_index=True) if batting_war_rows else pd.DataFrame()
    )
    pitching_war = (
        pd.concat(pitching_war_rows, ignore_index=True) if pitching_war_rows else pd.DataFrame()
    )
    return rosters, batting_war, pitching_war


def aggregate_players(data_dir: Path, divisions: list[int], years: list[int]) -> pd.DataFrame:
    rosters, batting_war, pitching_war = load_sources(data_dir, divisions, years)
    if rosters.empty and batting_war.empty and pitching_war.empty:
        return pd.DataFrame()

    pieces = []
    for frame in (rosters, batting_war, pitching_war):
        if frame.empty:
            continue
        pieces.append(frame)

    all_rows = pd.concat(pieces, ignore_index=True)
    all_rows = all_rows[all_rows["player_id"].map(is_valid)].copy()
    if all_rows.empty:
        return pd.DataFrame()

    all_rows = all_rows.sort_values(["source_priority", "year"], ascending=[True, False])

    result = all_rows.groupby("player_id", as_index=False).agg(
        {
            "player_name": first_valid,
            "team_name": first_valid,
            "bats": first_valid,
            "throws": first_valid,
        }
    )
    result["bats"] = result["bats"].map(standardize_hand)
    result["throws"] = result["throws"].map(standardize_hand)
    return result


def fill_missing_from_info(df: pd.DataFrame, info: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns or key not in info.columns:
        return df

    out = df.copy()
    if "bats" not in out.columns:
        out["bats"] = ""
    if "throws" not in out.columns:
        out["throws"] = ""

    merged = out.merge(
        info[[key, "bats", "throws"]].drop_duplicates(subset=[key]),
        on=key,
        how="left",
        suffixes=("", "__info"),
    )

    merged["bats"] = merged["bats"].map(standardize_hand)
    merged["throws"] = merged["throws"].map(standardize_hand)
    merged["bats__info"] = merged["bats__info"].map(standardize_hand)
    merged["throws__info"] = merged["throws__info"].map(standardize_hand)

    merged["bats"] = merged["bats"].where(merged["bats"].map(is_valid), merged["bats__info"])
    merged["throws"] = merged["throws"].where(
        merged["throws"].map(is_valid), merged["throws__info"]
    )
    return merged.drop(columns=["bats__info", "throws__info"], errors="ignore")


def merge_to_rosters(
    data_dir: Path, player_info: pd.DataFrame, divisions: list[int], years: list[int]
) -> None:
    info = player_info[["player_id", "bats", "throws"]].copy()
    for _, _, path in roster_paths(data_dir, divisions, years):
        roster = pd.read_csv(path, dtype={"player_id": str}, low_memory=False)
        merged = fill_missing_from_info(roster, info, key="player_id")
        merged.to_csv(path, index=False)
        logger.info("Updated %s", path)


def merge_to_war(
    data_dir: Path, player_info: pd.DataFrame, divisions: list[int], years: list[int]
) -> None:
    info = player_info[["player_id", "bats", "throws"]].copy()
    for suffix in ("batting", "pitching"):
        for _, _, path in war_paths(data_dir, divisions, years, suffix):
            war = pd.read_csv(path, dtype={"player_id": str}, low_memory=False)
            merged = fill_missing_from_info(war, info, key="player_id")
            merged.to_csv(path, index=False)
            logger.info("Updated %s", path)


def main(data_dir: str, divisions: list[int] | None = None, years: list[int] | None = None) -> None:
    data_dir_path = Path(data_dir)
    divisions = divisions or [1, 2, 3]

    if years is None:
        years = sorted(
            {
                int(path.stem.split("_")[-1])
                for _, _, path in roster_paths(data_dir_path, divisions, list(range(2000, 2101)))
            }
        )

    player_info = aggregate_players(data_dir_path, divisions, years)
    if player_info.empty:
        logger.info("No player data found.")
        return

    output_path = data_dir_path / "rosters" / "player_information.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_info.to_csv(output_path, index=False)
    logger.info("Saved %s players to %s", len(player_info), output_path)

    merge_to_rosters(data_dir_path, player_info, divisions, years)
    merge_to_war(data_dir_path, player_info, divisions, years)
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--years", nargs="+", type=int, default=None)
    args = parser.parse_args()

    main(args.data_dir, args.divisions, args.years)
