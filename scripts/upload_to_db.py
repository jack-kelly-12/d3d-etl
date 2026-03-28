import argparse
import fnmatch
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).parent.parent / "data"
DEFAULT_CONFIG = Path(__file__).parent / "db_table_config.json"

EXCLUDED_DIRS = {"_tmp", "headshots"}
YEAR_SUFFIX_RE = re.compile(r"_(\d{4})$")

DEDUP_KEYS = {
    "pbp": ["contest_id", "play_id"],
    "batting": ["player_id", "year", "division"],
    "pitching": ["player_id", "year", "division"],
    "batting_team": ["team_id", "year", "division"],
    "pitching_team": ["team_id", "year", "division"],
    "batting_lineups": ["player_id", "contest_id", "position"],
    "pitching_lineups": ["player_id", "contest_id"],
    "expected_runs": ["division", "year", "bases"],
    "guts_constants": ["division", "year"],
    "schedules": ["contest_id"],
}


def load_config(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        return json.load(f)


def match_table(rel_path: str, config: list[dict]) -> str | None:
    for entry in config:
        if fnmatch.fnmatch(rel_path, entry["pattern"]):
            return entry["table"]
    return None


def extract_year(path: Path) -> int | None:
    match = YEAR_SUFFIX_RE.search(path.stem)
    return int(match.group(1)) if match else None


def collect_files(year: int | None, config: list[dict]) -> dict[str, list[Path]]:
    tables: dict[str, list[Path]] = {}
    for path in sorted(DATA_ROOT.rglob("*.csv")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        file_year = extract_year(path)
        if year is not None and file_year is not None and file_year != year:
            continue
        rel = str(path.relative_to(DATA_ROOT))
        table = match_table(rel, config)
        if table is None:
            continue
        tables.setdefault(table, []).append(path)
    return tables


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def _dedup_sql(conn: sqlite3.Connection, table: str, keys: list[str]) -> int:
    key_cols = ", ".join(keys)
    conn.execute(
        f"DELETE FROM [{table}] WHERE rowid NOT IN "
        f"(SELECT MIN(rowid) FROM [{table}] GROUP BY {key_cols})"
    )
    removed = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return removed


def load_table(
    conn: sqlite3.Connection,
    table: str,
    files: list[Path],
    year: int | None,
) -> None:
    dedup_keys = DEDUP_KEYS.get(table)
    many_files = len(files) > 3 and dedup_keys is not None

    if many_files:
        _load_table_streaming(conn, table, files, year, dedup_keys)
    else:
        _load_table_bulk(conn, table, files, year, dedup_keys)


def _load_table_bulk(
    conn: sqlite3.Connection,
    table: str,
    files: list[Path],
    year: int | None,
    dedup_keys: list[str] | None,
) -> None:
    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  warning: {path.name}: {e}", file=sys.stderr)

    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)

    if dedup_keys and all(k in df.columns for k in dedup_keys):
        df = df.drop_duplicates(subset=dedup_keys, keep="first")

    if year is not None and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        year_df = df[df["year"] == year]
        if _table_exists(conn, table):
            conn.execute(f"DELETE FROM [{table}] WHERE year = ?", (year,))
            if not year_df.empty:
                year_df.to_sql(table, conn, if_exists="append", index=False)
        else:
            df.to_sql(table, conn, if_exists="replace", index=False)
    else:
        df.to_sql(table, conn, if_exists="replace", index=False)

    if dedup_keys and all(k in df.columns for k in dedup_keys) and _table_exists(conn, table):
        removed = _dedup_sql(conn, table, dedup_keys)
        if removed:
            print(f"  {table}: deduped {removed:,} rows on {dedup_keys}")

    count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0] if _table_exists(conn, table) else len(df)
    print(f"  {table}: {count:,} rows")


def _load_table_streaming(
    conn: sqlite3.Connection,
    table: str,
    files: list[Path],
    year: int | None,
    dedup_keys: list[str],
) -> None:
    total = 0
    first = True

    if year is not None and _table_exists(conn, table):
        conn.execute(f"DELETE FROM [{table}] WHERE year = ?", (year,))
        conn.commit()
        first = False

    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  warning: {path.name}: {e}", file=sys.stderr)
            continue

        if year is not None and "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] == year]
            if df.empty:
                continue

        mode = "replace" if first else "append"
        df.to_sql(table, conn, if_exists=mode, index=False)
        total += len(df)
        first = False
        print(f"    loaded {path.name} ({len(df):,} rows)")

    if total > 0:
        removed = _dedup_sql(conn, table, dedup_keys)
        total -= removed
        if removed:
            print(f"    deduped {removed:,} rows on {dedup_keys}")

    print(f"  {table}: {total:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load CSV data files into a SQLite database"
    )
    parser.add_argument("--db", required=True, help="SQLite database file path")
    parser.add_argument("--year", type=int, default=None, help="Only load files for this year")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Table config JSON file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    tables = collect_files(args.year, config)
    print(f"Loading {len(tables)} tables into {db_path}\n")

    with sqlite3.connect(db_path) as conn:
        for table, files in sorted(tables.items()):
            load_table(conn, table, files, args.year)

    print(f"\nDone: {db_path}")


if __name__ == "__main__":
    main()
