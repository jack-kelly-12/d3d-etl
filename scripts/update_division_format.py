import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DIVISION_MAP = {1: "ncaa_1", 2: "ncaa_2", 3: "ncaa_3"}


def update_division_in_file(path: Path, dry_run: bool = False) -> tuple[int, bool]:
    df = pd.read_csv(path, dtype={"division": "object"}, low_memory=False)

    if "division" not in df.columns:
        return 0, False

    before_unique = set(str(x) for x in df["division"].dropna().unique())

    def map_division(x):
        if pd.isna(x):
            return x
        if str(x).startswith("ncaa_"):
            return x
        try:
            x_int = int(float(x))
            if x_int in DIVISION_MAP:
                return DIVISION_MAP[x_int]
        except (ValueError, TypeError):
            pass
        return x

    df["division"] = df["division"].apply(map_division)

    after_unique = set(str(x) for x in df["division"].dropna().unique())
    changed = before_unique != after_unique

    if changed:
        if not dry_run:
            df.to_csv(path, index=False)
        return len(df), True
    return len(df), False


def main():
    parser = argparse.ArgumentParser(description="Update division format from 1/2/3 to ncaa_1/ncaa_2/ncaa_3")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/jackkelly/Desktop/d3d-etl/data",
        help="Root data directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    all_files = []
    for f in sorted(data_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(f, nrows=1, low_memory=False)
            if "division" in df.columns:
                all_files.append(f)
        except Exception:
            continue

    if not all_files:
        print(f"No files with division column found in {data_dir}")
        return

    print(f"Found {len(all_files)} files with division column")
    if args.dry_run:
        print("[DRY RUN] No files will be modified\n")

    total_rows = 0
    files_updated = 0

    for f in all_files:
        try:
            rows, changed = update_division_in_file(f, dry_run=args.dry_run)
            if changed:
                files_updated += 1
                total_rows += rows
                status = "[DRY RUN] " if args.dry_run else ""
                print(f"{status}{f.relative_to(data_dir)}: {rows} rows updated")
        except Exception as e:
            print(f"Error processing {f.relative_to(data_dir)}: {e}")

    print(f"\nTotal: {files_updated} files updated, {total_rows} rows affected")
    if not args.dry_run:
        print("Done! All division values have been updated to ncaa_1/ncaa_2/ncaa_3 format.")


if __name__ == "__main__":
    main()
