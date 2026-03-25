"""Rename all d{N}_* data files to ncaa_{N}_* across the data directory."""
import argparse
import re
from pathlib import Path

PATTERN = re.compile(r"^d([123])_(.+)$")


def rename_files(data_dir: Path, dry_run: bool = False) -> None:
    renamed = 0
    for path in sorted(data_dir.rglob("d[123]_*")):
        if not path.is_file():
            continue
        m = PATTERN.match(path.name)
        if not m:
            continue
        new_name = f"ncaa_{m.group(1)}_{m.group(2)}"
        new_path = path.parent / new_name
        if dry_run:
            print(f"  {path.relative_to(data_dir)}  ->  {new_path.relative_to(data_dir)}")
        else:
            path.rename(new_path)
            print(f"renamed  {path.relative_to(data_dir)}  ->  {new_name}")
        renamed += 1

    print(f"\n{'Would rename' if dry_run else 'Renamed'} {renamed} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename d{{N}}_* files to ncaa_{{N}}_*")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--dry_run", action="store_true", help="Print changes without renaming")
    args = parser.parse_args()

    rename_files(Path(args.data_dir), dry_run=args.dry_run)
