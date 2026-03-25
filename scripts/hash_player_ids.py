import argparse
import hashlib
from pathlib import Path

import pandas as pd

# Doesn't have to be secret, just needs to be consistent
SALT = "d3d-etl-player-id-salt"
STATS_DIR = Path("data/cube_stats")

def hash_player_id(player_id: int | str | float | None, salt: str = SALT) -> str:
    if player_id is None or pd.isna(player_id) or str(player_id).strip() in ("", "--", "nan"):
        return None
    if isinstance(player_id, float) and player_id == int(player_id):
        player_id = int(player_id)
    input_str = f"{salt}:{player_id}"
    return hashlib.sha256(input_str.encode()).hexdigest()[:16]


def generate_id_for_missing(
    player_name: str,
    team_id: int | str | None,
    division: str | None = None,
    salt: str = SALT,
) -> str:
    parts = []
    if player_name and not pd.isna(player_name):
        parts.append(str(player_name).strip().lower())
    if team_id and not pd.isna(team_id):
        parts.append(str(team_id).strip().lower())
    if division and not pd.isna(division):
        parts.append(str(division).strip().lower())

    if not parts:
        return None

    input_str = f"{salt}:missing:{':'.join(parts)}"
    return hashlib.sha256(input_str.encode()).hexdigest()[:16]


def process_file(path: Path, salt: str = SALT, dry_run: bool = False) -> tuple[int, int]:
    df = pd.read_csv(path, dtype={"player_id": "object"})

    hashed_count = 0
    generated_count = 0

    new_player_ids = []
    for _, row in df.iterrows():
        player_id = row.get("player_id")
        player_name = row.get("player_name")
        team_id = row.get("team_id")
        division = row.get("division")

        if player_id and not pd.isna(player_id) and str(player_id).strip() not in ("", "--", "nan"):
            try:
                pid_int = int(float(player_id))
                hashed_id = hash_player_id(pid_int, salt)
                new_player_ids.append(hashed_id)
                hashed_count += 1
            except (ValueError, TypeError):
                hashed_id = generate_id_for_missing(player_name, team_id, division, salt)
                new_player_ids.append(hashed_id)
                generated_count += 1
        else:
            hashed_id = generate_id_for_missing(player_name, team_id, division, salt)
            new_player_ids.append(hashed_id)
            generated_count += 1

    df["cube_player_id"] = df["player_id"]
    df["player_id"] = new_player_ids

    if not dry_run:
        df.to_csv(path, index=False)

    return hashed_count, generated_count


def main():
    parser = argparse.ArgumentParser(description="Hash player_id with salt and generate IDs for missing players")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/jackkelly/Desktop/d3d-etl/data/cube_stats",
        help="Directory containing cube stats CSV files",
    )
    parser.add_argument(
        "--salt",
        type=str,
        default=SALT,
        help="Salt for hashing (default: built-in salt)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    stats_dir = Path(args.data_dir)
    if not stats_dir.exists():
        print(f"Error: Directory not found: {stats_dir}")
        return

    stats_files = sorted(stats_dir.glob("*_batting_*.csv")) + sorted(stats_dir.glob("*_pitching_*.csv"))
    player_info_file = stats_dir / "cube_player_info.csv"

    files = stats_files
    if player_info_file.exists():
        files.append(player_info_file)

    if not files:
        print(f"No cube stats files found in {stats_dir}")
        return

    print(f"Processing {len(files)} files...")
    if args.dry_run:
        print("[DRY RUN] No files will be modified\n")

    total_hashed = 0
    total_generated = 0

    for f in files:
        try:
            if f.name == "cube_player_info.csv":
                df = pd.read_csv(f, dtype={"player_id": "object"})
                original_count = len(df)
                hashed_count = 0

                new_player_ids = []
                for _, row in df.iterrows():
                    player_id = row.get("player_id")
                    if player_id and not pd.isna(player_id) and str(player_id).strip() not in ("", "--", "nan"):
                        try:
                            pid_int = int(float(player_id))
                            hashed_id = hash_player_id(pid_int, args.salt)
                            new_player_ids.append(hashed_id)
                            hashed_count += 1
                        except (ValueError, TypeError):
                            new_player_ids.append(None)
                    else:
                        new_player_ids.append(None)

                df["cube_player_id"] = df["player_id"]
                df["player_id"] = new_player_ids
                if not args.dry_run:
                    df.to_csv(f, index=False)

                total_hashed += hashed_count
                status = "[DRY RUN] " if args.dry_run else ""
                print(f"{status}{f.name}: {hashed_count} hashed")
            else:
                hashed, generated = process_file(f, salt=args.salt, dry_run=args.dry_run)
                total_hashed += hashed
                total_generated += generated
                status = "[DRY RUN] " if args.dry_run else ""
                print(f"{status}{f.name}: {hashed} hashed, {generated} generated")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    print(f"\nTotal: {total_hashed} hashed, {total_generated} generated")
    if not args.dry_run:
        print("Done! All player_id values have been hashed/generated.")


if __name__ == "__main__":
    main()
