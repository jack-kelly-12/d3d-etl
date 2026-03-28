import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from processors.logging_utils import get_logger

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
DIVISIONS = ['ncaa_1', 'ncaa_2', 'ncaa_3']
logger = get_logger(__name__)


def run(cmd, outfile=None, skip=False):
    if skip and outfile is not None and Path(outfile).exists():
        logger.info("skipping %s", outfile)
        return
    logger.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _new_cube_ids(data_dir: str, divisions: list[str], year: int) -> bool:
    cube_stats_dir = Path(data_dir) / "cube_stats"
    info_path = cube_stats_dir / "cube_player_info.csv"

    done: set[int] = set()
    if info_path.exists():
        try:
            df = pd.read_csv(info_path, usecols=["cube_player_id"])
            done = set(df["cube_player_id"].dropna().astype(int))
        except Exception:
            pass

    for div in divisions:
        for kind in ("batting", "pitching"):
            path = cube_stats_dir / f"{div}_{kind}_{year}.csv"
            if not path.exists():
                continue
            for col in ("cube_player_id", "player_id"):
                try:
                    df = pd.read_csv(path, usecols=[col])
                    ids = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
                    if any(i not in done for i in ids):
                        return True
                    break
                except Exception:
                    continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/jackkelly/Desktop/d3d-etl/data")
    parser.add_argument("--divisions", nargs="+", type=str, default=DIVISIONS)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()

    data_dir = args.data_dir

    for year in args.years:
        if _new_cube_ids(data_dir, args.divisions, year):
            logger.info("New cube player IDs found for %d — scraping player info", year)
            run(
                [
                    sys.executable, "-m", "scrapers.collect_cube_player_info",
                    "--data_dir", data_dir,
                    "--out_file", str(Path(data_dir) / "cube_stats" / "cube_player_info.csv"),
                    "--run_remaining",
                    "--years", str(year),
                    "--divisions",
                ]
                + [str(d) for d in args.divisions]
            )
            run(
                [
                    sys.executable, "-m", "processors.reconcile_players",
                    "--data_dir", data_dir,
                ]
            )
        else:
            logger.info("No new cube player IDs for %d", year)

        run(
            [
                sys.executable, "-m", "processors.map_ncaa_to_cube",
                "--data_dir", data_dir,
                "--years", str(year),
                "--divisions",
            ]
            + [str(d) for d in args.divisions]
        )

        run(
            [
                sys.executable, "-m", "processors.pbp_parser",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions",
            ]
            + [str(d) for d in args.divisions]
        )

        for div in args.divisions:
            run([
                sys.executable, "-m", "processors.add_pbp_metrics",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions", str(div),
            ])

        for div in args.divisions:
            run([
                sys.executable, "-m", "processors.get_linear_weights",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions", str(div),
            ])

        run(
            [
                sys.executable, "-m", "processors.get_guts",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions",
            ]
            + [str(d) for d in args.divisions]
        )

        for div in args.divisions:
            run([
                sys.executable, "-m", "processors.get_er_matrix",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions", str(div),
            ])

        run(
            [
                sys.executable, "-m", "processors.get_war",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions",
            ]
            + [str(d) for d in args.divisions]
        )

        for div in args.divisions:
            run([
                sys.executable, "-m", "processors.leaderboards.main",
                "--data_dir", data_dir,
                "--year", str(year),
                "--divisions", str(div),
            ])


if __name__ == "__main__":
    main()
