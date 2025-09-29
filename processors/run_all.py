import subprocess
import sys
from pathlib import Path
import argparse

YEARS = [2021, 2022, 2023, 2024, 2025]


def run(cmd, outfile=None, skip=False):
    if skip and outfile is not None and Path(outfile).exists():
        print(f"skipping {outfile}")
        return
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../new_data")
    args = parser.parse_args()

    data_dir = args.data_dir

    for year in YEARS:

        run([sys.executable, "parse_pbp.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "get_er_matrix.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "get_linear_weights.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "add_pbp_metrics.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "get_guts.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "get_war.py",
             "--data_dir", data_dir,
             "--year", str(year)])

        run([sys.executable, "get_leaderboards.py",
             "--data_dir", data_dir,
             "--year", str(year)])


if __name__ == "__main__":
    main()
