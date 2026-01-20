import argparse
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3],
                        help="Divisions to process (default: 1 2 3)")
    args = parser.parse_args()

    data_dir = args.data_dir

    for year in YEARS:

        run([sys.executable, "pbp_parser/main.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "get_er_matrix.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "get_linear_weights.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "add_pbp_metrics.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "get_guts.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "get_war.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "get_leaderboards.py",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

    run([sys.executable, "aggregate_player_info.py",
         "--data_dir", data_dir,
         "--divisions"] + [str(d) for d in args.divisions] +
        ["--years"] + [str(y) for y in YEARS])


if __name__ == "__main__":
    main()
