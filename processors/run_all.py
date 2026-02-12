import argparse
import subprocess
import sys
from pathlib import Path

YEARS = [2021, 2022, 2023, 2024, 2025]
DIVISIONS = [1, 2, 3]


def run(cmd, outfile=None, skip=False):
    if skip and outfile is not None and Path(outfile).exists():
        print(f"skipping {outfile}")
        return
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/jackkelly/Desktop/d3d-etl/data")
    parser.add_argument("--divisions", nargs="+", type=int, default=DIVISIONS,
                        help="Divisions to process (default: 1 2 3)")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS,
                        help="Years to process (default: 2021-2025)")
    args = parser.parse_args()

    data_dir = args.data_dir

    for year in args.years:

     #    run([sys.executable, "-m", "processors.pbp_parser.main",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

     #    run([sys.executable, "-m", "processors.get_er_matrix",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

     #    run([sys.executable, "-m", "processors.get_linear_weights",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

     #    run([sys.executable, "-m", "processors.add_pbp_metrics",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

     #    run([sys.executable, "-m", "processors.get_guts",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

     #    run([sys.executable, "-m", "processors.get_war",
     #         "--data_dir", data_dir,
     #         "--year", str(year),
     #         "--divisions"] + [str(d) for d in args.divisions])

        run([sys.executable, "-m", "processors.leaderboards.main",
             "--data_dir", data_dir,
             "--year", str(year),
             "--divisions"] + [str(d) for d in args.divisions])

    run([sys.executable, "-m", "processors.aggregate_player_info",
         "--data_dir", data_dir,
         "--divisions"] + [str(d) for d in args.divisions] +
        ["--years"] + [str(y) for y in args.years])


if __name__ == "__main__":
    main()
