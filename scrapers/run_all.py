import argparse
import subprocess
import sys
from pathlib import Path

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
DIVISIONS = [1, 2, 3]


def run(cmd, outfile, skip):
    if skip and outfile.exists():
        print(f"skipping {outfile}")
        return
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--divisions", nargs="+", type=int, default=DIVISIONS)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--daily_budget", type=int, default=20000)
    args = parser.parse_args()

    common_args = [
        "--headless" if args.headless else "",
        "--base_delay", str(args.base_delay),
        "--daily_budget", str(args.daily_budget),
    ]
    common_args = [a for a in common_args if a]

    for year in args.years:
        for div in args.divisions:
            pbp_file = Path(f"../data/pbp/d{div}_pbp_{year}.csv")

            run([sys.executable, "collect_pbp.py",
                 "--year", str(year),
                 "--divisions", str(div),
                 "--indir", "../data/schedules",
                 "--outdir", "../data/pbp",
                 *common_args],
                pbp_file, args.skip_existing)


if __name__ == "__main__":
    main()
