import subprocess
import sys
from pathlib import Path
import argparse

YEARS = [2021, 2022, 2023, 2024, 2025]
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
    args = parser.parse_args()

    for year in YEARS:
        for div in DIVISIONS:
            sched_file = Path(f"../new_data/schedules/d{div}_schedules_{year}.csv")
            stats_file_b = Path(f"../new_data/stats/d{div}_batting_{year}.csv")
            stats_file_p = Path(f"../new_data/stats/d{div}_pitching_{year}.csv")
            pbp_file = Path(f"../new_data/pbp/d{div}_pbp_{year}.csv")
    
            run([sys.executable, "collect_schedules.py",
                 "--year", str(year),
                 "--divisions", str(div),
                 "--team_ids_file", "../new_data/ncaa_team_history.csv",
                 "--outdir", "../new_data/schedules"],
                sched_file, args.skip_existing)

            run([sys.executable, "collect_stats.py",
                 "--year", str(year),
                 "--divisions", str(div),
                 "--team_ids_file", "../new_data/ncaa_team_history.csv",
                 "--outdir", "../new_data/stats"],
                stats_file_b, args.skip_existing)

            run([sys.executable, "collect_pbp.py",
                 "--year", str(year),
                 "--divisions", str(div),
                 "--indir", "../new_data/schedules",
                 "--outdir", "../new_data/pbp"],
                pbp_file, args.skip_existing)

if __name__ == "__main__":
    main()
