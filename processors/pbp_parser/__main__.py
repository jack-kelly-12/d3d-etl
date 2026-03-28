import argparse

from .main import main

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="Root directory containing the data folders")
parser.add_argument("--year", required=True, type=int)
parser.add_argument("--divisions", required=True, nargs="+", type=str)
args = parser.parse_args()

main(args.data_dir, args.year, args.divisions)
