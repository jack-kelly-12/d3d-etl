import argparse
import subprocess
import sys
from pathlib import Path

from s3_utils.sync import sync_directory_to_s3, sync_s3_prefix_to_directory

from .pipeline_config import PipelineConfig
from .scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)


def _run_command(cmd: list[str], cwd: Path) -> bool:
    logger.info(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}")
        return False


def _ncaa_divs(cfg: PipelineConfig) -> list[str]:
    """Convert integer divisions [1,2,3] to ncaa_* strings expected by most scrapers."""
    return [f"ncaa_{d}" for d in cfg.divisions]


def _build_year_commands(project_root: Path, year: int, cfg: PipelineConfig) -> list[list[str]]:
    ncaa_divs = _ncaa_divs(cfg)

    return [
        [
            sys.executable,
            "-m",
            "scrapers.collect_schedules",
            "--year", str(year),
            "--divisions", *ncaa_divs,
            "--team_ids_file", str(cfg.team_ids_file),
            "--outdir", str(cfg.schedules_outdir),
            "--base_delay", str(cfg.base_delay),
        ],
        [
            sys.executable,
            "-m",
            "scrapers.collect_game",
            "--year", str(year),
            "--divisions", *ncaa_divs,
            "--indir", str(cfg.schedules_outdir),
            "--pbp_outdir", str(cfg.pbp_outdir),
            "--lineups_outdir", str(cfg.lineups_outdir),
            "--base_delay", str(cfg.base_delay),
        ],
    ]


def _build_cube_commands(cfg: PipelineConfig) -> list[list[str]]:
    years_args = [str(y) for y in cfg.years]
    ncaa_divs = _ncaa_divs(cfg)
    cube_stats_dir = cfg.data_root / "cube_stats"
    team_history_file = cfg.data_root / "cube_team_history.csv"

    cmds: list[list[str]] = []
    for div in ncaa_divs:
        cmds.append([
            sys.executable, "-m", "scrapers.collect_cube_stats",
            "--team_history_file", str(team_history_file),
            "--division", div,
            "--outdir", str(cube_stats_dir),
            "--years", *years_args,
            "--run_remaining",
        ])
    cmds.append([
        sys.executable, "-m", "scrapers.collect_cube_player_info",
        "--data_dir", str(cfg.data_root),
        "--out_file", str(cube_stats_dir / "cube_player_info.csv"),
        "--run_remaining",
        "--years", *years_args,
        "--divisions", *ncaa_divs,
    ])
    cmds.append([
        sys.executable, "-m", "processors.reconcile_players",
        "--data_dir", str(cfg.data_root),
    ])
    return cmds


def _build_resolve_commands(cfg: PipelineConfig) -> list[list[str]]:
    years_args = [str(y) for y in cfg.years]
    ncaa_divs = _ncaa_divs(cfg)
    return [
        [
            sys.executable, "-m", "processors.map_ncaa_to_cube",
            "--data_dir", str(cfg.data_root),
            "--years", *years_args,
            "--divisions", *ncaa_divs,
        ],
    ]


def _build_rankings_command(cfg: PipelineConfig) -> list[str]:
    years_args = [str(y) for y in cfg.years]
    return [
        sys.executable,
        "-m",
        "scrapers.collect_rankings",
        "--outdir", str(cfg.rankings_data_dir),
        "--years", *years_args,
        "--divisions", *_ncaa_divs(cfg),
        "--base_delay", str(cfg.base_delay),
    ]


def run_pipeline(cfg: PipelineConfig, project_root: Path) -> int:
    any_failed = False

    if cfg.s3_pull_before_run and cfg.s3_bucket:
        logger.info("Pulling existing data from S3 before scraper run")
        sync_s3_prefix_to_directory(
            bucket=cfg.s3_bucket,
            prefix=cfg.s3_prefix,
            local_dir=cfg.data_root,
            region=cfg.s3_region,
        )

    try:
        for year in cfg.years:
            logger.info(f"Starting scraper sequence for year={year}")
            for cmd in _build_year_commands(project_root, year, cfg):
                ok = _run_command(cmd, project_root)
                any_failed = any_failed or (not ok)

        for cmd in _build_cube_commands(cfg):
            ok = _run_command(cmd, project_root)
            any_failed = any_failed or (not ok)

        for cmd in _build_resolve_commands(cfg):
            ok = _run_command(cmd, project_root)
            any_failed = any_failed or (not ok)

        rankings_cmd = _build_rankings_command(cfg)
        ok = _run_command(rankings_cmd, project_root)
        any_failed = any_failed or (not ok)

    finally:
        if cfg.s3_upload_after_run and cfg.s3_bucket:
            logger.info("Uploading data to S3 after scraper run")
            sync_directory_to_s3(
                local_dir=cfg.data_root,
                bucket=cfg.s3_bucket,
                prefix=cfg.s3_prefix,
                region=cfg.s3_region,
                delete_extra=cfg.s3_sync_delete,
            )
        else:
            logger.info("Skipping S3 upload (S3_UPLOAD_AFTER_RUN is false or S3_BUCKET not set)")

    return 1 if any_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", default=None, help="Optional .env file path")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--divisions", nargs="+", type=int, default=None)
    parser.add_argument("--base_delay", type=float, default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--team_ids_file", default=None)
    parser.add_argument("--schedules_outdir", default=None)
    parser.add_argument("--stats_outdir", default=None)
    parser.add_argument("--pbp_outdir", default=None)
    parser.add_argument("--lineups_outdir", default=None)
    parser.add_argument("--rankings_data_dir", default=None)
    parser.add_argument("--s3_bucket", default=None)
    parser.add_argument("--s3_prefix", default=None)
    parser.add_argument("--s3_region", default=None)
    parser.add_argument("--s3_pull_before_run", action="store_true")
    parser.add_argument("--no_s3_upload", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig.from_env(args.env_file)

    if args.years:
        cfg.years = args.years
    if args.divisions:
        cfg.divisions = args.divisions
    if args.base_delay is not None:
        cfg.base_delay = args.base_delay

    if args.data_root:
        cfg.data_root = Path(args.data_root).expanduser().resolve()
    if args.team_ids_file:
        cfg.team_ids_file = Path(args.team_ids_file).expanduser().resolve()
    if args.schedules_outdir:
        cfg.schedules_outdir = Path(args.schedules_outdir).expanduser().resolve()
    if args.stats_outdir:
        cfg.stats_outdir = Path(args.stats_outdir).expanduser().resolve()
    if args.pbp_outdir:
        cfg.pbp_outdir = Path(args.pbp_outdir).expanduser().resolve()
    if args.lineups_outdir:
        cfg.lineups_outdir = Path(args.lineups_outdir).expanduser().resolve()
    if args.rankings_data_dir:
        cfg.rankings_data_dir = Path(args.rankings_data_dir).expanduser().resolve()

    if args.s3_bucket is not None:
        cfg.s3_bucket = args.s3_bucket
    if args.s3_prefix is not None:
        cfg.s3_prefix = args.s3_prefix
    if args.s3_region is not None:
        cfg.s3_region = args.s3_region
    if args.s3_pull_before_run:
        cfg.s3_pull_before_run = True
    if args.no_s3_upload:
        cfg.s3_upload_after_run = False

    project_root = Path(__file__).resolve().parent.parent

    for path in [
        cfg.schedules_outdir,
        cfg.stats_outdir,
        cfg.pbp_outdir,
        cfg.lineups_outdir,
        cfg.rankings_data_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return run_pipeline(cfg, project_root)


if __name__ == "__main__":
    print("[start] scrapers.run_all", flush=True)
    raise SystemExit(main())
