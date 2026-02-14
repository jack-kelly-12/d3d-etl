import argparse

from s3_utils.sync import sync_directory_to_s3, sync_s3_prefix_to_directory
from scrapers.scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["push", "pull"], required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--delete_extra", action="store_true")
    parser.add_argument("--sse_mode", default=None)
    parser.add_argument("--sse_kms_key_id", default=None)
    args = parser.parse_args()

    if args.mode == "push":
        sync_directory_to_s3(
            local_dir=args.local_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            region=args.region,
            delete_extra=args.delete_extra,
            sse_mode=args.sse_mode,
            sse_kms_key_id=args.sse_kms_key_id,
        )
        return 0

    sync_s3_prefix_to_directory(
        bucket=args.bucket,
        prefix=args.prefix,
        local_dir=args.local_dir,
        region=args.region,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
