from __future__ import annotations

import re
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from scrapers.scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)

_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def _key_matches_years(key: str, years: list[int]) -> bool:
    """Return True if the key should be downloaded given a year filter.

    A key passes if it contains no 4-digit year, or if it contains at least
    one year that is in *years*.
    """
    year_strs = {str(y) for y in years}
    found = _YEAR_RE.findall(key)
    if not found:
        return True
    return any(y in year_strs for y in found)


def _iter_local_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _normalize_prefix(prefix: str) -> str:
    p = (prefix or "").strip().strip("/")
    return f"{p}/" if p else ""


def _s3_key_for_file(local_file: Path, local_root: Path, prefix: str) -> str:
    rel = local_file.relative_to(local_root).as_posix()
    return f"{_normalize_prefix(prefix)}{rel}"


def _list_s3_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_normalize_prefix(prefix)):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)
    keys.sort()
    return keys


def sync_directory_to_s3(
    local_dir: str | Path,
    bucket: str,
    prefix: str = "",
    region: str | None = None,
    delete_extra: bool = False,
    sse_mode: str | None = None,
    sse_kms_key_id: str | None = None,
    include_files: list[str] | None = None,
) -> int:
    local_root = Path(local_dir).expanduser().resolve()
    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    files = _iter_local_files(local_root)
    if include_files is not None:
        include_set = set(include_files)
        files = [f for f in files if f.name in include_set]
    logger.info(
        f"S3 push start: {len(files)} local files from {local_root} to s3://{bucket}/{prefix}"
    )

    uploaded = 0
    extra_args: dict[str, str] = {}
    if sse_mode:
        extra_args["ServerSideEncryption"] = sse_mode
    if sse_kms_key_id:
        extra_args["SSEKMSKeyId"] = sse_kms_key_id

    for file_path in files:
        key = _s3_key_for_file(file_path, local_root, prefix)
        if extra_args:
            client.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(file_path), bucket, key)
        uploaded += 1

    if delete_extra:
        remote_keys = set(_list_s3_keys(client, bucket, prefix))
        local_keys = {_s3_key_for_file(p, local_root, prefix) for p in files}
        stale = sorted(remote_keys - local_keys)
        for key in stale:
            client.delete_object(Bucket=bucket, Key=key)
        if stale:
            logger.info(f"S3 push delete: removed {len(stale)} stale objects")

    logger.info(f"S3 push done: uploaded {uploaded} files")
    return uploaded


def sync_s3_prefix_to_directory(
    bucket: str,
    prefix: str,
    local_dir: str | Path,
    region: str | None = None,
    *,
    skip_forbidden: bool = False,
    years: list[int] | None = None,
) -> int:
    """
    Pull objects from S3 to local.

    If skip_forbidden=True, objects that return 403 AccessDenied are logged and skipped
    instead of failing the whole sync.
    """
    local_root = Path(local_dir).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    keys = _list_s3_keys(client, bucket, prefix)

    base_prefix = _normalize_prefix(prefix)
    downloaded = 0
    forbidden = 0

    logger.info(f"S3 pull start: {len(keys)} objects from s3://{bucket}/{prefix} to {local_root}")

    for _idx, key in enumerate(keys, start=1):
        if years is not None and not _key_matches_years(key, years):
            logger.debug(f"S3 pull skip (year filter): {key}")
            continue
        rel = key[len(base_prefix) :] if base_prefix and key.startswith(base_prefix) else key
        if not rel:
            continue

        target = local_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            head = client.head_object(Bucket=bucket, Key=key)

            sse = head.get("ServerSideEncryption")
            kms = head.get("SSEKMSKeyId")
            if sse == "aws:kms":
                logger.info(f"S3 object uses SSE-KMS: key={key} kms_key_id={kms}")

            client.download_file(bucket, key, str(target))
            downloaded += 1

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", "")
            msg = e.response.get("Error", {}).get("Message", "")

            if status == 403 or code in {"403", "AccessDenied"}:
                forbidden += 1
                logger.error(f"S3 pull forbidden (403) key={key} code={code} message={msg}")
                if skip_forbidden:
                    continue
                raise

            logger.error(f"S3 pull failed key={key} code={code} status={status} message={msg}")
            raise

    logger.info(f"S3 pull done: downloaded {downloaded} files (forbidden={forbidden})")
    return downloaded
