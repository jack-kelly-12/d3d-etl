from __future__ import annotations

from pathlib import Path

import boto3

from scrapers.scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)


def _iter_local_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _normalize_prefix(prefix: str) -> str:
    p = prefix.strip().strip("/")
    return f"{p}/" if p else ""


def _s3_key_for_file(local_file: Path, local_root: Path, prefix: str) -> str:
    rel = local_file.relative_to(local_root).as_posix()
    return f"{_normalize_prefix(prefix)}{rel}"


def _list_s3_keys(client, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_normalize_prefix(prefix)):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.add(key)
    return keys


def sync_directory_to_s3(
    local_dir: str | Path,
    bucket: str,
    prefix: str = "",
    region: str | None = None,
    delete_extra: bool = False,
    sse_mode: str | None = None,
    sse_kms_key_id: str | None = None,
) -> int:
    local_root = Path(local_dir).expanduser().resolve()
    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    files = _iter_local_files(local_root)
    logger.info(f"S3 push start: {len(files)} local files from {local_root} to s3://{bucket}/{prefix}")

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
        remote_keys = _list_s3_keys(client, bucket, prefix)
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
) -> int:
    local_root = Path(local_dir).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    keys = sorted(_list_s3_keys(client, bucket, prefix))

    base_prefix = _normalize_prefix(prefix)
    downloaded = 0
    logger.info(f"S3 pull start: {len(keys)} objects from s3://{bucket}/{prefix} to {local_root}")

    for key in keys:
        rel = key[len(base_prefix):] if base_prefix and key.startswith(base_prefix) else key
        if not rel:
            continue
        target = local_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(target))
        downloaded += 1

    logger.info(f"S3 pull done: downloaded {downloaded} files")
    return downloaded
