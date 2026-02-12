from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out or default


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()


def _normalize_bucket_and_prefix(bucket_raw: str | None, prefix_raw: str | None) -> tuple[str | None, str]:
    if not bucket_raw:
        return None, (prefix_raw or "").strip().strip("/")

    bucket_raw = bucket_raw.strip()
    prefix = (prefix_raw or "").strip().strip("/")

    if bucket_raw.startswith("s3://"):
        parsed = urlparse(bucket_raw)
        bucket = (parsed.netloc or "").strip()
        path_prefix = (parsed.path or "").strip().strip("/")
        if path_prefix:
            prefix = f"{path_prefix}/{prefix}".strip("/") if prefix else path_prefix
        return bucket or None, prefix

    return bucket_raw, prefix


@dataclass
class PipelineConfig:
    years: list[int]
    divisions: list[int]
    base_delay: float

    data_root: Path
    team_ids_file: Path
    schedules_outdir: Path
    stats_outdir: Path
    pbp_outdir: Path
    lineups_outdir: Path
    rankings_data_dir: Path

    s3_bucket: str | None
    s3_prefix: str
    s3_region: str | None
    s3_pull_before_run: bool
    s3_upload_after_run: bool
    s3_sync_delete: bool
    s3_sse_mode: str | None
    s3_sse_kms_key_id: str | None

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "PipelineConfig":
        load_dotenv(dotenv_path=env_file or ".env", override=False)

        data_root = _env_path("DATA_ROOT", "./data")
        s3_bucket, s3_prefix = _normalize_bucket_and_prefix(
            os.getenv("S3_BUCKET"),
            os.getenv("S3_PREFIX"),
        )

        return cls(
            years=_parse_int_list(os.getenv("YEARS"), [2026]),
            divisions=_parse_int_list(os.getenv("DIVISIONS"), [1, 2, 3]),
            base_delay=float(os.getenv("BASE_DELAY", "10.0")),
            data_root=data_root,
            team_ids_file=_env_path("TEAM_IDS_FILE", str(data_root / "ncaa_team_history.csv")),
            schedules_outdir=_env_path("SCHEDULES_OUTDIR", str(data_root / "schedules")),
            stats_outdir=_env_path("STATS_OUTDIR", str(data_root / "stats")),
            pbp_outdir=_env_path("PBP_OUTDIR", str(data_root / "pbp")),
            lineups_outdir=_env_path("LINEUPS_OUTDIR", str(data_root / "lineups")),
            rankings_data_dir=_env_path("RANKINGS_DATA_DIR", str(data_root)),
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or None,
            s3_pull_before_run=_parse_bool(os.getenv("S3_PULL_BEFORE_RUN"), False),
            s3_upload_after_run=_parse_bool(os.getenv("S3_UPLOAD_AFTER_RUN"), True),
            s3_sync_delete=_parse_bool(os.getenv("S3_SYNC_DELETE"), False),
            s3_sse_mode=(os.getenv("S3_SSE_MODE") or "").strip() or None,
            s3_sse_kms_key_id=(os.getenv("S3_SSE_KMS_KEY_ID") or "").strip() or None,
        )
