import argparse
import random
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from .scraper_utils import HardBlockError, ScraperConfig, ScraperSession

BASE = "https://masseyratings.com"


def build_url(year: int, division: int) -> str:
    return f"{BASE}/cbase{year}/ncaa-d{division}/ratings"


def _sleep_jitter(seconds: float, jitter_frac: float = 0.25) -> None:
    if seconds <= 0:
        return
    jitter = seconds * jitter_frac
    time.sleep(max(0.0, seconds + random.uniform(-jitter, jitter)))


def scrape_massey_rankings(
    data_dir: str | None,
    years: list[int],
    divisions: list[int],
    outdir: str | None = None,
    page_wait_s: float = 2.0,
    between_downloads_s: float = 4.0,
    between_divisions_s: float = 240.0,
    max_retries: int = 6,
    timeout_ms: int = 45000,
    base_delay: float = 8.0,
    hardblock_cooldown_s: float = 600.0,
):
    if outdir:
        outdir = Path(outdir)
    elif data_dir:
        outdir = Path(data_dir) / "rankings"
    else:
        raise ValueError("Either data_dir or outdir must be provided.")
    outdir.mkdir(parents=True, exist_ok=True)

    years = sorted({int(y) for y in years})
    divisions = sorted({int(d) for d in divisions})

    base_config = ScraperConfig(
        base_delay=base_delay,
        block_resources=False,
        max_retries=1,
        timeout_ms=timeout_ms,
        accept_downloads=True,
    )

    for year in years:
        for div in divisions:
            url = build_url(year, div)
            outpath = outdir / f"d{div}_rankings_{year}.csv"

            print(f"\n=== Massey export: year={year} div={div} ===")
            print(f"goto {url}")

            success = False

            for attempt in range(1, max_retries + 1):
                with ScraperSession(base_config) as session:
                    try:
                        html, status = session.fetch(
                            url,
                            wait_selector="select#pulldownlinks",
                            wait_timeout=timeout_ms,
                        )

                        if not html or (status is not None and status >= 400):
                            print(f"attempt {attempt}/{max_retries} failed: HTTP {status}")
                            _sleep_jitter(min(120.0, 2.0 ** (attempt - 1) * 5.0))
                            continue

                        _sleep_jitter(page_wait_s)

                        with session.page.expect_download(timeout=timeout_ms) as dl_info:
                            session.page.select_option("select#pulldownlinks", value="exportCSV")

                        download = dl_info.value
                        tmp_path = Path(download.path())

                        df = pd.read_csv(tmp_path)
                        df = normalize_massey_rankings(df, division=div, year=year)
                        df.to_csv(outpath, index=False)

                        try:
                            tmp_path.unlink()
                        except FileNotFoundError:
                            pass

                        print(f"saved {outpath} ({len(df)} rows)")
                        success = True

                        _sleep_jitter(between_downloads_s)
                        break

                    except HardBlockError:
                        cooldown = min(3600.0, hardblock_cooldown_s * (2 ** (attempt - 1)))
                        print(f"[403] cooldown ~{int(cooldown)}s; retrying (attempt {attempt}/{max_retries})")
                        _sleep_jitter(cooldown, jitter_frac=0.20)
                        continue

                    except (PlaywrightTimeoutError, Exception) as e:
                        print(f"attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
                        _sleep_jitter(min(180.0, 2.0 ** (attempt - 1) * 5.0))
                        continue

            if not success:
                print(f"FAILED year={year} div={div} url={url}")

            _sleep_jitter(between_divisions_s, jitter_frac=0.30)


def _parse_years(vals: list[str]) -> list[int]:
    out: list[int] = []
    for v in vals:
        v = str(v).strip()
        if not v:
            continue
        if "-" in v:
            a, b = v.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(v))
    return out


def normalize_massey_rankings(df: pd.DataFrame, division: int, year: int) -> pd.DataFrame:
    df = df.copy()

    cols = list(df.columns)

    base_cols = [
        ("massey_team", cols[0]),
        ("conference", cols[1]),
        ("record", cols[2]),
        ("win_pct", cols[3]),
    ]

    metric_names = ["rat", "pwr", "off", "def", "hfa", "sos"]

    start = 4
    new_cols = {}

    for i, metric in enumerate(metric_names):
        rank_col = cols[start + i * 2]
        val_col = cols[start + i * 2 + 1]

        new_cols[rank_col] = f"{metric}_rank"
        new_cols[val_col] = f"{metric}_val"

    rename_map = {orig: new for new, orig in base_cols}
    rename_map.update(new_cols)

    df = df.rename(columns=rename_map)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("%", "pct", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    for c in df.columns:
        if c.endswith("_rank"):
            numeric = pd.to_numeric(df[c], errors="coerce")
            df[c] = numeric.round().astype("Int64")
        elif c.endswith("_val") or c == "win_pct":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["division"] = division
    df["year"] = year

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--years", nargs="+", required=True)
    parser.add_argument("--divisions", nargs="+", type=int, required=True)
    parser.add_argument("--page_wait_s", type=float, default=5.0)
    parser.add_argument("--between_downloads_s", type=float, default=30.0)
    parser.add_argument("--between_divisions_s", type=float, default=240.0)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--timeout_ms", type=int, default=45000)
    parser.add_argument("--base_delay", type=float, default=8.0)
    parser.add_argument("--hardblock_cooldown_s", type=float, default=600.0)
    args = parser.parse_args()

    years = _parse_years(args.years)

    scrape_massey_rankings(
        data_dir=args.data_dir,
        years=years,
        divisions=args.divisions,
        outdir=args.outdir,
        page_wait_s=args.page_wait_s,
        between_downloads_s=args.between_downloads_s,
        between_divisions_s=args.between_divisions_s,
        max_retries=args.max_retries,
        timeout_ms=args.timeout_ms,
        base_delay=args.base_delay,
        hardblock_cooldown_s=args.hardblock_cooldown_s,
    )
