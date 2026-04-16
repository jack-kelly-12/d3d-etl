import argparse
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cloudscraper
import pandas as pd
import requests
from bs4 import BeautifulSoup

from scrapers.constants import (
    CUBE_BATTING_RENAMES,
    CUBE_COLUMN_RENAMES,
    CUBE_PITCHING_RENAMES,
    CUBE_STATS_YEARS,
)
from scripts.hash_player_ids import SALT, hash_player_id, generate_id_for_missing

BASE_URL = "https://thebaseballcube.com"
  
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DROP_COLS = {
    "draft_info", "hilvl", "mlb_years", "stat_years",
    "status", "cur_org", "cur_lev", "tbc"
}


class DuplicateColumnError(Exception):
    pass


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    new_cols: list[str] = []
    for col in df.columns:
        c = str(col)
        if c == "#":
            c = "number"
        c = c.replace("%", "_pct")
        c = re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()
        if not c:
            c = "unnamed"
        new_cols.append(c)

    seen: set[str] = set()
    for c in new_cols:
        if c in seen:
            raise DuplicateColumnError(
                f"Duplicate column '{c}' after normalization. Raw columns: {list(df.columns)}"
            )
        seen.add(c)

    df.columns = new_cols
    return df


def _apply_player_id_hash(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in df.columns:
        return df
    df = df.rename(columns={"player_id": "cube_player_id"})
    df["player_id"] = df["cube_player_id"].apply(lambda x: hash_player_id(x, SALT))
    mask = df["player_id"].isna()
    if mask.any():
        df.loc[mask, "player_id"] = df.loc[mask].apply(
            lambda r: generate_id_for_missing(r.get("player_name"), r.get("team_name"), salt=SALT),
            axis=1,
        )
    return df


def _extract_player_id(href: str) -> int | None:
    if not href:
        return None
    m = re.search(r"/player/(\d+)/", href)
    return int(m.group(1)) if m else None


def _stats_url(college_id: int, year: int) -> str:
    return f"{BASE_URL}/content/stats_college/{year}~{college_id}/"


def _parse_table(
    table, school_meta: dict[str, Any], table_type: str = "batting",
) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()

    header_rows = table.find_all("tr", class_=re.compile("header-row"))
    headers: list[str] = []
    for hr in header_rows:
        for td in hr.find_all("td"):
            text = td.get_text(strip=True)
            headers.append(text if text else f"col_{len(headers)}")

    rows: list[dict[str, Any]] = []
    for tr in table.find_all("tr", class_=re.compile("data-row")):
        cells = tr.find_all("td")
        if not cells:
            continue
        if "record(s)" in tr.get_text(strip=True).lower():
            continue

        row: dict[str, Any] = {"player_id": None, "player_url": None}
        for i, td in enumerate(cells):
            col = headers[i] if i < len(headers) else f"col_{i}"
            row[col] = td.get_text(strip=True)
            a = td.find("a")
            if a and a.get("href") and "player" in a["href"]:
                row["player_id"] = _extract_player_id(a["href"])
                row["player_url"] = urljoin(BASE_URL, a["href"])

        row.update(school_meta)
        rows.append(row)

    df = normalize_cols(pd.DataFrame(rows))
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    renames = {**CUBE_COLUMN_RENAMES, **(CUBE_BATTING_RENAMES if table_type == "batting" else CUBE_PITCHING_RENAMES)}
    df = df.rename(columns=renames)
    df = df.replace({"-": pd.NA, "--": pd.NA})
    df = df.where(df.notna(), other=pd.NA)
    return df


def _scrape_school_year(
    session,
    url: str,
    school_meta: dict[str, Any],
    timeout: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    resp = session.get(url, timeout=timeout)
    status = resp.status_code
    if status >= 400:
        return pd.DataFrame(), pd.DataFrame(), status

    soup = BeautifulSoup(resp.text, "html.parser")
    all_tables = {t.get("id"): t for t in soup.find_all("table") if t.get("id")}

    df_bat = _parse_table(all_tables.get("grid1"), school_meta, "batting")
    df_pit = _parse_table(all_tables.get("grid2"), school_meta, "pitching")
    return df_bat, df_pit, status


def _apply_ncaa_ids(
    outdir_path: Path,
    division: str,
    years: list[int],
    team_mappings_path: Path,
    ncaa_history_path: Path,
) -> None:
    team_map = pd.read_csv(team_mappings_path, usecols=["ncaa_team_name", "org_id", "cube_college_id", "ncaa_slug"])
    ncaa_history = pd.read_csv(ncaa_history_path, usecols=["org_id", "year", "conference", "division"])
    ncaa_history = ncaa_history.rename(columns={"division": "ncaa_division"})

    for year in years:
        for stat_type in ("batting", "pitching"):
            path = outdir_path / f"{division}_{stat_type}_{year}.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path)
            df = df.drop(columns=["conference"], errors="ignore")
            merged = df.merge(
                team_map.rename(columns={"cube_college_id": "team_id"}),
                on="team_id",
                how="left",
            ).merge(
                ncaa_history,
                on=["org_id", "year"],
                how="left",
            )

            merged["cube_player_id"] = pd.to_numeric(merged["cube_player_id"], errors="coerce")
            merged = merged.dropna(subset=["cube_player_id"])
            merged["cube_player_id"] = merged["cube_player_id"].astype(int)
            needs_hash = merged["player_id"].isna() if "player_id" in merged.columns else pd.Series(True, index=merged.index)
            merged.loc[needs_hash, "player_id"] = merged.loc[needs_hash, "cube_player_id"].apply(lambda x: hash_player_id(x, SALT))
            still_null = merged["player_id"].isna()
            if still_null.any():
                merged.loc[still_null, "player_id"] = merged.loc[still_null].apply(
                    lambda r: generate_id_for_missing(r.get("player_name"), r.get("ncaa_team_name"), salt=SALT),
                    axis=1,
                )

            has_ncaa_div = merged["ncaa_division"].notna()
            merged.loc[has_ncaa_div, "division"] = merged.loc[has_ncaa_div, "ncaa_division"]
            wrong_div = merged["division"] != division
            if wrong_div.any():
                bad_teams = merged.loc[wrong_div, "team_id"].unique()
                print(f"  removing {wrong_div.sum()} rows from {len(bad_teams)} teams not in {division}")
                merged = merged[~wrong_div]

            print(f"\n--- {path.name} ---")
            merged["team_id"] = merged["ncaa_slug"]
            merged["team_name"] = merged["ncaa_team_name"]
            merged = merged.drop(columns=["ncaa_slug", "ncaa_team_name", "org_id", "ncaa_division"])
            merged.to_csv(path, index=False)
            print(f"  saved {path.name}")


def scrape_cube_stats(
    team_history_file: str,
    division: str,
    outdir: str,
    years: list[int] | None = None,
    batch_size: int = 50,
    team_mappings_file: str = "/Users/jackkelly/Desktop/d3d-etl/data/team_mappings.csv",
    ncaa_history_file: str = "/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv",
):
    if years is None:
        years = list(CUBE_STATS_YEARS)

    years_sorted = sorted(years)
    history = pd.read_csv(team_history_file)

    all_ids: set = set()
    total_school_years = 0
    for year in years_sorted:
        yt = history[(history["division"] == division) & (history["year"] == year)]
        all_ids.update(yt["college_id"].dropna().unique())
        total_school_years += yt["college_id"].nunique()

    if not all_ids:
        print(f"[error] no teams found for division '{division}' in years {years_sorted}")
        print(f"  available: {sorted(history['division'].unique())}")
        return

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    all_bat: dict[int, list[pd.DataFrame]] = {y: [] for y in years_sorted}
    all_pit: dict[int, list[pd.DataFrame]] = {y: [] for y in years_sorted}

    n_distinct_schools = len(all_ids)
    print(
        f"\n=== {division} cube stats {years_sorted} — "
        f"{n_distinct_schools} distinct schools, {total_school_years} school-seasons "
        f"(year-by-year, chronological) ==="
    )

    session = cloudscraper.create_scraper()
    session.headers.update(HEADERS)

    try:
        n_scrapes = 0
        for year in years_sorted:
            yt = history[(history["division"] == division) & (history["year"] == year)]
            colleges_year = (
                yt[["college_id", "college_name"]]
                .drop_duplicates(subset=["college_id"])
                .sort_values("college_name")
                .reset_index(drop=True)
            )
            n_year = len(colleges_year)
            if n_year == 0:
                continue
            print(f"\n--- {year} ({division}): {n_year} schools ---")

            for idx, row in enumerate(colleges_year.itertuples(index=False)):
                college_name = row.college_name
                college_id = row.college_id

                print(f"\n  [{idx + 1}/{n_year}] {college_name}")

                url = _stats_url(college_id, year)
                school_meta = {"year": year, "college": college_name, "team_id": college_id, "division": division, "conference": ""}

                try:
                    df_bat, df_pit, status = _scrape_school_year(session, url, school_meta)
                except DuplicateColumnError as e:
                    print(f"  DUPLICATE COLUMN ERROR: {e}")
                    sys.exit(1)
                except requests.exceptions.HTTPError as e:
                    print(f"  BLOCKED: {e}")
                    return
                except Exception as e:
                    print(f"  ERROR ({year}): {e}")
                    time.sleep(2.0)
                    continue

                if status >= 400:
                    if status in (403, 429):
                        print(f"  HTTP {status} — rate limited, stopping")
                        return
                    continue

                if df_bat.empty and df_pit.empty:
                    print(f"  no data on TBC")
                    continue

                if not df_bat.empty:
                    all_bat[year].append(_apply_player_id_hash(df_bat))
                    print(f"  batting  -> {len(df_bat)} rows")
                if not df_pit.empty:
                    all_pit[year].append(_apply_player_id_hash(df_pit))
                    print(f"  pitching -> {len(df_pit)} rows")

                time.sleep(1.0)

                n_scrapes += 1
                if n_scrapes % batch_size == 0 and n_scrapes < total_school_years:
                    cooldown = random.uniform(15.0, 20.0)
                    print(f"\n[cooldown] sleeping {cooldown:.1f}s after {n_scrapes} school-seasons")
                    time.sleep(cooldown)

    finally:
        session.close()

    print("\n[writing] saving scraped data...")
    for year in years_sorted:
        if all_bat[year]:
            out = outdir_path / f"{division}_batting_{year}.csv"
            pd.concat(all_bat[year], ignore_index=True).to_csv(out, index=False)
            print(f"  wrote {out.name}")
        if all_pit[year]:
            out = outdir_path / f"{division}_pitching_{year}.csv"
            pd.concat(all_pit[year], ignore_index=True).to_csv(out, index=False)
            print(f"  wrote {out.name}")

    print("\n[ncaa ids] applying team mappings...")
    _apply_ncaa_ids(outdir_path, division, years_sorted, Path(team_mappings_file), Path(ncaa_history_file))


if __name__ == "__main__":
    print("[start] scrapers.collect_cube_stats", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_history_file", default="/Users/jackkelly/Desktop/d3d-etl/data/cube_team_history.csv")
    parser.add_argument("--division", required=True)
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/cube_stats")
    parser.add_argument("--year", "--years", dest="years", nargs="+", type=int, default=CUBE_STATS_YEARS)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--team_mappings_file", default="/Users/jackkelly/Desktop/d3d-etl/data/team_mappings.csv")
    parser.add_argument("--ncaa_history_file", default="/Users/jackkelly/Desktop/d3d-etl/data/ncaa_team_history.csv")
    args = parser.parse_args()

    scrape_cube_stats(
        team_history_file=args.team_history_file,
        division=args.division,
        outdir=args.outdir,
        years=args.years,
        batch_size=args.batch_size,
        team_mappings_file=args.team_mappings_file,
        ncaa_history_file=args.ncaa_history_file,
    )
