from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag
from playwright.sync_api import BrowserContext, Page, sync_playwright

from .bref_stats.constants import (
    BATTING_COLS,
    BATTING_MAP,
    BR_BASE,
    DEFAULT_HEADERS,
    DEFAULT_TIMEOUT_MS,
    HAND_MAP,
    PITCHING_COLS,
    PITCHING_MAP,
    REQUEST_DELAY_S,
)
from .scraper_utils import get_scraper_logger

META_COLS = {
    "division",
    "year",
    "team_name",
    "conference",
    "team_id",
    "bref_team_id",
    "ncaa_id",
}

logger = get_scraper_logger(__name__)


def _parse_id_from_href(href: str | None, query_key: str = "id") -> str | None:
    if not href:
        return None
    absolute = urljoin(BR_BASE, href)
    query_value = parse_qs(urlparse(absolute).query).get(query_key, [None])[0]
    if query_value:
        return query_value
    path = urlparse(absolute).path
    match = re.search(r"/([^/]+)$", path)
    if not match:
        return None
    tail = match.group(1)
    if "." in tail:
        tail = tail.split(".", 1)[0]
    return tail or None


def _find_table(soup: BeautifulSoup, table_id: str) -> Tag:
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        text = str(comment)
        if table_id not in text:
            continue
        comment_soup = BeautifulSoup(text, "lxml")
        table = comment_soup.find("table", id=table_id)
        if table is not None:
            return table
    raise ValueError(f"table not found: {table_id}")


def _cell(tr: Tag, data_stat: str) -> Any:
    cell = tr.find(["td", "th"], attrs={"data-stat": data_stat})
    return cell.get_text(strip=True) if cell else pd.NA


def _player_fields(td: Tag) -> dict[str, Any]:
    anchor = td.find("a")
    name = anchor.get_text(strip=True) if anchor else td.get_text(strip=True)
    href = anchor.get("href") if anchor else None
    player_url = urljoin(BR_BASE, href) if href else pd.NA
    full_text = td.get_text(strip=True) or ""
    marker = full_text[-1] if full_text else ""
    b_t = HAND_MAP.get(marker, "R")
    bref_player_id = _parse_id_from_href(href) if href else None
    return {
        "player_name": name,
        "b_t": b_t,
        "player_url": player_url,
        "bref_player_id": bref_player_id,
    }


class BrefBrowser:
    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS, delay_s: float = REQUEST_DELAY_S) -> None:
        self.timeout_ms = timeout_ms
        self.delay_s = delay_s
        self._playwright = None
        self._browser = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def __enter__(self) -> "BrefBrowser":
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self.context = self._browser.new_context(extra_http_headers=DEFAULT_HEADERS)
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.timeout_ms)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.page is not None:
            self.page.close()
            self.page = None
        if self.context is not None:
            self.context.close()
            self.context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def get_soup(self, url: str) -> BeautifulSoup:
        if self.page is None:
            raise RuntimeError("browser page is not initialized")
        time.sleep(self.delay_s)
        self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        html = self.page.content()
        return BeautifulSoup(html, "lxml")


def _scrape_table_rows(
    browser: BrefBrowser,
    url: str,
    table_id: str,
    out_cols: list[str],
    stat_map: dict[str, str],
    *,
    division: int | None,
    year: int,
    team_name: str,
    conference: str,
    team_id: int | None,
    bref_team_id: str | None,
    ncaa_id: int | None,
) -> pd.DataFrame:
    soup = browser.get_soup(url)
    table = _find_table(soup, table_id)

    rows: list[dict[str, Any]] = []
    for tr in table.select("tbody tr"):
        td = tr.find("td", attrs={"data-stat": "player"})
        if td is None:
            continue

        row: dict[str, Any] = {}
        row.update(_player_fields(td))
        row["class"] = pd.NA
        row["pos"] = pd.NA
        row["ht"] = pd.NA

        for out_col in out_cols:
            if out_col in row or out_col in META_COLS:
                continue
            stat = stat_map.get(out_col)
            row[out_col] = _cell(tr, stat) if stat else pd.NA

        row["division"] = division
        row["year"] = year
        row["team_name"] = team_name
        row["conference"] = conference
        row["team_id"] = team_id
        row["bref_team_id"] = bref_team_id
        row["ncaa_id"] = ncaa_id
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in out_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[out_cols]


def scrape_team_batting(
    browser: BrefBrowser,
    url: str,
    *,
    division: int | None,
    year: int,
    team_name: str,
    conference: str,
    team_id: int | None,
    bref_team_id: str | None,
    ncaa_id: int | None,
) -> pd.DataFrame:
    return _scrape_table_rows(
        browser,
        url,
        "team_batting",
        BATTING_COLS,
        BATTING_MAP,
        division=division,
        year=year,
        team_name=team_name,
        conference=conference,
        team_id=team_id,
        bref_team_id=bref_team_id,
        ncaa_id=ncaa_id,
    )


def scrape_team_pitching(
    browser: BrefBrowser,
    url: str,
    *,
    division: int | None,
    year: int,
    team_name: str,
    conference: str,
    team_id: int | None,
    bref_team_id: str | None,
    ncaa_id: int | None,
) -> pd.DataFrame:
    return _scrape_table_rows(
        browser,
        url,
        "team_pitching",
        PITCHING_COLS,
        PITCHING_MAP,
        division=division,
        year=year,
        team_name=team_name,
        conference=conference,
        team_id=team_id,
        bref_team_id=bref_team_id,
        ncaa_id=ncaa_id,
    )


def get_conference_teams_for_year(
    browser: BrefBrowser,
    conference_url: str,
    year: int,
) -> list[dict[str, str]]:
    soup = browser.get_soup(conference_url)
    table = _find_table(soup, "lg_history")

    teams: list[dict[str, str]] = []
    for tr in table.select("tbody tr"):
        year_col = tr.find(["th", "td"], attrs={"data-stat": "year_ID"})
        if year_col is None:
            continue
        year_text = year_col.get_text(strip=True)
        if not re.search(rf"\b{year}\b", year_text):
            continue

        team_col = tr.find("td", attrs={"data-stat": "team_ID"})
        if team_col is None:
            continue

        for anchor in team_col.find_all("a"):
            href = anchor.get("href") or ""
            if "/register/team.cgi" not in href:
                continue
            team_url = urljoin(BR_BASE, href)
            teams.append(
                {
                    "team_name": anchor.get_text(strip=True),
                    "team_url": team_url,
                    "bref_team_id": _parse_id_from_href(team_url, "id") or "",
                }
            )
        break
    return teams


def run(conferences_csv: str, year: int, division: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    confs = pd.read_csv(conferences_csv)
    if division is not None:
        confs["division"] = pd.to_numeric(confs["division"], errors="coerce").astype("Int64")
        confs = confs[confs["division"] == division].copy()
    logger.info(f"starting bref stats scrape: year={year} division={division} conferences={len(confs)}")
    all_batting: list[pd.DataFrame] = []
    all_pitching: list[pd.DataFrame] = []

    with BrefBrowser() as browser:
        for _, row in confs.iterrows():
            division = int(row["division"]) if pd.notna(row["division"]) else None
            conference = str(row["bref_conf_name"])
            conf_url = str(row["bref_conf_url"]).strip()

            teams = get_conference_teams_for_year(browser, conf_url, year)
            if not teams:
                logger.info(f"no teams for conference={conference} year={year}")
                continue

            logger.info(f"conference={conference} teams={len(teams)}")
            for team in teams:
                team_url = team["team_url"]
                team_name = team["team_name"]
                bref_team_id = team["bref_team_id"] or None

                try:
                    all_batting.append(
                        scrape_team_batting(
                            browser,
                            team_url,
                            division=division,
                            year=year,
                            team_name=team_name,
                            conference=conference,
                            team_id=None,
                            bref_team_id=bref_team_id,
                            ncaa_id=None,
                        )
                    )

                except Exception as exc:
                    logger.warning(f"failed batting team={team_name} bref_team_id={bref_team_id}: {exc}")

                try:
                    all_pitching.append(
                        scrape_team_pitching(
                            browser,
                            team_url,
                            division=division,
                            year=year,
                            team_name=team_name,
                            conference=conference,
                            team_id=None,
                            bref_team_id=bref_team_id,
                            ncaa_id=None,
                        )
                    )

                except Exception as exc:
                    logger.warning(f"failed pitching team={team_name} bref_team_id={bref_team_id}: {exc}")

    batting = (
        pd.concat(all_batting, ignore_index=True) if all_batting else pd.DataFrame(columns=BATTING_COLS)
    )
    pitching = (
        pd.concat(all_pitching, ignore_index=True)
        if all_pitching
        else pd.DataFrame(columns=PITCHING_COLS)
    )
    logger.info(
        f"completed bref stats scrape: year={year} division={division} batting_rows={len(batting)} pitching_rows={len(pitching)}"
    )
    return batting, pitching


if __name__ == "__main__":
    logger.info("[start] scrapers.collect_bref_stats")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conferences_csv",
        default="/Users/jackkelly/Desktop/d3d-etl/data/baseball_reference_conferences.csv",
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--divisions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--outdir", default="/Users/jackkelly/Desktop/d3d-etl/data/stats")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for division in sorted({int(d) for d in args.divisions}):
        batting, pitching = run(args.conferences_csv, year=args.year, division=division)
        batting_out = outdir / f"d{division}_batting_{args.year}.csv"
        pitching_out = outdir / f"d{division}_pitching_{args.year}.csv"
        batting.to_csv(batting_out, index=False)
        pitching.to_csv(pitching_out, index=False)

        logger.info(f"d{division} batting: {len(batting)} rows -> {batting_out}")
        logger.info(f"d{division} pitching: {len(pitching)} rows -> {pitching_out}")

