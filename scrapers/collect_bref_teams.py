from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag
from playwright.sync_api import BrowserContext, Page, sync_playwright

from .bref_stats.constants import BR_BASE, DEFAULT_HEADERS, DEFAULT_TIMEOUT_MS, REQUEST_DELAY_S
from .scraper_utils import get_scraper_logger

logger = get_scraper_logger(__name__)


def _parse_bref_team_id(href: str | None) -> str | None:
    if not href:
        return None
    absolute = urljoin(BR_BASE, href)
    query_value = parse_qs(urlparse(absolute).query).get("id", [None])[0]
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
        return BeautifulSoup(self.page.content(), "lxml")


def _conference_team_rows(
    browser: BrefBrowser,
    conference_url: str,
    *,
    division: int | None,
    bref_conf_id: str | None,
    conference: str,
    year_start: int,
    year_end: int,
) -> list[dict]:
    soup = browser.get_soup(conference_url)
    table = _find_table(soup, "lg_history")
    years = set(range(year_start, year_end + 1))
    rows: list[dict] = []

    for tr in table.select("tbody tr"):
        year_col = tr.find(["th", "td"], attrs={"data-stat": "year_ID"})
        if year_col is None:
            continue
        year_text = year_col.get_text(strip=True)
        m = re.search(r"\b(20\d{2})\b", year_text)
        if not m:
            continue
        year = int(m.group(1))
        if year not in years:
            continue

        team_col = tr.find("td", attrs={"data-stat": "team_ID"})
        if team_col is None:
            continue

        for anchor in team_col.find_all("a"):
            href = anchor.get("href") or ""
            if "/register/team.cgi" not in href:
                continue
            rows.append(
                {
                    "division": division,
                    "bref_conf_id": bref_conf_id,
                    "conference": conference,
                    "team_name": anchor.get_text(strip=True),
                    "bref_team_id": _parse_bref_team_id(href),
                    "year": year,
                }
            )
    return rows


def run(conferences_csv: str, year_start: int, year_end: int) -> pd.DataFrame:
    confs = pd.read_csv(conferences_csv)
    all_rows: list[dict] = []

    with BrefBrowser() as browser:
        for _, row in confs.iterrows():
            division = int(row["division"]) if pd.notna(row.get("division")) else None
            bref_conf_id = str(row["bref_conf_id"]).strip() if "bref_conf_id" in row else None
            conference = str(row["bref_conf_name"]).strip()
            conf_url = str(row["bref_conf_url"]).strip()
            try:
                rows = _conference_team_rows(
                    browser,
                    conf_url,
                    division=division,
                    bref_conf_id=bref_conf_id,
                    conference=conference,
                    year_start=year_start,
                    year_end=year_end,
                )
                all_rows.extend(rows)
                unique_team_names = len({str(x.get("team_name", "")).strip() for x in rows if x.get("team_name")})
                logger.info(
                    f"conference={conference} collected_rows={len(rows)} unique_team_names={unique_team_names}"
                )
            except Exception as exc:
                logger.warning(f"conference={conference} failed: {exc}")

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "division",
                "bref_conf_id",
                "conference",
                "team_name",
                "bref_team_id",
                "first_year",
                "last_year",
                "years_seen",
            ]
        )

    df = pd.DataFrame(all_rows)
    grouped = (
        df.groupby(["conference", "team_name"], as_index=False)
        .agg(
            division=("division", "first"),
            bref_conf_id=("bref_conf_id", "first"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .sort_values(["division", "conference", "team_name"])
        .reset_index(drop=True)
    )
    latest = (
        df.sort_values("year", ascending=False)
        .drop_duplicates(subset=["conference", "team_name"], keep="first")
        .loc[:, ["conference", "team_name", "bref_team_id"]]
    )
    out = grouped.merge(
        latest,
        on=["conference", "team_name"],
        how="left",
    )
    out["years_seen"] = (
        out["first_year"].astype("Int64").astype(str) + "-" + out["last_year"].astype("Int64").astype(str)
    )
    return out[
        [
            "division",
            "bref_conf_id",
            "conference",
            "team_name",
            "bref_team_id",
            "first_year",
            "last_year",
            "years_seen",
        ]
    ]


if __name__ == "__main__":
    logger.info("[start] scrapers.collect_bref_teams")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conferences_csv",
        default="/Users/jackkelly/Desktop/d3d-etl/data/baseball_reference_conferences.csv",
    )
    parser.add_argument("--year_start", type=int, default=2021)
    parser.add_argument("--year_end", type=int, default=2026)
    parser.add_argument("--outpath", default="/Users/jackkelly/Desktop/d3d-etl/data/bref_teams.csv")
    args = parser.parse_args()

    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = run(args.conferences_csv, year_start=args.year_start, year_end=args.year_end)
    df.to_csv(outpath, index=False)
    logger.info(f"saved {outpath} ({len(df)} rows)")

