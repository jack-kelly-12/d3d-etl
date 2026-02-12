import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, Route, sync_playwright

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"

def get_scraper_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


logger = get_scraper_logger(__name__)


class HardBlockError(RuntimeError):
    pass


@dataclass
class RequestMetadata:
    url: str
    content_hash: str
    fetched_at: str
    status: int
    bytes: int
    etag: str | None = None
    last_modified: str | None = None


class RateLimiter:
    def __init__(
        self,
        base_delay: float = 2.0,
        jitter_pct: float = 0.3,
        backoff_base: float = 5.0,
        max_backoff: float = 300.0,
    ):
        self.base_delay = base_delay
        self.jitter_pct = jitter_pct
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._consecutive_errors = 0
        self._last_request_time: float | None = None
        self._pause_until: float | None = None

    def _jittered_delay(self) -> float:
        jitter = random.uniform(-self.jitter_pct, self.jitter_pct)
        return self.base_delay * (1 + jitter)

    def wait(self):
        now = time.time()
        if self._pause_until and now < self._pause_until:
            sleep_time = self._pause_until - now
            logger.info(f"  [rate-limit] paused, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self._pause_until = None

        if self._last_request_time:
            elapsed = now - self._last_request_time
            delay = self._jittered_delay()
            if elapsed < delay:
                sleep_time = delay - elapsed
                time.sleep(sleep_time)

        self._last_request_time = time.time()

    def on_success(self):
        self._consecutive_errors = 0

    def on_error(self, status_code: int):
        self._consecutive_errors += 1
        if status_code in (403, 429):
            pause_seconds = random.uniform(120, 600)
            self._pause_until = time.time() + pause_seconds
            logger.info(f"  [rate-limit] got {status_code}, pausing for {pause_seconds:.0f}s")
        elif status_code >= 500:
            backoff = min(
                self.backoff_base * (2 ** (self._consecutive_errors - 1)),
                self.max_backoff
            )
            logger.info(f"  [rate-limit] got {status_code}, backing off {backoff:.1f}s")
            time.sleep(backoff)


class RequestCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "_metadata.json"
        self._metadata: dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                self._metadata = json.loads(self.metadata_file.read_text())
            except (OSError, json.JSONDecodeError):
                self._metadata = {}

    def _save_metadata(self):
        self.metadata_file.write_text(json.dumps(self._metadata, indent=2))

    def _cache_key(self, contest_id: int, page_type: str) -> str:
        return f"{contest_id}_{page_type}"

    def _cache_path(self, contest_id: int, page_type: str) -> Path:
        return self.cache_dir / f"{contest_id}" / f"{page_type}.html"

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha1(content.encode()).hexdigest()

    def get_cached(self, contest_id: int, page_type: str) -> str | None:
        path = self._cache_path(contest_id, page_type)
        if path.exists():
            return path.read_text()
        return None

    def is_cached(self, contest_id: int, page_type: str) -> bool:
        return self._cache_path(contest_id, page_type).exists()

    def save(
        self,
        contest_id: int,
        page_type: str,
        content: str,
        status: int,
        etag: str | None = None,
        last_modified: str | None = None
    ):
        path = self._cache_path(contest_id, page_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        key = self._cache_key(contest_id, page_type)
        self._metadata[key] = {
            "contest_id": contest_id,
            "page_type": page_type,
            "content_hash": self._hash_content(content),
            "fetched_at": datetime.now().isoformat(),
            "status": status,
            "bytes": len(content.encode()),
            "etag": etag,
            "last_modified": last_modified,
        }
        self._save_metadata()

    def get_metadata(self, contest_id: int, page_type: str) -> dict | None:
        return self._metadata.get(self._cache_key(contest_id, page_type))


BLOCKED_RESOURCE_TYPES = {"image", "font", "stylesheet", "media"}


def block_resources(route: Route):
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        route.abort()
    else:
        route.continue_()


def block_resources_by_url(route: Route):
    url = route.request.url.lower()
    blocked_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
                          ".woff", ".woff2", ".ttf", ".otf", ".eot",
                          ".css", ".mp4", ".webm", ".mp3", ".wav")
    blocked_domains = ("google-analytics.com", "googletagmanager.com", "facebook.com",
                       "doubleclick.net", "adsense", "analytics")

    if url.endswith(blocked_extensions):
        route.abort()
    elif any(domain in url for domain in blocked_domains):
        route.abort()
    else:
        route.continue_()


@dataclass
class ScraperConfig:
    base_delay: float = 2.0
    jitter_pct: float = 0.3
    max_retries: int = 3
    timeout_ms: int = 30000
    headless: bool = False
    block_resources: bool = False
    cache_html: bool = False
    cache_dir: Path | None = None
    concurrency: int = 1
    accept_downloads: bool = False


class ScraperSession:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.rate_limiter = RateLimiter(
            base_delay=self.config.base_delay,
            jitter_pct=self.config.jitter_pct
        )
        self.cache: RequestCache | None = None
        if self.config.cache_html and self.config.cache_dir:
            self.cache = RequestCache(self.config.cache_dir)

        self._request_count = 0
        self._playwright = None
        self._browser = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._hard_blocked = False

    @property
    def hard_blocked(self) -> bool:
        return self._hard_blocked



    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        self._context = self._browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            accept_downloads=self.config.accept_downloads,
        )
        self._page = self._context.new_page()

        if self.config.block_resources:
            self._page.route("**/*", block_resources)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Session not started. Use 'with ScraperSession() as session:'")
        return self._page

    def fetch(
        self,
        url: str,
        wait_selector: str | None = None,
        wait_timeout: int | None = None,
    ) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected (HTTP 403). Stopping scraper.")

        self.rate_limiter.wait()
        self._request_count += 1

        wait_timeout = wait_timeout or self.config.timeout_ms

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.page.goto(url, timeout=self.config.timeout_ms, wait_until="domcontentloaded")
                status = response.status if response else 0

                if status in (403,):
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError("Hard block detected (HTTP 403). Stopping scraper.")

                if wait_selector:
                    try:
                        self.page.wait_for_selector(wait_selector, timeout=wait_timeout)
                    except Exception:
                        pass

                self.rate_limiter.on_success()
                html = self.page.content()
                return html, status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(f"  [fetch] attempt {attempt}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    return None, 0

        return None, 0

    def fetch_with_api_request(
        self,
        url: str,
    ) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected (HTTP 403). Stopping scraper.")

        self.rate_limiter.wait()
        self._request_count += 1

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._context.request.get(
                    url,
                    timeout=self.config.timeout_ms,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                    }
                )
                status = response.status

                if status == 403:
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError("Hard block detected (HTTP 403). Stopping scraper.")

                if status >= 400:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                self.rate_limiter.on_success()
                return response.text(), status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(f"  [api-fetch] attempt {attempt}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    return None, 0

        return None, 0


class AsyncScraperSession:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.rate_limiter = RateLimiter(
            base_delay=self.config.base_delay,
            jitter_pct=self.config.jitter_pct
        )
        self._request_count = 0
        self._browser = None
        self._hard_blocked = False

    @property
    def hard_blocked(self) -> bool:
        return self._hard_blocked

    async def start(self, playwright):
        self._browser = await playwright.chromium.launch(headless=self.config.headless)
        return self

    async def close(self):
        if self._browser:
            await self._browser.close()

    async def new_page(self):
        context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        if self.config.block_resources:
            await page.route("**/*", lambda route: (
                route.abort() if route.request.resource_type in BLOCKED_RESOURCE_TYPES
                else route.continue_()
            ))

        return page

    async def fetch(
        self,
        page,
        url: str,
        wait_selector: str | None = None,
        wait_timeout: int | None = None,
    ) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected (HTTP 403). Stopping scraper.")

        self.rate_limiter.wait()
        self._request_count += 1

        wait_timeout = wait_timeout or self.config.timeout_ms

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await page.goto(url, timeout=self.config.timeout_ms, wait_until="domcontentloaded")
                status = response.status if response else 0

                if status == 403:
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError("Hard block detected (HTTP 403). Stopping scraper.")

                if status >= 400:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                if wait_selector:
                    try:
                        await page.wait_for_selector(wait_selector, timeout=wait_timeout)
                    except Exception:
                        pass

                self.rate_limiter.on_success()
                html = await page.content()
                return html, status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(f"  [async-fetch] attempt {attempt}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    return None, 0

        return None, 0


def get_completed_contest_ids(outdir: Path, div: int, year: int, filename_pattern: str) -> set[int]:
    fpath = outdir / filename_pattern.format(div=div, year=year)
    if not fpath.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(fpath)
        if "contest_id" in df.columns:
            return set(df["contest_id"].dropna().astype(int).unique())
    except Exception:
        pass
    return set()


def short_break_every(n: int, idx: int, sleep_s: float = 60.0) -> None:
    if n > 0 and idx > 0 and idx % n == 0:
        time.sleep(sleep_s)


def cooldown_between_batches(batch_end: int, total: int, cooldown_s: float) -> None:
    if cooldown_s > 0 and batch_end < total:
        time.sleep(cooldown_s)


def retry_backoff_sleep(attempt: int, cap_s: float = 10.0) -> None:
    time.sleep(min(cap_s, 2.0 ** (attempt - 1) + 1.0))
