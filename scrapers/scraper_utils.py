import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, Route, sync_playwright

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"


def get_scraper_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


logger = get_scraper_logger(__name__)
RATE_LIMIT_STATUSES = frozenset((403, 429, 430))
LONG_PAUSE_STATUSES = frozenset((503,))


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
        min_delay: float | None = None,
        max_delay: float | None = None,
        backoff_base: float = 5.0,
        max_backoff: float = 300.0,
    ):
        self.base_delay = base_delay
        self.jitter_pct = jitter_pct
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._consecutive_errors = 0
        self._last_request_time: float | None = None
        self._pause_until: float | None = None

    def _jittered_delay(self) -> float:
        if self.min_delay is not None and self.max_delay is not None:
            low = min(self.min_delay, self.max_delay)
            high = max(self.min_delay, self.max_delay)
            return random.uniform(low, high)
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
                time.sleep(delay - elapsed)

        self._last_request_time = time.time()

    def on_success(self):
        self._consecutive_errors = 0

    def on_error(self, status_code: int):
        self._consecutive_errors += 1
        if status_code in LONG_PAUSE_STATUSES:
            pause_seconds = random.uniform(500, 1000)
            self._pause_until = time.time() + pause_seconds
            logger.info(f"  [rate-limit] got {status_code}, pausing for {pause_seconds:.0f}s")
        elif status_code >= 500:
            backoff = min(
                self.backoff_base * (2 ** (self._consecutive_errors - 1)), self.max_backoff
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
        last_modified: str | None = None,
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


BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}

BLOCKED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp4",
    ".webm",
    ".mp3",
    ".wav",
    ".mov",
    ".avi",
    ".m4v",
}

BLOCKED_3P_HOST_SUBSTRINGS = {
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "adsystem",
    "adsense",
    "facebook.com",
    "connect.facebook.net",
    "facebook.net",
}

ALLOWED_3P_HOST_SUBSTRINGS = {
    "go-mpulse.net",
    "akamai",
    "akamaihd.net",
}


def looks_like_block_page(html: str | None) -> bool:
    if not html:
        return False
    s = html.lower()
    return (
        "access denied" in s
        or ("akamai" in s and "reference" in s)
        or "request blocked" in s
        or "perimeterx" in s
        or "attention required" in s
    )


def _should_abort_request(request) -> bool:
    url = request.url
    rtype = request.resource_type

    if rtype in {"document", "xhr", "fetch"}:
        return False

    if rtype in BLOCKED_RESOURCE_TYPES:
        return True

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()

    if any(path.endswith(ext) for ext in BLOCKED_EXTS):
        return True

    if host.endswith("stats.ncaa.org"):
        return False

    if any(s in host for s in ALLOWED_3P_HOST_SUBSTRINGS):
        return False

    if any(s in host for s in BLOCKED_3P_HOST_SUBSTRINGS):
        return True

    if rtype in {"script", "stylesheet", "other"}:
        return True

    return False


def smart_block(route: Route):
    if _should_abort_request(route.request):
        route.abort()
    else:
        route.continue_()


@dataclass
class ScraperConfig:
    base_delay: float = 2.0
    jitter_pct: float = 0.3
    min_request_delay: float | None = None
    max_request_delay: float | None = None
    max_retries: int = 3
    timeout_ms: int = 30000
    daily_request_budget: int | None = 20000
    headless: bool = False
    block_resources: bool = False
    cache_html: bool = False
    cache_dir: Path | None = None
    concurrency: int = 1
    accept_downloads: bool = False
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


class ScraperSession:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.rate_limiter = RateLimiter(
            base_delay=self.config.base_delay,
            jitter_pct=self.config.jitter_pct,
            min_delay=self.config.min_request_delay,
            max_delay=self.config.max_request_delay,
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

    @property
    def requests_made(self) -> int:
        return self._request_count

    @property
    def requests_remaining(self) -> int | float:
        budget = self.config.daily_request_budget
        if budget is None:
            return float("inf")
        return max(0, int(budget) - self._request_count)

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        self._context = self._browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080},
            accept_downloads=self.config.accept_downloads,
            locale="en-US",
            timezone_id="America/Chicago",
        )
        self._context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        self._page = self._context.new_page()

        if self.config.block_resources:
            self._page.route("**/*", smart_block)

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

    def reset_page(self, clear_cookies: bool = False) -> None:
        if self._context is None:
            raise RuntimeError("Session not started. Use 'with ScraperSession() as session:'")
        try:
            if self._page:
                try:
                    self._page.goto("about:blank", timeout=5000)
                except Exception:
                    pass
                self._page.close()
        except Exception:
            pass

        if clear_cookies:
            try:
                self._context.clear_cookies()
            except Exception:
                pass

        self._page = self._context.new_page()
        if self.config.block_resources:
            self._page.route("**/*", smart_block)

    def fetch(
        self,
        url: str,
        wait_selector: str | None = None,
        wait_timeout: int | None = None,
    ) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected. Stopping scraper.")

        self.rate_limiter.wait()
        self._request_count += 1

        wait_timeout = wait_timeout or self.config.timeout_ms

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.page.goto(
                    url, timeout=self.config.timeout_ms, wait_until="domcontentloaded"
                )
                status = response.status if response else 0

                if status in RATE_LIMIT_STATUSES:
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError(
                        "Rate limit detected (HTTP 403/429/430). Stopping scraper."
                    )

                if status in LONG_PAUSE_STATUSES:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                if status >= 400:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                if wait_selector:
                    try:
                        self.page.wait_for_selector(wait_selector, timeout=wait_timeout)
                    except Exception:
                        pass

                self.rate_limiter.on_success()
                html = self.page.content()

                if looks_like_block_page(html):
                    self._hard_blocked = True
                    raise HardBlockError("Block page detected (soft block). Stopping scraper.")

                return html, status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(f"  [fetch] attempt {attempt}/{self.config.max_retries} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(min(10.0, 2.0 ** (attempt - 1) + 1.0))
                else:
                    return None, 0

        return None, 0

    def fetch_with_api_request(self, url: str) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected. Stopping scraper.")
        if self._context is None:
            raise RuntimeError("Session not started.")

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
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                status = response.status

                if status in RATE_LIMIT_STATUSES:
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError(
                        "Rate limit detected (HTTP 403/429/430). Stopping scraper."
                    )

                if status in LONG_PAUSE_STATUSES:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                if status >= 400:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

                self.rate_limiter.on_success()
                text = response.text()

                if looks_like_block_page(text):
                    self._hard_blocked = True
                    raise HardBlockError("Block page detected (soft block). Stopping scraper.")

                return text, status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(
                    f"  [api-fetch] attempt {attempt}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(10.0, 2.0 ** (attempt - 1) + 1.0))
                else:
                    return None, 0

        return None, 0


class AsyncScraperSession:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.rate_limiter = RateLimiter(
            base_delay=self.config.base_delay,
            jitter_pct=self.config.jitter_pct,
            min_delay=self.config.min_request_delay,
            max_delay=self.config.max_request_delay,
        )
        self._request_count = 0
        self._browser = None
        self._hard_blocked = False

    @property
    def hard_blocked(self) -> bool:
        return self._hard_blocked

    @property
    def requests_made(self) -> int:
        return self._request_count

    @property
    def requests_remaining(self) -> int | float:
        budget = self.config.daily_request_budget
        if budget is None:
            return float("inf")
        return max(0, int(budget) - self._request_count)

    async def start(self, playwright):
        self._browser = await playwright.chromium.launch(headless=self.config.headless)
        return self

    async def close(self):
        if self._browser:
            await self._browser.close()

    async def new_page(self):
        context = await self._browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/Chicago",
        )
        await context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        page = await context.new_page()

        if self.config.block_resources:
            await page.route(
                "**/*",
                lambda route: route.abort()
                if _should_abort_request(route.request)
                else route.continue_(),
            )

        return page

    async def fetch(
        self, page, url: str, wait_selector: str | None = None, wait_timeout: int | None = None
    ) -> tuple[str | None, int]:
        if self._hard_blocked:
            raise HardBlockError("Hard block already detected. Stopping scraper.")

        self.rate_limiter.wait()
        self._request_count += 1

        wait_timeout = wait_timeout or self.config.timeout_ms

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await page.goto(
                    url, timeout=self.config.timeout_ms, wait_until="domcontentloaded"
                )
                status = response.status if response else 0

                if status in RATE_LIMIT_STATUSES:
                    self._hard_blocked = True
                    self.rate_limiter.on_error(status)
                    raise HardBlockError(
                        "Rate limit detected (HTTP 403/429/430). Stopping scraper."
                    )

                if status in LONG_PAUSE_STATUSES:
                    self.rate_limiter.on_error(status)
                    if attempt < self.config.max_retries:
                        continue
                    return None, status

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

                if looks_like_block_page(html):
                    self._hard_blocked = True
                    raise HardBlockError("Block page detected (soft block). Stopping scraper.")

                return html, status

            except HardBlockError:
                raise
            except Exception as e:
                logger.info(
                    f"  [async-fetch] attempt {attempt}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries:
                    import asyncio

                    await asyncio.sleep(min(10.0, 2.0 ** (attempt - 1) + 1.0))
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
