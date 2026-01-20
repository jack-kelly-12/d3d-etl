BASE = "https://stats.ncaa.org"

DEFAULT_BASE_DELAY = 2.0
DEFAULT_JITTER_PCT = 0.3
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_MS = 30000
DEFAULT_DAILY_BUDGET = 20000

BLOCKED_RESOURCE_TYPES = {"image", "font", "stylesheet", "media", "other"}
BLOCKED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".css", ".mp4", ".webm", ".mp3", ".wav"
)
BLOCKED_DOMAINS = (
    "google-analytics.com", "googletagmanager.com", "facebook.com",
    "doubleclick.net", "adsense", "analytics"
)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
