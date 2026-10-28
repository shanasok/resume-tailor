"""Lever board DISCOVERY — "which companies have Lever boards?".

A thin wrapper over the shared board_discovery engine (same pattern as greenhouse_discovery).
We supply only the Lever-specific pieces: the URLs to harvest (jobs.lever.co/*), how to
validate a token against the Postings API, and where to cache the directory.

Build it occasionally:  python -m src.ats.lever_discovery
Everyday job search just reads the cached directory via load_directory().
"""
import requests                                      # used by the Lever-specific validator

from src.ats import lever                            # reuse its BASE url for the postings API
from src.ats import board_discovery                  # the shared discovery engine

# ── The Lever-specific pieces the generic engine needs ──
SOURCE = "lever"                                     # board_directory rows are stored under this name
HOST_PATTERNS = ["jobs.lever.co/*"]                  # public Lever boards live at jobs.lever.co/{token}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# IMPORTANT — which Common Crawl snapshots to harvest from:
# Lever's CURRENT robots.txt blocks crawlers (Content-Signal: no), so the LATEST crawls
# contain only jobs.lever.co/robots.txt — zero company board pages. OLDER snapshots predate
# that block and are full of jobs.lever.co/{company} URLs (we measured ~450-480 unique slugs
# per crawl in 2023-2025, vs 0 in 2026). So we harvest a spread of these PRE-BLOCK crawls and
# union them; every harvested slug is then validated against Lever's LIVE API below, so only
# companies that still have a real board today survive. (These crawl IDs are permanent;
# browse/adjust at https://index.commoncrawl.org/collinfo.json — unlike Greenhouse, which is
# crawlable, so greenhouse_discovery just uses the newest crawl.)
DEFAULT_COLLECTIONS = ["CC-MAIN-2025-26", "CC-MAIN-2024-42", "CC-MAIN-2023-23"]


def validate_token(token: str) -> int | None:
    """Return the posting count if `token` is a live Lever board, else None. One light call."""
    try:
        resp = requests.get(f"{lever.BASE}/{token}", params={"mode": "json"},
                            headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return len(data) if isinstance(data, list) else None   # the API returns a list
    except (requests.RequestException, ValueError):
        pass
    return None


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the Lever board directory (forwards to the shared engine).
    Defaults to harvesting the PRE-BLOCK crawls in DEFAULT_COLLECTIONS (see the note above);
    pass collections=[...] to override."""
    kwargs.setdefault("collections", DEFAULT_COLLECTIONS)   # use pre-block crawls unless overridden
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token, **kwargs)


def load_directory() -> dict:
    """Load the stored Lever board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.lever_discovery`.
if __name__ == "__main__":
    result = build_directory(limit_per_query=None, max_workers=16, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live Lever boards in the directory.")
