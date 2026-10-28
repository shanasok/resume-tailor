"""Rippling board DISCOVERY — "which companies have Rippling boards?".

A thin wrapper over the shared board_discovery engine (same pattern as greenhouse_discovery
and lever_discovery). We supply only the Rippling-specific pieces: the URLs to harvest
(ats.rippling.com/*), how to validate a slug against the public ATS API, and where to cache
the directory.

Build/refresh it occasionally:  python -m src.ats.rippling_discovery
Everyday job search just reads the cached directory via load_directory().

Note: harvested first-path segments include Rippling locale prefixes (e.g. "en-CA") as well as
real company slugs; the locales simply fail validation below and are dropped, so only live
boards survive into the directory.
"""
import requests                                        # used by the Rippling-specific validator

from src.ats import rippling                           # reuse its BASE url for the ATS API
from src.ats import board_discovery                    # the shared discovery engine

# ── The Rippling-specific pieces the generic engine needs ──
SOURCE = "rippling"                                    # board_directory rows are stored under this name
HOST_PATTERNS = ["ats.rippling.com/*"]                 # public Rippling boards live at ats.rippling.com/{slug}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def validate_token(token: str) -> int | None:
    """Return the job count if `token` is a live Rippling board, else None. One light call."""
    try:
        resp = requests.get(f"{rippling.BASE}/{token}/jobs", headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return len(data) if isinstance(data, list) else None   # the API returns a list
    except (requests.RequestException, ValueError):
        pass
    return None


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the Rippling board directory (forwards to the shared engine).
    Merges with any boards already in the directory, so a rebuild only grows the set or prunes
    dead boards — never loses good ones (e.g. a manually-seeded slug)."""
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token, **kwargs)


def load_directory() -> dict:
    """Load the stored Rippling board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.rippling_discovery`.
if __name__ == "__main__":
    result = build_directory(limit_per_query=None, max_workers=16, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live Rippling boards in the directory.")
