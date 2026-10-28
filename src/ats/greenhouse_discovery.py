"""Greenhouse board DISCOVERY — "which companies have Greenhouse boards?".

This is now a THIN wrapper: the generic harvest → validate → cache machinery lives in
board_discovery.py (shared with Lever). Here we only supply the three Greenhouse-specific
pieces: which URLs to harvest, how to validate a token, and where to cache the directory.

Everyday job search just reads the cached directory (load_directory); building it is the
slow, occasional job (run `python -m src.ats.greenhouse_discovery`).
"""
import requests                                      # used by the Greenhouse-specific validator

from src.ats import greenhouse                       # reuse its BASE url for the jobs API
from src.ats import board_discovery                  # the shared discovery engine

# ── The three Greenhouse-specific pieces the generic engine needs ──
SOURCE = "greenhouse"                                 # board_directory rows are stored under this name
HOST_PATTERNS = ["boards.greenhouse.io/*", "job-boards.greenhouse.io/*"]   # URLs to harvest
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def validate_token(token: str) -> int | None:
    """Return the job count if `token` is a live Greenhouse board, else None. One light call.
    (Passed to the generic engine, which calls it for every harvested token.)"""
    try:
        resp = requests.get(f"{greenhouse.BASE}/{token}/jobs", headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            return len(resp.json().get("jobs", []))   # how many jobs this board has
    except (requests.RequestException, ValueError):
        pass
    return None                                       # not a live/valid board


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the Greenhouse board directory. Forwards to the shared
    engine with the Greenhouse-specific patterns/validator/path. Accepts the same keyword
    options as board_discovery.build_directory (limit_per_query, max_workers, …)."""
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token, **kwargs)


def load_directory() -> dict:
    """Load the stored Greenhouse board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.greenhouse_discovery`.
if __name__ == "__main__":
    # Standard operating procedure: FULL-SCALE build — harvest the entire latest crawl
    # (no per-query cap), validate every token, merge with the existing directory.
    result = build_directory(limit_per_query=None, max_workers=16, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live boards in the directory.")
