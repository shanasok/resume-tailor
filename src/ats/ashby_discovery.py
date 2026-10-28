"""Ashby board DISCOVERY — "which companies have public Ashby boards?".

A thin wrapper over the shared board_discovery engine (same pattern as greenhouse_discovery /
lever_discovery / rippling_discovery). Ashby's board identity is a single slug that lives in
the FIRST path segment (jobs.ashbyhq.com/{slug}), so — unlike Workday — the engine's DEFAULT
first-path-segment token extractor works as-is; we supply only the Ashby-specific pieces: the
URLs to harvest, how to validate a slug against the posting API, and where to cache.

Build/refresh it occasionally:  python -m src.ats.ashby_discovery
Everyday job search just reads the cached directory via load_directory().
"""
import requests                                        # used by the Ashby-specific validator

from src.ats import ashby                              # reuse its BASE url + error type
from src.ats import board_discovery                    # the shared discovery engine

SOURCE = "ashby"                                       # board_directory rows are stored under this name
HOST_PATTERNS = ["jobs.ashbyhq.com/*"]                 # public Ashby boards live at jobs.ashbyhq.com/{slug}
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}


def validate_token(token: str) -> int | None:
    """Return the job count if `token` is a live Ashby board, else None. One light call
    (a live board → 200 with a jobs list; an unknown slug → 404)."""
    try:
        resp = requests.get(f"{ashby.BASE}/{token}", params={"includeCompensation": "false"},
                            headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            jobs = resp.json().get("jobs")
            if isinstance(jobs, list):
                return len(jobs)
    except (requests.RequestException, ValueError):
        pass
    return None


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the Ashby board directory (forwards to the shared engine).
    Merges with any boards already in the directory, so a rebuild only grows the set or prunes
    dead boards — never loses good ones."""
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token, **kwargs)


def load_directory() -> dict:
    """Load the stored Ashby board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.ashby_discovery`.
if __name__ == "__main__":
    result = build_directory(limit_per_query=None, max_workers=16, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live Ashby boards in the directory.")
