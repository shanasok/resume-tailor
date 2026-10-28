"""SmartRecruiters board DISCOVERY — "which companies have public SmartRecruiters boards?".

A thin wrapper over the shared board_discovery engine (same pattern as ashby_discovery /
greenhouse_discovery / lever_discovery). A SmartRecruiters board's identity is a single
company identifier that lives in the FIRST path segment (jobs.smartrecruiters.com/{Company}),
so — unlike Workday — the engine's DEFAULT first-path-segment token extractor works as-is.

One subtlety handled by the default extractor for free: SmartRecruiters URLs preserve the
identifier's original casing ("BoschGroup"), but the Posting API is CASE-INSENSITIVE, so the
default extractor's lowercasing ("boschgroup") is fine — no custom token_from_url needed.

Build/refresh it occasionally:  python -m src.ats.smartrecruiters_discovery
Everyday job search just reads the cached directory via load_directory().
"""
import requests                                        # used by the SmartRecruiters-specific validator

from src.ats import board_discovery                    # the shared discovery engine
from src.ats import smartrecruiters                    # reuse its BASE url + error type

SOURCE = "smartrecruiters"                             # board_directory rows are stored under this name
# Public SmartRecruiters boards are linked from both hostnames; harvest company slugs from each.
HOST_PATTERNS = ["jobs.smartrecruiters.com/*", "careers.smartrecruiters.com/*"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}


def validate_token(token: str) -> int | None:
    """Return the (US) job count if `token` is a live SmartRecruiters board, else None. One
    light call (limit=1); a live board → 200 with a numeric totalFound, an unknown identifier
    also → 200 but totalFound 0 (filtered to None here so empty boards aren't stored).

    Counts US postings only (?country=us), matching what the scan will actually walk — a board
    with jobs but none in the US is not worth storing for a US-only search."""
    try:
        resp = requests.get(f"{smartrecruiters.BASE}/{token}/postings",
                            params={"limit": 1, "country": smartrecruiters.COUNTRY},
                            headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            total = resp.json().get("totalFound")
            if isinstance(total, (int, float)) and total > 0:
                return int(total)
    except (requests.RequestException, ValueError):
        pass
    return None


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the SmartRecruiters board directory (forwards to the shared
    engine). Merges with any boards already in the directory, so a rebuild only grows the set
    or prunes dead boards — never loses good ones."""
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token, **kwargs)


def load_directory() -> dict:
    """Load the stored SmartRecruiters board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.smartrecruiters_discovery`.
if __name__ == "__main__":
    result = build_directory(limit_per_query=None, max_workers=16, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live SmartRecruiters boards in the directory.")
