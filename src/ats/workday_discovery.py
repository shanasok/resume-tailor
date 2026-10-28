"""Workday board DISCOVERY — "which companies have public Workday career sites?".

A wrapper over the shared board_discovery engine (same pattern as the other
*_discovery modules), with ONE twist: a Workday board's identity is three parts —
tenant + data-center host + site — and two of them live in the HOSTNAME
(nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite), so this module supplies its
own token extractor via the engine's `token_from_url_fn` hook instead of the default
first-path-segment one. Tokens are packed "{tenant}.{dc}@{site}" (see workday.py).

Build/refresh it occasionally:  python -m src.ats.workday_discovery
Everyday job search just reads the cached directory via load_directory().

BE EXTRA POLITE here: *.myworkdayjobs.com covers a huge share of the Fortune 500, so
the Common Crawl harvest is large and validation probes many distinct companies. Use
modest parallelism (max_workers ~8, not the default 16) and do a first run with
limit_per_query set (e.g. 200) as a smoke test before a full build. The directory is
cached long-term — rebuilds are occasional, never per-scan.
"""
import re                                              # host/site pattern matching
from urllib.parse import urlparse, unquote             # pull tenant/dc/site out of a URL

import requests                                        # used by the Workday-specific validator

from src.ats import board_discovery                    # the shared discovery engine
from src.ats import workday                            # reuse its token parsing + API base

SOURCE = "workday"                                     # board_directory rows are stored under this name
HOST_PATTERNS = ["*.myworkdayjobs.com/*"]              # every public Workday career site
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}

# Career-site hosts look like {tenant}.{wdN}.myworkdayjobs.com. Rejecting anything else
# also drops the non-board subdomains (www, impl-*, resources, …) in one check.
_HOST_RE = re.compile(r"^([a-z0-9-]+)\.(wd\d+)\.myworkdayjobs\.com$")
# Public URLs are often locale-prefixed: /en-US/{site}/job/… — skip that segment.
_LOCALE_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$", re.IGNORECASE)
# A plausible career-site name: the segment right after the (optional) locale.
_SITE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Path segments that are never a site name (API paths, static assets, …).
_NON_SITES = {"wday", "assets", "favicon.ico", "robots.txt", "sitemap.xml"}


def token_from_url(url: str) -> str | None:
    """Extract a packed "{tenant}.{dc}@{site}" token from a harvested Workday URL, or
    None for junk. Handles both plain and locale-prefixed paths:
        https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/…
        https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/…
    (Passed to the engine as its `token_from_url_fn` hook — the default extractor
    only reads the first path segment and can't see the host.)"""
    parsed = urlparse(url)
    host_match = _HOST_RE.match((parsed.hostname or "").lower())
    if not host_match:
        return None
    tenant, dc = host_match.group(1), host_match.group(2)

    segments = [unquote(s) for s in parsed.path.strip("/").split("/") if s]
    if segments and _LOCALE_RE.match(segments[0]):     # drop a leading "en-US"-style segment
        segments = segments[1:]
    if not segments:
        return None
    site = segments[0]
    if site.lower() in _NON_SITES or not _SITE_RE.match(site):
        return None                                    # API path / asset / corrupted link
    return f"{tenant}.{dc}@{site}"                     # site keeps its case (URLs are case-sensitive)


def validate_token(token: str) -> int | None:
    """Return the job count if `token` is a live Workday board, else None. One light
    POST (limit=1). Verified failure modes: bad tenant → 422, wrong site → 404."""
    try:
        tenant, dc, site = workday._parse_token(token)
    except workday.WorkdayError:
        return None
    try:
        resp = requests.post(f"{workday._cxs_base(tenant, dc, site)}/jobs",
                             json={"appliedFacets": {}, "limit": 1, "offset": 0,
                                   "searchText": ""},
                             headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            total = resp.json().get("total")
            if isinstance(total, (int, float)):
                return int(total)
    except (requests.RequestException, ValueError):
        pass
    return None


def build_directory(**kwargs) -> dict:
    """Harvest → validate → cache the Workday board directory (forwards to the shared
    engine, supplying the custom host-aware token extractor). Merges with any boards
    already in the directory, so a rebuild only grows the set or prunes dead boards."""
    return board_discovery.build_directory(SOURCE, HOST_PATTERNS, validate_token,
                                           token_from_url_fn=token_from_url, **kwargs)


def load_directory() -> dict:
    """Load the stored Workday board directory (raises FileNotFoundError if not built yet)."""
    return board_discovery.load_directory(SOURCE)


# Runs only when you launch this file directly: `python -m src.ats.workday_discovery`.
if __name__ == "__main__":
    # Modest parallelism on purpose — see the module docstring. For a first smoke test,
    # temporarily pass limit_per_query=200.
    result = build_directory(limit_per_query=None, max_workers=8, on_progress=print)
    print(f"BUILD COMPLETE — {result['board_count']} live Workday boards in the directory.")
