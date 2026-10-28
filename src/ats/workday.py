"""Workday career-site connector (READ side) — where the Fortune 500 posts jobs.

Every Workday customer's public career site (e.g. nvidia.wd5.myworkdayjobs.com) is
backed by the same unauthenticated JSON endpoints the site's own pages call — no
login, no key, and the sites' robots.txt explicitly allows crawling (they publish
sitemaps so Google indexes every posting).

  List jobs:  POST https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
              body {"appliedFacets": {...}, "limit": 20, "offset": N, "searchText": ""}
  Job detail: GET  https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/{job path}

BOARD TOKEN: a Workday board's identity is THREE parts — the tenant, its data-center
host (wd1/wd5/wd12…), and the career-site name — packed as "{tenant}.{dc}@{site}",
e.g. "nvidia.wd5@NVIDIAExternalCareerSite". No colons (the tracker packs job ids as
"source:token:id", so ':' is reserved).

UNLIKE the other boards, listing is PAGINATED: Workday hard-caps each request at 20
jobs (bigger limits error out), so a large board takes total/20 POSTs. Two mitigations:
  • US-only facet: each board's first page carries a "facets" list; we look for the
    "United States of America" value and re-query with it applied. The facet ids are
    PER-TENANT (verified: NVIDIA's id fails on Salesforce), hence dynamic detection
    with an unfiltered fallback — worst case is slower, never broken.
  • MAX_JOBS safety cap + early stop when a page comes back empty ("total" isn't
    trusted: some tenants report a suspiciously round display cap like 2000).

LIKE GREENHOUSE/RIPPLING: the list entries are lightweight (title/location/relative
date, no description), so list_jobs() returns descriptionless summaries and the deep
screen pulls full postings per-survivor via get_job(). Salary is never structured —
it lives in the description text (US pay-transparency ranges), which the matcher's
parse_salary_range already reads.

Every job is normalized to the shared shape the matcher + pipeline expect (same as
greenhouse.py / lever.py / rippling.py): source/board_token/id/title/company/location/
url/description/salary_min/salary_max/salary_text/date_posted.
"""
import html                                            # un-escape HTML entities (&amp; → &)
import re                                              # relative-date + salary-snippet parsing
import time                                            # pause politely between result pages
from datetime import date, timedelta                   # turn "Posted 3 Days Ago" into a real date

import requests                                        # HTTP requests
from bs4 import BeautifulSoup                          # turn the HTML description into plain text

from src.ats import ATSError                           # shared base error (see src/ats/__init__.py)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}
_TIMEOUT = 15                                          # seconds to wait on any single request

LIST_LIMIT = 20            # Workday's hard per-request cap (verified: limit=50 errors out)
MAX_JOBS = 2000            # safety cap per board, so one giant tenant can't stall a scan
_PAGE_DELAY = 0.2          # polite pause between result pages within one board

# Same salary-snippet regex as greenhouse.py/rippling.py — display-only fallback; the
# matcher does its own thorough parse_salary_range() on the description.
_SALARY_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?(?:\s?[-–to]+\s?\$?\s?\d{2,3}(?:,\d{3})?)?(?:\s?(?:k|/yr|per year|annually|USD))?",
    re.IGNORECASE,
)

# "Posted Today" / "Posted Yesterday" / "Posted 6 Days Ago" → a number of days.
# "Posted 30+ Days Ago" deliberately does NOT match (the '+' means "we don't know how
# much older"), so those return "" and the age filter keeps rather than false-drops them.
_DAYS_AGO_RE = re.compile(r"(\d+)\s*days?\s*ago", re.IGNORECASE)


class WorkdayError(ATSError):
    """Raised when a Workday board can't be read (bad token, network error, non-200)."""


# ── Token handling ───────────────────────────────────────────────────────────

def _parse_token(token: str) -> tuple[str, str, str]:
    """Unpack "{tenant}.{dc}@{site}" → (tenant, dc, site). The site keeps its case
    (Workday site names are case-sensitive in URLs); tenant/dc are lowercase."""
    host, _, site = (token or "").partition("@")
    tenant, _, dc = host.partition(".")
    if not (tenant and dc and site):
        raise WorkdayError(
            f"Bad Workday token '{token}' — expected 'tenant.wdN@SiteName' "
            "(e.g. 'nvidia.wd5@NVIDIAExternalCareerSite').")
    return tenant.lower(), dc.lower(), site


def _cxs_base(tenant: str, dc: str, site: str) -> str:
    """The JSON API base for one board (the same endpoints the career page itself calls)."""
    return f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"


def _company_name(tenant: str) -> str:
    """Workday has no cheap company-name endpoint, so derive a display name from the
    tenant slug: "nvidia" → "Nvidia", "bostonscientific" stays best-effort. Editable
    later in the tracker row (same approach as rippling/lever)."""
    pretty = tenant.replace("-", " ").replace("_", " ").strip().title()
    return pretty or tenant


# ── Field helpers ────────────────────────────────────────────────────────────

def _relative_date(posted_on: str, today: date = None) -> str:
    """Workday's LIST stage only gives relative text ("Posted Today", "Posted 3 Days
    Ago", "Posted 30+ Days Ago"). Convert to "YYYY-MM-DD" best-effort; "" when the
    text is open-ended ("30+") or unrecognized — the age filter treats "" as
    unknown-and-kept, which is safer than guessing a stale date."""
    text = (posted_on or "").strip().lower()
    today = today or date.today()
    if not text:
        return ""
    if "today" in text or "just posted" in text:
        return today.isoformat()
    if "yesterday" in text:
        return (today - timedelta(days=1)).isoformat()
    if "+" in text:                                    # "30+ days ago" — unbounded, don't guess
        return ""
    m = _DAYS_AGO_RE.search(text)
    if m:
        return (today - timedelta(days=int(m.group(1)))).isoformat()
    return ""


def _html_to_text(content: str) -> str:
    """Un-escape HTML entities then strip tags to plain text (same approach as greenhouse.py)."""
    if not content:
        return ""
    return BeautifulSoup(html.unescape(str(content)), "html.parser").get_text(separator="\n", strip=True)


def _find_salary(description: str) -> str:
    """Return the first salary-like snippet in the description (for review only)."""
    match = _SALARY_RE.search(description or "")
    return match.group(0).strip() if match else ""


def _us_facet(facets) -> dict:
    """Find this board's own "United States" filter in its facets list, so pagination
    only walks US jobs. Returns {"facetParameter": ["value-id"]} or {} when no US
    value exists — the ids are PER-TENANT, so this must be detected per board, and it
    must NEVER raise (fallback = unfiltered pagination: slower, not broken).

    The facets shape is undocumented and varies, so everything is .get()-based and
    nested values/facets lists are walked recursively. A hit under a parameter named
    like "...country..." wins; otherwise any parameter (e.g. NVIDIA's verified
    locationHierarchy1) is accepted."""
    matches = []                                       # (facet_parameter, value_id) pairs

    def walk(node, param):
        if isinstance(node, list):
            for item in node:
                walk(item, param)
            return
        if not isinstance(node, dict):
            return
        param = node.get("facetParameter") or param    # nearest enclosing parameter name
        descriptor = str(node.get("descriptor") or "").strip().lower()
        value_id = node.get("id")
        if value_id and descriptor in ("united states of america", "united states"):
            matches.append((param, value_id))
        for key in ("values", "facets"):
            if isinstance(node.get(key), list):
                walk(node[key], param)

    try:
        walk(facets, None)
        for param, value_id in matches:                # prefer an explicit country facet
            if param and "country" in str(param).lower():
                return {param: [value_id]}
        for param, value_id in matches:                # else take any (e.g. locationHierarchy1)
            if param:
                return {param: [value_id]}
    except Exception:                                  # defensive: unknown shapes must not break a scan
        pass
    return {}


# ── Normalization ────────────────────────────────────────────────────────────

def _external_id(posting: dict) -> str:
    """A list entry's stable id = its externalPath minus the leading slash, e.g.
    "job/US-CA-Remote/Senior-Engineer_JR2018101". The WHOLE path is kept (not just the
    last segment) because get_job() must rebuild the detail URL from token+id alone.
    Slashes are fine in job ids (they only travel in form bodies); a ':' would break
    the packed "source:token:id" format, so such a posting is skipped upstream."""
    return (posting.get("externalPath") or "").lstrip("/")


def _normalize_summary(posting: dict, token: str, tenant: str, dc: str, site: str,
                       company: str) -> dict:
    """Normalize a job-list entry (no description available at this stage)."""
    external_id = _external_id(posting)
    return {
        "source": "workday",
        "board_token": token,
        "id": external_id,
        "title": (posting.get("title") or "").strip(),
        "company": company,
        "location": (posting.get("locationsText") or "").strip(),
        # The human-facing page URL (what you'd click), not the JSON endpoint.
        "url": f"https://{tenant}.{dc}.myworkdayjobs.com/{site}/{external_id}",
        "description": "",                             # not available in the list view
        "salary_text": "",
        "date_posted": _relative_date(posting.get("postedOn")),
    }


# ── The two contract functions ───────────────────────────────────────────────

def _post_page(url: str, facets: dict, offset: int) -> dict:
    """One page of the jobs list (Workday caps each request at LIST_LIMIT=20)."""
    try:
        resp = requests.post(url, json={"appliedFacets": facets, "limit": LIST_LIMIT,
                                        "offset": offset, "searchText": ""},
                             headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        raise WorkdayError(f"Could not list Workday jobs at {url} (offset {offset}): {e}") from e


def list_jobs(token: str) -> list[dict]:
    """List a board's jobs (lightweight summaries — titles/locations, no descriptions).

    Paginated at 20/request. The first page's facets are inspected for this board's
    own "United States" filter; when found, the walk restarts filtered (typically
    30-50% fewer pages). Early-stops on an empty page ("total" can be a display cap)
    and at MAX_JOBS. Raises WorkdayError on a bad token / network failure."""
    tenant, dc, site = _parse_token(token)
    url = f"{_cxs_base(tenant, dc, site)}/jobs"
    company = _company_name(tenant)

    first = _post_page(url, {}, 0)
    facets = _us_facet(first.get("facets") or [])
    if facets:
        first = _post_page(url, facets, 0)             # restart the walk US-only

    total = first.get("total")
    total = int(total) if isinstance(total, (int, float, str)) and str(total).isdigit() else 0
    postings = list(first.get("jobPostings") or [])

    offset = LIST_LIMIT
    while offset < min(total, MAX_JOBS):
        time.sleep(_PAGE_DELAY)                        # polite pacing between pages
        page = _post_page(url, facets, offset).get("jobPostings") or []
        if not page:                                   # "total" over-reported — stop early
            break
        postings.extend(page)
        offset += LIST_LIMIT

    jobs = []
    for posting in postings:
        job = _normalize_summary(posting, token, tenant, dc, site, company)
        if not job["id"] or ":" in job["id"]:          # ':' is reserved by the packed job_id
            continue
        jobs.append(job)
    return jobs


def get_job(token: str, job_id: str) -> dict:
    """Fetch one job WITH its full description (used on title-pass survivors and when
    re-generating a résumé for a tracked job). `job_id` is the externalPath kept by
    list_jobs, so the detail URL is just base + id."""
    tenant, dc, site = _parse_token(token)
    try:
        resp = requests.get(f"{_cxs_base(tenant, dc, site)}/{job_id}",
                            headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        info = resp.json().get("jobPostingInfo") or {}
    except (requests.RequestException, ValueError) as e:
        raise WorkdayError(f"Could not fetch Workday job {job_id} on '{token}': {e}") from e

    description = _html_to_text(info.get("jobDescription"))
    # Primary + additional locations joined for the matcher ("US, CA, Remote, US, NY, Remote").
    locations = [str(info.get("location") or "").strip()]
    locations += [str(x).strip() for x in info.get("additionalLocations") or []]
    # startDate is the REAL posting date (verified live); relative text is the fallback.
    date_posted = str(info.get("startDate") or "")[:10] or _relative_date(info.get("postedOn"))
    return {
        "source": "workday",
        "board_token": token,
        "id": job_id,
        "title": (info.get("title") or "").strip(),
        "company": _company_name(tenant),
        "location": ", ".join(loc for loc in locations if loc),
        "url": (info.get("externalUrl") or "").strip()
               or f"https://{tenant}.{dc}.myworkdayjobs.com/{site}/{job_id}",
        "description": description,
        "salary_min": None,                            # Workday never structures pay
        "salary_max": None,                            # Workday never structures pay
        "salary_text": _find_salary(description),      # matcher parses it from the text
        "date_posted": date_posted,
    }
