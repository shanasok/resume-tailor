"""SmartRecruiters public Posting API connector (READ side).

SmartRecruiters powers the public careers pages of many large employers
(jobs.smartrecruiters.com/{Company}). Every public posting is served by an
unauthenticated JSON "Posting API" — no auth, no key — so, like Greenhouse/Lever/
Rippling/Ashby, it's meant to be read openly.

  List jobs:  GET https://api.smartrecruiters.com/v1/companies/{token}/postings
              ?limit=100&offset=N&country=us
  Job detail: GET https://api.smartrecruiters.com/v1/companies/{token}/postings/{id}

`{token}` is the company's identifier (the bit in jobs.smartrecruiters.com/{token},
e.g. "BoschGroup"). The API is CASE-INSENSITIVE on the identifier, so discovery can use
the default lowercasing token extractor (unlike Workday). Discovery of which identifiers
exist is handled separately (Common Crawl harvest + validation); this module just reads a
known board.

LIKE GREENHOUSE/WORKDAY (not Ashby/Lever): the LIST endpoint returns lightweight summaries
(title/location/date, no description), so list_jobs() returns descriptionless summaries and
the deep screen pulls full postings per-survivor via get_job(). Two niceties over Workday:
  • the list is US-filterable server-side (?country=us), so a huge board (Bosch: 4,721 →
    272 US) is walked in a fraction of the pages — the same US-only intent as Workday's
    dynamic facet, but built in and reliable; and
  • get_job() returns STRUCTURED compensation ({min,max,currency,period}), valuable because
    the matcher hard-drops jobs with no stated pay. Falls back to a description snippet.

Every job is normalized to the shared shape the matcher + pipeline expect (same as
greenhouse.py / lever.py / rippling.py / workday.py / ashby.py): source/board_token/id/
title/company/location/url/description/salary_min/salary_max/salary_text/date_posted.
"""
import html                                            # un-escape HTML entities (&amp; → &)
import re                                              # find a salary snippet in the description
import time                                            # pause politely between result pages
import requests                                        # HTTP requests
from bs4 import BeautifulSoup                           # turn HTML description sections into plain text

from src.ats import ATSError                           # shared base error (see src/ats/__init__.py)

BASE = "https://api.smartrecruiters.com/v1/companies"   # the public Posting API base
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}
_TIMEOUT = 15                                           # seconds to wait on any single request

LIST_LIMIT = 100          # SmartRecruiters caps each list page at 100 (bigger is silently clamped)
MAX_JOBS = 2000           # safety cap per board, so one giant employer can't stall a scan
COUNTRY = "us"            # server-side US filter — the scan is US-only (see phase2-job-criteria)
_PAGE_DELAY = 0.2         # polite pause between result pages within one board

# The description sections we stitch into the posting text, most job-relevant first. The
# boilerplate companyDescription comes last so title/keyword matching sees the real role first.
_SECTION_ORDER = ("jobDescription", "qualifications", "additionalInformation", "companyDescription")

# Same salary-snippet regex as the other connectors — a display-only fallback for the posting
# whose structured compensation is absent; the matcher still parses the description itself.
_SALARY_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?(?:\s?[-–to]+\s?\$?\s?\d{2,3}(?:,\d{3})?)?(?:\s?(?:k|/yr|per year|annually|USD))?",
    re.IGNORECASE,
)


class SmartRecruitersError(ATSError):
    """Raised when a SmartRecruiters board can't be read (bad identifier, network error,
    non-200). Subclasses the shared ATSError, so callers can catch it either way."""


def _company_name(posting: dict, token: str) -> str:
    """Display name from the posting's own `company.name` (e.g. "Bosch Group"); falls back to
    a title-cased slug then the raw token, so a summary always carries a readable company."""
    name = str(((posting.get("company") or {}).get("name")) or "").strip()
    return name or token.replace("-", " ").replace("_", " ").strip().title() or token


def _location(posting: dict) -> str:
    """Build one location string the matcher can classify (US-ness + remote/hybrid/onsite).

    SmartRecruiters gives a structured location object; `fullLocation` ("Bethlehem, PA,
    United States") already reads cleanly for classify_location, and the boolean remote/
    hybrid flags are appended as words so classify_workplace can see them."""
    loc = posting.get("location") or {}
    parts = [str(loc.get("fullLocation") or "").strip()]
    if not parts[0]:                                    # older/sparse postings: assemble from pieces
        parts = [", ".join(p for p in (str(loc.get("city") or "").strip(),
                                       str(loc.get("region") or "").strip(),
                                       str(loc.get("country") or "").strip().upper()) if p)]
    if loc.get("remote"):
        parts.append("Remote")
    if loc.get("hybrid"):
        parts.append("Hybrid")
    return " · ".join(p for p in parts if p)


def _salary(posting: dict) -> tuple:
    """Best-effort (salary_min:int|None, salary_max:int|None, display_text:str) from
    SmartRecruiters' structured `compensation` ({min,max,currency,period}). Only an ANNUAL
    (YEARLY) USD range sets a figure — an hourly/monthly range must NOT be read as a yearly
    salary. Returns (None, None, '') when no usable structured pay is present."""
    comp = posting.get("compensation")
    if not isinstance(comp, dict):
        return None, None, ""
    if str(comp.get("currency") or "USD").upper() not in ("USD", ""):
        return None, None, ""
    if str(comp.get("period") or "YEARLY").upper() not in ("YEARLY", "YEAR", "ANNUAL", "ANNUALLY"):
        return None, None, ""                           # hourly/monthly — not a yearly floor figure
    lo = comp.get("min")
    if lo in (None, ""):
        return None, None, ""
    try:
        lo = int(float(lo))
    except (TypeError, ValueError):
        return None, None, ""
    hi = comp.get("max")
    try:
        hi = int(float(hi)) if hi not in (None, "") else lo
    except (TypeError, ValueError):
        hi = lo
    return lo, hi, f"${lo:,} - ${hi:,}"                 # the :, adds thousands separators


def _description(posting: dict) -> str:
    """Full posting text as plain text: the jobAd's sections (job description, qualifications,
    additional info, then company boilerplate) unescaped, tag-stripped, and joined."""
    sections = ((posting.get("jobAd") or {}).get("sections")) or {}
    chunks = []
    for key in _SECTION_ORDER:
        text = _html_to_text(((sections.get(key) or {}).get("text")) or "")
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _html_to_text(content: str) -> str:
    """Un-escape HTML entities then strip tags to plain text (same approach as greenhouse.py)."""
    if not content:
        return ""
    return BeautifulSoup(html.unescape(str(content)), "html.parser").get_text(separator="\n", strip=True)


def _find_salary(description: str) -> str:
    """Return the first salary-like snippet in the description (display fallback only)."""
    match = _SALARY_RE.search(description or "")
    return match.group(0).strip() if match else ""


def _posted_date(posting: dict) -> str:
    """Publish date as YYYY-MM-DD from the ISO `releasedDate` ("" if absent)."""
    raw = str(posting.get("releasedDate") or "")
    return raw[:10] if raw else ""                      # [:10] keeps just the "YYYY-MM-DD" part


def _normalize_summary(posting: dict, token: str) -> dict:
    """Normalize a job-list entry (no description available at this stage)."""
    return {
        "source": "smartrecruiters",
        "board_token": token,
        "id": str(posting.get("id", "")),
        "title": (posting.get("name") or "").strip(),
        "company": _company_name(posting, token),
        "location": _location(posting),
        "url": (posting.get("postingUrl") or posting.get("applyUrl") or "").strip(),
        "description": "",                              # not available in the list view
        "salary_text": "",
        "date_posted": _posted_date(posting),
    }


def _get_page(token: str, offset: int) -> dict:
    """One page of a board's postings list (US-filtered, up to LIST_LIMIT)."""
    try:
        resp = requests.get(f"{BASE}/{token}/postings",
                            params={"limit": LIST_LIMIT, "offset": offset, "country": COUNTRY},
                            headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        raise SmartRecruitersError(
            f"Could not list jobs for board '{token}' (offset {offset}): {e}") from e


def list_jobs(token: str) -> list[dict]:
    """List a board's US jobs (lightweight summaries — titles/locations/dates, no descriptions).

    Paginated at LIST_LIMIT=100 and filtered to US postings server-side (?country=us), so even
    a very large employer is walked in a handful of pages. Early-stops on an empty page and at
    MAX_JOBS. Fetch full descriptions only for survivors via get_job(). Raises
    SmartRecruitersError on a non-200 / bad identifier."""
    first = _get_page(token, 0)
    total = first.get("totalFound")
    total = int(total) if isinstance(total, (int, float, str)) and str(total).isdigit() else 0
    postings = list(first.get("content") or [])

    offset = LIST_LIMIT
    while offset < min(total, MAX_JOBS):
        time.sleep(_PAGE_DELAY)                         # polite pacing between pages
        page = _get_page(token, offset).get("content") or []
        if not page:                                    # total over-reported — stop early
            break
        postings.extend(page)
        offset += LIST_LIMIT

    jobs = []
    for posting in postings:
        job = _normalize_summary(posting, token)
        if not job["id"] or ":" in job["id"]:           # ':' is reserved by the packed job_id
            continue
        jobs.append(job)
    return jobs


def get_job(token: str, job_id) -> dict:
    """Fetch one job WITH its full description + structured compensation (used on title-pass
    survivors and when re-generating a résumé for a tracked job). Raises SmartRecruitersError
    on a non-200 / missing posting."""
    try:
        resp = requests.get(f"{BASE}/{token}/postings/{job_id}", headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise SmartRecruitersError(
            f"Could not fetch job {job_id} on board '{token}': {e}") from e

    description = _description(data)
    salary_min, salary_max, salary_text = _salary(data)
    return {
        "source": "smartrecruiters",
        "board_token": token,
        "id": str(data.get("id", "") or job_id),
        "title": (data.get("name") or "").strip(),
        "company": _company_name(data, token),
        "location": _location(data),
        "url": (data.get("postingUrl") or data.get("applyUrl") or "").strip(),
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_text": salary_text or _find_salary(description),
        "date_posted": _posted_date(data),
    }
