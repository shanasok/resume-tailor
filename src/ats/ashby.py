"""Ashby public job-board connector (READ side).

Ashby powers many well-funded tech startups' careers pages (jobs.ashbyhq.com/{board}). Every
public board is backed by an unauthenticated JSON "posting API" — no auth, no key — so, like
Greenhouse/Lever/Rippling, it's meant to be read openly.

  List jobs:  GET https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true

`{token}` is the company's board slug (the bit in jobs.ashbyhq.com/{token}, e.g. "ramp").
There is NO public single-job endpoint (it returns 401), so get_job() re-reads the board and
finds the posting by id — the whole board comes back in one call anyway.

LIKE LEVER (not Greenhouse/Rippling/Workday): the LIST endpoint already returns each posting's
FULL description AND structured compensation, so list_jobs() returns fully-populated jobs and
the deep screen needs no second fetch. This is the ONLY board with reliably structured salary
(`compensation.summaryComponents`) — valuable because the matcher hard-drops jobs with no
stated pay.

Every job is normalized to the shared shape the matcher + pipeline expect (same as
greenhouse.py / lever.py / rippling.py / workday.py): source/board_token/id/title/company/
location/url/description/salary_min/salary_max/salary_text/date_posted.
"""
import html                                            # un-escape HTML entities (&amp; → &)
import re                                              # find a salary snippet in the description
import requests                                        # HTTP requests
from bs4 import BeautifulSoup                           # turn HTML description into plain text

from src.ats import ATSError                           # shared base error (see src/ats/__init__.py)

BASE = "https://api.ashbyhq.com/posting-api/job-board"   # the public posting-API base
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"}
_TIMEOUT = 15                                           # seconds to wait on any single request

# Same salary-snippet regex as the other connectors — a display-only fallback for the rare
# board whose structured comp is empty; the matcher still parses the description itself.
_SALARY_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?(?:\s?[-–to]+\s?\$?\s?\d{2,3}(?:,\d{3})?)?(?:\s?(?:k|/yr|per year|annually|USD))?",
    re.IGNORECASE,
)

# Ashby comp intervals we treat as an annual salary (the matcher's floor is a yearly figure,
# so an hourly "1 HOUR" minValue like 55 must NOT be read as salary_min).
_ANNUAL_INTERVALS = {"1 YEAR", "YEAR", "ANNUAL", "ANNUALLY", "YEARLY"}


class AshbyError(ATSError):
    """Raised when an Ashby board can't be read (bad slug, network error, non-200)."""


def _company_name(token: str) -> str:
    """Ashby's posting API has no company-name field, so derive a display name from the slug:
    "some-co" → "Some Co". Best-effort — editable on the tracked row later (same as lever/rippling)."""
    pretty = token.replace("-", " ").replace("_", " ").strip().title()
    return pretty or token


def _country_label(address: dict) -> str:
    """Pull a country phrase the matcher recognizes out of Ashby's nested address, mapping the
    common US spellings to "United States" ("" when absent)."""
    country = ""
    if isinstance(address, dict):
        country = str((address.get("postalAddress") or {}).get("addressCountry") or "").strip()
    if country.upper() in ("US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"):
        return "United States"
    return country


def _location(job: dict) -> str:
    """Build a single location string the matcher can classify (US-ness + remote/hybrid/onsite),
    folding Ashby's separate signals into one string: the primary location, the workplace type
    (Remote/Hybrid/OnSite), the country, and any secondary office locations (so a remote role
    with an office in the user's commute zone still surfaces as commutable-hybrid)."""
    parts = [(job.get("location") or "").strip(),
             (job.get("workplaceType") or "").strip(),
             _country_label(job.get("address"))]
    for sec in job.get("secondaryLocations") or []:
        if isinstance(sec, dict):
            parts.append((sec.get("location") or "").strip())
    seen, unique = set(), []                            # keep order, drop empties + duplicates
    for p in parts:
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            unique.append(p)
    return " · ".join(unique)


def _salary(job: dict) -> tuple:
    """Best-effort (salary_min:int|None, salary_max:int|None, display_text:str) from Ashby's
    structured compensation.

    Prefers `summaryComponents` (the roll-up across pay tiers); falls back to walking each
    tier's `components`. Only an ANNUAL USD Salary component sets salary_min/salary_max —
    equity/bonus and hourly components are ignored. A component with no `maxValue` falls back
    to its own min; across multiple components/tiers, salary_min is the lowest low and
    salary_max is the highest high. The display text prefers Ashby's own clean
    `scrapeableCompensationSalarySummary`, else the richer `compensationTierSummary`."""
    comp = job.get("compensation")
    if not isinstance(comp, dict):
        return None, None, ""

    display = (str(comp.get("scrapeableCompensationSalarySummary") or "").strip()
               or str(comp.get("compensationTierSummary") or "").strip())

    def usd_annual_salary(component: dict):
        """Return (min, max) for an ANNUAL USD Salary component, else None. When the
        component has no maxValue, max == min."""
        if not isinstance(component, dict):
            return None
        if str(component.get("compensationType") or "") != "Salary":
            return None
        if str(component.get("currencyCode") or "USD").upper() not in ("USD", ""):
            return None
        if str(component.get("interval") or "").upper() not in _ANNUAL_INTERVALS:
            return None
        lo = component.get("minValue")
        if lo in (None, ""):
            return None
        try:
            lo = int(float(lo))
        except (TypeError, ValueError):
            return None
        hi = component.get("maxValue")
        try:
            hi = int(float(hi)) if hi not in (None, "") else lo
        except (TypeError, ValueError):
            hi = lo
        return lo, hi

    # 1) the summary roll-up, 2) each tier's components. Span them: lowest low, highest high.
    pairs = []
    for component in comp.get("summaryComponents") or []:
        got = usd_annual_salary(component)
        if got is not None:
            pairs.append(got)
    if not pairs:
        for tier in comp.get("compensationTiers") or []:
            for component in (tier.get("components") if isinstance(tier, dict) else None) or []:
                got = usd_annual_salary(component)
                if got is not None:
                    pairs.append(got)
    salary_min = min(lo for lo, _ in pairs) if pairs else None
    salary_max = max(hi for _, hi in pairs) if pairs else None
    return salary_min, salary_max, display


def _description(job: dict) -> str:
    """Full posting text as plain text. Ashby gives descriptionPlain (already plain); fall back
    to stripping descriptionHtml when only the HTML form is present."""
    plain = (job.get("descriptionPlain") or "").strip()
    if plain:
        return plain
    return _html_to_text(job.get("descriptionHtml") or "")


def _html_to_text(content: str) -> str:
    """Un-escape HTML entities then strip tags to plain text (same approach as greenhouse.py)."""
    if not content:
        return ""
    return BeautifulSoup(html.unescape(str(content)), "html.parser").get_text(separator="\n", strip=True)


def _find_salary(description: str) -> str:
    """Return the first salary-like snippet in the description (display fallback only)."""
    match = _SALARY_RE.search(description or "")
    return match.group(0).strip() if match else ""


def _posted_date(job: dict) -> str:
    """Publish date as YYYY-MM-DD from Ashby's ISO `publishedAt` ("" if absent)."""
    raw = str(job.get("publishedAt") or "")
    return raw[:10] if raw else ""                      # [:10] keeps just the "YYYY-MM-DD" part


def _normalize(job: dict, token: str) -> dict:
    """Turn one raw Ashby posting into the project's shared job shape (description included)."""
    salary_min, salary_max, salary_text = _salary(job)
    description = _description(job)
    return {
        "source": "ashby",
        "board_token": token,
        "id": str(job.get("id", "")),
        "title": (job.get("title") or "").strip(),
        "company": _company_name(token),
        "location": _location(job),
        "url": (job.get("jobUrl") or "").strip(),
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_text": salary_text or _find_salary(description),
        "date_posted": _posted_date(job),
    }


def _fetch_board(token: str) -> list[dict]:
    """Fetch a board's raw postings list (with compensation). Raises AshbyError on a non-200 /
    bad slug / unexpected shape."""
    try:
        resp = requests.get(f"{BASE}/{token}", params={"includeCompensation": "true"},
                            headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise AshbyError(f"Could not list jobs for Ashby board '{token}': {e}") from e
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        raise AshbyError(f"Unexpected response shape for Ashby board '{token}'.")
    return jobs


def list_jobs(token: str) -> list[dict]:
    """List a board's LISTED postings, fully populated (titles, locations, descriptions, AND
    structured salary). One API call returns everything the matcher needs. Unlisted/internal
    postings (isListed=false) are skipped. Raises AshbyError on a non-200 / bad slug."""
    return [_normalize(j, token) for j in _fetch_board(token) if j.get("isListed", True)]


def get_job(token: str, job_id) -> dict:
    """Fetch a single posting by id (used when re-generating a résumé for one tracked job).
    Ashby has no public per-job endpoint, so this re-reads the board and finds the posting —
    including unlisted ones, so a since-unlisted tracked job can still be re-generated. Returns
    the same normalized shape as list_jobs()."""
    for job in _fetch_board(token):
        if str(job.get("id", "")) == str(job_id):
            return _normalize(job, token)
    raise AshbyError(f"Ashby job {job_id} not found on board '{token}'.")
