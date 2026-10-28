"""Rippling ATS public job-board connector (READ side).

Rippling's public careers pages (ats.rippling.com/{slug}/jobs) are backed by a public JSON
API — no auth, no key — so, like Greenhouse and Lever, it's meant to be read openly.

  List jobs:  GET https://api.rippling.com/platform/api/ats/v1/board/{token}/jobs
  Job detail: GET https://api.rippling.com/platform/api/ats/v1/board/{token}/jobs/{uuid}

`{token}` is the company's board slug (the bit in ats.rippling.com/{token}/jobs, e.g.
"riot-platforms-careers"); the job id is a UUID. Discovery of which slugs exist is handled
separately (rippling_discovery.py); this module just reads a known board.

LIKE GREENHOUSE (not Lever): the LIST endpoint is lightweight — titles + locations only, no
description — so list_jobs() returns descriptionless summaries and the full posting is pulled
per-survivor via get_job(). The list endpoint also omits the company name, so the summary uses
a slug-derived name; get_job() upgrades it to the real `companyName` from the detail payload
(and the deep-screen stage in job_search replaces the summary with that fuller job, so the
tracked/logged company ends up correct).

Every job is normalized to the shared shape the matcher + pipeline expect (same as
greenhouse.py / lever.py): source/board_token/id/title/company/location/url/description/
salary_min/salary_max/salary_text/date_posted.
"""
import html                                            # un-escape HTML entities (&amp; → &)
import re                                              # find a salary snippet in the description
import requests                                        # HTTP requests
from bs4 import BeautifulSoup                          # turn the HTML description into plain text

from src.ats import ATSError                           # shared base error (see src/ats/__init__.py)

BASE = "https://api.rippling.com/platform/api/ats/v1/board"   # the public ATS API base
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_TIMEOUT = 15                                           # seconds to wait on any single request

# Same salary-snippet regex as greenhouse.py — Rippling rarely fills structured pay, so we
# also scrape a "$120,000 - $150,000"-style mention from the description for display only.
_SALARY_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?(?:\s?[-–to]+\s?\$?\s?\d{2,3}(?:,\d{3})?)?(?:\s?(?:k|/yr|per year|annually|USD))?",
    re.IGNORECASE,
)


class RipplingError(ATSError):
    """Raised when a Rippling board can't be read (bad slug, network error, non-200)."""


def _company_name(token: str) -> str:
    """Rippling's LIST endpoint has no company name, so derive a display name from the slug
    for the title pass: "riot-platforms-careers" → "Riot Platforms Careers". Best-effort — the
    real name from get_job() replaces this downstream (and you can edit the row later)."""
    pretty = token.replace("-", " ").replace("_", " ").strip().title()
    return pretty or token


def _location(job: dict) -> str:
    """Single location string for the matcher. The list endpoint gives workLocation {label};
    the detail endpoint gives workLocations [label, …]. Handle both, joined with ', '."""
    locs = job.get("workLocations")
    if isinstance(locs, list) and locs:
        # detail: a list of plain strings (e.g. ["Remote (United States)"])
        return ", ".join(str(x).strip() for x in locs if str(x).strip())
    single = job.get("workLocation") or {}             # list summary: {label, id}
    return (single.get("label") or "").strip()


def _description(data: dict) -> str:
    """Full posting text as plain text. Rippling's `description` is a dict of HTML sections
    (typically {'company': …, 'role': …}); stitch them in reading order and strip the HTML."""
    desc = data.get("description")
    if isinstance(desc, str):                          # defensive: some boards may send a string
        return _html_to_text(desc)
    if not isinstance(desc, dict):
        return ""
    # Preserve a sensible reading order (company blurb, then the role), then any other sections.
    ordered = ["company", "role"]
    parts = [_html_to_text(desc[k]) for k in ordered if desc.get(k)]
    parts += [_html_to_text(v) for k, v in desc.items() if k not in ordered and v]
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _salary(data: dict) -> tuple:
    """Best-effort (min_salary:int|None, max_salary:int|None, display_text:str) from Rippling's
    optional payRangeDetails. Its exact shape varies and is often empty, so we probe several
    likely key names (USD only) and degrade to (None, None, '') when nothing usable is present."""
    for rng in data.get("payRangeDetails") or []:
        if not isinstance(rng, dict):
            continue
        currency = str(rng.get("currency") or rng.get("currencyCode") or "USD").upper()
        if currency not in ("USD", ""):
            continue
        lo = rng.get("min") or rng.get("minValue") or rng.get("minPay") or rng.get("lowValue")
        hi = rng.get("max") or rng.get("maxValue") or rng.get("maxPay") or rng.get("highValue")
        if lo:
            try:
                lo = int(float(lo))
                hi = int(float(hi)) if hi else lo
                return lo, hi, f"${lo:,} - ${hi:,}"    # the :, adds thousands separators
            except (TypeError, ValueError):
                continue
    return None, None, ""


def _posted_date(data: dict) -> str:
    """Publish date as YYYY-MM-DD from Rippling's ISO `createdOn` ("" if absent)."""
    raw = data.get("createdOn") or ""
    return raw[:10] if raw else ""                     # [:10] keeps just the "YYYY-MM-DD" part


def _html_to_text(content: str) -> str:
    """Un-escape HTML entities then strip tags to plain text (same approach as greenhouse.py)."""
    if not content:
        return ""
    return BeautifulSoup(html.unescape(str(content)), "html.parser").get_text(separator="\n", strip=True)


def _find_salary(description: str) -> str:
    """Return the first salary-like snippet in the description (for review only)."""
    match = _SALARY_RE.search(description or "")
    return match.group(0).strip() if match else ""


def _normalize_summary(job: dict, token: str, company: str) -> dict:
    """Normalize a job-list entry (no description available at this stage)."""
    return {
        "source": "rippling",
        "board_token": token,
        "id": str(job.get("uuid", "")),
        "title": (job.get("name") or "").strip(),
        "company": company,
        "location": _location(job),
        "url": (job.get("url") or "").strip(),
        "description": "",                             # not available in the list view
        "salary_text": "",
    }


def list_jobs(token: str) -> list[dict]:
    """List all jobs on a board (lightweight — titles + locations, no descriptions). One API
    call returns every posting, which is what the matcher needs for the cheap first-pass
    title/location filter. Fetch full descriptions only for survivors via get_job(). Raises
    RipplingError on a non-200 / bad slug."""
    try:
        resp = requests.get(f"{BASE}/{token}/jobs", headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        jobs = resp.json()                             # the API returns a JSON LIST of jobs
    except (requests.RequestException, ValueError) as e:
        raise RipplingError(f"Could not list jobs for Rippling board '{token}': {e}") from e
    if not isinstance(jobs, list):                     # defensive: unexpected shape
        raise RipplingError(f"Unexpected response shape for Rippling board '{token}'.")
    company = _company_name(token)
    return [_normalize_summary(j, token, company) for j in jobs]


def get_job(token: str, job_id) -> dict:
    """Fetch a single job WITH its full description text (used on title-pass survivors and when
    re-generating a résumé for one tracked job). Returns the shared normalized shape, with the
    REAL company name from the detail payload."""
    try:
        resp = requests.get(f"{BASE}/{token}/jobs/{job_id}", headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise RipplingError(f"Could not fetch Rippling job {job_id} on board '{token}': {e}") from e

    description = _description(data)
    salary_min, salary_max, salary_text = _salary(data)
    return {
        "source": "rippling",
        "board_token": token,
        "id": str(data.get("uuid", "")) or str(job_id),
        "title": (data.get("name") or "").strip(),
        "company": (data.get("companyName") or "").strip() or _company_name(token),
        "location": _location(data),
        "url": (data.get("url") or "").strip(),
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_text": salary_text or _find_salary(description),
        "date_posted": _posted_date(data),
    }
