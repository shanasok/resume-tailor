"""Greenhouse Job Board API connector (READ side).

Greenhouse's public Job Board API powers companies' public careers pages, so it's
meant to be read openly — no auth, no API key, no LinkedIn-style ToS gray area.

  List jobs:  GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs
  Job detail: GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{id}?questions=true
  Board meta: GET https://boards-api.greenhouse.io/v1/boards/{token}

`{token}` is the company's board token (e.g. "stripe"). Discovery of which tokens
exist is handled separately (Common Crawl harvest + validation); this module just
reads a known board.

A "connector" is a module whose whole job is to talk to ONE external service and hand
back clean data in the project's shape. Every job is normalized to this shared shape the
matcher + pipeline expect:
    {
        "source":      "greenhouse",
        "board_token": str,
        "id":          str,
        "title":       str,
        "company":     str,
        "location":    str,   # free-text, e.g. "Remote - US"
        "url":         str,
        "description": str,   # plain text; "" in list summaries (fetch via get_job)
        "salary_text": str,   # salary snippet pulled from the description, else ""
    }
"""
import html                                          # to un-escape HTML entities (&amp; → &)
import re                                            # regular expressions (find a salary snippet)
import requests                                      # HTTP requests
from bs4 import BeautifulSoup                        # parse HTML descriptions into plain text

from src.ats import ATSError                         # shared base error (see src/ats/__init__.py)

BASE = "https://boards-api.greenhouse.io/v1/boards"  # the API's base address
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_TIMEOUT = 15                                         # seconds to wait on any single request

# Board token -> company display name, so repeated calls don't re-fetch board meta.
# This is a simple in-memory CACHE (a dict that remembers answers we've already looked up).
_board_name_cache: dict[str, str] = {}

# Grab a salary mention (a $ figure, optionally a range) plus a little context, so
# the reviewer can eyeball pay. We do NOT filter on it (see phase2-job-criteria).
_SALARY_RE = re.compile(
    r"\$\s?\d{2,3}(?:,\d{3})?(?:\s?[-–to]+\s?\$?\s?\d{2,3}(?:,\d{3})?)?(?:\s?(?:k|/yr|per year|annually|USD))?",
    re.IGNORECASE,
)


class GreenhouseError(ATSError):
    """Raised when a board can't be read (bad token, network error, non-200). Subclasses
    the shared ATSError, so callers can catch it as either GreenhouseError or ATSError."""


def board_name(token: str) -> str:
    """Company display name for a board token (cached). Falls back to the token itself."""
    if token in _board_name_cache:                   # already looked this up? reuse the answer
        return _board_name_cache[token]
    try:
        resp = requests.get(f"{BASE}/{token}", headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        # Use the API's "name" field; if it's blank/missing, fall back to the token.
        name = (resp.json().get("name") or "").strip() or token
    except (requests.RequestException, ValueError):
        name = token                                 # any failure → just use the token as the name
    _board_name_cache[token] = name                  # remember it for next time
    return name


def list_jobs(token: str) -> list[dict]:
    """List all jobs on a board (lightweight — titles + locations, no descriptions).

    One API call returns every posting, which is exactly what the matcher needs for
    the cheap first-pass title/location filter. Fetch full descriptions only for the
    survivors via get_job(). Raises GreenhouseError on a non-200 / bad token."""
    try:
        resp = requests.get(f"{BASE}/{token}/jobs", headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])           # the API returns {"jobs": [...]}
    except (requests.RequestException, ValueError) as e:
        raise GreenhouseError(f"Could not list jobs for board '{token}': {e}") from e

    company = board_name(token)
    # Convert each raw API job into our normalized shape (list comprehension).
    return [_normalize_summary(j, token, company) for j in jobs]


def get_job(token: str, job_id) -> dict:
    """Fetch a single job WITH its full description text. Use on jobs that already
    passed the cheap title/location filter, to avoid pulling every description."""
    try:
        resp = requests.get(
            f"{BASE}/{token}/jobs/{job_id}",
            params={"questions": "true", "pay_transparency": "true"},  # ask for pay info too
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise GreenhouseError(f"Could not fetch job {job_id} on board '{token}': {e}") from e

    company = board_name(token)
    description = _html_to_text(data.get("content", ""))   # the description comes as HTML → text
    # Prefer Greenhouse's STRUCTURED pay (many boards put salary here, not in the
    # description text); fall back to a snippet scraped from the description.
    salary_min, salary_max, salary_text = _extract_structured_pay(data)
    return {
        "source": "greenhouse",
        "board_token": token,
        "id": str(data.get("id", "")),
        "title": (data.get("title") or "").strip(),
        "company": company,
        # location is a nested object {"name": ...}; (... or {}) guards against it being missing.
        "location": ((data.get("location") or {}).get("name") or "").strip(),
        "url": (data.get("absolute_url") or "").strip(),
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_text": salary_text or _find_salary(description),
        "date_posted": _posted_date(data),
    }


def _posted_date(data: dict) -> str:
    """Date the job was first published, as YYYY-MM-DD (falls back to updated_at)."""
    raw = data.get("first_published") or data.get("updated_at") or ""
    return raw[:10] if raw else ""                   # [:10] keeps just the "YYYY-MM-DD" part


def _extract_structured_pay(data: dict) -> tuple:
    """Pull a (min_salary:int|None, max_salary:int|None, display_text:str) from Greenhouse's
    structured pay fields (USD only): pay_input_ranges, then a currency_range / 'Pay Range' in
    metadata. Returns (None, None, '') when no structured pay is present."""
    # A tiny helper to turn a pay amount into a whole-dollar integer.
    def usd(amount) -> int:
        return int(float(amount))

    # Greenhouse sometimes provides explicit pay ranges. Check those first.
    for rng in data.get("pay_input_ranges") or []:
        cur = (rng.get("currency_type") or rng.get("currency") or "USD").upper()
        if cur not in ("USD", ""):                   # only handle US dollars
            continue
        if rng.get("min_cents") is not None:         # amounts given in cents
            lo = usd(rng["min_cents"]) // 100        # // is integer division (cents → dollars)
            hi = usd(rng.get("max_cents") or rng["min_cents"]) // 100
            return lo, hi, f"${lo:,} - ${hi:,}"      # the :, adds thousands separators
        if rng.get("min_value") is not None:         # amounts given in dollars
            lo, hi = usd(rng["min_value"]), usd(rng.get("max_value") or rng["min_value"])
            return lo, hi, f"${lo:,} - ${hi:,}"

    # Otherwise look through the job's freeform "metadata" entries for a pay range.
    pay_text = ""
    for md in data.get("metadata") or []:
        value, name = md.get("value"), (md.get("name") or "").lower()
        if md.get("value_type") == "currency_range" and isinstance(value, dict):
            if (value.get("unit") or "USD").upper() == "USD" and value.get("min_value"):
                lo = usd(value["min_value"])
                hi = usd(value.get("max_value") or value["min_value"])
                return lo, hi, f"${lo:,} - ${hi:,}"
        if ("pay range" in name or "salary range" in name) and isinstance(value, str):
            pay_text = value                          # a text pay range (no parsed number)
    return None, None, pay_text.strip()


def _normalize_summary(job: dict, token: str, company: str) -> dict:
    """Normalize a job-list entry (no description available at this stage)."""
    return {
        "source": "greenhouse",
        "board_token": token,
        "id": str(job.get("id", "")),
        "title": (job.get("title") or "").strip(),
        "company": company,
        "location": ((job.get("location") or {}).get("name") or "").strip(),
        "url": (job.get("absolute_url") or "").strip(),
        "description": "",                            # not available in the list view
        "salary_text": "",
    }


def _html_to_text(content: str) -> str:
    """Greenhouse returns the description as an HTML-escaped string; unescape the
    entities, then strip tags to plain text."""
    if not content:
        return ""
    unescaped = html.unescape(content)               # turn &amp; back into &, etc.
    return BeautifulSoup(unescaped, "html.parser").get_text(separator="\n", strip=True)


def _find_salary(description: str) -> str:
    """Return the first salary-like snippet in the description (for review only)."""
    match = _SALARY_RE.search(description or "")
    return match.group(0).strip() if match else ""   # group(0) = the whole matched text
