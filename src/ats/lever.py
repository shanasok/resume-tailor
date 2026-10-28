"""Lever Postings API connector (READ side).

Lever's public Postings API powers companies' public careers pages, so — like Greenhouse —
it's meant to be read openly: no auth, no API key, no ToS gray area.

  List jobs:  GET https://api.lever.co/v0/postings/{token}?mode=json
  Job detail: GET https://api.lever.co/v0/postings/{token}/{id}?mode=json

`{token}` is the company's Lever slug (the bit in jobs.lever.co/{token}, e.g. "spotify").
Discovery of which tokens exist is handled separately (Common Crawl harvest + validation,
in lever_discovery.py); this module just reads a known board.

KEY DIFFERENCE FROM GREENHOUSE: Lever's LIST endpoint already returns each posting's full
description and structured fields, so list_jobs() returns fully-populated jobs (no second
fetch needed). get_job() exists for the re-fetch-by-id path (used when regenerating a
résumé for a single tracked Lever job).

Every job is normalized to the shared shape the matcher + pipeline expect (same as
greenhouse.py): source/board_token/id/title/company/location/url/description/
salary_text/salary_min/salary_max/date_posted.
"""
from datetime import datetime, timezone             # convert Lever's epoch-millis timestamp to a date

import html                                          # un-escape HTML entities in the description
import requests                                      # HTTP requests
from bs4 import BeautifulSoup                        # turn the HTML description into plain text

from src.ats import ATSError                         # shared base error (see src/ats/__init__.py)

BASE = "https://api.lever.co/v0/postings"            # the Postings API base address
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_TIMEOUT = 15                                         # seconds to wait on any single request


class LeverError(ATSError):
    """Raised when a Lever board can't be read (bad token, network error, non-200)."""


def _company_name(token: str) -> str:
    """Lever has no 'board name' endpoint, so derive a display name from the slug:
    "some-co" → "Some Co". Best-effort — you can edit it on the tracked row later."""
    pretty = token.replace("-", " ").replace("_", " ").strip().title()
    return pretty or token


def _posted_date(created_at) -> str:
    """Lever's createdAt is epoch MILLISECONDS. Convert to a "YYYY-MM-DD" string ("" if absent)."""
    if not created_at:
        return ""
    try:
        # /1000 converts milliseconds → seconds; utcfromtimestamp builds a UTC date.
        return datetime.fromtimestamp(int(created_at) / 1000, tz=timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return ""


def _location(posting: dict) -> str:
    """Build a single location string the matcher can read, combining Lever's structured
    fields: the city/region, the workplace type (remote/onsite/hybrid), and the country.

    Why combine them: the matcher (matching.py) classifies US-ness and remote-ness from the
    job's `location` text, so we fold Lever's separate signals into one string it understands
    (e.g. workplaceType "remote" makes "remote" appear; country "US" → "United States")."""
    cats = posting.get("categories") or {}
    # Prefer the single location; fall back to joining allLocations if that's all we have.
    place = (cats.get("location") or "").strip() or ", ".join(cats.get("allLocations") or [])
    workplace = (posting.get("workplaceType") or "").strip()   # "remote" / "onsite" / "hybrid"
    country = (posting.get("country") or "").strip()
    # Map the US country code to a phrase the matcher recognizes as US; pass others through.
    country_label = "United States" if country.upper() == "US" else country
    # Keep only the non-empty pieces and join them with " · ".
    parts = [p for p in (place, workplace, country_label) if p]
    return " · ".join(parts)


def _salary(posting: dict) -> tuple:
    """Best-effort (min_salary:int|None, max_salary:int|None, display_text:str) from Lever's
    optional salaryRange (USD only), falling back to salaryDescriptionPlain text. Returns
    (None, None, '') when absent."""
    rng = posting.get("salaryRange")
    if isinstance(rng, dict):
        currency = (rng.get("currency") or "USD").upper()
        lo, hi = rng.get("min"), rng.get("max")
        if currency in ("USD", "") and lo is not None:
            lo = int(lo)
            hi = int(hi) if hi is not None else lo
            return lo, hi, f"${lo:,} - ${hi:,}"          # the :, adds thousands separators
    # No structured range → use any plain-text salary note Lever provides.
    return None, None, (posting.get("salaryDescriptionPlain") or "").strip()


def _description(posting: dict) -> str:
    """The full posting text as plain text. Lever gives descriptionPlain (already plain) plus
    `lists` (responsibilities etc.) and additionalPlain; we stitch the plain parts together."""
    parts = [posting.get("descriptionPlain") or ""]
    # `lists` is an array of {text: heading, content: HTML}. Strip the HTML to text.
    for section in posting.get("lists") or []:
        heading = (section.get("text") or "").strip()
        body = _html_to_text(section.get("content") or "")
        if heading or body:
            parts.append(f"{heading}\n{body}".strip())
    parts.append(posting.get("additionalPlain") or "")
    # Join the non-empty parts with blank lines between them.
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _html_to_text(content: str) -> str:
    """Un-escape HTML entities then strip tags to plain text (same approach as greenhouse.py)."""
    if not content:
        return ""
    return BeautifulSoup(html.unescape(content), "html.parser").get_text(separator="\n", strip=True)


def _normalize(posting: dict, token: str) -> dict:
    """Turn one raw Lever posting (from the API) into the project's shared job shape."""
    salary_min, salary_max, salary_text = _salary(posting)
    return {
        "source": "lever",
        "board_token": token,
        "id": str(posting.get("id", "")),
        "title": (posting.get("text") or "").strip(),       # Lever calls the title "text"
        "company": _company_name(token),
        "location": _location(posting),
        "url": (posting.get("hostedUrl") or "").strip(),
        "description": _description(posting),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_text": salary_text,
        "date_posted": _posted_date(posting.get("createdAt")),
    }


def list_jobs(token: str) -> list[dict]:
    """List all postings on a Lever board, fully populated (titles, locations, AND
    descriptions). One API call returns everything the matcher needs. Raises LeverError
    on a non-200 / bad token."""
    try:
        resp = requests.get(f"{BASE}/{token}", params={"mode": "json"},
                            headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        postings = resp.json()                        # Lever returns a JSON LIST of postings
    except (requests.RequestException, ValueError) as e:
        raise LeverError(f"Could not list jobs for Lever board '{token}': {e}") from e
    if not isinstance(postings, list):                # defensive: unexpected shape
        raise LeverError(f"Unexpected response shape for Lever board '{token}'.")
    return [_normalize(p, token) for p in postings]


def get_job(token: str, job_id) -> dict:
    """Fetch a single posting by id (used when re-generating a résumé for one tracked job).
    Returns the same normalized shape as list_jobs()."""
    try:
        resp = requests.get(f"{BASE}/{token}/{job_id}", params={"mode": "json"},
                            headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise LeverError(f"Could not fetch Lever job {job_id} on board '{token}': {e}") from e
    # The single-posting endpoint may return the posting directly, or wrapped in a list.
    if isinstance(data, list):
        if not data:
            raise LeverError(f"Lever job {job_id} on board '{token}' not found.")
        data = data[0]
    return _normalize(data, token)
