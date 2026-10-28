"""LinkedIn 1st-degree connections lookup.

Loads `data/Connections.csv` (the standard LinkedIn "Connections" export:
First Name, Last Name, URL, Email Address, Company, Position, Connected On) and
indexes it by a normalized company name so the UI can show, for each job, the
people you already know at that company — with a link to their profile so you can
ask them to refer you.
"""

import csv
import os
import re
from functools import lru_cache

CONNECTIONS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "Connections.csv")

# Common corporate suffixes/filler words stripped before matching so that a job's
# "Meta" matches a connection's "Meta Platforms", "Acme Inc." matches "Acme", etc.
_NOISE = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|company|co|gmbh|plc|sa|ag|nv|"
    r"group|holdings|platforms|technologies|technology|labs|software|systems)\b",
    re.I,
)


def _norm(name: str) -> str:
    """Normalize a company name for fuzzy matching (lowercase, strip punctuation/suffixes)."""
    if not name:
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _index(mtime: float):
    """Build {normalized_company: [ {name, url, position}, ... ]}.

    Keyed on the file's mtime so edits to Connections.csv are picked up without a
    restart (a new mtime is a cache miss). Returns an empty dict if the file is absent.
    """
    idx: dict[str, list[dict]] = {}
    if not os.path.exists(CONNECTIONS_CSV):
        return idx
    with open(CONNECTIONS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            company = _norm(row.get("Company", ""))
            url = (row.get("URL") or "").strip()
            if not company or not url:
                continue
            name = f"{(row.get('First Name') or '').strip()} {(row.get('Last Name') or '').strip()}".strip()
            idx.setdefault(company, []).append(
                {"name": name or url, "url": url, "position": (row.get("Position") or "").strip()}
            )
    return idx


def connections_for(company: str) -> list[dict]:
    """Return the list of connections at `company` (each {name, url, position}); [] if none."""
    key = _norm(company)
    if not key:
        return []
    try:
        mtime = os.path.getmtime(CONNECTIONS_CSV)
    except OSError:
        return []
    return _index(mtime).get(key, [])
