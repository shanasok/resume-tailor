"""Generic ATS board DISCOVERY engine — shared by greenhouse_discovery and lever_discovery.

The problem (for every board): there's no "list all companies" endpoint, but every public
board is a public web page, so we can HARVEST company tokens from the public web and VALIDATE
them against the board's API. This module holds the board-AGNOSTIC machinery; each board's
own discovery module supplies just three things:
  • host_patterns — which URLs to harvest (e.g. "jobs.lever.co/*")
  • validate_fn(token) — returns a job count if the token is a live board, else None
  • source — the name to store the directory under (e.g. "lever")

Pipeline (same for every board):
  1. HARVEST candidate tokens from Common Crawl's URL index (free, no key).
  2. VALIDATE each token against the board's API — keep only live boards.
  3. STORE survivors in the shared `board_directory` table (via src/db.py). The scan reads
     the same table to know which boards to walk.

The directory used to be cached as flat data/<source>_boards.json files; it now lives in
Postgres like the rest of the app's data (migration v11 imported the old files once).

All pure Python / HTTP — no LLM. Be a polite client: this hits Common Crawl and many board
endpoints, so it's rate-limited and meant to run occasionally.
"""
import json                                          # parse Common Crawl's JSON lines
import re                                            # validate token characters
from concurrent.futures import ThreadPoolExecutor, as_completed  # validate many tokens in parallel
from datetime import date                            # stamp the returned directory with today's date
from urllib.parse import urlparse, unquote           # pull the token out of a URL

import requests                                      # HTTP requests

from src import db                                   # the board_directory table lives here (all SQL in db.py)

CC_COLLINFO = "https://index.commoncrawl.org/collinfo.json"   # lists Common Crawl collections
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Path segments that are never a board token (so we don't mistake them for company slugs).
_NON_TOKENS = {"", "embed", "v1", "assets", "favicon.ico", "robots.txt"}
# Valid board tokens are lowercase alphanumerics + hyphen/underscore. Anything else is a
# corrupted/junk path link harvested from the web. (^ = start of string, $ = end.)
_VALID_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _noop(_msg):
    """Do-nothing default for the optional progress callback."""
    pass


def _dedupe_case_insensitive(tokens: set[str]) -> set[str]:
    """Collapse tokens that differ ONLY in letter case, keeping the variant with the most
    uppercase letters. Workday's site name is case-preserving, so one board can harvest as
    both "3m.wd1@Search" and "3m.wd1@search"; the CamelCase form is the canonical one Workday
    actually serves and an all-lowercase copy is usually a degraded link. A no-op for boards
    whose tokens are already lowercased by the default extractor (no case-only duplicates)."""
    best: dict[str, str] = {}
    for token in tokens:
        key = token.lower()
        current = best.get(key)
        if current is None or sum(c.isupper() for c in token) > sum(c.isupper() for c in current):
            best[key] = token
    return set(best.values())


# ── 1. Harvest ───────────────────────────────────────────────────────────────
def latest_collections(n: int = 1) -> list[str]:
    """Return the IDs of the n most recent Common Crawl monthly collections."""
    resp = requests.get(CC_COLLINFO, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return [c["id"] for c in resp.json()][:n]         # newest first; take the first n IDs


def token_from_url(url: str) -> str | None:
    """Extract a board token from a board URL's first path segment.

    URL-decodes the segment and rejects anything outside the valid token charset, so
    corrupted page links (e.g. "1pyra)mid_health&care") don't leak in. Returns None for junk."""
    # urlparse(url).path is the "/something/else" part; strip slashes, take the first segment,
    # URL-decode it (unquote), and lowercase it.
    first = unquote(urlparse(url).path.strip("/").split("/")[0]).lower()
    if first in _NON_TOKENS or not _VALID_TOKEN.match(first):
        return None
    return first


def harvest_tokens(host_patterns: list[str], collections: list[str] = None,
                   limit_per_query: int = None, on_progress=_noop,
                   token_from_url_fn=None) -> set[str]:
    """Harvest candidate board tokens from Common Crawl's URL index for the given host
    patterns (e.g. ["jobs.lever.co/*"]).

    Streams results line-by-line so large responses don't have to fit in memory.
    `limit_per_query` caps results per (collection, host) pair — handy for a quick build;
    pass None for a full harvest. Returns a SET of unique tokens (sets auto-deduplicate).
    `token_from_url_fn` overrides the default first-path-segment extractor for boards whose
    token also depends on the HOST (e.g. Workday's tenant.dcN@Site)."""
    collections = collections or latest_collections(1)   # default to the newest collection
    extract = token_from_url_fn or token_from_url         # board-specific hook, else the default
    tokens: set[str] = set()

    for collection in collections:
        for pattern in host_patterns:
            params = {"url": pattern, "output": "json"}
            if limit_per_query:
                params["limit"] = str(limit_per_query)
            on_progress(f"Harvesting {pattern} from {collection}…")
            try:
                # stream=True lets us read the (potentially huge) response line by line.
                with requests.get(f"https://index.commoncrawl.org/{collection}-index",
                                  params=params, headers=_HEADERS, timeout=120,
                                  stream=True) as resp:
                    if resp.status_code != 200:
                        on_progress(f"  {pattern}: HTTP {resp.status_code}, skipping.")
                        continue
                    for line in resp.iter_lines():    # one JSON record per line
                        if not line:
                            continue
                        try:
                            url = json.loads(line).get("url", "")
                        except json.JSONDecodeError:
                            continue                  # skip any malformed line
                        token = extract(url)
                        if token:
                            tokens.add(token)
            except requests.RequestException as e:
                on_progress(f"  {pattern}: harvest error ({e}), skipping.")

    on_progress(f"Harvested {len(tokens)} unique candidate token(s).")
    return tokens


# ── 2 + 3. Validate + build + store ──────────────────────────────────────────
def build_directory(source: str, host_patterns: list[str], validate_fn,
                    collections: list[str] = None, limit_per_query: int = None,
                    min_jobs: int = 1, max_workers: int = 16,
                    merge_existing: bool = True, on_progress=_noop,
                    token_from_url_fn=None) -> dict:
    """Harvest → validate → store. Returns the directory dict and writes it to the
    `board_directory` table under `source`.

    `validate_fn(token)` is the board-specific check: it returns the job count if the token
    is a live board, else None. `max_workers` validates tokens concurrently (kept polite).
    Only boards with >= `min_jobs` live postings are kept. With `merge_existing` (default),
    the harvest is unioned with the tokens already stored for this source and everything is
    re-validated, so a rebuild only grows the set or prunes dead boards — never loses good ones.
    `token_from_url_fn` overrides the default token extractor (see harvest_tokens)."""
    tokens = harvest_tokens(host_patterns, collections, limit_per_query, on_progress,
                            token_from_url_fn=token_from_url_fn)

    if merge_existing:
        # Union (|) the freshly-harvested tokens with the ones already stored for this source.
        with db.connect() as conn:
            previous = {r["token"] for r in db.fetch_board_directory(conn, source)}
        new_count = len(tokens - previous)            # how many are genuinely new (set difference)
        tokens |= previous                            # |= adds all of `previous` into `tokens`
        if previous:
            on_progress(f"Merged {len(previous)} existing board token(s) "
                        f"({new_count} newly harvested) → {len(tokens)} to validate.")

    before = len(tokens)
    tokens = _dedupe_case_insensitive(tokens)          # collapse case-only duplicates (Workday sites)
    if len(tokens) < before:
        on_progress(f"Collapsed {before - len(tokens)} case-duplicate token(s) → {len(tokens)} to validate.")

    boards = []
    on_progress(f"Validating {len(tokens)} token(s) against the API…")
    # Validate many tokens at once using a pool of worker threads (network calls wait a lot,
    # so doing them in parallel is much faster than one at a time). Network only — the DB
    # write happens once, after, so no connection is held during the slow validation.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(validate_fn, t): t for t in tokens}   # one job per token
        for future in as_completed(futures):          # process results as they finish
            token = futures[future]
            count = future.result()                   # the job count, or None
            if count is not None and count >= min_jobs:
                boards.append({"token": token, "job_count": count})

    boards.sort(key=lambda b: b["job_count"], reverse=True)   # busiest boards first
    with db.connect() as conn:
        db.replace_board_directory(conn, source, boards)   # upsert survivors, prune dead boards
    on_progress(f"Kept {len(boards)} live board(s) → board_directory[{source}]")
    return {"generated_at": date.today().isoformat(), "board_count": len(boards), "boards": boards}


def load_directory(source: str) -> dict:
    """Load a source's board directory from the `board_directory` table. Raises
    FileNotFoundError (kept for the caller's existing except-clause) with guidance if the
    source has no boards yet, so the scan can tell the user to run the build first."""
    with db.connect() as conn:
        boards = db.fetch_board_directory(conn, source)
    if not boards:
        raise FileNotFoundError(
            f"No boards stored for '{source}' yet. Build it first with "
            f"`python -m src.ats.{source}_discovery`."
        )
    return {"board_count": len(boards), "boards": boards}
