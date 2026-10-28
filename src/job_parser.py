# src/job_parser.py — turns "where the job posting is" into clean job-description TEXT.
# The source can be a URL, a file path, or pasted text. For URLs this fetches the page
# SAFELY (guarding against SSRF attacks — see docs/WORKFLOW.md → "Safety note") and extracts
# just the description, throwing away menus/footers/ads.

import ipaddress                                     # parse/classify IP addresses (for the SSRF guard)
import json                                          # parse JSON-LD job data embedded in pages
import re                                            # regular expressions (pattern matching in text)
import socket                                        # resolve a hostname to its IP address(es)
from pathlib import Path                             # file paths
from urllib.parse import urljoin, urlparse           # break apart / combine URLs

import requests                                      # HTTP requests
from bs4 import BeautifulSoup                        # "BeautifulSoup": parse HTML and pull text out

from src.ats import ashby, greenhouse, lever, rippling, workday, ATSError   # connectors + shared error, to re-fetch/verify a job
from src import db                                   # look up a job's board identity

# Maps a job's source name to the connector module that can re-fetch it. Add future boards
# here (one line) and fetch_job_description below handles them automatically.
_CONNECTORS = {"greenhouse": greenhouse, "lever": lever, "rippling": rippling,
               "workday": workday, "ashby": ashby}

# A fake browser User-Agent so sites return normal pages instead of blocking "a script".
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Safety net: even a good parse occasionally grabs more than the description.
# The actual job description is ~400-900 words; cap well above that so we never
# blow up the prompt (which gets re-sent through ~6-8 LLM calls per generation).
_MAX_WORDS = 1200

# Refuse to download more than this from a job URL (defence against huge responses).
_MAX_FETCH_BYTES = 3_000_000                          # 3 MB (underscores are just digit separators)


def _is_blocked_ip(ip: str) -> bool:
    """True for any address we must never let the server fetch (SSRF guard):
    private, loopback, link-local, reserved, multicast, and CGNAT (100.64.0.0/10 —
    which Tailscale uses, so this also blocks reaching the internal LLM).

    SSRF = "Server-Side Request Forgery": a trick where a bad URL makes our server fetch
    an internal/private address it shouldn't. We refuse those addresses here."""
    addr = ipaddress.ip_address(ip)                  # turn the text IP into an address object
    if (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
        return True
    # CGNAT / shared address space is not flagged "private" by ipaddress — block it explicitly.
    return addr in ipaddress.ip_network("100.64.0.0/10") or addr in ipaddress.ip_network("::ffff:0:0/96")


def _assert_public_url(url: str) -> None:
    """Validate a URL is safe to fetch server-side: http(s) only, and every IP the
    host resolves to is publicly routable. Raises ValueError otherwise (SSRF guard).
    ('assert' here means: check, and blow up with an error if the check fails.)"""
    parts = urlparse(url)                            # split the URL into scheme/host/port/path/etc.
    if parts.scheme not in ("http", "https"):        # block file://, ftp://, etc.
        raise ValueError(f"Blocked URL scheme {parts.scheme!r}; only http/https are allowed.")
    host = parts.hostname
    if not host:
        raise ValueError("URL has no host.")
    port = parts.port or (443 if parts.scheme == "https" else 80)   # default ports if none given
    try:
        # Look up every IP address this hostname points to.
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}")
    for info in infos:
        ip = info[4][0]                              # dig the IP string out of the result tuple
        if _is_blocked_ip(ip):
            raise ValueError(f"Blocked request to non-public address {ip} (host {host!r}).")


def _safe_get(url: str, **kwargs) -> requests.Response:
    """requests.get hardened against SSRF: validates the URL (and each redirect hop)
    against private/loopback/CGNAT addresses, follows redirects manually, enforces a
    timeout, and aborts on oversized responses.

    `**kwargs` means "any extra keyword arguments" — they get passed through to requests.get."""
    kwargs.setdefault("headers", _HEADERS)           # use our headers unless the caller gave some
    kwargs.setdefault("timeout", 15)                 # give up after 15s unless told otherwise
    kwargs["allow_redirects"] = False                # we follow redirects ourselves (to re-check each)
    for _ in range(5):                               # initial request + up to 4 redirects (the "_" = unused counter)
        _assert_public_url(url)                      # re-check safety on EVERY hop
        resp = requests.get(url, **kwargs)
        if resp.is_redirect and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])   # resolve the redirect target and loop
            continue
        cl = resp.headers.get("Content-Length")      # how big the server says the response is
        if cl and cl.isdigit() and int(cl) > _MAX_FETCH_BYTES:
            raise ValueError("Job posting response is too large to process.")
        return resp
    raise ValueError("Too many redirects while fetching the job posting.")


def parse_job_posting(source: str) -> str:
    """Accept a URL, file path, or raw text string. Returns clean job posting text.
    This is the main entry point the pipeline calls."""
    if source.startswith("http://") or source.startswith("https://"):
        return _fetch_url(source)                    # it's a URL → fetch + extract

    # Treat as a file path only if it plausibly is one — a short, single-line string.
    # Pasted job descriptions are long/multi-line; running them through Path.exists()
    # raises "File name too long" (the OS rejects a 4KB filename), so guard both the
    # length/shape and any OSError before falling back to raw text.
    if len(source) < 256 and "\n" not in source:
        try:
            path = Path(source)
            if path.exists():
                return path.read_text(encoding="utf-8")   # it's a file → read it
        except OSError:
            pass

    return source                                    # otherwise: it's already the posting text


def _board_identity(job_id: str) -> tuple[str, str, str] | None:
    """A tracked job's (source, board_token, external_id) from its identity columns —
    or None for manual/unknown jobs (board_token is NULL for those). One indexed PK
    lookup; replaces the old parse-the-packed-job_id-string approach."""
    if not job_id:
        return None
    with db.connect() as conn:
        row = db.fetch_job_identity(conn, job_id)
    if row and row["board_token"]:
        return row["source"], row["board_token"], row["external_id"]
    return None


def fetch_job_description(job_id: str, job_url: str) -> tuple[str, str]:
    """Best-effort recovery of a tracked job's posting text + canonical source URL.

    Re-scoring and on-demand resume generation sometimes need the original description
    after the fact. We recover it from whatever identifier we stored: a Greenhouse Job ID
    (re-fetch via the API, which also yields the canonical URL) or a plain job URL
    (re-parse). Returns ("", source_url) when the text can't be recovered — e.g. a job
    added by pasting text, whose original posting was never stored. Never raises: every
    lookup failure degrades to ("", url) so callers can simply check the returned text."""
    identity = _board_identity(job_id)               # (source, token, id) for a board job, else None
    if identity:
        source, token, gid = identity
        connector = _CONNECTORS.get(source)          # pick the right board connector (greenhouse/lever)
        if connector:
            try:
                full = connector.get_job(token, gid)
                return full.get("description", "") or "", full.get("url") or job_url
            except Exception:
                pass                                 # API failed → fall through to the URL attempt
    if (job_url or "").startswith(("http://", "https://")):
        try:
            return parse_job_posting(job_url) or "", job_url
        except Exception:
            return "", job_url
    # No board API and no URL to re-parse (a pasted-text job) — fall back to the posting
    # text we saved when the job was added, if any. Never raises: any DB hiccup → "".
    try:
        with db.connect() as conn:
            stored = db.fetch_job_description_text(conn, job_id)
        if stored:
            return stored, job_url
    except Exception:
        pass
    return "", job_url                               # nothing we can recover from


def _board_error_is_gone(err: Exception) -> bool:
    """True when an ATS connector error means the posting is GONE (taken down), rather than
    a transient hiccup (timeout, 500, rate-limit). Connectors raise their error `from e`, so
    the original HTTP error is chained on as __cause__ — a 404/410 status there means the
    posting was removed. We also treat an explicit 'not found' message as gone (Lever's
    empty-result path raises that without a chained HTTP error)."""
    cause = getattr(err, "__cause__", None)          # the original requests error, if any
    resp = getattr(cause, "response", None)          # its HTTP response, if any
    status = getattr(resp, "status_code", None)      # its status code, if any
    if status in (404, 410):                         # 404 Not Found / 410 Gone → removed
        return True
    return "not found" in str(err).lower()


def job_is_active(job_id: str, job_url: str) -> tuple[bool, str]:
    """Best-effort check that a tracked job is STILL live, run right before we spend LLM
    tokens tailoring a résumé for it. Jobs are often discovered days before their résumé is
    generated (Phase 2 discovery → a scheduled batch run), and a posting can close in that
    gap.

    Returns (active, reason). We report INACTIVE only on a POSITIVE 'this is gone' signal —
    a board 404/410, or a 404/410 on the posting URL. ANY uncertainty (a network blip, a
    timeout, a board error, a pasted-text job with no URL to check) returns active=True, so a
    transient failure never makes us skip a perfectly good job.
      • Board jobs (Greenhouse/Lever): re-fetch via the connector API — a removed posting 404s.
      • Plain-URL jobs: GET the posting URL (SSRF-guarded) and treat 404/410 as closed.
    """
    identity = _board_identity(job_id)               # (source, token, id) for a board job, else None
    if identity:
        source, token, gid = identity
        connector = _CONNECTORS.get(source)          # greenhouse / lever
        if connector:
            try:
                connector.get_job(token, gid)        # a live posting fetches without error
                return True, "active"
            except ATSError as e:
                if _board_error_is_gone(e):
                    return False, "the posting was taken down on its job board"
                return True, "active"                # board/network error → don't skip
            except Exception:
                return True, "active"                # anything unexpected → don't skip
    if (job_url or "").startswith(("http://", "https://")):
        try:
            resp = _safe_get(job_url)                # SSRF-guarded GET (doesn't raise on 404)
            if resp.status_code in (404, 410):
                return False, f"the posting URL is no longer available (HTTP {resp.status_code})"
            return True, "active"
        except Exception:
            return True, "active"                    # couldn't reach it → assume active, don't skip
    return True, "active"                             # pasted-text job / no URL → nothing to verify


def _fetch_url(url: str) -> str:
    """Fetch a job posting from a URL, extracting ONLY the job description text.

    Strategy, cheapest/cleanest first:
      1. LinkedIn → pull the job ID and hit the guest endpoint, which returns
         just the description fragment (no nav / "people also viewed" / footer).
      2. Any page with a JSON-LD `JobPosting` schema → use its `description`.
      3. Fallback → largest meaningful content block, scrubbed of boilerplate.
    The result is capped at `_MAX_WORDS` so a bad parse can't balloon the prompt.
    """
    linkedin = _linkedin_description(url)            # try strategy 1
    if linkedin:
        return _cap(linkedin)

    response = _safe_get(url)                        # safely fetch the page
    response.raise_for_status()                      # error out if the server said "not OK"
    soup = BeautifulSoup(response.text, "html.parser")   # parse the HTML

    jsonld = _jsonld_description(soup)               # try strategy 2
    if jsonld:
        return _cap(jsonld)

    return _cap(_main_content(soup))                 # strategy 3 (fallback)


def _linkedin_description(url: str) -> str | None:
    """If `url` is a LinkedIn job posting, fetch just the description via the
    public guest endpoint. Returns None if not LinkedIn or if the fetch fails."""
    if "linkedin.com" not in url.lower():
        return None                                  # not a LinkedIn URL

    job_id = _linkedin_job_id(url)                   # pull the numeric job ID out of the URL
    if not job_id:
        return None

    # LinkedIn exposes a "guest" endpoint that returns just the description fragment.
    endpoint = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    try:
        resp = requests.get(endpoint, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None                                  # network/HTTP problem → give up gracefully

    soup = BeautifulSoup(resp.text, "html.parser")
    # The guest fragment wraps the description in a dedicated container; targeting
    # it strips the surrounding sign-in / apply chrome the endpoint also returns.
    # select_one(...) finds the first element matching a CSS selector; "or" tries a backup.
    markup = soup.select_one("div.show-more-less-html__markup") or soup.select_one(
        "div.description__text"
    )
    text = (markup or soup).get_text(separator="\n", strip=True)
    return text or None                              # return the text, or None if it's empty


def _linkedin_job_id(url: str) -> str | None:
    """Extract the 10ish-digit job ID from a LinkedIn job URL.

    Handles both `/jobs/view/4402180403/` and `?currentJobId=4402180403` forms.
    """
    # re.search looks for the pattern anywhere in the string. \d+ means "one or more digits";
    # the parentheses "capture" those digits so .group(1) can return them.
    match = re.search(r"/jobs/view/(\d+)", url)
    if match:
        return match.group(1)
    match = re.search(r"currentJobId=(\d+)", url)
    if match:
        return match.group(1)
    return None


def _jsonld_description(soup: BeautifulSoup) -> str | None:
    """Many job boards embed a schema.org JobPosting as JSON-LD with a clean
    `description` field. Find it and return that description as text.
    (JSON-LD is structured data sites tuck into <script> tags for search engines.)"""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")      # parse the script's JSON content
        except (json.JSONDecodeError, TypeError):
            continue                                 # not valid JSON → skip this tag
        for node in _iter_jsonld_nodes(data):        # walk every object in that JSON
            if isinstance(node, dict) and node.get("@type") == "JobPosting":
                description = node.get("description")
                if description:
                    # `description` is usually an HTML string — strip the tags to plain text.
                    return BeautifulSoup(description, "html.parser").get_text(
                        separator="\n", strip=True
                    )
    return None


def _iter_jsonld_nodes(data):
    """Walk JSON-LD, which may be a single object, a list, or a @graph wrapper.

    This is a GENERATOR: `yield` hands back values one at a time as they're found, and
    `yield from` does that for a nested call. It lets the loop above visit every object
    no matter how the JSON is nested."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_nodes(item)      # recurse into each list item
    elif isinstance(data, dict):
        yield data                                   # hand back this object…
        if "@graph" in data:
            yield from _iter_jsonld_nodes(data["@graph"])   # …and any nested @graph list


def _main_content(soup: BeautifulSoup) -> str:
    """Fallback extraction: drop chrome, then prefer the page's main content
    container; if none, take the largest text block on the page."""
    # Remove page "chrome" (navigation, styling, scripts, etc.) so it doesn't pollute the text.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()                              # delete the tag from the parsed document

    main = soup.find("main") or soup.find("article") # prefer a semantic main/article container
    if main:
        return main.get_text(separator="\n", strip=True)

    # No semantic container — pick the densest <div>/<section> by text length.
    candidates = soup.find_all(["div", "section"])
    if candidates:
        # max(..., key=...) returns the candidate with the most text. The lambda measures each.
        best = max(candidates, key=lambda t: len(t.get_text(strip=True)))
        return best.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True) # last resort: all text on the page


def _cap(text: str) -> str:
    """Trim to `_MAX_WORDS` words so a noisy parse can't bloat the prompt."""
    words = text.split()                             # split on whitespace into a list of words
    if len(words) <= _MAX_WORDS:
        return text                                  # already short enough
    return " ".join(words[:_MAX_WORDS])              # keep only the first _MAX_WORDS words
