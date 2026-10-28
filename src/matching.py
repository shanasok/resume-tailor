"""Pure-Python job matcher for Phase 2 (the "meaning net" adds one AI-embedding hook).

Filters normalized jobs (see src/ats/greenhouse.py for the shape) against the wide-net
criteria. This module stays deterministic Python so scanning thousands of jobs is free;
the one semantic step (embedding keyword-less titles) lives in job_search.py and calls
back in with semantic_pass=True.

Criteria (a job qualifies when ALL hold):
  1. TITLE SIEVE — the title has any ONE domain keyword / qualifying phrase /
     priority / high-value term (generic words like "manager" never count alone), OR
     the meaning net admitted it (semantic_pass=True).
  2. Location is confidently US (or the description confirms US).
  3. Workplace: remote passes from anywhere; HYBRID passes only when the location is
     within the user's configured commute zone (commutable_locations); ON-SITE always drops.
  4. Salary: a stated starting salary >= SALARY_FLOOR is required — no stated salary
     drops the job (once the description is available).
  5. No gig/contract/part-time signals; posting not older than MAX_POSTING_AGE_DAYS.

Ambiguity at the title-only stage is flagged, not dropped: a location that isn't
clearly US / clearly remote still qualifies but carries a flag, and the deep screen
settles it against the description.
"""
import re                                            # "regular expressions" — text pattern matching
from datetime import date, datetime                  # to judge how old a posting is (Date Posted)

from src import filter_config                        # the editable keyword lists this matcher uses

# ── BEGINNER ORIENTATION ───────────────────────────────────────────────────────────
# This is the "does this job fit me?" engine. It's pure Python (NO AI), so scanning
# thousands of jobs is free. Read docs/WORKFLOW.md → "How a job is judged" for the rules.
#
# A few Python/idea notes that recur below:
#  • A "job" is a dict with keys like "title", "location", "description".
#  • match_job(job) returns a VERDICT dict. Its most important key is "qualifies" (True/False).
#    It also carries "reject_reasons" (why it was dropped) and scores for ranking.
#  • "regex" (the re module): a mini-language for finding patterns in text. You'll see
#    r"\b...\b" a lot — \b means "word boundary", so r"\bus\b" matches the whole word "us"
#    but not the "us" inside "business". The r"..." prefix is a "raw string" (backslashes
#    are literal), which regex patterns need.
#  • { } with no key:value pairs is a SET (a bag of unique items), used for fast "is X in here?"
#  • Most checks are case-insensitive: the code lowercases text first, and keywords are
#    stored lowercase.
# ───────────────────────────────────────────────────────────────────────────────────

# ── Per-user config object (replaces the old module globals) ──────────────────────────
# The matcher used to read module globals set by reload_config(). That's unsafe once each
# user has their own filters and scans can run concurrently — one user's config would
# clobber another's. Instead, a Filters object is built per scan from the user's saved
# config (filter_config.load(user_id)) and threaded into every match function.

class Filters:
    """One user's loaded matching config — the keyword lists + thresholds match_job runs
    on. Build it once per scan (load_filters) and pass it down; never share global state."""

    def __init__(self, cfg: dict):
        self.salary_floor = cfg.get("salary_floor", 0)
        self.max_posting_age_days = cfg.get("max_posting_age_days", 90)
        self.domain_keywords = cfg.get("domain_keywords", [])
        self.qualifying_phrases = cfg.get("qualifying_phrases", [])
        self.generic_keywords = cfg.get("generic_keywords", [])
        self.commutable_locations = cfg.get("commutable_locations", [])
        # config stores the meaning-net threshold as a whole percent; matcher wants 0-1.
        self.title_similarity = cfg.get("title_similarity_percent", 45) / 100.0
        self.priority_keywords = cfg.get("priority_keywords", [])
        self.high_value_keywords = cfg.get("high_value_keywords", [])
        self.title_negative_keywords = cfg.get("title_negative_keywords", [])
        self.employment_exclude_keywords = cfg.get("employment_exclude_keywords", [])
        self.description_exclude_phrases = cfg.get("description_exclude_phrases", [])
        self.technical_terms = cfg.get("technical_terms", [])


def load_filters(user_id: int) -> Filters:
    """Build a user's Filters from their saved config (or neutral defaults). One DB read;
    call once per scan and thread the result through the match functions."""
    return Filters(filter_config.load(user_id))

# Plausible annual-salary bounds, so we ignore non-salary dollar figures in the text
# (e.g. "$20/hr" gigs, "$30M in savings", training budgets).
_SALARY_MIN_PLAUSIBLE, _SALARY_MAX_PLAUSIBLE = 30000, 1000000

# Best-effort location signals (matched whole-word, case-insensitive on the location).
_US_STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}
_US_MARKERS = ["united states", "usa", "u.s.a", "u.s.", " us ", "remote - us", "remote, us", "us-remote"]
_US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}
# Comprehensive non-US places. Matched WHOLE-WORD (so "india" can't hit "Indiana", etc.)
# and only AFTER the US checks above, so any US-state collision (e.g. "Georgia", or a US
# city sharing a foreign name like "Paris, TX") resolves to US first. "georgia" is
# intentionally omitted here — it's a US state by default. "puerto rico" is omitted too
# (US territory). This is the authoritative "confidently non-US" list.
_NON_US_MARKERS = [
    # countries + common alternates
    "afghanistan", "albania", "algeria", "andorra", "angola", "argentina", "armenia",
    "australia", "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados",
    "belarus", "belgium", "belize", "benin", "bhutan", "bolivia", "bosnia", "botswana",
    "brazil", "brunei", "bulgaria", "burkina faso", "burundi", "cambodia", "cameroon",
    "canada", "cape verde", "chad", "chile", "china", "colombia", "comoros", "congo",
    "costa rica", "croatia", "cuba", "cyprus", "czech republic", "czechia", "denmark",
    "djibouti", "dominican republic", "ecuador", "egypt", "el salvador", "eritrea",
    "estonia", "eswatini", "ethiopia", "fiji", "finland", "france", "gabon", "gambia",
    "germany", "ghana", "greece", "grenada", "guatemala", "guinea", "guyana", "haiti",
    "honduras", "hong kong", "hungary", "iceland", "india", "indonesia", "iran", "iraq",
    "ireland", "israel", "italy", "ivory coast", "jamaica", "japan", "jordan",
    "kazakhstan", "kenya", "kosovo", "kuwait", "kyrgyzstan", "laos", "latvia", "lebanon",
    "lesotho", "liberia", "libya", "liechtenstein", "lithuania", "luxembourg", "macau",
    "madagascar", "malawi", "malaysia", "maldives", "mali", "malta", "mauritania",
    "mauritius", "mexico", "moldova", "monaco", "mongolia", "montenegro", "morocco",
    "mozambique", "myanmar", "namibia", "nepal", "netherlands", "new zealand",
    "nicaragua", "niger", "nigeria", "north korea", "north macedonia", "norway", "oman",
    "pakistan", "palestine", "panama", "papua new guinea", "paraguay", "peru",
    "philippines", "poland", "portugal", "qatar", "romania", "russia", "rwanda", "samoa",
    "san marino", "saudi arabia", "senegal", "serbia", "sierra leone", "singapore",
    "slovakia", "slovenia", "somalia", "south africa", "south korea", "korea", "spain",
    "sri lanka", "sudan", "suriname", "sweden", "switzerland", "syria", "taiwan",
    "tajikistan", "tanzania", "thailand", "togo", "trinidad", "tunisia", "turkey",
    "turkmenistan", "uganda", "ukraine", "united arab emirates", "uae", "united kingdom",
    "great britain", "england", "scotland", "wales", "northern ireland", "uruguay",
    "uzbekistan", "vanuatu", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe",
    # regions
    "emea", "apac", "latam", "mena", "europe", "asia", "africa", "oceania",
    "middle east", "asia pacific", "asia-pacific", "latin america", "south america",
    "central america", "caribbean", "scandinavia", "balkans", "nordics",
    # common foreign cities / subnationals
    "uk", "london", "dublin", "ontario", "toronto", "vancouver", "montreal", "berlin",
    "munich", "paris", "amsterdam", "madrid", "barcelona", "lisbon", "milan", "rome",
    "zurich", "geneva", "stockholm", "copenhagen", "oslo", "helsinki", "warsaw", "prague",
    "vienna", "brussels", "bengaluru", "bangalore", "mumbai", "new delhi", "hyderabad",
    "pune", "chennai", "sydney", "melbourne", "tokyo", "seoul", "shanghai", "beijing",
    "shenzhen", "manila", "jakarta", "bangkok", "ho chi minh", "kuala lumpur", "dubai",
    "abu dhabi", "tel aviv", "sao paulo", "são paulo", "buenos aires", "mexico city",
    "bogota", "lima", "santiago", "cairo", "lagos", "nairobi", "istanbul", "moscow",
]
# One word-boundary regex for all of the above (longest first so phrases win).
_NON_US_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(_NON_US_MARKERS, key=len, reverse=True)) + r")\b"
)
_REMOTE_MARKERS = ["remote", "distributed", "work from home", "wfh", "anywhere"]
_HYBRID_MARKERS = ["hybrid"]
_ONSITE_MARKERS = ["on-site", "onsite", "on site", "in office", "in-office",
                   "in person", "in-person"]

# The excluded-title / non-full-time / description-gig phrase lists are per-user settings
# carried on the Filters object (f.title_negative_keywords, f.employment_exclude_keywords,
# f.description_exclude_phrases), built from filter_config.load(user_id).

# Ranking: seniority signal in the title boosts a job so the strongest fits are
# generated first. Highest matching tier wins (not summed).
_SENIORITY_WEIGHTS = [
    (["chief", "vp", "vice president", "head of"], 6),
    (["director", "principal"], 5),
    (["senior", "sr.", "sr ", "staff", "lead"], 2),
    (["manager"], 1),
]


# ── Posting age ──────────────────────────────────────────────────────────────
def posting_age_days(date_posted: str) -> int | None:
    """How many days ago the job was posted, from a 'YYYY-MM-DD' Date Posted string.
    Returns None when the date is missing/unparseable — so callers KEEP such jobs
    (we don't drop a posting just because its age can't be determined)."""
    if not date_posted:
        return None
    try:
        posted = datetime.strptime(str(date_posted)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    return (date.today() - posted).days


# ── Title screens (quality + employment type) ────────────────────────────────
def _title_hits(title: str, keywords) -> list[str]:
    """Return which of `keywords` appear in `title` as WHOLE words, case-insensitively.
    e.g. keyword "sales" matches "Sales Lead" but not "Wholesaler"."""
    text = (title or "").lower()                     # lowercase so matching ignores case
    # For each keyword, keep it if the regex finds it as a standalone word. re.escape(kw)
    # treats the keyword literally; (?<!\w) and (?!\w) require non-word chars on both sides
    # (i.e. a word boundary). This is a list comprehension (PRIMER §6).
    return [kw for kw in keywords if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text)]


# Some negative terms are FORGIVEN when the title also shows the right context.
# "architect" normally disqualifies (Solutions/Software/Data Architect aren't target
# roles) but "Learning Solutions Architect" is on-target; "sales" disqualifies
# ("Sales Manager") but "Sales Enablement" / "Sales Engineering" are real targets.
# Map: negative term → context terms that forgive it when ALSO present in the title.
_LEARNING_CONTEXT_TERMS = ["learning", "instructional", "curriculum", "l&d",
                           "training", "education", "enablement"]
_NEGATIVE_CARVEOUTS = {
    "architect": _LEARNING_CONTEXT_TERMS,
    "architecture": _LEARNING_CONTEXT_TERMS,
    "sales": ["enablement", "engineer", "engineering", "training"],
}


def negative_hits(title: str, f: "Filters") -> list[str]:
    """Gig-work / wrong-seniority terms in the title (a non-empty list disqualifies).
    A term with a carve-out (architect, sales, …) is forgiven when the title also
    contains one of its context terms (e.g. 'Learning Technology Architecture',
    'Sales Enablement Manager')."""
    hits = _title_hits(title, f.title_negative_keywords)   # all negative terms found in the title
    # Staff/principal-level IC ENGINEER roles are above the target level. The list has
    # the contiguous phrases ('staff engineer', …), but titles like 'Staff Machine
    # Learning Engineer' or 'Principal Backend Engineer' split them — so also drop any
    # title where staff/principal co-occurs with engineer as words. 'Engineering' is
    # deliberately NOT matched ('Staff Engineering Manager' is a manager role).
    if (_title_hits(title, ["staff", "principal"])
            and _title_hits(title, ["engineer", "engineers"])
            and not hits):
        hits = ["staff/principal-level engineer"]
    # For each hit that has a carve-out, forgive it (drop it from the list) when any of
    # its context terms is also in the title.
    return [h for h in hits
            if not (h in _NEGATIVE_CARVEOUTS and _title_hits(title, _NEGATIVE_CARVEOUTS[h]))]


def non_fulltime_hits(title: str, f: "Filters") -> list[str]:
    """Non-full-time employment terms in the title (a non-empty list disqualifies)."""
    return _title_hits(title, f.employment_exclude_keywords)


def seniority_score(title: str) -> int:
    """Highest seniority weight signaled by the title (0 if none)."""
    text = (title or "").lower()
    for terms, weight in _SENIORITY_WEIGHTS:
        if any(t in text for t in terms):
            return weight
    return 0


def description_exclude_hits(description: str, f: "Filters") -> list[str]:
    """Employment-type / gig phrases found in the job DESCRIPTION (a non-empty list
    disqualifies). Contextual phrases only — so legit roles that merely mention interns,
    trainees, or vendor contracts are NOT dropped."""
    text = (description or "").lower()
    return [p for p in f.description_exclude_phrases if p in text]


# A pay amount — handles $ OR USD OR a bare number, with commas/decimals and an optional
# K suffix:  $200,000 · 200000 · $200K · USD 200,000 · 200,000.00 · 200k
_AMOUNT = r"(?:\$\s?|usd\s?)?(\d{2,3}(?:,\d{3})+|\d{4,7}|\d{2,3})(?:\.\d+)?\s?([kK])?"
_SALARY_RANGE_RE = re.compile(_AMOUNT + r"\s*(?:to|through|and|[-–—]+)\s*" + _AMOUNT, re.I)
_AMOUNT_RE = re.compile(_AMOUNT, re.I)
# Terms that signal the surrounding numbers are pay (case-insensitive).
_SALARY_CONTEXT_RE = re.compile(
    r"salary|base pay|base salary|\bbase\b|compensation|\bcomp\b|pay range|salary range|"
    r"pay rate|\bpay\b|\bwage\b|/ ?year|per year|per yr|annually|annual|/ ?yr\b|"
    r"range for this|range is|expected (?:pay|salary|compensation)|hiring range|"
    r"compensation range|on-target earnings|\bote\b|\busd\b", re.I)


def _amount_value(num_str: str, k_suffix: str) -> int:
    n = float(num_str.replace(",", ""))
    return int(n * 1000) if k_suffix else int(n)


def _plausible_salary(n: int) -> bool:
    return _SALARY_MIN_PLAUSIBLE <= n <= _SALARY_MAX_PLAUSIBLE


def parse_salary_range(description: str) -> tuple[int | None, int | None, str]:
    """Best-effort extraction of the pay range from a job description.

    Returns (min_salary, max_salary, matched_text). Strategy: prefer explicit pay
    RANGES ("$151,000 to $200,000", "Pay range: 150,000–200,000 USD"). ALL plausible
    ranges are scanned (a posting may list different pay per zone/level/area) and the
    result spans them: min = the lowest low, max = the HIGHEST high — so a floor check
    screens on the top of the best range. When any range exists, single amounts
    (stray OTE/equity one-offs) are IGNORED. Only when NO range is present do we fall
    back to a single plausible amount near a pay term ("salary", "/year", "USD"…),
    with min == max. Implausible figures ($20/hr, $30M) are ignored. Returns
    (None, None, "") when no pay is stated."""
    text = description or ""
    # Don't misread retirement plans ("401k", "401(k)", "403b", "457b") as pay.
    text = re.sub(r"\b401\s*\(?k\)?|\b403\s*\(?b\)?|\b457\s*\(?b\)?", " ", text, flags=re.I)

    # 1) Explicit ranges. Collect EVERY plausible range; span them (lowest low, highest
    #    high). The two-number range pattern is a strong pay signal on its own, so no
    #    nearby keyword is required.
    los, highest = [], None            # highest = (hi, matched_text) with the biggest hi
    for m in _SALARY_RANGE_RE.finditer(text):
        low = _amount_value(m.group(1), m.group(2))
        high = _amount_value(m.group(3), m.group(4))
        lo, hi = min(low, high), max(low, high)
        if _plausible_salary(lo) and _plausible_salary(hi):
            los.append(lo)
            if highest is None or hi > highest[0]:
                highest = (hi, m.group(0).strip())
    if highest is not None:
        return min(los), highest[0], highest[1]

    # 2) Single plausible amount within 120 chars of a pay term (min == max).
    kw_positions = [m.start() for m in _SALARY_CONTEXT_RE.finditer(text)]
    if kw_positions:
        for m in _AMOUNT_RE.finditer(text):
            value = _amount_value(m.group(1), m.group(2))
            if _plausible_salary(value) and any(abs(m.start() - kp) <= 120 for kp in kw_positions):
                return value, value, m.group(0).strip()

    return None, None, ""


# The top-tier (→ 5★) and secondary (→ 4★) term lists are per-user settings on the Filters
# object (f.priority_keywords, f.high_value_keywords).


# ── Title matching ───────────────────────────────────────────────────────────
def _distinct_hits(text_lower: str, keywords) -> list[str]:
    """Whole-word matches of `keywords` in already-lowercased text, with phrase
    suppression: a matched multi-word phrase removes its contained single words, so
    overlapping terms (e.g. "product manager" + "manager") count once."""
    # Find every keyword present as a whole word.
    hits = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)]
    # Collect the individual words that make up any matched MULTI-word phrase. This is a "set
    # comprehension" ({ }): for each multi-word hit (" " in kw), add each of its words.
    phrase_tokens = {tok for kw in hits if " " in kw for tok in kw.split()}
    # Keep a hit if it's a phrase, OR a single word that ISN'T already part of a matched
    # phrase. So "product manager" + "manager" collapses to just "product manager" (counts once).
    return [kw for kw in hits if " " in kw or kw not in phrase_tokens]


# The exact reject-reason string the title sieve produces. job_search.py matches on it
# to find "meaning net" candidates (jobs whose ONLY problem is a keyword-less title).
TITLE_SIEVE_REJECT = "title has no qualifying domain keyword/phrase"


def matched_keywords(title: str, f: "Filters") -> list[str]:
    """Distinct keywords of ANY tier (domain + phrases + generic) present in `title`.
    This is RANKING signal only — qualification is title_qualifies(). dict.fromkeys
    dedupes terms that appear in more than one list, keeping first-seen order."""
    terms = list(dict.fromkeys([*f.domain_keywords, *f.qualifying_phrases, *f.generic_keywords]))
    return _distinct_hits((title or "").lower(), terms)


def title_qualifies(title: str, f: "Filters") -> tuple[bool, list[str]]:
    """The wide-net title sieve: (does this title qualify?, which terms hit).
    Any ONE domain keyword or qualifying phrase passes. Generic words (manager,
    director, software, …) are deliberately absent — they never qualify alone."""
    terms = list(dict.fromkeys([*f.domain_keywords, *f.qualifying_phrases]))
    hits = _distinct_hits((title or "").lower(), terms)
    return bool(hits), hits


def high_value_hits(title: str, f: "Filters") -> list[str]:
    """Distinct SECONDARY high-value (4★) terms present in `title`."""
    return _distinct_hits((title or "").lower(), f.high_value_keywords)


def priority_hits(title: str, f: "Filters") -> list[str]:
    """Distinct TOP-priority (5★) terms present in `title` — the user's core most-wanted
    role terms."""
    return _distinct_hits((title or "").lower(), f.priority_keywords)


def title_desirability(title: str, f: "Filters") -> int:
    """A 1-5 'how close to my desired role' score, from the title alone:
      5 = ANY top-priority term (the user's core role family)
      4 = a secondary high-value term, OR 4+ keywords
      3 = 3 keywords        2 = 2 keywords        1 = barely qualifying.
    Seniority is kept out of the score (it's only a sort tiebreaker in rank_score)."""
    if priority_hits(title, f):                       # user's top-priority terms → top
        return 5
    if high_value_hits(title, f):                     # other strong adjacent roles → 4★
        return 4
    kw = len(matched_keywords(title, f))
    if kw >= 4:
        return 4
    if kw >= 3:
        return 3
    if kw >= 2:
        return 2
    return 1


def has_technical_terms(text: str, f: "Filters") -> bool:
    """True if `text` (title and/or description) contains any of the user's technical_terms.
    Used ONLY to set the informational 'Technical' flag on a job — it does not change the
    fit score."""
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in f.technical_terms)


# ── Location classification (best-effort) ────────────────────────────────────
def _has_marker(text: str, markers) -> bool:
    return any(m in text for m in markers)


def classify_location(location: str) -> str:
    """'usa' | 'non_usa' | 'unverified' — best-effort from the free-text location."""
    text = f" {(location or '').lower()} "
    if _has_marker(text, _US_MARKERS):
        return "usa"
    # A "City, ST" with a US state abbreviation is a strong US signal (e.g. "Austin, TX").
    # The abbreviation must be UPPERCASE — case-insensitive matching let lowercase English
    # words after a comma collide with state codes (the ", or" in "Canada - Remote (ON, AB,
    # BC, or NS Only)" looked like Oregon, mislabeling Canadian jobs as US). Connector
    # locations always capitalize the state code, so requiring uppercase loses nothing real.
    if any(re.search(rf",\s*{abbr}\b", location or "") for abbr in _US_STATE_ABBR):
        return "usa"
    # ...or a full US state name anywhere in the location ("Chicago, Illinois").
    if any(re.search(rf"\b{re.escape(name)}\b", text) for name in _US_STATE_NAMES):
        return "usa"
    if _NON_US_RE.search(text):
        return "non_usa"
    return "unverified"


def names_non_us_place(location: str) -> bool:
    """True if the location text names a confidently non-US country/region/city (e.g.
    'Canada - Remote'). Unlike classify_location, this does NOT first try to resolve the
    string to a US state — so it still fires when a loose 'City, ST' collision (e.g. the
    ', or' in 'ON, AB, BC, or NS' looking like Oregon) would otherwise mask the country.
    Used to break duplicate-posting ties toward the US twin."""
    return bool(_NON_US_RE.search(f" {(location or '').lower()} "))


# Explicit US signals safe to match in free DESCRIPTION text. Deliberately excludes the
# bare word "us" (→ "join us", "contact us") and 2-letter state abbreviations (OR, IN,
# OK, ME, HI… are common English words), which would falsely confirm US.
_US_DESC_MARKERS = [
    "united states", "u.s.a", "u.s.", "u. s.", "us-based", "u.s.-based",
    "based in the us", "within the us", "across the us", "us only", "us-only",
    "authorized to work in the united states",
]


def description_confirms_us(description: str) -> bool:
    """True if the DESCRIPTION explicitly indicates a US-based role. Conservative —
    explicit country markers, 'USA' as a whole word, or a full US state name only."""
    text = (description or "").lower()
    if any(m in text for m in _US_DESC_MARKERS):
        return True
    if re.search(r"\busa\b", text):
        return True
    return any(re.search(rf"\b{re.escape(name)}\b", text) for name in _US_STATE_NAMES)


def classify_workplace(location: str, description: str = "") -> str:
    """'remote' | 'hybrid' | 'onsite' | 'unverified' — where the work happens.

    The LOCATION field decides first (it's short, deliberate text). Precedence there is
    onsite > hybrid > remote, so "Denver, CO (Hybrid)" reads hybrid even when the
    company boilerplate says remote. If the location says nothing, the DESCRIPTION is
    consulted — remote-first there, because long postings that offer "remote or hybrid
    options" can be done remotely and shouldn't be held to the commute-zone rule."""
    loc = (location or "").lower()
    if _has_marker(loc, _ONSITE_MARKERS):
        return "onsite"
    if _has_marker(loc, _HYBRID_MARKERS):
        return "hybrid"
    if _has_marker(loc, _REMOTE_MARKERS):
        return "remote"
    desc = (description or "").lower()
    if desc:
        if re.search(r"\bremote\b", desc):
            return "remote"
        if re.search(r"\bhybrid\b", desc):
            return "hybrid"
        if _has_marker(desc, _ONSITE_MARKERS):
            return "onsite"
    return "unverified"


def is_commutable(location: str, f: "Filters") -> bool:
    """True when the location names somewhere in the user's commute zone (their editable
    commutable_locations list). Only consulted for HYBRID jobs — remote passes from anywhere."""
    text = (location or "").lower()
    return any(marker in text for marker in f.commutable_locations)


# ── Top-level match ──────────────────────────────────────────────────────────
def match_job(job: dict, f: "Filters", semantic_pass: bool = False) -> dict:
    """Evaluate one normalized job against a user's wide-net criteria (`f`).

    Returns a verdict dict; `qualifies` is True when the title passes the sieve
    (title_qualifies — or semantic_pass=True, meaning the embedding "meaning net"
    already admitted this title), there are no negative/non-full-time terms, the
    location is US, the workplace rule holds (remote anywhere / hybrid only when
    commutable / never on-site), and — once a description exists — a stated salary
    >= the user's salary floor. Unverified location/workplace still qualifies at the
    title-only stage but is flagged. `rank_score` orders the strongest fits first."""
    # ── Look at the TITLE first (cheap; works even before we have a description) ──
    title = job.get("title", "")
    sieve_ok, _sieve_hits = title_qualifies(title, f)   # the wide-net sieve (domain/phrase)
    keywords = matched_keywords(title, f)               # ALL tiers — ranking signal only
    priority = priority_hits(title, f)                  # top-tier terms (→ 5★)
    high_value = high_value_hits(title, f)              # secondary "most-wanted" terms (→ 4★)
    negatives = negative_hits(title, f)
    non_fulltime = non_fulltime_hits(title, f)
    # Description-based screens only run when a description is present (i.e. on full
    # jobs fetched via get_job); title-only summaries skip them harmlessly.
    description = job.get("description", "")
    has_description = bool(description.strip())
    desc_has_remote = "remote" in description.lower()
    desc_confirms_us = has_description and description_confirms_us(description)
    desc_excludes = description_exclude_hits(description, f)
    # Prefer the connector's STRUCTURED pay (e.g. Greenhouse's pay-range metadata);
    # fall back to parsing the description text only when no structured figure exists.
    if job.get("salary_min") is not None:
        min_salary = job["salary_min"]
        max_salary = job.get("salary_max") or min_salary
        salary_match = job.get("salary_text", "")
    else:
        min_salary, max_salary, salary_match = parse_salary_range(description)
    location_status = classify_location(job.get("location", ""))
    workplace = classify_workplace(job.get("location", ""), description)

    flags = []
    if location_status == "unverified" and not has_description:
        flags.append("location unverified — confirm US in description")
    # Workplace unverified: flag (and check the description) — only at the
    # title-only stage, where there's no description yet to confirm against.
    if workplace == "unverified" and not has_description:
        flags.append("remote unverified — confirm in description")
    # A commutable hybrid job passes but gets marked, so it's visibly not fully remote.
    if workplace == "hybrid" and is_commutable(job.get("location", ""), f):
        flags.append("hybrid — commutable")
    # Reasons a job is rejected (kept for transparency/debugging).
    reject = []
    # Title sieve: ONE domain keyword / qualifying phrase / priority / high-value term
    # is enough — the net is deliberately wide; the AI ranking sorts the catch after
    # storage. Generic words alone never pass. semantic_pass=True means the embedding
    # meaning net already admitted this title, so the sieve is skipped.
    if not (sieve_ok or priority or high_value or semantic_pass):
        reject.append(TITLE_SIEVE_REJECT)
    if negatives:
        reject.append("title gig/seniority term(s): " + ", ".join(negatives))
    if non_fulltime:
        reject.append("title non-full-time term(s): " + ", ".join(non_fulltime))
    if desc_excludes:
        reject.append("description employment/gig phrase(s): " + ", ".join(desc_excludes))
    # Salary comes from a connector's structured pay or a parse of the description. Drop a
    # job whose TOP pay is below the floor, OR that states no pay at all (once we have the
    # description). Screening on the max keeps a wide range like $125k–$250k for a $200k floor.
    if f.salary_floor and max_salary is not None and max_salary < f.salary_floor:
        reject.append(f"salary tops out at ${max_salary:,} (< ${f.salary_floor:,})")
    elif f.salary_floor and has_description and max_salary is None:
        reject.append("no salary stated in description")
    # Posting age: drop a job whose Date Posted is older than the configured limit, so we
    # don't tailor resumes for stale listings. A missing/unknown date is kept (age is None).
    age_days = posting_age_days(job.get("date_posted", ""))
    if f.max_posting_age_days and age_days is not None and age_days > f.max_posting_age_days:
        reject.append(f"posted {age_days} days ago (> {f.max_posting_age_days}-day limit)")
    if location_status == "non_usa":
        reject.append("location is non-US")
    elif location_status == "unverified" and has_description and not desc_confirms_us:
        # Location didn't confirm US AND the description has no explicit US signal → drop.
        reject.append("US not confirmed in location or description")
    # Workplace rule: remote passes from anywhere (US); HYBRID only when the location is
    # in the user's commute zone; ON-SITE always drops.
    if workplace == "onsite":
        reject.append("on-site role")
    elif workplace == "hybrid" and not is_commutable(job.get("location", ""), f):
        reject.append("hybrid outside your commute zone")
    elif workplace == "unverified" and has_description and not desc_has_remote:
        # Location didn't settle it AND the description never says "remote" → drop.
        reject.append("no 'remote' found in location or description")

    # The job QUALIFIES only if there are zero reject reasons. "not reject" is True when the
    # reject list is empty (an empty list is falsy — see PRIMER §5).
    qualifies = not reject
    # FIT (1-5): 5★ for top-priority (learning/training/enablement/upskilling) roles, 4★ for
    # secondary high-value terms or keyword-dense titles, down to 1★. `technical` is still
    # computed for the informational 'Technical' flag, but no longer changes the fit.
    technical = has_technical_terms(f"{title} {description}", f)
    desirability = title_desirability(title, f)
    # rank_score sorts the strongest fits first. Multiplying fit by 1000 and seniority by 10
    # makes fit the dominant factor, then seniority, then keyword breadth as a final nudge.
    rank_score = desirability * 1000 + seniority_score(title) * 10 + len(keywords)

    return {
        "qualifies": qualifies,
        "matched_keywords": keywords,
        "keyword_count": len(keywords),
        "high_value": high_value,
        "technical": technical,
        "desirability": desirability,
        "seniority": seniority_score(title),
        "rank_score": rank_score,
        "location_status": location_status,
        "workplace": workplace,
        "semantic_pass": semantic_pass,              # admitted by the meaning net, not keywords
        "description_excludes": desc_excludes,
        "min_salary": min_salary,
        "max_salary": max_salary,
        "flags": flags,
        "reject_reasons": reject,
        # Prefer the parsed salary range/figure for display; fall back to any snippet.
        "salary_text": salary_match or job.get("salary_text", ""),
    }


def filter_jobs(jobs: list[dict], f: "Filters") -> list[dict]:
    """Return jobs qualifying under `f`, each annotated with its match verdict under
    'match', sorted strongest-first by rank_score (so a max_resumes cap spends the token
    budget on the best fits, not whatever was scanned first)."""
    out = []
    for job in jobs:
        verdict = match_job(job, f)                    # judge each job for this user
        if verdict["qualifies"]:
            # {**job, "match": verdict} makes a new dict = all of job's keys PLUS a "match"
            # key holding the verdict. (The ** "spreads" job's key/values into the new dict.)
            out.append({**job, "match": verdict})
    # Sort best-first by the rank score we computed. reverse=True = highest first.
    out.sort(key=lambda j: j["match"]["rank_score"], reverse=True)
    return out


def keyword_only_rejects(jobs: list[dict], f: "Filters") -> list[dict]:
    """Jobs whose ONLY problem is a keyword-less title (every other title-stage screen
    passed). These are the "meaning net" candidates: job_search.py embeds their titles
    and re-admits any whose meaning is close enough to the target-roles profile."""
    out = []
    for job in jobs:
        verdict = match_job(job, f)
        if verdict["reject_reasons"] == [TITLE_SIEVE_REJECT]:
            out.append({**job, "match": verdict})
    return out
