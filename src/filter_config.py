"""Editable filter configuration for Phase 2 matching.

The keyword lists + thresholds the matcher uses are stored PER USER in the database
(the user_settings + user_filter_terms tables; all SQL in src/db.py). This module owns
the neutral, role-agnostic DEFAULTS a brand-new user starts from, the UI SCHEMA, and the
load(user_id)/save(user_id, cfg) helpers that assemble/persist a user's config.
Keywords are matched case-insensitively, so store them lowercase.

Per-user, not global: a user who has saved filters gets exactly theirs; a user who
hasn't gets the DEFAULTS below. Nothing here is tied to any one person's job search —
each user's own tuning lives in their user rows (seeded by the v8 migration), NOT
in this file.

HOW A TITLE QUALIFIES (the wide-net sieve — see matching.title_qualifies):
  • it contains ANY ONE domain keyword or qualifying phrase, OR any priority /
    high-value term (those are the 5★/4★ role families), OR
  • the "meaning net": its AI embedding is close enough to the target_roles_text
    paragraph — so a creatively-titled role you'd never list still gets through.
GENERIC words (manager, director, software, …) never qualify a title by themselves —
they only add ranking signal when a real domain word is also present.
"""

# DEFAULTS: the NEUTRAL starter config a user with no saved filters gets. Role/field/
# location lists are empty on purpose (each user fills in their own field, target roles,
# commute zone, and salary floor via the Settings page). Only the job-TYPE screens
# (non-full-time / contract / gig phrases) carry generic, broadly-useful values, since
# those describe employment type, not any particular job field. A dict mapping each
# setting name to its value; lists are keyword lists, numbers are thresholds,
# target_roles_text is a free-text paragraph.
DEFAULTS = {
    # No salary floor by default (0 = keep jobs regardless of stated pay). Each user sets
    # their own floor. NOTE: a stated pay BELOW a non-zero floor still drops the job.
    'salary_floor': 0,
    # Drop a posting older than this many days (by its Date Posted). 90 ≈ 3 months.
    # 0 disables the age filter. Postings with no date are kept (age can't be judged).
    # Applies at BOTH ends: the scan won't add an older posting (matching.match_job), and
    # the home page won't list one it already tracks (db._AGE_VISIBLE) — otherwise a job
    # that was fresh when found would sit in the list forever, which is what made the
    # table slow. Applied jobs and jobs with a generated résumé are always exempt.
    'max_posting_age_days': 90,

    # AI Fit at or below this puts a job in the home page's "Under Qualified" tab instead
    # of "To Apply". Deliberately NOT in SCHEMA: it's a display setting, edited in the
    # home page's own Tools → Settings modal, not a matcher rule on the Settings page.
    'underqualified_at_or_below': 5,

    # The paragraph the title "meaning net" compares against — a user describes their
    # target roles here. Empty by default → the meaning net is inactive until set.
    'target_roles_text': '',
    # Meaning-net threshold as a whole percent (0-100). A title whose embedding is at
    # least this similar to target_roles_text qualifies. Scan logs print each admitted
    # title's similarity so you can tune this after a real run.
    'title_similarity_percent': 45,

    # DOMAIN keywords: any ONE of these in a title qualifies the job for the deep screen.
    # Empty by default — each user adds the keywords for THEIR field.
    'domain_keywords': [],

    # QUALIFYING PHRASES: multi-word titles that qualify on their own even though their
    # individual words are generic ("manager" alone never qualifies, but "Engineering
    # Manager" does). Empty by default.
    'qualifying_phrases': [],

    # GENERIC words: common words that also appear in unrelated titles. They NEVER qualify
    # a title alone — only ranking signal next to a domain word. Empty by default.
    'generic_keywords': [],

    # HYBRID jobs are kept only when their location mentions one of these (the user's
    # commute zone). Remote jobs pass from anywhere in the US; on-site jobs always drop.
    # Empty by default — each user sets their own commutable locations.
    'commutable_locations': [],

    # TOP tier (rated 5/5): a user's core most-wanted role terms. Empty by default.
    'priority_keywords': [],
    # SECONDARY tier (rated 4/5): strong adjacent role terms. Empty by default.
    'high_value_keywords': [],
    # ANY of these in a title drops the job (wrong field / seniority). Empty by default —
    # these are inherently personal to a user's field and level, so a new user adds their
    # own. ('architect'/'sales' get the learning/engineering carve-out in
    # matching._NEGATIVE_CARVEOUTS when a user does add them.)
    'title_negative_keywords': [],
    # Non-full-time employment TYPE terms (title). Generic + broadly useful, so kept as a
    # sensible default (not personal to any field). A user seeking contract work can clear it.
    'employment_exclude_keywords': [   'part-time',
                                       'part time',
                                       'contract',
                                       'contractor',
                                       'temporary',
                                       'temp',
                                       'seasonal',
                                       'per diem',
                                       'prn',
                                       'hourly'],
    'description_exclude_phrases': [   'part-time position',
                                       'part time position',
                                       'part-time role',
                                       'part time role',
                                       'part-time opportunity',
                                       'contract position',
                                       'contract role',
                                       'contract opportunity',
                                       'contract assignment',
                                       'this is a contract',
                                       'contract-to-hire',
                                       'contract to hire',
                                       'fixed-term',
                                       'fixed term contract',
                                       'month contract',
                                       'temporary position',
                                       'temporary role',
                                       'temporary assignment',
                                       'temp-to-perm',
                                       'temp to perm',
                                       'seasonal position',
                                       'seasonal role',
                                       'per diem',
                                       '1099 ',
                                       'w-2 contract',
                                       'corp-to-corp',
                                       'hourly rate',
                                       'per hour',
                                       '/hour',
                                       '/hr ',
                                       'hourly position',
                                       'no experience required',
                                       'no prior experience',
                                       'no experience necessary',
                                       'entry-level position',
                                       'entry level position',
                                       'data labeling',
                                       'data annotation',
                                       'label data',
                                       'annotate data',
                                       'side hustle',
                                       'gig work',
                                       'earn extra'],
    # Informational 'Technical' flag terms — field-specific, so empty by default (a user
    # adds the tech signals relevant to their field). Does not affect the fit score.
    'technical_terms': []}

# UI metadata: (key, label, type, help). Order = display order on the Settings page.
# Each tuple describes one editable setting so settings.html can render a labeled field
# for it. "list" settings show a text box of keywords; "number" settings show a number
# box; "text" settings show a free-text paragraph box.
# The SCHEMA order is also the display order, arranged in three plain-language groups
# (settings.html prints a heading before each group's first key). help text is deliberately
# terse: one line, plain words, says exactly what the filter does.
SCHEMA = [
    # ── Group 1: Roles you want (what qualifies a title + how it ranks) ──
    ("target_roles_text", "Describe your ideal roles", "text",
     "One or two plain sentences. The AI matches each job title's MEANING against this — so "
     "a good title with none of your keywords still gets through."),
    ("title_similarity_percent", "Meaning-match strictness (%)", "number",
     "How close a title's meaning must be to your description above (0–100). Lower = wider net."),
    ("domain_keywords", "Must-have title words", "list",
     "Any ONE of these in a title qualifies it. One term per line."),
    ("qualifying_phrases", "Qualifying phrases", "list",
     "Whole phrases that qualify on their own — 'Engineering Manager' passes, bare 'Manager' doesn't."),
    ("priority_keywords", "Top picks (★5)", "list",
     "Your most-wanted roles. These titles score 5/5 and sort to the top."),
    ("high_value_keywords", "Strong picks (★4)", "list",
     "Close runner-up roles. These score 4/5."),
    ("generic_keywords", "Vague words (rank only)", "list",
     "Common words like 'manager', 'director'. Never qualify a title alone — they only nudge ranking."),

    # ── Group 2: Roles you skip (drops) ──
    ("title_negative_keywords", "Skip these title words", "list",
     "Any ONE in a title drops the job (wrong field or level)."),
    ("employment_exclude_keywords", "Skip these job types (title)", "list",
     "Any ONE in a title drops it as not full-time — contract, temp, part-time…"),
    ("description_exclude_phrases", "Skip these phrases (job text)", "list",
     "Any ONE found in the posting text drops the job (contract / part-time / gig)."),

    # ── Group 3: Location, pay & freshness (hard rules on the full posting) ──
    ("commutable_locations", "Your commute zone", "list",
     "HYBRID jobs are kept only if the location matches one of these. Remote passes anywhere in "
     "the US; on-site always drops."),
    ("salary_floor", "Minimum salary (USD)", "number",
     "Drop jobs paying below this — and jobs that state no pay. 0 = keep all."),
    ("max_posting_age_days", "Max posting age (days)", "number",
     "Hide postings older than this, in your job list AND on new scans (90 ≈ 3 months). "
     "Jobs you applied to or made a résumé for always stay. No date = kept. 0 = off."),
]
# NOTE: `technical_terms` is intentionally NOT in SCHEMA — it only tags a job's `technical`
# flag, which nothing in the current UI shows or uses, so it's hidden from Settings. The key
# stays in DEFAULTS (and matching still computes the flag) in case the Technical column returns.
# Pull the setting names of each field type out of SCHEMA (handy elsewhere).
# The "_" names are the tuple parts we don't need here (label and help text).
LIST_KEYS = [k for k, _, t, _ in SCHEMA if t == "list"]
NUMBER_KEYS = [k for k, _, t, _ in SCHEMA if t == "number"]
TEXT_KEYS = [k for k, _, t, _ in SCHEMA if t == "text"]


def defaults() -> dict:
    """A fresh COPY of the neutral DEFAULTS (lists copied so callers can't mutate them)."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}


def load(user_id: int) -> dict:
    """A user's active filter config: their saved DB values, or the neutral DEFAULTS when
    they haven't configured filters yet. This is what the matcher runs on.

    A user is "configured" once they have a user_settings row; from then on their DB values
    are authoritative — including an EMPTY keyword list (they cleared it), which must not
    silently fall back to a default. A user with no settings row gets pure defaults."""
    from src import db                                # local import avoids an import cycle
    cfg = defaults()
    with db.connect() as conn:
        settings = db.fetch_user_settings(conn, user_id)
        terms = db.fetch_user_filter_terms(conn, user_id)
    if settings:                                      # this user has saved filters → authoritative
        for k in db.SETTINGS_COLS:
            if settings.get(k) is not None:           # NULL column → keep the default
                cfg[k] = settings[k]
        for list_name in LIST_KEYS:                   # a missing list means "empty", not default
            cfg[list_name] = terms.get(list_name, [])
    return cfg


def save(user_id: int, cfg: dict) -> None:
    """Persist a user's filter config to the DB (the Settings page calls this). Writes the
    scalar/text settings row plus every keyword list, so it also marks the user as
    'configured' (see load)."""
    from src import db
    with db.connect() as conn:
        db.upsert_user_settings(conn, user_id, cfg)
        for list_name in LIST_KEYS:
            db.replace_user_filter_terms(conn, user_id, list_name, cfg.get(list_name, []))
