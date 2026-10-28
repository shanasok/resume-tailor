"""Unified job + resume tracker, backed by the normalized PostgreSQL schema.

Storage (see src/db.py): a job POSTING lives in `jobs`; its generated RÉSUMÉ (if any)
in `resumes` (FK → jobs); the fact you APPLIED (if you did) in `applications`
(FK → jobs). The `job_overview` view joins the three back into one flat row per job.

Both flows share the store: manual resume generation (Source='manual', resume
attached immediately) and Phase 2 automated discovery (Source='greenhouse', no
resume until you choose to make one). A resume is just data attached to a job, so
both flows share dedupe, applied tracking, and one source of truth.

Lifecycle status is DERIVED, not stored:  Found (no resume row) → Generated (resume
row exists) → Applied (application row exists).

Rows flow through the app as flat dicts keyed by the old display names ("Job ID",
"Company", "HTML", …), but the VALUES keep their real database types: dates are
`datetime.date`, Match Score a float, Gen Seconds an int, Technical a bool, and None
means "not set". Templates render these directly (Jinja shows a date as ISO, and the
app's finalize hook turns None into ""); no string round-tripping anywhere.
"""
import threading                                     # locking, so concurrent web requests stay safe
from datetime import date, timedelta                 # today's date + the posting-age window
from pathlib import Path                             # OS-independent file paths

from src import db                                   # the PostgreSQL store (schema + row helpers)
from src import embeddings                           # base-résumé vector → the Similarity column
from src.renderer import _slug                        # company+role → the filename-safe slug (manual ids)

# Serializes log_application's find-then-write sequence so two threads in THIS process
# can't both miss the lookup and insert duplicate manual jobs. (The other writers are
# single atomic statements and don't need it.)
# An RLock is a "reentrant lock" — only one thread can hold it at a time.
_write_lock = threading.RLock()


def _synchronized(fn):
    """A DECORATOR (see PRIMER §11). Wrapping a function with @_synchronized makes it grab
    the lock before running, so its read-the-rows → decide → write-them sequence can't be
    interrupted by another thread doing the same thing (which could insert a duplicate)."""
    def wrapper(*args, **kwargs):                     # this replaces the original function…
        with _write_lock:                            # …grabbing the lock for the duration…
            return fn(*args, **kwargs)               # …then calling the real function.
    wrapper.__name__ = fn.__name__                   # copy the name/docs so it still looks like the original
    wrapper.__doc__ = fn.__doc__
    return wrapper


# The project root folder (paths in rows are stored relative to it).
PROJECT_ROOT = Path(__file__).parent.parent

# The display field names every row dict carries — the app-wide contract (templates,
# stats, job_search all read these keys). "Flags" is a list, "Stage LLMs" a
# {stage: backend} dict; the rest are scalars (see _to_fields). "Similarity" is the
# 0-1 embedding match to the base résumé (None = job not embedded / no résumé vector);
# "AI Fit Score"/"AI Fit Reason" are ChatGPT's 1-10 verdict (None/"" = not scored yet).
FIELDS = ["Job ID", "Source", "Date Matched", "Date Posted", "Company", "Role", "Location",
          "Job URL", "Flags", "Salary", "Technical", "Match Score", "HTML", "PDF",
          "Applied", "Applied Date", "Gen Seconds", "Stage LLMs",
          "Similarity", "AI Fit Score", "AI Fit Reason", "Critic Findings"]

# The four LLM-driven generation stages, in display order (owned by the schema — each
# is a real column on resumes).
STAGE_ORDER = db.STAGES


def _rel(path) -> str:
    """Store paths relative to the project root for a clean, portable log
    (e.g. "resumes/generated_resumes/x.html" instead of a long absolute path)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:                               # path isn't under the project → store as-is
        return str(path)


def _job_url(source: str) -> str:
    """Decide what to record in the job's URL column. For a real URL, record it; for a
    file or pasted text, record a short note instead."""
    if source.startswith("http://") or source.startswith("https://"):
        return source
    # Only probe the filesystem for short, single-line strings — a pasted job
    # description is long/multi-line and would make Path.exists() raise
    # "File name too long" (the OS rejects a multi-KB filename).
    if len(source) < 256 and "\n" not in source:
        try:
            if Path(source).exists():
                return f"(file: {source})"
        except OSError:
            pass
    return "(pasted text)"


def resume_key(html_path) -> str:
    """Stable per-resume identifier: just the generated HTML filename (no folder)."""
    return Path(str(html_path)).name


def manual_job_id(user_id: int, html_path) -> str:
    """Job ID for a user's manually generated resume (stable per user + company+role slug).
    The user_id keeps manual postings distinct across users (board postings dedupe on a
    global id; manual ones are user-authored, so two users' "acme_director_resume" must not
    collide). .stem is the filename without its extension, e.g. "acme_director_resume"."""
    return f"manual:{user_id}:{Path(str(html_path)).stem}"


def make_job_id(job: dict) -> str:
    """Stable dedupe key for a discovered job: "source:board:id" — e.g.
    "greenhouse:stripe:42" or "lever:spotify:abc-123". Works for ANY board, because the
    source comes from the job's own "source" field."""
    return f"{job.get('source', 'greenhouse')}:{job['board_token']}:{job['id']}"


def status_of(row: dict) -> str:
    """Work out a job's lifecycle status from which fields are filled in (it's not stored
    as its own column). Applied wins, then a resume exists (Generated), else just Found."""
    if row.get("Applied"):
        return "Applied"
    if row.get("HTML"):
        return "Generated"
    return "Found"


# ── The read boundary: view rows → the app's flat dicts (keys renamed, types KEPT) ────

def _to_fields(row: dict) -> dict:
    """Convert one job_overview row into the app-wide flat dict keyed by FIELDS.
    Values keep their real database types — dates are datetime.date, Match Score a
    float, Gen Seconds an int, Technical a bool; None means "not set". Text columns
    are plain strings, with "" standing in when the joined résumé/application row
    doesn't exist yet (callers treat these as paths/labels, so str keeps them safe)."""
    return {
        "Job ID": row["job_id"],
        "Source": row["source"],
        "Date Matched": row["date_matched"],         # datetime.date | None (per-user)
        "Date Posted": row["date_posted"],           # datetime.date | None
        "Company": row["company"],
        "Role": row["role"],
        "Location": row["location"],
        "Job URL": row["job_url"],
        "Flags": list(row["flags"] or []),           # matcher warnings (job_flags rows)
        "Salary": row["salary"],
        "Technical": row["technical"],               # bool | None (None = never assessed)
        # NUMERIC comes back as Decimal; float is what callers pass in and jsonify likes.
        "Match Score": None if row["match_score"] is None else float(row["match_score"]),
        "HTML": row["html_path"] or "",              # NULL when no résumé row is joined
        "PDF": row["pdf_path"] or "",
        "Applied": row["applied_via"] or "",         # NULL when no application row is joined
        "Applied Date": row["applied_date"],         # datetime.date | None
        "Gen Seconds": row["gen_seconds"],           # int | None
        # {stage: backend} for the stages that ran — built from the four <stage>_llm
        # columns (empty/absent stages omitted).
        "Stage LLMs": {s: v for s in STAGE_ORDER if (v := row[f"{s}_llm"])},
        # ChatGPT's fit verdict (None/"" until the job is scored after a scan).
        "AI Fit Score": row["ai_fit_score"],         # int 1-10 | None
        "AI Fit Reason": row["ai_fit_reason"] or "",
        # Filled in by read_applications (needs the base-résumé vector, one query for
        # ALL rows); stays None on single-row reads like get_job.
        "Similarity": None,
        # The critic's report on the generated résumé ({kind: [finding, …]}, kinds
        # fidelity/overstatement/style/revision/strength) — filled in by read_applications/get_job
        # from resume_findings; {} until a generation stores one.
        "Critic Findings": {},
    }


def read_applications(user_id: int) -> list[dict]:
    """This user's tracked jobs/resumes, most recent first by Date Matched. Each row is
    annotated with a derived 'Status' (Found/Generated/Applied) for the UI, and with
    'Similarity' — the embedding match to THEIR base résumé — when both the résumé vector
    and the job's embedding exist (no OpenAI call here; the résumé vector is only ever
    computed/refreshed at scan time)."""
    resume_vec = embeddings.cached_base_resume_embedding(user_id)   # stored vector or None
    with db.connect() as conn:
        rows = db.fetch_overview(conn, user_id, newest_first=True)
        # One indexed query gives {job_id: similarity} for every embedded job THEY track.
        sims = (db.fetch_similarity_map(conn, user_id, resume_vec,
                                        embeddings.active_model())
                if resume_vec else {})
        # …and one gives every résumé's critic report (findings tooltip on the score).
        findings = db.fetch_findings_map(conn, user_id)
    out = []
    for r in rows:
        row = _to_fields(r)
        row["Similarity"] = sims.get(r["job_id"])    # None when unembedded/unrankable
        row["Critic Findings"] = findings.get(r["job_id"], {})
        row["Status"] = status_of(row)               # attach the derived status for display
        out.append(row)
    return out


# ── The home page's paged read ────────────────────────────────────────────────────────

def _age_cutoff(max_age_days: int | None):
    """The earliest posting date still shown, or None when the age filter is off.

    Computed HERE, in Python, so it uses the app's own idea of today. The docker Postgres
    runs UTC and `date_posted` is written with the app's local date, so asking SQL for
    CURRENT_DATE would shift the window by a day on US evenings.

    0 / None turn the filter off — the same convention the Settings page documents."""
    if not max_age_days:
        return None
    return date.today() - timedelta(days=int(max_age_days))


def read_job_page(user_id: int, *, tab: str, max_age_days: int | None = None,
                  underqualified_at_or_below: int = 5, company: str | None = None,
                  sort: str | None = None, descending: bool = False,
                  limit: int | None = None, offset: int = 0) -> list[dict]:
    """One rendered page of this user's jobs table, as flat display dicts with 'Status'.

    Unlike read_applications this reads only the rows the page will actually show — the
    posting-age window, one tab, the company search, ordered and sliced in SQL. It also
    skips the similarity map: nothing displays it, and ordering now happens in the
    database, so a page load no longer runs a vector query over every tracked job."""
    with db.connect() as conn:
        rows = db.fetch_overview_page(
            conn, user_id, posted_on_or_after=_age_cutoff(max_age_days), tab=tab,
            underqualified_at_or_below=underqualified_at_or_below, company=company,
            sort=sort, descending=descending, limit=limit, offset=offset)
        findings = db.fetch_findings_map(conn, user_id)
    out = []
    for r in rows:
        row = _to_fields(r)
        row["Critic Findings"] = findings.get(r["job_id"], {})
        row["Status"] = status_of(row)
        out.append(row)
    return out


def count_job_tabs(user_id: int, *, max_age_days: int | None = None,
                   underqualified_at_or_below: int = 5,
                   company: str | None = None) -> dict[str, int]:
    """{tab: row count} for the four tabs, under the same window read_job_page uses —
    so the badge on a tab always matches what opening it shows."""
    with db.connect() as conn:
        return db.count_overview_tabs(
            conn, user_id, posted_on_or_after=_age_cutoff(max_age_days),
            underqualified_at_or_below=underqualified_at_or_below, company=company)


def job_summary(user_id: int, *, max_age_days: int | None = None,
                company: str | None = None) -> dict[str, int]:
    """The summary-bar tallies (total / applied / generated / …) over the same visible
    rows the table pages through."""
    with db.connect() as conn:
        return db.count_overview_summary(
            conn, user_id, posted_on_or_after=_age_cutoff(max_age_days), company=company)


def get_job(user_id: int, job_id: str) -> dict | None:
    """One of this user's tracked jobs as a flat display dict (with Status), or None.
    Prefer this over scanning read_applications() when you already know the Job ID — it's
    one indexed lookup instead of loading and joining every row."""
    with db.connect() as conn:
        row = db.fetch_overview_row(conn, user_id, job_id)
        findings = db.fetch_resume_findings(conn, user_id, job_id) if row else {}
    if row is None:
        return None
    out = _to_fields(row)
    out["Critic Findings"] = findings
    out["Status"] = status_of(out)
    return out


def tracked_job_urls(user_id: int) -> set[str]:
    """Every URL this user already tracks — the cheap dedupe set for their board scan."""
    with db.connect() as conn:
        return db.fetch_job_urls(conn, user_id)


@_synchronized                                       # ← lock while we read-decide-write
def log_application(user_id: int, company: str, role: str, source: str, html_path, pdf_path,
                    job_id: str = None, match_score=None, gen_seconds=None,
                    stage_llms: dict = None, critic_findings: dict = None) -> None:
    """Record/refresh a generated resume. Attaches the résumé row to an existing job so
    a resume for a discovered job updates it in place instead of duplicating; the job's
    application status and Source are untouched.

    `job_id`, when provided (the per-row 'Generate Resume' button knows exactly which job
    was clicked), is the AUTHORITATIVE match — the resume always attaches to that exact
    job. Without it we fall back to matching by Job URL, then resume filename, then the
    derived manual Job ID; if nothing matches, a brand-new 'manual' job is created.

    `gen_seconds` (whole seconds the generation took) and `stage_llms` ({stage: backend}
    for the four generation stages — each stored in its own column) are recorded for the
    Stats page. Each is only written when supplied, so a re-render that doesn't re-generate
    (which passes none of them) won't wipe the résumé row's existing timing/per-stage data.

    `critic_findings` ({kind: [finding, …]}, kinds fidelity/overstatement/style/revision/strength) is
    the critic's report on this résumé; when supplied it REPLACES the stored report (it
    describes exactly one generation). Omitted → the existing report is kept."""
    # Only treat the source as a URL if it actually is one. .startswith((...,...)) checks both.
    url = source if source.startswith(("http://", "https://")) else None

    # ── The résumé data itself (only the supplied bits — see docstring) ──
    resume_values = {"html_path": _rel(html_path), "pdf_path": _rel(pdf_path)}
    if match_score is not None:
        resume_values["match_score"] = match_score
    if gen_seconds is not None:
        resume_values["gen_seconds"] = gen_seconds
    if stage_llms is not None:
        for stage in STAGE_ORDER:
            if stage_llms.get(stage):
                resume_values[f"{stage}_llm"] = stage_llms[stage]

    with db.connect() as conn:
        # Targeted, indexed lookups (see find_job_for_resume for the priority order) —
        # no need to load the whole table to find one of this user's jobs.
        target = db.find_job_for_resume(conn, user_id, job_id, url, resume_key(html_path),
                                        manual_job_id(user_id, html_path))
        if target:
            # ── Refresh the shared posting's display fields, attach/refresh THIS user's résumé ──
            job_updates = {"company": company, "role": role}
            existing = db.fetch_overview_row(conn, user_id, target)
            if url and not existing["job_url"]:
                job_updates["job_url"] = url
            # job_id / source / date_matched / application row / location / salary / flags untouched.
            db.update_job(conn, target, job_updates)
            db.upsert_resume(conn, user_id, target, resume_values)
            if critic_findings is not None:
                db.replace_resume_findings(conn, user_id, target, critic_findings)
        else:
            # ── No existing job → add a brand-new 'manual' posting + this user's overlay + résumé ──
            new_id = manual_job_id(user_id, html_path)
            _, board_token, external_id = db.split_job_id(new_id)
            db.upsert_posting(conn, {
                "job_id": new_id,
                "source": "manual",
                "board_token": board_token,          # holds the user id for manual ids
                "external_id": external_id,          # the résumé slug
                "company": company,
                "role": role,
                "job_url": _job_url(source),
            })
            db.upsert_user_job(conn, user_id, new_id, date.today())
            db.upsert_resume(conn, user_id, new_id, resume_values)
            if critic_findings is not None:
                db.replace_resume_findings(conn, user_id, new_id, critic_findings)


def upsert_found_jobs(user_id: int, jobs: list[dict]) -> dict:
    """Add discovered jobs (from find_matches) to this user's list. Does NOT generate
    resumes. Dedupes by Job ID: postings new to this user get an overlay row (no resume);
    ones they already track get the shared display fields refreshed but keep their
    application/resume/date_matched. Returns {'added': n, 'refreshed': n} (relative to
    THIS user's list).

    No lock needed here: each job is atomic upserts (shared row + this user's overlay),
    so concurrent scans can't duplicate or clobber a row."""
    added = refreshed = 0                            # counters (both start at 0)

    with db.connect() as conn:
        for job in jobs:
            match = job.get("match", {})             # the matcher's verdict dict (may be missing)
            inserted = db.upsert_discovered_job(conn, user_id, {
                "job_id": make_job_id(job),          # the stable ID for this discovered job
                "source": job.get("source", "greenhouse"),
                "board_token": job["board_token"],   # the identity parts, as real columns
                "external_id": str(job["id"]),
                "company": job.get("company", ""),
                "role": job.get("title", ""),
                "location": job.get("location", ""),
                "job_url": job.get("url", ""),
                "salary": job.get("salary_text", ""),
                "technical": bool(match.get("technical")),
                # The app machine's date, NOT the DB server's CURRENT_DATE — the docker
                # Postgres runs on UTC, which is already "tomorrow" on US evenings.
                "date_found": date.today(),
                "date_posted": db.parse_date(job.get("date_posted")),
            }, flags=list(match.get("flags", [])))   # child rows, replaced atomically
            # Persist the posting text so scoring/rewriting never has to re-fetch it
            # from the board (boards rate-limit bulk re-fetches).
            if job.get("description"):
                db.upsert_job_description(conn, make_job_id(job), job["description"])
            # Persist the description embedding so the job gets a Match % on the home
            # page. Absent when no OpenAI key.
            if job.get("embedding"):
                db.upsert_job_embedding(conn, make_job_id(job), job["embedding"],
                                        job.get("embedding_model", ""))
            if inserted:
                added += 1
            else:
                refreshed += 1

    return {"added": added, "refreshed": refreshed}


@_synchronized                                       # ← lock: find-then-write, like log_application
def add_manual_job(user_id: int, company: str, role: str, source: str) -> dict:
    """Add a job the user found elsewhere to their tracked list WITHOUT generating a résumé.

    This backs the Manual tab's "Add job" button: it drops a new 'Found' row into the list
    so the user can queue many jobs quickly, then generate résumés later (the per-row
    'Generate Resume' button, or a scheduled batch). `source` is a URL or pasted description.

    The job_id matches what a résumé generated for this company+role later would carry
    (`manual:<user_id>:<slug>_resume`), so the per-row generate attaches its résumé to THIS
    row instead of creating a duplicate. Pasted-text descriptions are stored so a later
    generate/AI-fit can recover them (a URL is re-fetched on demand instead).

    Returns {'ok', 'added' (False when the user already tracks it), 'job_id', 'message'}."""
    job_id = manual_job_id(user_id, f"{_slug(company, role)}_resume")
    _, board_token, external_id = db.split_job_id(job_id)
    with db.connect() as conn:
        db.upsert_posting(conn, {
            "job_id": job_id,
            "source": "manual",
            "board_token": board_token,              # holds the user id for manual ids
            "external_id": external_id,              # the résumé slug
            "company": company,
            "role": role,
            "job_url": _job_url(source),
        })
        added = db.upsert_user_job(conn, user_id, job_id, date.today())
        # On a FRESH add, keep pasted text so a later résumé generation / AI-fit scoring can
        # recover it (a real URL is re-fetched on demand, so only store non-URL text). A
        # re-add of a job the user already tracks must NOT clobber the saved description.
        if added and not source.startswith(("http://", "https://")) and source.strip():
            db.upsert_job_description(conn, job_id, source)
    if not added:
        return {"ok": True, "added": False, "job_id": job_id,
                "message": f"{company} — {role} is already in your list."}
    return {"ok": True, "added": True, "job_id": job_id,
            "message": f"Added {company} — {role} to your list. "
                       "Generate a résumé for it when you're ready."}


def mark_applied(user_id: int, job_id: str) -> bool:
    """Mark this user's job `job_id` as applied (manually) today — i.e. create its row in
    the applications table. Returns True if the user tracks that job, False otherwise."""
    with db.connect() as conn:
        if not db.user_job_exists(conn, user_id, job_id):
            return False                             # this user doesn't track that job
        db.upsert_application(conn, user_id, job_id, "manual", date.today())
        return True


def delete_job(user_id: int, job_id: str, reason: str = "deleted manually") -> dict | None:
    """Remove this job from THIS user's list — their résumé/application rows go with it
    (ON DELETE CASCADE); the shared posting stays for other users. Returns the deleted row
    as a flat dict (so the caller can also delete that résumé's generated files), or None
    if the user doesn't track that job.

    The deletion is also REMEMBERED for this user (rejected_jobs, by URL): deleting a job
    means "don't show ME this again", so none of their future scans re-discovers the same
    posting. Pasted-text manual rows have no real URL — nothing to remember, and the
    scanner could never re-pull them anyway."""
    with db.connect() as conn:
        target = db.fetch_overview_row(conn, user_id, job_id)
        if target is None:
            return None
        url = target.get("job_url") or ""
        if url.startswith(("http://", "https://")):
            db.record_rejection(conn, user_id, url, target.get("company", ""),
                                target.get("role", ""), reason)
        db.delete_job_row(conn, user_id, job_id)
        return _to_fields(target)


def reset_scan(user_id: int) -> dict:
    """Return this user's "Retrieve Jobs" scan to a fresh state: remove every scan-discovered
    job they haven't acted on (no résumé, no application; manual jobs are always kept) and
    clear their permanent rejection memory, so the next Retrieve Jobs re-discovers everything.
    Generated résumés, applications, filters, and the shared job catalog are untouched.
    Returns {"jobs_deleted", "rejections_cleared"}."""
    with db.connect() as conn:
        return db.reset_scan_state(conn, user_id)
