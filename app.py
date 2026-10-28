#!/usr/bin/env python3
"""
Resume Tailor — Web UI (Phase 1)

A simple local front end over the resume-tailoring pipeline.
Run:  source .venv/bin/activate && python app.py
Then open http://127.0.0.1:5000

This file is the WEB version of the app. It uses Flask (a small web framework). Each
function marked with @app.route(...) is a "route" — code that runs when the browser visits
a particular web address. Like main.py, the heavy lifting is delegated to src/pipeline.py
and the other modules; this file mostly handles the browser ↔ Python plumbing.
"""
import os                                            # atomic file swap for the base-resume save
import shutil                                          # copy résumé files when archiving a version
import json                                           # read/write archived résumé version data
import re                                            # regular expressions (the email check)
import threading                                     # run the slow job scan on a background thread
import time                                          # timestamps for the scan's elapsed time
import uuid                                          # unique ids for background generation tasks
from io import BytesIO                               # validate uploaded bytes without saving first
from pathlib import Path                             # file paths
from dotenv import load_dotenv                       # load .env (API keys) into the environment

load_dotenv()                                        # must run BEFORE importing modules that read keys

# Flask gives us the web server plus these helpers (explained where each is used below).
from flask import (Flask, render_template, request, redirect, url_for, flash,
                   send_from_directory, abort, jsonify)

# Project modules — the actual functionality the routes call into.
from src.pipeline import generate                    # run the Phase 1 résumé pipeline
from src.renderer import load_cached_data, render_html, render_pdf  # re-render saved résumé data
from src.tracker import (read_applications, get_job, mark_applied, reset_scan,
                         log_application, delete_job, make_job_id,
                         add_manual_job, read_job_page, count_job_tabs,
                         job_summary)  # the jobs store (Postgres)
from src import tailoring                    # is_local_llm / local_tag / list_local_models
from src import db                                    # DB health check for /health
from src import health_checks                   # local LLM health-check registry + runner
from src.job_search import find_matches, source_names, add_job_manually  # Phase 2 job scan
from src import filter_config                        # the editable filter settings (per user)
from src.current_user import current_user_id         # the one place "who is this request?" is decided
from src.chat import chat_turn                       # résumé-enhancement chat
from src.connections import connections_for           # LinkedIn connections at a company
from src.critic import score_resume                  # re-score a résumé against a job
from src.job_parser import fetch_job_description, job_is_active  # recover a job's text; verify it's still live
from src import embeddings                            # base-résumé vector for Match % ranking
from src import fit_verdict                           # ChatGPT 1-10 fit scores for top matches
from src import scheduler                             # in-process batch scheduler (Phase 2)
from src import run_log                                # append-only DB audit of every generation
from src import resume_parser                        # base-resume location + upload validation
from src import resume_extract                        # safe .docx → structured profile extraction
from src import base_resume                            # this user's base résumé (DB profile → text)
from src import scan_audit                              # BUCKET_ORDER for the scan-audit event filters

# Simple, permissive email check — enough to catch typos without an AI/library call.
# This regex means: some non-space chars, "@", more non-space chars, ".", more non-space chars.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# The folder generated résumés live in. .resolve() makes it an absolute path (used below to
# prevent serving files outside this folder).
PROJECT_ROOT = Path(__file__).parent
GENERATED_DIR = (PROJECT_ROOT / "resumes" / "generated_resumes").resolve()

# Developer Options: a hidden Tools → Developer Options page, gated by an .env flag (off by
# default). Only when this is truthy does the button render and the /developer routes work.
DEVELOPER_OPTIONS = os.environ.get("DEVELOPER_OPTIONS", "0").lower() in ("1", "true", "yes", "on")

# Date-stamp format for the per-day sub-folder scheduled (cron) résumés are filed under,
# e.g. "20260626" for 2026-06-26. All cron runs on the same calendar day share one folder.
DATED_FOLDER_FMT = "%Y%m%d"


def _dated_dir() -> Path:
    """Today's dated sub-folder under generated_resumes (created on demand by the renderer).
    Scheduled runs write here so every resume made by cron on a given day is grouped, with
    no duplicate folders for the same date."""
    from datetime import date
    return GENERATED_DIR / date.today().strftime(DATED_FOLDER_FMT)


def _generated_relpath(stored: str) -> str:
    """A stored résumé path (relative or absolute) → its path RELATIVE to GENERATED_DIR,
    as a forward-slash string for use in /files/<path> links — e.g.
    "20260626/acme_resume.pdf" for a cron résumé, or "acme_resume.pdf" for a flat one.
    Falls back to the bare filename if the path isn't under GENERATED_DIR."""
    if not stored:
        return ""
    p = Path(stored)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    try:
        return p.resolve().relative_to(GENERATED_DIR).as_posix()
    except ValueError:
        return Path(stored).name


def _resume_dir_for(company: str, role: str) -> Path:
    """The folder a tracked résumé's files live in (its dated sub-folder for cron résumés,
    else GENERATED_DIR). Derived from the matching row's stored HTML path so re-render/
    re-read operations (Regenerate, Set email) target the right place. Defaults to
    GENERATED_DIR when there's no row or no stored file."""
    row = next((r for r in read_applications(current_user_id())
                if r.get("Company") == company and r.get("Role") == role and r.get("HTML")), None)
    if not row:
        return GENERATED_DIR
    html = Path(row["HTML"])
    if not html.is_absolute():
        html = PROJECT_ROOT / html
    parent = html.resolve().parent
    # Stay inside GENERATED_DIR no matter what was stored.
    return parent if str(parent).startswith(str(GENERATED_DIR)) else GENERATED_DIR

app = Flask(__name__)                                # create the web application
app.secret_key = "resume-handler-local"  # local-only app; just enables flash messages
# App-wide upload ceiling: reject any request body over this BEFORE it's read into memory
# (defends every upload route from oversized-file / memory-exhaustion DoS). A résumé .docx
# is well under this; env-overridable for unusual cases.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
# Tracker rows carry real types (dates, numbers, None for "not set"). Render None as ""
# in templates so a NULL never shows up as the literal text "None".
app.jinja_env.finalize = lambda v: "" if v is None else v


def _snapshot_current_resume(user_id: int, job_id: str, note: str):
    """Archive the CURRENT résumé for this job BEFORE a regenerate overwrites it, so a worse
    rewrite can be undone. Copies the live .html/.pdf/.json into a `versions/` sub-folder next
    to the résumé (named `<slug>_resume_v<N>.*`) and records the snapshot — its match score,
    critic report, and per-stage LLMs — in resume_versions. `note` says why ("before AI
    rewrite", "before restoring v2"). Best-effort: never raises, never blocks the regenerate;
    returns the new version number, or None when there's nothing to archive."""
    try:
        row = get_job(user_id, job_id)
        if not row or not row.get("HTML"):
            return None                                   # no résumé on disk yet → nothing to save
        html_src = (GENERATED_DIR / _generated_relpath(row["HTML"])).resolve()
        if not str(html_src).startswith(str(GENERATED_DIR)) or not html_src.is_file():
            return None                                   # stored path escaped the dir or is gone
        pdf_rel = _generated_relpath(row.get("PDF", "")) if row.get("PDF") else ""
        pdf_src = (GENERATED_DIR / pdf_rel).resolve() if pdf_rel else None
        json_src = html_src.with_suffix(".json")          # cached data lives beside the HTML
        versions_dir = html_src.parent / "versions"
        stem = html_src.stem                              # e.g. acme_corp_role_resume
        with db.connect() as conn:
            version = db.next_resume_version(conn, user_id, job_id)
            versions_dir.mkdir(parents=True, exist_ok=True)

            def _archive(src: Path, suffix: str) -> str:
                """Copy one file into versions/ as <stem>_v<N><suffix>; return its path
                relative to GENERATED_DIR (so the /files route can serve it), or ""."""
                if src and Path(src).is_file():
                    dst = versions_dir / f"{stem}_v{version}{suffix}"
                    shutil.copy2(src, dst)
                    return dst.resolve().relative_to(GENERATED_DIR).as_posix()
                return ""

            db.insert_resume_version(conn, user_id, job_id, version, {
                "html_path": _archive(html_src, ".html"),
                "pdf_path": _archive(pdf_src, ".pdf"),
                "json_path": _archive(json_src, ".json"),
                "match_score": row.get("Match Score"),
                "findings": row.get("Critic Findings") or {},
                "stage_llms": row.get("Stage LLMs") or {},
                "note": note,
            })
        return version
    except Exception as e:                                # a failed archive must never block a regen
        app.logger.warning("Could not archive résumé version for %s: %s", job_id, e)
        return None


def critic_findings_tooltip(cf: dict) -> str:
    """Build the Match-cell hover tooltip that explains WHY a résumé scored what it did, from a
    critic-findings dict ({kind: [finding,…]}, kinds fidelity/overstatement/style/revision/strength). Blank
    "" when there are no findings. Kept in ONE place so the server-rendered table
    (_job_row.html) and the async score updates after a generation/regeneration
    (/generate-status, /regenerate-resume, /rewrite-resume) show the SAME explanation — the
    async path used to leave the freshly-updated cell with no tooltip at all."""
    cf = cf or {}
    parts = []
    if cf.get("fidelity"):
        parts.append(f"⚠ Truthfulness ({len(cf['fidelity'])}): " + " • ".join(cf["fidelity"]))
    if cf.get("overstatement"):
        parts.append(f"ℹ️ Scope notes ({len(cf['overstatement'])}) — true, stated broadly; "
                     "does not lower the score: " + " • ".join(cf["overstatement"]))
    if cf.get("style"):
        parts.append(f"🤖 AI-style ({len(cf['style'])}): " + " • ".join(cf["style"]))
    if cf.get("revision"):
        parts.append("🔧 Suggested fixes: " + " • ".join(cf["revision"]))
    if cf.get("strength"):
        parts.append("✓ Strengths: " + " • ".join(cf["strength"]))
    return "\n\n".join(parts)


# Make the tooltip builder available to templates (the home-page table renders it inline).
app.jinja_env.globals["critic_findings_tooltip"] = critic_findings_tooltip


# How many job rows one page of the table renders. A long-running search accumulates
# thousands of tracked jobs, and rendering them all is what made the page crawl — the
# browser had to parse every row just to hide most of them behind a tab.
JOBS_PER_PAGE = 250


@app.route("/")                                      # the home page (http://127.0.0.1:5001/)
def index():
    """The jobs table: ONE tab, ONE page of rows, filtered and ordered by the database.

    Everything the browser used to do after the fact — hide aged-out postings, split rows
    into tabs, search by company, sort a column — is a query parameter now, so the page
    only ever ships the rows it displays."""
    uid = current_user_id()
    cfg = filter_config.load(uid)
    # The user's "Max posting age (days)" now governs the LIST as well as the scan: a job
    # that was fresh when found shouldn't sit in the table forever. Applied jobs and jobs
    # with a generated résumé are exempt (see db._AGE_VISIBLE).
    max_age_days = cfg.get("max_posting_age_days") or 0
    threshold = cfg.get("underqualified_at_or_below") or 5

    tab = request.args.get("tab", "pending")
    if tab not in db.JOB_TABS:                       # a stale bookmark must not 500
        tab = "pending"
    company = (request.args.get("q") or "").strip()
    sort = request.args.get("sort") or None
    if sort not in db.SORT_COLUMNS:                  # unknown column → the tab's natural order
        sort = None
    # A first click on a column header sorts ascending, as the old in-browser sort did;
    # ?dir=desc is the second click. With no ?sort at all this is ignored — the tab's own
    # natural order applies (see db._TAB_DEFAULT_SORT).
    descending = request.args.get("dir") == "desc"
    per_page = app.config.get("JOBS_PER_PAGE", JOBS_PER_PAGE)
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    counts = count_job_tabs(uid, max_age_days=max_age_days,
                            underqualified_at_or_below=threshold, company=company)
    # Past the last page (a stale link, or rows deleted since) → show the last real one.
    pages = max(1, -(-counts[tab] // per_page))      # ceiling division
    page = min(page, pages)
    applications = read_job_page(
        uid, tab=tab, max_age_days=max_age_days, underqualified_at_or_below=threshold,
        company=company, sort=sort, descending=descending,
        limit=per_page, offset=(page - 1) * per_page)
    for a in applications:
        # Download links must address files in dated sub-folders too, so store each résumé's
        # path RELATIVE to generated_resumes (works for both flat and dated-folder résumés).
        a["HTMLrel"] = _generated_relpath(a.get("HTML", ""))
        a["PDFrel"] = _generated_relpath(a.get("PDF", ""))
        # 1st-degree LinkedIn connections at this company — surfaced so you can ask
        # someone you already know for a referral.
        a["Connections"] = connections_for(a.get("Company", ""))

    summary = job_summary(uid, max_age_days=max_age_days, company=company)
    # What the tab strip, pager and sort arrows need to render themselves and build links.
    view = {"tab": tab, "counts": counts, "company": company, "sort": sort,
            "dir": "desc" if descending else "asc", "page": page, "pages": pages,
            "per_page": per_page, "total": counts[tab], "threshold": threshold,
            "max_age_days": max_age_days}
    # render_template fills templates/index.html with this data and returns the HTML page.
    # `sources` drives the "Find Jobs Automatically" checklist (all ticked by default).
    return render_template("index.html", applications=applications, summary=summary,
                           view=view, sources=source_names(),
                           local_models=available_local_models(),
                           developer_options=DEVELOPER_OPTIONS)


@app.route("/underqualified-threshold", methods=["POST"])
def underqualified_threshold_route():
    """Save the AI Fit score at or below which a job counts as "Under Qualified".

    It used to live in the browser's localStorage, which meant only the browser knew which
    tab a job belonged to — so the server had to send every row. Storing it per user is
    what lets the table be split in SQL."""
    uid = current_user_id()
    try:
        value = int(request.form.get("threshold", ""))
    except ValueError:
        value = 0
    if not 1 <= value <= 10:
        flash("Enter a whole number from 1 to 10.", "error")
        return redirect(url_for("index"))
    cfg = filter_config.load(uid)
    cfg["underqualified_at_or_below"] = value
    filter_config.save(uid, cfg)
    return redirect(url_for("index", tab=request.form.get("tab") or "pending"))


@app.route("/job-row/<path:job_id>")
def job_row_route(job_id):
    """ONE rendered jobs-table row (templates/_job_row.html — the same markup the home
    page uses). The scan's live feed fetches this for each newly-stored job and inserts
    it into the table, so finds appear in the list without a page reload."""
    a = get_job(current_user_id(), job_id)
    if a is None:
        return "", 404
    # The same per-row annotations index() applies to every row.
    a["HTMLrel"] = _generated_relpath(a.get("HTML", ""))
    a["PDFrel"] = _generated_relpath(a.get("PDF", ""))
    a["Connections"] = connections_for(a.get("Company", ""))
    return render_template("_job_row.html", app=a)


# Non-local backends that are always valid. Local backends are validated by prefix
# (tailoring.is_local_llm), so the selectable set is no longer a fixed enumeration.
_CLOUD_LLMS = {"claude", "chatgpt"}
# The four LLM-driven pipeline stages, mapped to the form field the UI submits for each.
_STAGE_FIELDS = {"generate": "llm_generate", "external": "llm_external",
                 "check": "llm_check", "regenerate": "llm_regenerate"}


def _normalize_llm(raw, allow_off: bool = False) -> str:
    """Map a UI-supplied backend choice to a valid LLM key. Unknown values fall back to
    'claude'. `allow_off` permits 'off' (the skippable external-review stage). A local
    choice is any 'local:<tag>' (or a legacy alias) that tailoring can actually dispatch."""
    v = (raw or "").strip()
    low = v.lower()
    if allow_off and low == "off":
        return "off"
    if low in _CLOUD_LLMS:
        return low
    if tailoring.is_local_llm(v):
        return v                       # keep original case — tags are case-sensitive
    return "claude"


def _chosen_llms() -> dict:
    """The per-stage LLM backends selected in the UI: a dict with keys generate/external/
    check/regenerate. The external stage may be 'off' (skip). request.form holds the
    browser's submitted form data; missing fields normalize to 'claude'."""
    return {stage: _normalize_llm(request.form.get(field), allow_off=(stage == "external"))
            for stage, field in _STAGE_FIELDS.items()}


def _parse_keyword_list(raw: str) -> list:
    """Split a textarea (newline- or comma-separated) into deduped, lowercased terms."""
    seen, out = set(), []                            # `seen` tracks dupes; `out` keeps the order
    for part in (raw or "").replace(",", "\n").splitlines():   # treat commas as line breaks
        item = part.strip().lower()
        if item and item not in seen:                # skip blanks and duplicates
            seen.add(item)
            out.append(item)
    return out


@app.route("/settings", methods=["GET", "POST"])     # this route handles both viewing and saving
def settings_route():
    """View/edit this user's filter keyword lists + thresholds (stored per user in the DB)."""
    uid = current_user_id()
    if request.method == "POST":                     # the user clicked "Save"
        cfg = filter_config.load(uid)
        # Walk every editable setting and read its new value from the submitted form.
        for key, _label, ftype, _help in filter_config.SCHEMA:
            raw = request.form.get(key, "")
            if ftype == "list":
                cfg[key] = _parse_keyword_list(raw)
            elif ftype == "text":                    # free-text paragraph, saved as-is
                cfg[key] = str(raw).strip()
            else:  # number
                try:
                    # Strip $ and commas, then convert to an int. If it's not a number, skip it.
                    cfg[key] = int(str(raw).replace(",", "").replace("$", "").strip())
                except ValueError:
                    pass
        filter_config.save(uid, cfg)                  # persist this user's filters to the DB
        # No process-wide reload needed anymore: each scan / fit run loads the user's
        # filters fresh from the DB, so the save takes effect on the next Retrieve/Fit.
        flash("Filters saved — applied to your next Fit scoring and Retrieve.", "success")
        return redirect(url_for("settings_route"))    # reload the page (avoids re-submitting on refresh)

    # GET request → just show the settings page, pre-filled with this user's config.
    return render_template("settings.html", cfg=filter_config.load(uid),
                           schema=filter_config.SCHEMA)


@app.route("/developer")
def developer_route():
    """Hidden Developer Options page — only reachable when DEVELOPER_OPTIONS is on (else 404,
    so it can't be opened by typing the URL even though the Tools button is also hidden)."""
    if not DEVELOPER_OPTIONS:
        abort(404)
    return render_template("developer.html")


@app.route("/developer/reset-scan", methods=["POST"])
def reset_scan_route():
    """Reset this user's "Retrieve Jobs" scan to a fresh state (delete unacted scan jobs +
    clear rejection memory). Gated behind DEVELOPER_OPTIONS. Requires the type-to-confirm
    'RESET' value (the button enforces it client-side; re-checked here so a stray POST no-ops)."""
    if not DEVELOPER_OPTIONS:
        abort(404)
    if request.form.get("confirm", "").strip() != "RESET":
        flash("Reset not confirmed — type RESET to proceed.", "error")
        return redirect(url_for("developer_route"))
    try:
        counts = reset_scan(current_user_id())
    except Exception:
        app.logger.exception("reset-scan failed")
        flash("Reset failed — no changes were made. See the server log.", "error")
        return redirect(url_for("developer_route"))
    flash(f"Removed {counts['jobs_deleted']} discovered job(s) and cleared "
          f"{counts['rejections_cleared']} rejection(s) — your next Retrieve Jobs starts fresh.",
          "success")
    return redirect(url_for("developer_route"))


@app.route("/developer/scan-audits")
def developer_scan_audits_route():
    """Gated scan-audit list — newest scans with derived drop-funnel tallies."""
    if not DEVELOPER_OPTIONS:
        abort(404)
    page = max(1, request.args.get("page", 1, type=int))
    per = 50
    with db.connect() as conn:
        scans = db.fetch_scans(conn, current_user_id(), limit=per, offset=(page - 1) * per)
    return render_template("scan_audits.html", scans=scans, page=page, per=per)


def _group_events_by_job(rows: list) -> list:
    """Fold a flat event page into scans → jobs → events for the card view.

    Rows arrive ordered (scan_id DESC, seq ASC). A single job's stage events are NOT
    contiguous, though: `seq` is one global counter shared by all ~6 concurrent board
    threads, and even within a board a job's list-stage and later deep-screen/store
    events are separated by other jobs' events — so grouping must key on job identity
    across the whole page, not just against the previous row. We key on
    (company, title, source, board_token) — the fields stable across a job's whole
    funnel (location can differ between the listing and the fetched posting, so it is
    NOT part of the key). Dict insertion order preserves first-appearance order, which
    (rows being scan_id DESC, seq ASC) is newest-scan-first and, within a scan,
    earliest-seq-first — i.e. jobs in the order they entered the funnel. Each job's
    summary is read off its highest-seq event (the funnel endpoint). A job whose events
    straddle the page boundary shows as two partial cards on adjacent pages — rare (one
    per ~100 events) and harmless."""
    scans: dict = {}                                  # scan_id → scan (insertion-ordered)
    for e in rows:
        scan = scans.get(e["scan_id"])
        if scan is None:
            scan = scans[e["scan_id"]] = {
                "scan_id": e["scan_id"], "started_at": e.get("started_at"), "jobs": {}}
        key = (e.get("company"), e.get("title"), e.get("source"), e.get("board_token"))
        job = scan["jobs"].get(key)
        if job is None:
            job = scan["jobs"][key] = {
                "company": e.get("company"), "title": e.get("title"),
                "source": e.get("source"), "board_token": e.get("board_token"),
                "location": e.get("location"),
                "external_id": e.get("external_id"), "url": e.get("url"),
                "events": []}
        job["events"].append(e)                       # rows are seq-asc → events stay ordered
    out: list = []
    for scan in scans.values():
        jobs = list(scan["jobs"].values())
        for job in jobs:
            last = job["events"][-1]                  # highest seq = funnel endpoint
            job["final_stage"] = last["stage"]
            job["final_outcome"] = last["outcome"]
            job["final_detail"] = last["detail"]
            job["stored"] = any(ev["stage"] == "store" and ev["outcome"] == "stored"
                                for ev in job["events"])
            job["manually_added"] = any(ev["stage"] == "manual_add" for ev in job["events"])
        scan["jobs"] = jobs
        out.append(scan)
    return out


EVENTS_PER_PAGE_CHOICES = (20, 40, 80, 200, 400, 1000, "all")  # jobs per page; "all" = no paging
EVENTS_PER_PAGE_DEFAULT = 40
_ALL_JOBS_LIMIT = 1_000_000                           # effectively unbounded for the "All" choice


def _resolve_per(raw):
    """Validate the ?per choice against the allowlist; junk → the default. Returns an int
    jobs-per-page or the literal 'all'."""
    if raw == "all":
        return "all"
    if raw is not None and raw.isdigit() and int(raw) in EVENTS_PER_PAGE_CHOICES:
        return int(raw)
    return EVENTS_PER_PAGE_DEFAULT


@app.route("/developer/scan-audits/events")
def developer_scan_events_route():
    """Gated scan-audit event view. ?scan_id drills into one scan; without it the
    filters search across all of this user's scans. Server-side filtered + paginated
    (?per jobs per page, or 'all' for no paging); the page's events are grouped into
    per-job cards (scans → jobs → stage trace)."""
    if not DEVELOPER_OPTIONS:
        abort(404)
    args = request.args
    scan_id = args.get("scan_id", type=int)
    filters = {k: (args.get(k) or None) for k in
               ("stage", "outcome", "reason_bucket", "source")}
    filters["company"] = args.get("company") or None
    page = max(1, args.get("page", 1, type=int))
    per = _resolve_per(args.get("per"))               # int choice or 'all'
    if per == "all":
        page, limit, offset = 1, _ALL_JOBS_LIMIT, 0   # one page, every job
    else:
        limit, offset = per, (page - 1) * per
    with db.connect() as conn:
        rows, total = db.fetch_scan_job_events(
            conn, current_user_id(), scan_id=scan_id, limit=limit,
            offset=offset, **filters)
    scans = _group_events_by_job(rows)
    return render_template("scan_audit_events.html", scans=scans, total=total,
                           scan_id=scan_id, page=page, per=per, args=args,
                           per_options=EVENTS_PER_PAGE_CHOICES,
                           has_prev=(per != "all" and page > 1),
                           has_next=(per != "all" and page * per < total),
                           buckets=scan_audit.BUCKET_ORDER)


@app.route("/developer/scan-audits/add-job", methods=["POST"])
def developer_scan_add_job_route():
    """Gated 'Track this job' rescue: re-fetch a scan-dropped posting and store it into this
    user's tracked jobs, recording a manual_add audit event on its scan. JSON in, JSON out."""
    if not DEVELOPER_OPTIONS:
        abort(404)
    f = request.form
    source = (f.get("source") or "").strip()
    board_token = (f.get("board_token") or "").strip()
    external_id = (f.get("external_id") or "").strip()
    company = (f.get("company") or "").strip() or None
    title = (f.get("title") or "").strip() or None
    posting_url = (f.get("url") or "").strip() or None
    scan_id = request.form.get("scan_id", type=int)
    fit_llm = _normalize_llm(f.get("llm_fit") or "chatgpt")
    if not (source and board_token and external_id and scan_id):
        return jsonify(ok=False, error="missing job identity"), 400
    try:
        result = add_job_manually(current_user_id(), scan_id, source, board_token,
                                  external_id, fit_llm=fit_llm,
                                  audit_company=company, audit_title=title,
                                  audit_url=posting_url)
    except Exception as e:                               # no connector / posting closed / etc.
        app.logger.exception("manual add-job failed")
        return jsonify(ok=False, error=str(e))
    msg = f"Tracked {result['company']} — {result['role']}"
    if result.get("ai_fit_score"):
        msg += f" (AI Fit {result['ai_fit_score']}/10)"
    return jsonify(ok=True, message=msg)


def _lines(raw: str) -> list:
    """A textarea into a list of trimmed, non-empty lines (case preserved — this is résumé
    content, not keywords)."""
    return [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]


def _parse_resume_form(form) -> dict:
    """Build a base-résumé profile dict (tailoring._SCHEMA shape) from the submitted form.
    Repeatable sections arrive as parallel lists (request.form.getlist), aligned by index;
    fully-blank rows are dropped so an empty 'Add' block never becomes a stored entry."""
    # Repeatable experience blocks: company/title/dates/location + a bullets textarea each.
    experience = []
    companies = form.getlist("exp_company")
    for i, company in enumerate(companies):
        entry = {
            "company": company.strip(),
            "title": _at(form.getlist("exp_title"), i),
            "dates": _at(form.getlist("exp_dates"), i),
            "location": _at(form.getlist("exp_location"), i),
            "bullets": _lines(_at(form.getlist("exp_bullets"), i)),
        }
        if entry["company"] or entry["title"] or entry["bullets"]:   # skip empty blocks
            experience.append(entry)

    # Repeatable education blocks.
    education = []
    for i, inst in enumerate(form.getlist("edu_institution")):
        entry = {"institution": inst.strip(), "degree": _at(form.getlist("edu_degree"), i),
                 "date": _at(form.getlist("edu_date"), i),
                 "detail": _at(form.getlist("edu_detail"), i)}
        if entry["institution"] or entry["degree"]:
            education.append(entry)

    # Repeatable skill rows (category + comma-separated items).
    skills = []
    for i, cat in enumerate(form.getlist("skill_category")):
        items = _at(form.getlist("skill_items"), i)
        if cat.strip() or items:
            skills.append({"category": cat.strip(), "items": items})

    return {
        "name": (form.get("name") or "").strip(),
        "contact": {"location": (form.get("location") or "").strip(),
                    "phone": (form.get("phone") or "").strip(),
                    "email": (form.get("email") or "").strip(),
                    "website": (form.get("website") or "").strip()},
        "summary": (form.get("summary") or "").strip(),
        "competencies": _lines(form.get("competencies")),
        "experience": experience,
        "education": education,
        "certifications": _lines(form.get("certifications")),
        "skills": skills,
    }


def _at(items: list, i: int) -> str:
    """The i-th value of a parallel form list, trimmed; '' when the list is short."""
    return items[i].strip() if i < len(items) else ""


# An empty résumé skeleton so the entry form renders the same whether or not a profile
# exists yet — 'enter' and 'edit' are the one screen.
_EMPTY_RESUME = {"name": "", "contact": {"location": "", "phone": "", "email": "", "website": ""},
                 "summary": "", "competencies": [], "experience": [], "education": [],
                 "certifications": [], "skills": []}


@app.route("/base-resume", methods=["GET", "POST"])
def base_resume_route():
    """Enter or edit this user's structured base résumé (stored per user in the DB — the
    resume_profiles + section tables). GET shows the form pre-filled with their current
    résumé (empty when they have none yet); POST saves it."""
    uid = current_user_id()
    if request.method == "POST":
        profile = _parse_resume_form(request.form)
        with db.connect() as conn:
            db.upsert_resume_profile(conn, uid, profile)
        flash("Base résumé saved.", "success")
        return redirect(url_for("base_resume_route"))

    with db.connect() as conn:
        profile = db.fetch_resume_profile(conn, uid)
    return render_template("base_resume.html", profile=profile or _EMPTY_RESUME,
                           has_profile=profile is not None)


# Résumé .docx uploads are capped well below the app-wide limit: they're plain documents,
# and a smaller cap further limits the text handed to the extraction model.
_RESUME_UPLOAD_MAX_BYTES = int(os.environ.get("RESUME_UPLOAD_MAX_BYTES", 2 * 1024 * 1024))


@app.route("/base-resume/import", methods=["POST"])
def base_resume_import_route():
    """Pre-fill this user's structured résumé by extracting fields from an uploaded .docx.
    SECURITY: the file is validated as a real Word doc first; its text is extracted and
    handed to the model as UNTRUSTED data (see src/resume_extract), and the model's output
    is strictly sanitized before it's stored. Import REPLACES the current profile, then
    lands the user on the editable screen to review — nothing downstream runs automatically."""
    uid = current_user_id()
    file = request.files.get("resume_file")
    if file is None or not file.filename:
        flash("Choose a .docx file to import.", "error")
        return redirect(url_for("base_resume_route"))
    if not file.filename.lower().endswith(".docx"):
        flash("Only .docx files are supported — export your résumé as Word format.", "error")
        return redirect(url_for("base_resume_route"))

    data = file.read()
    if len(data) > _RESUME_UPLOAD_MAX_BYTES:
        flash("That file is too large — please upload a résumé .docx under 2 MB.", "error")
        return redirect(url_for("base_resume_route"))
    try:
        text = resume_parser.extract_docx_text(BytesIO(data))     # in-memory: proves it's a real .docx
    except Exception:
        flash("That file couldn't be read as a Word document — is it a real .docx?", "error")
        return redirect(url_for("base_resume_route"))

    # Which backend reads the file: chosen on the home page ("LLM selection", saved in
    # localStorage) and carried here as a hidden field. Missing/unknown → claude.
    llm = _normalize_llm(request.form.get("import_llm"))
    try:
        profile = resume_extract.extract_profile_from_text(text, llm=llm)  # untrusted → LLM → sanitized
    except resume_extract.ExtractionError as exc:
        # exc carries only a safe, generic message (never raw model output).
        flash(f"Import didn't work: {exc} You can still enter your résumé by hand below.",
              "error")
        return redirect(url_for("base_resume_route"))

    with db.connect() as conn:
        db.upsert_resume_profile(conn, uid, profile)
    flash(f"Imported {len(profile['experience'])} role(s) and "
          f"{len(profile['skills'])} skill group(s) from your résumé — review and correct "
          "everything below before relying on it.", "success")
    return redirect(url_for("base_resume_route"))


@app.errorhandler(413)
def _too_large(_err):
    """Friendly response when an upload exceeds MAX_CONTENT_LENGTH (Flask aborts with 413
    before the route runs)."""
    flash("That upload is too large.", "error")
    return redirect(url_for("base_resume_route"))


def llm_label(v) -> str:
    """Human-readable label for a backend value, for the Stats page + 'Made by' summary."""
    if not v:
        return "—"
    if v == "off":
        return "off — skipped"
    if v in ("claude", "chatgpt"):
        return {"claude": "Claude", "chatgpt": "ChatGPT"}[v]
    tag = tailoring.local_tag(v)
    return f"Local — {tag}" if tag else v


def llm_css(v) -> str:
    """CSS badge class for a backend value (see stats.html)."""
    if v in ("claude", "chatgpt"):
        return f"llm-{v}"
    if tailoring.is_local_llm(v):
        return "llm-local"
    return "llm-none"                  # off, None, unknown


# The four generation stages and their column labels, in display order (matches
# tracker.STAGE_ORDER). Drives the per-stage breakdown shown for each résumé on Stats.
STAGE_LABELS = [("generate", "Generate"), ("external", "External review"),
                ("check", "Truthfulness check"), ("regenerate", "Regenerate")]


def _stage_badges(row) -> list:
    """Build the per-stage LLM display cells for one Stats row: a list of
    {stage, llm, display, css} in STAGE_LABELS order. "Stage LLMs" is already a
    {stage: backend} dict (pre-stage-tracking résumés were backfilled by migration v3,
    so there's no legacy fallback to apply here)."""
    decoded = row.get("Stage LLMs") or {}
    out = []
    for stage, label in STAGE_LABELS:
        llm = decoded.get(stage)                      # None if this stage's backend is unknown
        out.append({"stage": label, "llm": llm,
                    "display": llm_label(llm), "css": llm_css(llm)})
    return out


def _fmt_gen_time(raw) -> str:
    """Format a stored 'Gen Seconds' value as a short duration ("45s", "2m 03s"), or "—"
    when it's missing/blank (older resumes generated before timing was tracked)."""
    try:
        secs = int(float(raw))
    except (ValueError, TypeError):
        return "—"
    if secs <= 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60:02d}s"


# Docker creates /.dockerenv inside every container — the standard "am I in a
# container?" probe. Surfaced via /health so you can always tell WHICH copy of the
# app answered on port 5001 (the container, or a local `python app.py`).
IN_DOCKER = Path("/.dockerenv").exists()


@app.route("/health")
def health_route():
    """Liveness + database reachability, for docker-compose's healthcheck (and you).
    `runtime` says which instance answered: "docker" or "local"."""
    runtime = "docker" if IN_DOCKER else "local"
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
        return {"ok": True, "runtime": runtime}
    except Exception as e:
        return {"ok": False, "runtime": runtime, "error": str(e)}, 503


# /api/chat is a GENERATION call (unlike the fast metadata pings), so it gets its own
# timeout — a bounded default so a wedged server fails fast instead of hanging. Both an
# env override and a per-request field (the chat card's "timeout" input) can change it.
HEALTH_CHAT_TIMEOUT = int(os.environ.get("HEALTH_CHAT_TIMEOUT", "600"))


@app.route("/health-checks")
def health_checks_page():
    """Diagnostic page: probe the local Ollama server. Fixed metadata checks (tested via
    Test-all) plus an interactive /api/chat card driven by its own Send button."""
    return render_template("health_checks.html",
                           checks=health_checks.list_checks(),
                           chat_params=health_checks.CHAT_PARAMS,
                           chat_models=health_checks.chat_models(),
                           chat_timeout_default=HEALTH_CHAT_TIMEOUT)


@app.route("/health-checks/run/<check_id>")
def health_checks_run(check_id):
    """Run ONE health check by id and return its result as JSON. The browser fires these
    (one per check, in parallel) to drive Test-all / Re-test-failed / per-item Test."""
    try:
        return jsonify(health_checks.run_check(check_id))
    except KeyError:
        return jsonify({"error": "unknown check", "id": check_id}), 404


@app.route("/health-checks/chat", methods=["POST"])
def health_checks_chat():
    """Forward a user-composed request to the local /api/chat endpoint, synchronously.
    Body: {"body": {...ollama request...}, "timeout": <seconds>}. A missing/non-positive/
    non-numeric timeout falls back to HEALTH_CHAT_TIMEOUT. Never raises out to the browser."""
    data = request.get_json(force=True, silent=True) or {}
    body = data.get("body") or {}
    try:
        timeout = int(data.get("timeout"))
        if timeout <= 0:
            timeout = HEALTH_CHAT_TIMEOUT
    except (TypeError, ValueError):
        timeout = HEALTH_CHAT_TIMEOUT
    return jsonify(health_checks.run_chat(body, timeout))


# In-process cache of the discovered local-model list. This is a cache of live external
# state (what `ollama list` shows), NOT domain data — so it lives in memory, never the DB.
# `models` is None until the first fetch; after a failed fetch it's [] (so page loads don't
# re-block) with a human message in `error`. The refresh button (?refresh=1) forces a re-fetch.
_local_models_cache = {"models": None, "error": None}
_local_models_lock = threading.Lock()


def _reset_local_models_cache():
    """Test hook: clear the module-level cache so each test starts cold."""
    with _local_models_lock:
        _local_models_cache["models"] = None
        _local_models_cache["error"] = None


def _refresh_local_models_locked():
    """Re-fetch the model list into the cache. MUST hold _local_models_lock."""
    try:
        _local_models_cache["models"] = tailoring.list_local_models()
        _local_models_cache["error"] = None
    except Exception as e:                     # transport error, bad URL, unexpected shape
        _local_models_cache["models"] = []     # mark fetched-but-empty → don't re-block every load
        _local_models_cache["error"] = f"Could not reach Ollama at {tailoring.local_base_url()}: {e}"


def available_local_models() -> list:
    """The cached list of local model tags, for server-rendering the dropdowns. Fetches
    once lazily on a cold cache (short timeout, degrades to [])."""
    with _local_models_lock:
        if _local_models_cache["models"] is None:
            _refresh_local_models_locked()
        return list(_local_models_cache["models"] or [])


@app.route("/local-models")
def local_models_route():
    """JSON list of the user's local Ollama models for the refresh button. `?refresh=1`
    busts the in-process cache and re-queries the server."""
    if request.args.get("refresh"):
        with _local_models_lock:
            _refresh_local_models_locked()
    models = available_local_models()
    with _local_models_lock:
        error = _local_models_cache["error"]
    return jsonify({"models": models, "error": error})


@app.route("/stats")
def stats_route():
    """Per-resume generation stats: how long each resume took to make and which LLM made
    it. Only rows that actually have a resume are shown (a job with no resume has nothing
    to report). Data comes from the same jobs/resumes store as the main table. The
    "final" LLM is the regenerate-stage backend — the one whose output is on disk."""
    rows = [a for a in read_applications(current_user_id()) if a.get("HTML")]   # only generated resumes
    for a in rows:
        final_llm = a["Stage LLMs"].get("regenerate")
        a["GenTimeDisplay"] = _fmt_gen_time(a.get("Gen Seconds"))
        a["LLMDisplay"] = llm_label(final_llm)
        a["StageBadges"] = _stage_badges(a)           # per-stage LLM breakdown for the table
    # How many resumes were made on each LLM (for the summary line at the top).
    claude_n = sum(1 for a in rows if a["Stage LLMs"].get("regenerate") == "claude")
    local_n = sum(1 for a in rows if tailoring.is_local_llm(a["Stage LLMs"].get("regenerate")))
    chatgpt_n = sum(1 for a in rows if a["Stage LLMs"].get("regenerate") == "chatgpt")
    summary = {"total": len(rows), "claude": claude_n, "local": local_n, "chatgpt": chatgpt_n}
    return render_template("stats.html", rows=rows, summary=summary)


# Background retrieve job state. A full board scan can take many minutes; running it
# inside the HTTP request meant the browser's fetch() connection dropped ("Failed to
# fetch") before it finished. Instead we run the scan on a background thread and let the
# UI poll /retrieve-status, so the work no longer depends on one long-lived connection.
_retrieve_lock = threading.Lock()                    # protects the shared state dict below
_retrieve_state = {"running": False, "done": False, "ok": True, "message": "", "started": 0.0,
                   "stage": "", "done_count": 0, "total": 0, "found": 0,
                   "recent": []}                     # newest-first feed of stored+scored jobs


def _run_retrieve(sources, user_id, fit_llm, use_audit=True):
    """The actual scan, run on a background thread FOR one user. Updates _retrieve_state as
    it goes so the web page can poll for progress. `sources` is the list of board sources
    the user ticked; `user_id` is captured from the request before the thread starts (a
    background thread has no request context to resolve the current user itself). `use_audit`
    is False when the user chose to continue past an unwritable logs/ dir (see the route)."""
    # A small callback the scanner calls to report progress; it updates the shared state.
    def tick(stage, done, total, found):
        with _retrieve_lock:
            _retrieve_state.update(stage=stage, done_count=done, total=total, found=found)

    # Called by the scan the moment a job is stored (+scored): feed it to the UI so the
    # scan card shows finds live, newest first (the page keeps only the last handful).
    # job_id lets the page fetch /job-row/<id> and insert the row into the table live.
    def on_found(job):
        entry = {"company": job.get("company", ""), "role": job.get("title", ""),
                 "score": job.get("ai_fit_score"), "new": job.get("stored") == "added",
                 "job_id": make_job_id(job)}
        with _retrieve_lock:
            _retrieve_state["recent"] = [entry] + _retrieve_state["recent"][:9]
    try:
        # Warm/refresh the base-résumé embedding FIRST (also clears stale AI verdicts if
        # the résumé changed). Failures here must not block the scan — jobs would just
        # land without a Match %.
        if embeddings.available() and base_resume.has_base_resume(user_id):
            try:
                embeddings.base_resume_embedding(user_id)
            except (embeddings.EmbeddingsError, FileNotFoundError) as e:
                print(f"[retrieve] couldn't refresh the base-résumé embedding: {e}")

        # The scan stores + embeds + AI-scores each survivor AS IT PASSES (see
        # job_search._store_and_score), so the jobs table fills up while this runs.
        # on_progress=print puts every stored job, drop reason, and per-job failure in
        # the app log — nothing fails invisibly.
        matches = find_matches(user_id, sources=sources, on_tick=tick, on_progress=print,
                               on_found=on_found, fit_llm=fit_llm, use_audit=use_audit)
        added = sum(1 for m in matches if m.get("stored") == "added")
        scored = sum(1 for m in matches if m.get("ai_fit_score") is not None)
        ok = True
        message = (f"Added {added} new job(s) "
                   f"({len(matches)} matched this run"
                   + (f"; AI-scored {scored}" if scored else "") + ").")
    except FileNotFoundError:
        # Special case: the board directory hasn't been built yet — give a helpful message.
        ok, message = False, ("No board directory yet — build it first "
                              "(python -m src.ats.greenhouse_discovery).")
    except Exception as e:
        ok, message = False, f"Job retrieval failed: {e}"
    with _retrieve_lock:
        _retrieve_state.update(running=False, done=True, ok=ok, message=message)


@app.route("/retrieve-jobs", methods=["POST"])
def retrieve_jobs_route():
    """Kick off the FREE Phase 2 funnel (no LLM, no resumes) on a background thread and
    return immediately. The UI polls /retrieve-status for completion."""
    # The user ticks which board sources to scan (all are checked by default). Keep only
    # names we actually recognize; an empty selection falls back to "all" so a scan is never
    # a no-op just because the checklist round-tripped oddly.
    valid = source_names()
    selected = [s for s in request.form.getlist("sources") if s in valid]
    sources = selected or list(valid)
    if not selected and request.form.getlist("sources"):
        # They sent sources but none matched — bad input rather than an unchecked default.
        return jsonify({"ok": False, "message": "Pick at least one job board to scan."})

    # Which backend AI-scores each stored job (the "Job scan - AI Fit verdict" picker).
    fit_llm = _normalize_llm(request.form.get("llm_fit") or "chatgpt")

    # Preflight: the scan records a per-scan drop-funnel audit in the DB. If the DB is
    # unreachable, don't crash mid-scan — WARN and let the user choose to run without the
    # audit. The browser re-POSTs with allow_no_audit=1.
    allow_no_audit = request.form.get("allow_no_audit") == "1"
    use_audit = True
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
    except Exception:
        if not allow_no_audit:
            return jsonify({"ok": False, "needs_confirm": True,
                            "message": "Can't reach the database to record the scan audit. "
                                       "Run the scan anyway, without the audit?"})
        use_audit = False                             # user confirmed: proceed, no audit

    with _retrieve_lock:
        if _retrieve_state["running"]:                # don't start a second scan on top of one
            return jsonify({"ok": True, "running": True,
                            "message": "A scan is already running — hang tight."})
        # Mark the scan as started and reset the progress fields.
        _retrieve_state.update(running=True, done=False, ok=True,
                               message="Scanning boards…", started=time.time(),
                               stage="scan", done_count=0, total=0, found=0, recent=[])

    # Resolve the current user HERE (in the request), then hand it to the thread — a
    # background thread has no request context to resolve it later.
    threading.Thread(target=_run_retrieve, args=(sources, current_user_id(), fit_llm, use_audit),
                     daemon=True).start()
    return jsonify({"ok": True, "running": True,
                    "message": "Scanning boards… this can take a few minutes."})


@app.route("/fit-verdicts", methods=["POST"])
def fit_verdicts_route():
    """The "Score fits" button: have ChatGPT score the most-similar STORED jobs that
    don't have an AI fit verdict yet (backfill for jobs from before this feature, or
    scans that ran without a key). Runs as a background generation task — the page
    polls /generate-status with the returned task id."""
    if not embeddings.available():
        return jsonify({"ok": False, "message": "No OpenAI key configured — add "
                        "CHATGPT_API_KEY to .env to enable AI fit scoring."})

    uid = current_user_id()                          # captured for the background task
    llm = _normalize_llm(request.form.get("llm_fit") or "chatgpt")

    def work(on_progress=None):
        lines = []
        def log(msg):                                # show in the app log AND keep for the UI
            print(msg)
            lines.append(msg)
            if on_progress:                          # surface the latest line to the poller
                on_progress(msg)
        scored = fit_verdict.run_stored_verdicts(uid, on_progress=log, llm=llm)
        skipped = sum(1 for line in lines
                      if line.lstrip().startswith(("(skipped", "(couldn't")))
        detail = lines[-1] if (lines and not scored) else (
            f"AI-scored {scored} job(s)."
            + (f" {skipped} couldn't be scored — details in the app log." if skipped else ""))
        return {"ok": True, "message": detail, "category": "success"}

    task_id = _start_generation(work)
    return jsonify({"ok": True, "task_id": task_id,
                    "message": "Scoring stored jobs against your résumé…"})


@app.route("/retrieve-status")
def retrieve_status_route():
    """Current state of the background board scan (polled by the UI every second or so)."""
    with _retrieve_lock:
        state = dict(_retrieve_state)                 # copy it so we can add a field safely
    if state.get("started"):
        state["elapsed"] = int(time.time() - state["started"])   # seconds since the scan began
    return jsonify(state)


# ── Background résumé generation (async, like the job scan) ──────────────────────
# Full generation runs the multi-minute pipeline (tailor → external review → critic loop).
# Holding the browser's request open that long is fragile — the fetch() drops ("Failed to
# fetch") even though the server finishes the work and saves the files. So each generation
# runs on a background thread; the page gets a task id and polls /generate-status until it's
# done. Results are kept briefly in memory and handed back on the first "done" poll.
_gen_lock = threading.Lock()                          # guards the task registry below
_gen_tasks: dict = {}                                 # task_id -> {"done": bool, "result": dict|None, "started": float}

# Watchdog: the longest a single generation may run before /generate-status abandons it.
# A background thread can stop making progress WITHOUT ever marking its task done — e.g. an
# LLM HTTP call that wedges because the local model host slept mid-request (this happened with
# a slow all-local Ōura run). Without this cap the browser polls {done:false} forever and the
# spinner never stops. A full all-local generation is several LLM calls (tailor + external +
# critic + regenerate loop), each capped by LOCAL_LLM_TIMEOUT (default 1 hour) on slow home
# hardware — so the whole run can legitimately take hours. Default 6 hours leaves headroom;
# raise GEN_TASK_TIMEOUT_SEC (and it should exceed roughly LOCAL_LLM_TIMEOUT × the number of
# stage calls) for slower setups, or lower it if you only run fast cloud LLMs.
GEN_TASK_TIMEOUT_SEC = int(os.environ.get("GEN_TASK_TIMEOUT_SEC", str(6 * 60 * 60)))


def _start_generation(work) -> str:
    """Run `work(on_progress)` on a background thread and return a task id the page can poll.
    `work` is a one-arg callable: it receives an `on_progress(msg)` function it should forward
    to the pipeline so each stage's status ("Pass 2 — refining (attempt 2/3)…") is surfaced to
    the polling browser. It must capture everything else it needs (llm, form values) UP FRONT,
    since it runs outside the Flask request context, and return a JSON-ready result dict."""
    task_id = uuid.uuid4().hex[:12]
    with _gen_lock:
        _gen_tasks[task_id] = {"done": False, "result": None, "started": time.time(),
                               "progress": ""}

    # Records the latest pipeline status on the task slot so /generate-status can hand it to
    # the browser. Guarded by the lock because the poll thread reads the same slot.
    def on_progress(msg):
        with _gen_lock:
            slot = _gen_tasks.get(task_id)
            if slot is not None:
                slot["progress"] = str(msg).strip()

    def runner():
        try:
            result = work(on_progress)
        except Exception as e:                        # a blow-up must still resolve the task
            # Log the FULL traceback to the app log too — the browser only ever sees the
            # one-shot message below, and the task slot is dropped after one poll, so without
            # this a background failure leaves no diagnosable trace anywhere.
            app.logger.exception("Background generation failed (task %s)", task_id)
            result = {"ok": False, "message": f"Generation failed: {e}", "category": "error"}
        with _gen_lock:
            slot = _gen_tasks.get(task_id)
            if slot is not None:
                slot["done"], slot["result"] = True, result

    threading.Thread(target=runner, daemon=True).start()
    return task_id


@app.route("/generate-status")
def generate_status_route():
    """Poll a background generation started by /generate or /generate-for-job. Returns
    {"done": false} while it's still running, or {"done": true, "result": {...}} once
    finished — where `result` is the SAME payload these routes used to return synchronously.
    The finished task is dropped once handed back (the page stops polling on 'done')."""
    task_id = (request.args.get("task_id") or "").strip()
    with _gen_lock:
        task = _gen_tasks.get(task_id)
        if task is None:
            # Unknown/expired id (e.g. the app restarted mid-run). The work may still have
            # completed and saved its files, so point the user at a refresh rather than "error".
            return jsonify({"done": True, "result": {
                "ok": False, "category": "error",
                "message": "That generation is no longer tracked (the app may have restarted). "
                           "Refresh the page — it may have finished."}})
        if not task["done"]:
            # Watchdog: if the task has run past the cap it has almost certainly wedged (the
            # thread can't be force-killed, but if it ever finishes runner() finds no slot and
            # no-ops). Abandon it so the browser gets a definitive answer instead of spinning.
            if time.time() - task.get("started", 0) > GEN_TASK_TIMEOUT_SEC:
                _gen_tasks.pop(task_id, None)
                return jsonify({"done": True, "result": {
                    "ok": False, "category": "error",
                    "message": (f"Generation exceeded the {GEN_TASK_TIMEOUT_SEC // 60}-minute "
                                "limit and was abandoned — the background run likely stalled "
                                "(e.g. the local LLM host became unreachable). Refresh the page: "
                                "if a resume was produced it may already be saved.")}})
            # Still running — hand back the latest pipeline stage so the spinner can show it.
            return jsonify({"done": False, "progress": task.get("progress", "")})
        _gen_tasks.pop(task_id, None)                 # hand the result back once, then forget it
        return jsonify({"done": True, "result": task["result"]})


@app.route("/add-job", methods=["POST"])
def add_job_route():
    """Add a manually-found job to the tracked list — WITHOUT generating a résumé. Backs the
    Manual tab's "Add job" button so the user can queue many jobs quickly and generate
    résumés later (the per-row 'Generate Resume' button, or a scheduled batch). Returns JSON
    for the AJAX form; a plain post falls back to a flash + redirect."""
    ajax = bool(request.form.get("ajax"))
    # Read each form field, defaulting to "" and trimming whitespace.
    company = (request.form.get("company") or "").strip()
    role = (request.form.get("role") or "").strip()
    job_url = (request.form.get("job_url") or "").strip()
    job_text = (request.form.get("job_text") or "").strip()

    # A small helper to report a failure either as JSON (AJAX) or a flash+redirect (normal).
    def fail(msg):
        if ajax:
            return jsonify({"ok": False, "message": msg, "category": "error"})
        flash(msg, "error")
        return redirect(url_for("index"))

    if not company or not role:
        return fail("Company and Role are required.")

    source = job_url or job_text                      # use whichever the user provided
    if not source:
        return fail("Provide a job posting — either a URL or pasted text.")

    try:
        result = add_manual_job(current_user_id(), company, role, source)
    except Exception as e:
        return fail(f"Could not add the job: {e}")

    if ajax:
        return jsonify({"ok": True, "message": result["message"], "category": "success"})
    flash(result["message"], "success")               # show a banner on the page
    return redirect(url_for("index"))


def _generation_message(company, role, meta):
    """Build the (message, category) describing a generation result. 'category' is "success"
    or "error", which the page uses to color the banner."""
    # f-strings with a small conditional handle singular/plural ("1 page" vs "2 pages").
    pg = f"{meta['pages']} page{'s' if meta['pages'] != 1 else ''}"
    att = f"{meta['attempts']} attempt{'s' if meta['attempts'] != 1 else ''}"
    gpt = " · ChatGPT-reviewed" if meta.get("external_review_used") else ""
    if meta["accepted"]:
        return (f"Tailored resume generated for {company} — {role}: "
                f"{pg}, quality score {meta['score']}/100 ({att}{gpt}).", "success")
    # The truthfulness critic never returned a usable verdict (e.g. its stage LLM couldn't produce
    # a real critique). The résumé rendered fine but was NOT audited, so its 0/100 score and empty
    # flag lists are meaningless — say so plainly instead of implying a clean-but-low result.
    if meta.get("unaudited"):
        return (f"Generated for {company} — {role} ({pg}), but it could NOT be truthfulness-checked "
                f"— the check-stage model returned no usable review. Treat this résumé as UNVERIFIED: "
                f"read it carefully for invented skills/tools before sending, or regenerate with a "
                f"different check LLM.", "error")
    # Not accepted → explain why (length, low score, fidelity/AI-style flags) as a warning.
    reasons = []
    if not meta["fit"]:
        reasons.append(f"{pg} (over target)")
    if meta.get("too_short"):
        reasons.append(f"only {pg} (too thin — content was dropped)")
    reasons.append(f"score {meta['score']}/100")
    if meta["fidelity_issues"]:
        reasons.append(f"{len(meta['fidelity_issues'])} truthfulness flag(s)")
    if meta.get("style_issues"):
        reasons.append(f"{len(meta['style_issues'])} AI-style flag(s)")
    return (f"Generated for {company} — {role}, best of {att}: "
            f"{', '.join(reasons)}. Hover the Match score for the critic's findings; "
            f"review before sending.",
            "error")


def _primary_llm(llms) -> str:
    """The single backend to record in audit logs for a (possibly per-stage) selection: the
    Stage-4 regenerate backend that authored the final résumé. Accepts a per-stage dict or a
    plain backend string (older scheduled runs stored just a string)."""
    if isinstance(llms, dict):
        return llms.get("regenerate") or llms.get("generate") or "claude"
    return llms


def _generate_for_job(user_id: int, job_id: str, llms, out_dir: Path = None,
                      source: str = "manual", force: bool = False, on_progress=None) -> dict:
    """Generate a resume for ONE of this user's tracked jobs, recording the outcome to the
    audit log. Thin wrapper around `_generate_for_job_core` that times the attempt and appends
    one row to the generation log (so manual AND scheduled runs leave a durable trail of what
    ran and why it failed). `llms` is a per-stage backend dict (or a plain backend string).
    `source` is "manual" (per-row button), "scheduled" (batch run), or "rewrite" (the
    'Regenerate with LLM' full rewrite). `force` re-runs even if the job already has a resume
    (used by the rewrite button). Never raises."""
    started = time.monotonic()
    result = _generate_for_job_core(user_id, job_id, llms, out_dir, force=force,
                                    on_progress=on_progress)
    run_log.log(user_id, source=source, job_id=job_id, llm=_primary_llm(llms), result=result,
                duration_sec=time.monotonic() - started)
    return result


def _generate_for_job_core(user_id: int, job_id: str, llms, out_dir: Path = None,
                           force: bool = False, on_progress=None) -> dict:
    """Generate a resume for ONE tracked job. The shared core used by the per-row 'Generate
    Resume' button, the batch scheduler, AND the 'Regenerate with LLM' full rewrite. `out_dir`
    chooses where the files are written (the scheduler passes today's dated sub-folder; the
    manual button leaves it None → the default generated_resumes folder). `force=True` (the
    rewrite button) re-runs even when the job already has a resume, and — because we're
    deliberately rewriting an existing résumé — skips the "posting still live?" check so a
    closed posting can't delete the row/files out from under the rewrite. Returns a JSON-ready
    dict: {ok, message, category, job_id, company, role, match_score, html, pdf}; on a job that
    already has a resume (and not force) it returns {ok: False, skipped: True} so the scheduler
    can mark it 'skipped' rather than 'error'. Never raises — failures come back as ok=False."""
    # Guard shared by the per-row button AND scheduled batch runs: without a base résumé
    # (structured profile OR .docx) there is nothing to tailor, so refuse politely.
    if not base_resume.has_base_resume(user_id):
        return {"ok": False, "category": "error",
                "message": "No base résumé found — add one on the Base Résumé screen "
                           "(enter it or import a .docx)."}

    row = get_job(user_id, job_id)                    # one indexed lookup, not a full-table scan
    if not row:
        return {"ok": False, "message": "Could not find that job.", "category": "error"}
    if row.get("HTML") and not force:                 # already has a résumé → don't redo it
        return {"ok": False, "skipped": True,
                "message": "That job already has a resume.", "category": "error"}

    company, role = row["Company"], row["Role"]

    # A forced re-run means the user is rewriting an EXISTING résumé ("With LLM"). Archive the
    # current one first so this rewrite is undoable if it comes out worse (see /resume-versions).
    if force and row.get("HTML"):
        _snapshot_current_resume(user_id, job_id, "before AI rewrite")

    # Verify the posting is STILL live before spending tokens on it. A job discovered days
    # ago (Phase 2) may have closed by the time a scheduled batch reaches it; generating a
    # résumé for a dead posting wastes LLM credits. job_is_active only reports inactive on a
    # positive 'gone' signal (a board 404), so a transient network error won't skip a good job.
    # Skipped for a forced rewrite: the user is intentionally rewriting an existing résumé, so
    # we must NOT delete the row/files even if the posting has since closed.
    if not force:
        active, reason = job_is_active(job_id, row.get("Job URL") or "")
        if not active:
            # The posting is gone and we never generated a résumé for it (the HTML check above
            # already returned for jobs that have one), so there's nothing to keep — drop the row
            # from the tracked list. No generated files exist on this path, so none are deleted.
            # The CSV audit log still records this as status=closed with the reason.
            delete_job(user_id, job_id, reason=f"posting no longer active: {reason}")
            return {"ok": False, "closed": True, "category": "error",
                    "company": company, "role": role,
                    "message": f"Removed {company} — {role} (posting no longer active): {reason}."}

    try:
        # Recover the job's description + canonical URL from whatever we stored. This works for
        # ANY board (Greenhouse, Lever, …) via fetch_job_description's connector dispatch, and
        # falls back to re-parsing the stored URL for non-board rows.
        job_text, source = fetch_job_description(job_id, row.get("Job URL") or "")
        if not source:
            source = row.get("Job URL") or ""
        # A full rewrite needs the job description. Board/URL jobs can be re-fetched, but a
        # manual pasted-text job has nothing to recover — refuse clearly rather than tailoring
        # against garbage. (Re-render from JSON still works for those.)
        if force and not (job_text or "").strip() and not str(source).startswith(("http://", "https://")):
            return {"ok": False, "category": "error", "company": company, "role": role,
                    "message": "Couldn't recover this job's description to rewrite it (it looks "
                               "like a pasted-text job). Use “Regenerate from JSON” instead."}
        # `llms` may be a per-stage dict (from the UI) or a plain backend string (older
        # scheduled runs); pass whichever the pipeline expects.
        stage_llms = llms if isinstance(llms, dict) else None
        base_llm = "claude" if isinstance(llms, dict) else llms
        # allow_resume: if a previous attempt for this job failed partway, the pipeline saved its
        # finished stages (evidence map, Pass-1 draft, external review, latest draft) — reuse them
        # and continue instead of re-paying for every earlier Claude call. Cleared on success, so a
        # later fresh regenerate still starts clean.
        html_path, pdf_path, meta = generate(user_id, company, role, source=source,
                                             job_text=job_text or None, job_id=job_id,
                                             llm=base_llm, stage_llms=stage_llms,
                                             out_dir=out_dir, allow_resume=True,
                                             on_progress=on_progress)
    except Exception as e:
        # The finished stages are checkpointed on disk, so re-running continues from here rather
        # than starting over — tell the user so they know a retry is cheap.
        return {"ok": False, "category": "error", "company": company, "role": role,
                "message": f"Generation failed: {e} — click the same button again to continue "
                           "from where it stopped (finished steps are saved)."}

    message, category = _generation_message(company, role, meta)
    return {
        "ok": True,
        "job_id": job_id,
        "company": company,
        "role": role,
        "message": message,
        "category": category,
        "match_score": round(meta["score"] / 10, 1),
        # The critic's findings tooltip, so the async cell update explains the score too.
        "match_tip": critic_findings_tooltip(meta.get("critic_findings")),
        "html": _generated_relpath(str(html_path)),   # path under generated_resumes (for the link)
        "pdf": _generated_relpath(str(pdf_path)),
    }


@app.route("/generate-for-job", methods=["POST"])
def generate_for_job_route():
    """Generate a resume for a single tracked job (the per-row 'Generate Resume' button).
    Runs on a background thread and returns a task id the page polls via /generate-status,
    so the multi-minute pipeline can't drop the browser request. Spends Claude/ChatGPT
    credits — triggered manually, one job at a time."""
    job_id = (request.form.get("job_id") or "").strip()
    llms = _chosen_llms()                             # capture before the thread (no request ctx there)
    uid = current_user_id()                           # ditto — resolve who this is for now
    task_id = _start_generation(lambda p: _generate_for_job(uid, job_id, llms, on_progress=p))
    return jsonify({"ok": True, "task_id": task_id})


@app.route("/rewrite-resume", methods=["POST"])
def rewrite_resume_route():
    """Full AI rewrite of an EXISTING résumé for a job ('Regenerate with LLM'). Re-runs the
    whole tailoring pipeline using the per-stage LLMs chosen at the top of the page — unlike
    'Regenerate from JSON', which only re-renders the cached JSON + re-scores. Runs on a
    background thread the page polls via /generate-status. Spends LLM credits."""
    job_id = (request.form.get("job_id") or "").strip()
    company = (request.form.get("company") or "").strip()
    role = (request.form.get("role") or "").strip()
    if not job_id:
        return jsonify({"ok": False, "message": "Missing job id for rewrite.", "category": "error"})
    llms = _chosen_llms()                             # capture before the thread (no request ctx there)
    uid = current_user_id()                           # ditto
    # Write the rewrite back to the existing résumé's folder (its dated sub-folder for cron
    # résumés, else the default) so it replaces the current files in place.
    out_dir = _resume_dir_for(company, role) if company and role else None
    task_id = _start_generation(
        lambda p: _generate_for_job(uid, job_id, llms, out_dir=out_dir, source="rewrite",
                                    force=True, on_progress=p))
    return jsonify({"ok": True, "task_id": task_id})


def _version_llm_label(stage_llms: dict) -> str:
    """A short label for the LLMs behind a saved version — one backend name when every stage
    used the same one, a per-stage list otherwise, or "" when unknown (older/From-JSON snapshots)."""
    used = {v for v in (stage_llms or {}).values() if v and v != "off"}
    if not used:
        return ""
    if len(used) == 1:
        return llm_label(next(iter(used)))
    return ", ".join(f"{s}: {llm_label(v)}" for s, v in stage_llms.items() if v)


@app.route("/resume-versions")
def resume_versions_route():
    """List the saved previous versions of one résumé (newest first) for the History modal.
    Read-only JSON; the files themselves are served by the /files route."""
    uid = current_user_id()
    job_id = (request.args.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"ok": False, "versions": []})
    with db.connect() as conn:
        versions = db.list_resume_versions(conn, uid, job_id)
    out = [{
        "version": v["version"],
        "created_at": v["created_at"].strftime("%b %-d, %Y %-I:%M %p") if v["created_at"] else "",
        "match_score": v["match_score"],
        "note": v["note"],
        "llms": _version_llm_label(v["stage_llms"]),
        "html": v["html_path"],
        "pdf": v["pdf_path"],
        "can_restore": bool(v["json_path"]),              # restore re-renders from the saved data
        "is_current": False,
    } for v in versions]
    # Prepend the CURRENT (live) résumé so the panel always shows which LLM produced it — the
    # resume_versions table only holds PREVIOUS (archived) snapshots, so without this the
    # first/original résumé's LLM would never appear here (it's only archived once you
    # regenerate over it). No Restore button: it's already the live one.
    row = get_job(uid, job_id)
    if row and row.get("HTML"):
        out.insert(0, {
            "version": None,
            "created_at": "",
            "match_score": row.get("Match Score"),
            "note": "Current version",
            "llms": _version_llm_label(row.get("Stage LLMs") or {}),
            "html": _generated_relpath(row.get("HTML", "")),
            "pdf": _generated_relpath(row.get("PDF", "")) if row.get("PDF") else "",
            "can_restore": False,
            "is_current": True,
        })
    return jsonify({"ok": True, "versions": out})


@app.route("/save-resume-json", methods=["POST"])
def save_resume_json_route():
    """Save hand-edited résumé data (from the Compare screen's edit mode). Archives the current
    résumé first (so the edit is undoable via History), then re-renders the HTML + PDF from the
    edited JSON. NO LLM call and NO re-score — the Match score is deliberately left as-is
    (editing changed the content, so click "↻ From JSON" afterward to recompute it). Body is
    JSON: {company, role, job_id, data}. Renders synchronously (no rewrite, just Playwright)
    like restore. Returns JSON."""
    uid = current_user_id()
    payload = request.get_json(silent=True) or {}
    company = (payload.get("company") or "").strip()
    role = (payload.get("role") or "").strip()
    job_id = (payload.get("job_id") or "").strip()
    data = payload.get("data")
    if not company or not role:
        return jsonify({"ok": False, "message": "Missing company or role.", "category": "error"})
    if not isinstance(data, dict):
        return jsonify({"ok": False, "message": "The edited résumé data is not valid.", "category": "error"})
    # Minimal shape guard so a mangled edit can't render an unusable résumé.
    if not str(data.get("name") or "").strip() or not isinstance(data.get("experience"), list):
        return jsonify({"ok": False, "category": "error",
                        "message": "The résumé needs at least a name and an experience list."})
    # Archive the current résumé BEFORE the re-render overwrites it, so the edit is undoable.
    if job_id:
        _snapshot_current_resume(uid, job_id, "before manual edit")
    try:
        out_dir = _resume_dir_for(company, role)
        html_path = render_html(data, company, role, out_dir)   # also re-writes the cached JSON
        pdf_path = render_pdf(html_path)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Could not save your edits: {e}", "category": "error"})
    return jsonify({"ok": True, "category": "success",
                    "message": f"Saved your edits to {company} — {role} (PDF updated). "
                               f'Match score unchanged — click "↻ From JSON" to recompute it.',
                    "company": company, "role": role,
                    "html": _generated_relpath(str(html_path)),
                    "pdf": _generated_relpath(str(pdf_path))})


@app.route("/resume-diff")
def resume_diff_route():
    """Side-by-side "what changed" view: the user's BASE résumé (left) vs the GENERATED
    résumé for one job (right), with added/removed/changed items colour-coded. Also the place
    to EDIT the tailored résumé: an "Edit tailored résumé" toggle turns the right side into a
    form (populated from `gen`) that POSTs to /save-resume-json. Both sides are the same
    tailoring._SCHEMA dict, so src.resume_diff.diff_resumes aligns them field-by-field. Opened
    in a new tab from the résumé cell's ⇄ Compare link."""
    from src import resume_diff
    uid = current_user_id()
    company = (request.args.get("company") or "").strip()
    role = (request.args.get("role") or "").strip()
    if not company or not role:
        abort(404)
    base = base_resume.structured_profile(uid)
    try:
        gen = load_cached_data(company, role, _resume_dir_for(company, role))
    except FileNotFoundError:
        gen = None
    if base is None or gen is None:
        # No base profile, or no generated résumé cached for this job — nothing to compare.
        return render_template("resume_diff.html", company=company, role=role,
                               diff=None, missing_base=base is None, missing_gen=gen is None)
    diff = resume_diff.diff_resumes(base, gen)
    # The job_id lets the editor snapshot the current résumé to History before overwriting it.
    row = next((r for r in read_applications(uid)
                if r.get("Company") == company and r.get("Role") == role), None)
    return render_template("resume_diff.html", company=company, role=role, diff=diff,
                           missing_base=False, missing_gen=False,
                           base_data=base, gen_data=gen,
                           job_id=(row or {}).get("Job ID") or "")


@app.route("/restore-resume-version", methods=["POST"])
def restore_resume_version_route():
    """Make a saved version the CURRENT résumé again. Archives the current one first (so a
    restore is itself undoable), then re-renders the chosen version's saved data over the live
    files and points the résumé row back at that version's score/report/LLMs. Returns JSON."""
    uid = current_user_id()
    job_id = (request.form.get("job_id") or "").strip()
    try:
        version = int(request.form.get("version") or 0)
    except ValueError:
        version = 0
    if not job_id or not version:
        return jsonify({"ok": False, "message": "Missing job or version.", "category": "error"})
    row = get_job(uid, job_id)
    if not row:
        return jsonify({"ok": False, "message": "Could not find that job.", "category": "error"})
    company, role = row.get("Company", ""), row.get("Role", "")
    with db.connect() as conn:
        ver = db.get_resume_version(conn, uid, job_id, version)
    if not ver:
        return jsonify({"ok": False, "message": "That version no longer exists.", "category": "error"})
    # Restore re-renders from the version's saved JSON data (robust to template changes), so we
    # need that file. Keep it inside GENERATED_DIR (path-traversal guard).
    v_json = (GENERATED_DIR / ver["json_path"]).resolve() if ver["json_path"] else None
    if not v_json or not str(v_json).startswith(str(GENERATED_DIR)) or not v_json.is_file():
        return jsonify({"ok": False, "category": "error",
                        "message": "That version's saved data is missing, so it can't be restored."})
    # Snapshot the CURRENT résumé first — restoring is just another overwrite, and should be undoable.
    _snapshot_current_resume(uid, job_id, f"before restoring version {version}")
    try:
        data = json.loads(v_json.read_text(encoding="utf-8"))
        out_dir = _resume_dir_for(company, role)
        html_path = render_html(data, company, role, out_dir)   # re-writes the live files + cached JSON
        pdf_path = render_pdf(html_path)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Could not restore that version: {e}",
                        "category": "error"})
    # Point the live résumé row back at the restored files and this version's score/report/LLMs.
    # (No re-scoring — we're deliberately reinstating the version's own numbers.)
    source = row.get("Job URL") or ""
    log_application(uid, company, role, source, html_path, pdf_path, job_id=job_id,
                    match_score=ver["match_score"], stage_llms=ver["stage_llms"] or None,
                    critic_findings=ver["findings"] or {})
    return jsonify({"ok": True, "category": "success",
                    "message": f"Restored version {version} for {company} — {role}. "
                               f"(Your latest version was saved to history first.)",
                    "company": company, "role": role,
                    "html": _generated_relpath(str(html_path)),
                    "pdf": _generated_relpath(str(pdf_path)),
                    "match_score": ver["match_score"],
                    "match_tip": critic_findings_tooltip(ver["findings"] or {})})


@app.route("/schedule-run", methods=["POST"])
def schedule_run_route():
    """Create a scheduled batch run: generate resumes for the selected tracked jobs at the
    chosen Eastern-Time moment, using the chosen per-stage LLMs. Body (form): job_ids
    (repeated), fire_at ('YYYY-MM-DDTHH:MM', Eastern), the four llm_* stage fields. Returns JSON."""
    job_ids = [j.strip() for j in request.form.getlist("job_ids") if j.strip()]
    fire_at = (request.form.get("fire_at") or "").strip()
    llms = _chosen_llms()

    if not job_ids:
        return jsonify({"ok": False, "message": "Select at least one job to schedule."})
    if not scheduler.is_valid_fire(fire_at):
        return jsonify({"ok": False, "message": "Pick a valid date and time."})

    # Resolve each selected Job ID to its row, skipping any that already have a resume
    # (nothing to generate) or that no longer exist.
    rows = {r.get("Job ID"): r for r in read_applications(current_user_id())}
    jobs, skipped = [], 0
    for jid in job_ids:
        row = rows.get(jid)
        if not row or row.get("HTML"):                # missing, or already generated
            skipped += 1
            continue
        jobs.append({"job_id": jid, "company": row.get("Company", ""),
                     "role": row.get("Role", "")})
    if not jobs:
        return jsonify({"ok": False,
                        "message": "None of the selected jobs need a resume (already generated)."})

    scheduler.create_run(fire_at, llms, jobs)
    note = f" ({skipped} skipped — already have a resume)" if skipped else ""
    return jsonify({"ok": True,
                    "message": f"Scheduled {len(jobs)} resume(s) for {fire_at} ET, "
                               f"using {_llms_summary(llms)}{note}."})


def _llms_summary(llms: dict) -> str:
    """A short human description of a per-stage backend selection for status messages —
    "Claude" when every stage uses one backend, else "per-stage LLMs (generate: …, …)"."""
    used = {v for v in llms.values() if v != "off"}
    if len(used) == 1:
        return llm_label(next(iter(used)))
    parts = ", ".join(f"{stage}: {llm_label(v)}" for stage, v in llms.items())
    return f"per-stage LLMs ({parts})"


@app.route("/scheduled-runs")
def scheduled_runs_route():
    """Current scheduled runs + their per-job status (polled by the UI). `keep_awake` tells
    the UI whether we're currently holding the Mac awake for an upcoming run."""
    return jsonify({"ok": True, "runs": scheduler.list_runs(),
                    "keep_awake": scheduler.is_keeping_awake()})


@app.route("/update-scheduled-run", methods=["POST"])
def update_scheduled_run_route():
    """Edit a still-pending run's job list before it fires. action='remove' drops one job
    (job_id); action='add' folds the currently-selected jobs (job_ids) into the run. A run
    that's already running/done can't be edited."""
    run_id = (request.form.get("run_id") or "").strip()
    action = (request.form.get("action") or "").strip()

    if action == "remove":
        job_id = (request.form.get("job_id") or "").strip()
        res = scheduler.remove_job(run_id, job_id)
        if res == "run_deleted":
            return jsonify({"ok": True, "message": "Removed the last job — the run was deleted."})
        if res == "removed":
            return jsonify({"ok": True, "message": "Removed that job from the scheduled run."})
        return jsonify({"ok": False,
                        "message": "Couldn't remove it — the run may have already started."})

    if action == "add":
        job_ids = [j.strip() for j in request.form.getlist("job_ids") if j.strip()]
        if not job_ids:
            return jsonify({"ok": False, "message": "Tick some jobs in the table first."})
        rows = {r.get("Job ID"): r for r in read_applications(current_user_id())}
        jobs, skipped = [], 0
        for jid in job_ids:
            row = rows.get(jid)
            if not row or row.get("HTML"):            # missing, or already has a resume
                skipped += 1
                continue
            jobs.append({"job_id": jid, "company": row.get("Company", ""),
                         "role": row.get("Role", "")})
        if not jobs:
            return jsonify({"ok": False,
                            "message": "None of the selected jobs need a resume (already generated)."})
        run = scheduler.add_jobs(run_id, jobs)
        if not run:
            return jsonify({"ok": False,
                            "message": "Couldn't add — the run may have already started."})
        note = f" ({skipped} skipped — already have a resume)" if skipped else ""
        return jsonify({"ok": True, "message": f"Added {len(jobs)} job(s) to the run{note}."})

    return jsonify({"ok": False, "message": "Unknown action."})


@app.route("/cancel-scheduled-run", methods=["POST"])
def cancel_scheduled_run_route():
    """Cancel a still-pending run, or remove a finished/cancelled one from the list."""
    run_id = (request.form.get("run_id") or "").strip()
    action = (request.form.get("action") or "cancel").strip()
    if action == "delete":
        ok = scheduler.delete_run(run_id)
        return jsonify({"ok": ok, "message": "Removed." if ok else "Run not found."})
    ok = scheduler.cancel_run(run_id)
    return jsonify({"ok": ok,
                    "message": "Cancelled." if ok else "Too late — that run already started."})


@app.route("/chat", methods=["POST"])
def chat_route():
    """Resume-enhancement chat for a single job (gap suggestions + save real experience
    to the base resume). Body: {job_id, messages:[{role,content}], llm}."""
    # This route receives JSON (not a form). get_json reads it; "or {}" guards against None.
    data = request.get_json(force=True, silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    row = get_job(current_user_id(), job_id)          # one indexed lookup, not a full-table scan
    if not row:
        return jsonify({"ok": False, "reply": "Couldn't find that job."})
    llm = _normalize_llm(data.get("llm"))
    try:
        out = chat_turn(current_user_id(), job_id, row.get("Job URL", ""),
                        row.get("Company", ""), row.get("Role", ""),
                        data.get("messages") or [], llm=llm)
    except Exception as e:
        return jsonify({"ok": False, "reply": f"Chat failed: {e}"})
    return jsonify({"ok": True, **out})               # **out merges chat_turn's reply fields in


@app.route("/update-email", methods=["POST"])
def update_email_route():
    """Rewrite a single resume's contact email — pure code, NO AI. Loads the cached
    tailored JSON, swaps contact.email, and re-renders the HTML + PDF."""
    company = (request.form.get("company") or "").strip()
    role = (request.form.get("role") or "").strip()
    email = (request.form.get("email") or "").strip()

    if not company or not role:
        flash("Missing resume identity; cannot update email.", "error")
        return redirect(url_for("index"))
    if not EMAIL_RE.match(email):                     # reject obviously-invalid emails
        flash(f"'{email}' is not a valid email address.", "error")
        return redirect(url_for("index"))

    try:
        out_dir = _resume_dir_for(company, role)      # the résumé's folder (dated sub-folder for cron)
        data = load_cached_data(company, role, out_dir)  # the saved résumé data
        # setdefault("contact", {}) ensures a "contact" dict exists, then set its email.
        data.setdefault("contact", {})["email"] = email
        html_path = render_html(data, company, role, out_dir)  # also re-caches the JSON
        render_pdf(html_path)
    except FileNotFoundError:
        flash(f"No cached resume found for {company} — {role}; re-generate it first.", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Could not update email: {e}", "error")
        return redirect(url_for("index"))

    flash(f"Email set to {email} for {company} — {role}. Resume rewritten (no AI used).",
          "success")
    return redirect(url_for("index"))


@app.route("/regenerate-resume", methods=["POST"])
def regenerate_resume_route():
    """Re-render HTML + PDF from the cached tailored JSON, then recompute the Match score
    by re-running the critic against the job description (the cached resume may have been
    hand-edited). Returns JSON so the UI can update the row inline. Re-scoring uses one LLM
    call; if the job description can't be recovered (pasted-text jobs), the existing score
    is kept.

    Runs on a background thread and returns a task id the page polls via /generate-status —
    the re-score step is an LLM call that can be slow (especially on the local model), so we
    don't hold the browser request open for it.
    """
    company = (request.form.get("company") or "").strip()
    role = (request.form.get("role") or "").strip()
    # Regenerate re-renders (no LLM) then re-scores — a "check" operation — so it uses the
    # Stage-3 truthfulness-check backend. Capture before the thread (no request ctx there).
    llm = _chosen_llms()["check"]

    if not company or not role:
        return jsonify({"ok": False, "message": "Missing company or role.", "category": "error"})

    uid = current_user_id()                           # capture before the thread (no request ctx there)
    task_id = _start_generation(lambda p: _regenerate_core(uid, company, role, llm, on_progress=p))
    return jsonify({"ok": True, "task_id": task_id})


def _regenerate_core(user_id, company, role, llm, on_progress=None) -> dict:
    """Re-render HTML+PDF from the cached résumé JSON and best-effort re-score it. Returns a
    JSON-ready result dict (the shape the page expects). `on_progress` forwards status to the
    polling browser. Never raises."""
    def log(msg):
        if on_progress:
            on_progress(msg)
    # Archive the current résumé BEFORE the re-render overwrites its files, so this regenerate
    # is undoable (see /resume-versions). Look the row up now for its Job ID; reused for scoring.
    row = next((r for r in read_applications(user_id)
                if r.get("Company") == company and r.get("Role") == role), None)
    if row and row.get("Job ID") and row.get("HTML"):
        _snapshot_current_resume(user_id, row["Job ID"], "before re-render from JSON")
    try:
        log("Re-rendering PDF from saved JSON…")
        out_dir = _resume_dir_for(company, role)      # the résumé's folder (dated sub-folder for cron)
        data = load_cached_data(company, role, out_dir)
        # render_html also re-writes the cached JSON so it's safe to call
        html_path = render_html(data, company, role, out_dir)
        pdf_path = render_pdf(html_path)
    except FileNotFoundError:
        return {"ok": False, "message": f"No cached resume found for {company} — {role}; re-generate it first.", "category": "error"}
    except Exception as e:
        return {"ok": False, "message": f"Could not regenerate: {e}", "category": "error"}

    # The PDF/HTML are now regenerated and saved — that part is DONE regardless of what
    # happens next. Recomputing the Match score is a best-effort extra: it needs an LLM
    # call, which can fail (quota, timeout, local LLM down). A scoring failure must NEVER
    # undo the regeneration, so it's isolated here and surfaced as its own warning popup.
    # (`row` was already resolved above for the pre-render snapshot.)
    match_score = None
    match_tip = None                                  # critic findings tooltip; set when re-scored
    base = f"Regenerated resume for {company} — {role} (PDF updated)."
    # Try to recover the job description so we can re-score. (row or {}) guards against None.
    job_desc, source = fetch_job_description(
        (row or {}).get("Job ID") or "", (row or {}).get("Job URL") or "")
    if job_desc:
        try:
            log("Re-scoring against the job description…")
            review = score_resume(data, job_desc, base_resume.resume_source_text(user_id),
                                  company, role, llm=llm)
            match_score = round(review["overall_score"] / 10, 1)
            findings = {
                "fidelity": review["fidelity_issues"],
                "overstatement": review.get("overstatement_notes", []),
                "style": review["style_issues"],
                "revision": review["revisions"],
                "strength": review["strengths"],
            }
            log_application(user_id, company, role, source, html_path, pdf_path,
                            job_id=(row or {}).get("Job ID"), match_score=match_score,
                            critic_findings=findings)
            match_tip = critic_findings_tooltip(findings)
            message, category = f"{base} Match score recalculated: {match_score}/10.", "success"
        except Exception as e:
            # Resume IS regenerated; only the score step failed — warn, don't fail.
            message = (f"{base} Match score NOT recalculated — the scoring step failed "
                       f"({e}). Your previous score was kept.")
            category = "error"
    else:
        message = f"{base} Match score left unchanged (no stored job description to score against)."
        category = "success"

    return {
        "ok": True,                 # regeneration succeeded even when scoring didn't
        "message": message,
        "category": category,
        "company": company,
        "role": role,
        "html": _generated_relpath(str(html_path)),
        "pdf": _generated_relpath(str(pdf_path)),
        "match_score": match_score,  # null when not recalculated → JS leaves the cell as-is
        "match_tip": match_tip,      # critic findings for the tooltip (null when not re-scored)
    }


def _delete_generated_files(row) -> list:
    """Delete a row's generated résumé files — the HTML, the PDF, and the cached JSON (which
    shares the HTML's stem). Handles files in dated sub-folders too (uses each path relative
    to the generated dir). Best-effort, and strictly confined to the generated dir (so a bad
    stored path can never delete anything outside it). Returns the relative paths removed."""
    html, pdf = row.get("HTML") or "", row.get("PDF") or ""
    rels = set()
    if html:
        html_rel = _generated_relpath(html)
        rels.add(html_rel)                                       # <dir>/slug_resume.html
        rels.add(str(Path(html_rel).with_suffix(".json")))      # <dir>/slug_resume.json (cache)
    if pdf:
        rels.add(_generated_relpath(pdf))                       # <dir>/slug_resume.pdf
    removed = []
    for rel in rels:
        if not rel:
            continue
        target = (GENERATED_DIR / rel).resolve()
        # Path-traversal guard: only touch files that really live inside the generated dir.
        if str(target).startswith(str(GENERATED_DIR)) and target.is_file():
            try:
                target.unlink()
                removed.append(rel)
            except OSError:
                pass
    return removed


@app.route("/delete-job", methods=["POST"])
def delete_job_route():
    """Remove a tracked job row AND delete its generated résumé files. Intended for the
    'To Apply' tab; applied jobs are refused (they're history). Returns JSON so the page can
    drop the row inline without a full reload."""
    uid = current_user_id()
    job_id = (request.form.get("job_id") or "").strip()
    row = get_job(uid, job_id)                         # one indexed lookup, not a full-table scan
    if not row:
        return jsonify({"ok": False, "message": "Could not find that job to delete."})
    if row.get("Applied"):
        return jsonify({"ok": False, "message": "Applied jobs can't be deleted here."})

    deleted = delete_job(uid, job_id)
    files = _delete_generated_files(deleted) if deleted else []
    suffix = f" (removed {len(files)} résumé file{'s' if len(files) != 1 else ''})." if files else "."
    return jsonify({"ok": True, "job_id": job_id,
                    "message": f"Deleted {row.get('Company','')} — {row.get('Role','')}{suffix}"})


@app.route("/applied", methods=["POST"])
def applied_route():
    """Mark a job as applied (today's date), triggered by the 'Applied manually' button."""
    key = (request.form.get("key") or "").strip()
    if key and mark_applied(current_user_id(), key):  # mark_applied returns True if it found the row
        flash("Marked as applied. ✓", "success")
    else:
        flash("Could not find that resume to mark as applied.", "error")
    return redirect(url_for("index"))


@app.route("/files/<path:filename>")                 # <path:filename> captures the requested filename
def files(filename):
    """Serve a generated HTML or PDF for preview/download."""
    target = (GENERATED_DIR / filename).resolve()
    # Security: prevent "path traversal" (e.g. ../../etc/passwd) — only serve files that
    # actually live inside GENERATED_DIR and exist. abort(404) returns a "not found" page.
    if not str(target).startswith(str(GENERATED_DIR)) or not target.exists():
        abort(404)
    as_download = request.args.get("download") == "1"  # ?download=1 forces a download vs. preview
    return send_from_directory(GENERATED_DIR, filename, as_attachment=as_download)


# Runs only when you launch this file directly: `python app.py`.
if __name__ == "__main__":
    # Default to 5001: on macOS, port 5000 is taken by AirPlay Receiver (ControlCenter),
    # which intercepts http://localhost:5000 over IPv6 and hides this app. Override with
    # the PORT env var if needed.
    import os
    port = int(os.environ.get("PORT", "5001"))
    # In Docker the app must listen on all interfaces (0.0.0.0) to be reachable from the
    # host, and the reloader/debugger are turned off (compose sets HOST + FLASK_DEBUG).
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    use_reloader = debug
    # Start the batch scheduler thread. With the reloader on, Flask runs TWO processes;
    # only the worker (WERKZEUG_RUN_MAIN="true") should fire runs, or each scheduled run
    # would generate twice. Without the reloader there's one process, so start it directly.
    if not use_reloader or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        # Cron-generated résumés are filed under today's dated sub-folder (computed per job
        # at generation time, so a batch that runs near midnight uses the actual run date).
        # No request context in the scheduler thread; current_user_id() resolves to the
        # default user (there's no auth yet), which is who scheduled runs belong to.
        scheduler.start(lambda job_id, llms: _generate_for_job(
            current_user_id(), job_id, llms, out_dir=_dated_dir(), source="scheduled"))
    # threaded=True so a long resume generation doesn't block other requests (clicking
    # links, loading files, or generating another resume in parallel).
    app.run(debug=debug, host=host, port=port, threaded=True, use_reloader=use_reloader)
