"""Phase 2 orchestrator: cached board directories → match → dedupe → generate().

Ties the pieces together across ALL supported job boards (Greenhouse, Lever, Rippling,
Workday, Ashby today):
  1. Read each board's directory from the board_directory table (built by its *_discovery module).
  2. For each board, list jobs (one cheap call) and filter with the pure-Python
     matcher — no LLM, so scanning thousands of jobs is free.
  3. Dedupe against the tracker's jobs table so we never re-process a job already logged.
  4. For each deep-screen survivor, IMMEDIATELY: store it (job row + full description),
     embed it (Match %), and AI-score it (1-10 fit verdict) — one job at a time, so the
     home page fills up with scored jobs WHILE the scan runs, and an interrupted scan
     keeps everything it processed. The description is stored so nothing ever needs to
     re-fetch it from the board (boards rate-limit bulk re-fetches).
  5. (Optional) For each new match, run the existing Phase 1 generate() on demand.

The filters stay deterministic Python; ChatGPT is only reached by jobs that already
passed every rule. This file is the "conductor" for search (like pipeline.py is
the conductor for résumé generation).

ADDING A NEW BOARD later is a one-line change: write its connector + *_discovery module,
then add an entry to _SOURCES below. The rest of this file (and the matcher/tracker) is
board-agnostic because every connector returns the same normalized job shape.
"""
import sys                                           # stderr fallback when audit bookkeeping fails
import threading                                     # one lock guards the scan's shared tallies
import time                                          # to pause politely between API calls
from concurrent.futures import ThreadPoolExecutor    # several boards scanned at once

from src import db                                   # store fit verdicts as they're written
from src import embeddings                           # meaning net + Match % vectors (OpenAI)
from src import fit_verdict                          # per-job ChatGPT 1-10 fit scoring
from src import matching                             # read live config (TITLE_SIMILARITY) at call time
from src.ats import ATSError                         # shared base error for any connector failure
from src.ats import (ashby, greenhouse, lever, rippling, smartrecruiters,
                     workday)   # the board connectors (list/fetch jobs)
from src.ats import (ashby_discovery, greenhouse_discovery, lever_discovery,
                     rippling_discovery, smartrecruiters_discovery,
                     workday_discovery)   # each board's cached directory loader
from src.matching import (classify_location, is_commutable, match_job,
                          names_non_us_place, TITLE_SIEVE_REJECT)  # pure-Python matcher
from src.pipeline import generate                    # Phase 1 résumé generation (only used on demand)
from src import base_resume                          # this user's base résumé (DB profile → text)
from src.scan_audit import NullAudit, ScanAudit       # per-scan drop-funnel audit log
from src.tailoring import chatgpt_key                # is the ChatGPT backend configured?
from src.tracker import make_job_id, tracked_job_urls, upsert_found_jobs

# The registry of boards we scan: (source name, connector module, directory loader).
# To add a board, append one line here once its connector + discovery module exist.
_SOURCES = [
    ("greenhouse", greenhouse, greenhouse_discovery.load_directory),
    ("lever", lever, lever_discovery.load_directory),
    ("rippling", rippling, rippling_discovery.load_directory),
    ("workday", workday, workday_discovery.load_directory),
    ("ashby", ashby, ashby_discovery.load_directory),
    ("smartrecruiters", smartrecruiters, smartrecruiters_discovery.load_directory),
]
# Quick lookup of source name → connector, for re-fetching a job by its source during the
# deep screen / generation (a job dict carries its own "source" field).
_CONNECTORS = {name: connector for name, connector, _loader in _SOURCES}


def source_names() -> list[str]:
    """The names of every board source we can scan (e.g. ['greenhouse', 'lever']). Drives the
    UI's "Find Jobs Automatically" source checklist so it always reflects _SOURCES — add a
    board to _SOURCES and it shows up in the UI automatically."""
    return [name for name, _connector, _loader in _SOURCES]


def _noop(_msg):
    """A do-nothing default for the optional progress callback (so callers can omit it).
    "noop" = "no operation". The "_msg" name signals the argument is intentionally ignored."""
    pass


def _location_rank(job: dict, f: "matching.Filters") -> int:
    """How much we'd prefer to KEEP this listing when collapsing duplicate same-company/
    same-title postings that differ only by location. Higher = keep:
      3 = inside the user's own commute zone (their editable commutable_locations)
      2 = some other confidently-US location
      1 = location we can't confirm as US
      0 = confidently non-US (e.g. a Canada-only twin of a US-Remote posting)
    An empty commute zone simply yields no rank-3 (neutral). Ties keep the first-seen
    (strongest-ranked) listing."""
    location = job.get("location", "")
    if is_commutable(location, f):
        return 3
    # A confidently non-US twin (e.g. a 'Canada - Remote' clone of a 'US - Remote' posting)
    # always loses, even if a loose 'City, ST' collision would let classify_location call it US.
    if names_non_us_place(location):
        return 0
    status = classify_location(location)
    return {"usa": 2, "unverified": 1, "non_usa": 0}[status]


def _dedupe_company_title(jobs: list[dict], f: "matching.Filters") -> list[dict]:
    """Collapse postings that share the same company AND title (differing only by location)
    down to ONE. Some boards list the same role once per office, so a single opening can
    appear many times (e.g. 'Human Agency' surfaced 8 identical roles, one per city) — without
    this we'd generate identical resumes for each. Keeps the best listing by _location_rank
    (the user's commute zone first, then any US, never a non-US twin when a US one exists);
    ties keep the first seen, which (since the list is rank-sorted) is the strongest-ranked.
    First-seen order is preserved so the overall best-first ordering is unchanged."""
    best: dict[tuple, dict] = {}                      # (company, title) → the listing we'll keep
    order: list[tuple] = []                           # the keys in first-seen order
    for job in jobs:
        # Normalize company + title so trivial case/whitespace differences still collapse.
        key = (job.get("company", "").strip().lower(), job.get("title", "").strip().lower())
        if key not in best:
            best[key] = job
            order.append(key)
        elif _location_rank(job, f) > _location_rank(best[key], f):
            best[key] = job                          # upgrade to the better-located listing
    return [best[key] for key in order]


def _meaning_net(borderline: list[dict], target_vec, f, on_progress=_noop, audit=None) -> list[dict]:
    """The title MEANING NET: give keyword-less titles one more chance via embeddings.

    `borderline` = this board's jobs whose ONLY reject reason was the title sieve. Each
    distinct title is embedded (one small batched OpenAI call per board) and compared
    to `target_vec` — the embedded target-roles paragraph, computed ONCE per scan by
    the caller (None = the net is off, e.g. no OpenAI key). Titles at least
    f.title_similarity similar are re-admitted with semantic_pass=True. Every admitted
    title is logged WITH its similarity %, so the Settings threshold can be tuned.
    Returns an empty list when the net is off or nothing is borderline."""
    if not borderline or target_vec is None:         # off (no key) → caller logged it once
        return []
    try:
        titles = sorted({j["title"] for j in borderline})     # each distinct title once
        vecs = embeddings.embed_texts(titles)                 # one small call per board
        sims = {t: embeddings.cosine(v, target_vec) for t, v in zip(titles, vecs)}
    except embeddings.EmbeddingsError as e:
        on_progress(f"  (meaning net skipped for this board: {e})")
        return []
    admitted = []
    for job in borderline:
        sim = sims.get(job["title"], 0.0)
        if sim >= f.title_similarity:                # this user's meaning-net threshold
            verdict = match_job(job, f, semantic_pass=True)   # re-judge without the sieve
            if verdict["qualifies"]:
                admitted.append({**job, "match": verdict})
                on_progress(f"  meaning net admitted: {job.get('company', '?')} — "
                            f"{job['title']} (similarity {sim:.0%})")
                if audit is not None:
                    audit.sieve_pass(job, f"meaning-net ({sim:.0%})")
                    audit.stage(job, "list_screen", "advanced")
    return admitted


def _prepare_fit_scoring(user_id: int, on_progress=_noop, fit_llm: str = "chatgpt") -> dict | None:
    """Once per scan: everything the per-job AI fit scoring needs (the résumé text, this
    user's set of already-scored job ids, and which LLM backend scores). Returns None —
    with ONE log line, not one per job — when scoring can't run at all (chatgpt backend
    without a ChatGPT/OpenAI key, or no base résumé). Claude and the local models need
    no OpenAI key."""
    if fit_llm == "chatgpt" and not chatgpt_key():
        on_progress("  (no ChatGPT/OpenAI key — jobs will be stored without AI fit scores)")
        return None
    try:
        resume_text = base_resume.resume_source_text(user_id)
    except FileNotFoundError:
        on_progress("  (no base résumé for this user — jobs will be stored without AI fit scores)")
        return None
    with db.connect() as conn:                       # re-discovered jobs keep their old score
        already = db.fetch_verdicted_ids(conn, user_id)
    return {"resume_text": resume_text, "already": already, "llm": fit_llm}


def _store_and_score(job: dict, user_id: int, scoring: dict | None, on_progress=_noop) -> None:
    """Handle ONE deep-screen survivor, immediately: embed its description (Match %),
    store the job row WITH the description, then AI-score it. Each step degrades
    gracefully — a failure is logged and the job still lands in the database, so an
    interrupted scan loses nothing. Annotates the job dict with what happened
    ('stored' = added|refreshed; 'ai_fit_score'/'ai_fit_reason' when scored)."""
    company, title = job.get("company", "?"), job.get("title", "?")
    description = job.get("description", "")

    # 1. Embed (one small OpenAI call) BEFORE storing, so the row lands complete —
    #    with its Match % — rather than being patched later.
    if embeddings.available() and description:
        try:
            text = f"{title}\n{company}\n{job.get('location', '')}\n\n{description}"
            job["embedding"] = embeddings.embed_texts([text])[0]
            job["embedding_model"] = embeddings.active_model()
        except embeddings.EmbeddingsError as e:
            on_progress(f"  (couldn't embed {company} — {title}: {e})")

    # 2. Store: shared posting + this user's overlay + flags + description + embedding.
    result = upsert_found_jobs(user_id, [job])
    job["stored"] = "added" if result["added"] else "refreshed"

    # 3. AI-score it right now, while the description is in hand. A job that can't be
    #    scored (no description / ChatGPT hiccup) stays stored and unscored — the
    #    Score-fits backfill can pick it up later.
    if scoring is None or not description or make_job_id(job) in scoring["already"]:
        on_progress(f"  ✓ {job['stored']}: {company} — {title}")
        return
    for attempt in (1, 2):                           # one retry: most failures are blips
        try:
            score, reason = fit_verdict.verdict_for(scoring["resume_text"], company, title,
                                                    description, llm=scoring["llm"])
            with db.connect() as conn:
                db.set_fit_verdict(conn, user_id, make_job_id(job), score, reason)
            job["ai_fit_score"], job["ai_fit_reason"] = score, reason
            on_progress(f"  ✓ {job['stored']} + scored {score}/10: {company} — {title}")
            return
        except Exception as e:
            if attempt == 1:
                time.sleep(2.0)                      # brief pause, then try once more
            else:
                on_progress(f"  (couldn't AI-score {company} — {title}: {e})")


def add_job_manually(user_id: int, scan_id: int, source: str, board_token: str,
                     external_id: str, *, fit_llm: str = "chatgpt",
                     audit_company: str | None = None,
                     audit_title: str | None = None,
                     audit_url: str | None = None) -> dict:
    """Rescue a scan-dropped job into this user's tracked list. Re-fetches the live posting
    from its board connector, stores it via the same embed→store→AI-score path the scan uses
    (degrading gracefully without an OpenAI key / base résumé), clears the URL-keyed rejection
    so future scans stay consistent, and appends a distinct `manual_add` event to `scan_id`.
    Returns {company, role, ai_fit_score, stored}. Raises ValueError if `source` has no
    re-fetch connector; propagates the connector's error if the posting can't be fetched.
    audit_company/audit_title, when given, label the manual_add event with the job's
    originally-audited (list-stage) company/title so it groups with its dropped events —
    the re-fetched detail title can differ. audit_url is the url the scan actually rejected
    under (the card's stored url); we clear BOTH it and the re-fetched url, since a board's
    list url can differ from its detail url and the rejection was keyed on the former.
    """
    connector = _CONNECTORS.get(source)
    if connector is None:
        raise ValueError(f"no re-fetch connector for source '{source}'")
    full = connector.get_job(board_token, external_id)   # the live posting (may raise)

    scoring = _prepare_fit_scoring(user_id, fit_llm=fit_llm)   # None → stored without a score
    _store_and_score(full, user_id, scoring)                  # embeds/stores/scores, annotates full

    # Best-effort bookkeeping — the job is already stored; these must never fail the add.
    try:
        with db.connect() as conn:
            for u in {audit_url or "", full.get("url") or ""}:   # deduped; blank is a no-op
                db.clear_rejection(conn, user_id, u)
            db.append_scan_event(conn, scan_id, {
                "company": audit_company or full.get("company"),
                "title": audit_title or full.get("title"),
                "source": source, "board_token": board_token,
                "location": full.get("location"), "external_id": external_id,
                "url": full.get("url"), "stage": "manual_add", "outcome": "stored",
                "detail": "added manually", "reason_bucket": None})
    except Exception as e:                                     # observability must not break the add
        print(f"[add_job_manually] post-store bookkeeping failed: {e}", file=sys.stderr)

    return {"company": full.get("company", ""), "role": full.get("title", ""),
            "ai_fit_score": full.get("ai_fit_score"), "stored": full.get("stored")}


def _available_boards(on_progress, sources=None) -> list[tuple]:
    """Gather (source_name, connector, token) for every board across the selected sources
    whose directory has been built. `sources` (a set/list of source names, e.g. {"greenhouse"})
    limits which sources to scan; None means all. Notes any source whose directory is missing
    (so a half-set-up install still scans what it can). Raises FileNotFoundError only if NO
    selected source has a directory yet — so the UI can tell you to build one."""
    boards = []
    missing = []
    for name, connector, load_dir in _SOURCES:
        if sources is not None and name not in sources:
            continue                                  # source not selected in the UI — skip it
        try:
            directory = load_dir()                    # raises FileNotFoundError if not built yet
        except FileNotFoundError:
            missing.append(name)
            continue
        for b in directory["boards"]:
            boards.append((name, connector, b["token"]))
    if not boards:
        raise FileNotFoundError(
            "No board directories built yet. Build at least one, e.g. "
            "`python -m src.ats.greenhouse_discovery` or `python -m src.ats.lever_discovery`."
        )
    if missing:
        # At least one source works, but not all — note the missing one(s) and keep going.
        on_progress(f"(note: no directory yet for {', '.join(missing)} — skipping it; build with "
                    f"`python -m src.ats.{missing[0]}_discovery`)")
    return boards


def _sieve_and_screen(jobs, f, seen_urls, audit):
    """Split the title sieve from the rest of the list-stage screen so a sieve-passer
    dropped by a non-sieve rule is LOGGED, not silently discarded.

    Returns (candidates, borderline): candidates are sieve-passers that also clear the
    list screen (each annotated with its 'match' verdict); borderline are sieve-FAILERS
    whose only reject was the sieve — the meaning net's inputs. Jobs failing the sieve for
    ANY other reason, and already-seen URLs, are dropped WITHOUT a log line (per the design:
    never log a title that didn't pass the sieve)."""
    candidates, borderline = [], []
    for job in jobs:
        if job.get("url") in seen_urls:
            continue
        verdict = match_job(job, f)                       # ONE match_job call per job
        reasons = verdict["reject_reasons"]
        if TITLE_SIEVE_REJECT in reasons:                 # did NOT pass the title sieve
            if reasons == [TITLE_SIEVE_REJECT]:           # sieve was the ONLY problem
                borderline.append({**job, "match": verdict})   # → meaning net
            continue                                      # silent: no sieve pass, no log
        audit.sieve_pass(job, "keyword")                  # cleared the sieve → start its trace
        if verdict["qualifies"]:
            audit.stage(job, "list_screen", "advanced")
            candidates.append({**job, "match": verdict})
        else:
            audit.stage(job, "list_screen", "dropped", reasons)
    return candidates, borderline


def _screen_store_score(job: dict, f, user_id: int, scoring: dict | None,
                        on_found, on_progress, audit) -> dict | None:
    """Take ONE title-sieve survivor all the way to the end, right now:
    fetch its full description (if the board's list call didn't include one) → deep
    screen (salary / US / remote / age / contract rules) → if it passes, store it
    (with the description) + embed + AI-score via _store_and_score, and notify the
    UI feed. Returns the stored match dict, or None if the deep screen dropped it."""
    connector = _CONNECTORS.get(job.get("source"))
    # If we already have the description (Lever) — or have no connector to re-fetch with —
    # use the job as-is; otherwise fetch the full posting (Greenhouse), retrying once
    # (fetch hiccups are usually blips, and a jobless retry costs a stored-but-unscored row).
    if job.get("description") or connector is None:
        full = job
    else:
        full = None
        for attempt in (1, 2):
            try:
                full = connector.get_job(job["board_token"], job["id"])   # the FULL description
                break
            except ATSError:
                if attempt == 1:
                    time.sleep(1.0)
        if full is None:
            # Still can't fetch the description — keep the title-passed job rather than
            # lose it (stored without description/score; the backfill finishes it later).
            audit.stage(job, "fetch", "failed")
            _store_and_score(job, user_id, scoring, on_progress)
            audit.stage(job, "store", "stored")
            if on_found:
                on_found(job)
            return job

    audit.stage(full, "fetch", "ok")
    # Carry forward whether the meaning net admitted this title, so the deep screen
    # doesn't re-reject it for having no keywords (all OTHER screens still apply).
    verdict = match_job(full, f, semantic_pass=job["match"].get("semantic_pass", False))
    if not verdict["qualifies"]:
        reason = "; ".join(verdict["reject_reasons"])
        audit.stage(full, "deep_screen", "dropped", verdict["reject_reasons"])
        on_progress(f"  dropped: {full['company']} — {full['title']} [{reason}]")
        # Remember the rejection so none of this user's future scans re-fetch this posting.
        with db.connect() as conn:
            db.record_rejection(conn, user_id, full.get("url", ""), full.get("company", ""),
                                full.get("title", ""), reason)
        return None
    # Keep the FULL job (with description) plus its fresh verdict, then store, embed,
    # and AI-score it RIGHT NOW — the home page feed shows it immediately.
    audit.stage(full, "deep_screen", "advanced")
    match = {**full, "salary_text": verdict["salary_text"], "match": verdict}
    _store_and_score(match, user_id, scoring, on_progress)
    score = match.get("ai_fit_score")
    audit.stage(match, "store", "stored", [f"ai_fit {score}/10"] if score is not None else None)
    if on_found:
        on_found(match)
    return match


def find_matches(user_id: int, boards_limit: int = None, polite_delay: float = 0.2,
                 on_progress=_noop, on_tick=None,
                 sources=None, on_found=None, workers: int = 6,
                 fit_llm: str = "chatgpt", use_audit: bool = True) -> list[dict]:
    """Scan the cached board directories for one USER and return their qualifying,
    NOT-yet-seen jobs (judged against that user's filters).

    ONE interleaved pass, board by board — every job goes through the FULL pipeline
    the moment its board is listed (no more "list all 3,000+ boards first, process
    later"):
        list this board's jobs (one cheap call)
        → title sieve + this board's meaning net (keyword-less titles vs the embedded
          target-roles paragraph) + collapse per-office duplicate listings
        → for each candidate: fetch the description, deep screen it (salary / US /
          remote / age / contract), and if it passes: STORE it (with the description)
          + embed (Match %) + AI-score — all before moving on.
    So real, scored matches land in the database (and the UI's live feed) within the
    first minutes of a scan, and an interrupted scan keeps everything processed so
    far. Each returned job dict is annotated with 'stored' ('added'|'refreshed') and,
    when scored, 'ai_fit_score'/'ai_fit_reason'.

    Speed: `workers` boards are scanned CONCURRENTLY (each on its own thread — the work
    is almost all waiting on board/OpenAI network calls, which threads overlap well);
    confidently non-US candidates are dropped from list data BEFORE paying for a
    description fetch; and every deep-screen rejection is remembered (`rejected_jobs`),
    so later scans never re-fetch a posting that already failed the rules.

    `boards_limit` caps how many boards to scan in total across all sources (None = all).
    `on_tick`, if given, is called with ("scan", boards_done, boards_total, stored_count)
    for the UI progress bar — "found" now counts STORED matches, not title candidates.
    `on_found`, if given, is called with each annotated job dict right after it's
    stored (the UI's live "found so far" feed)."""
    # If no on_tick was given, use a throwaway function that ignores its arguments.
    tick = on_tick or (lambda *a, **k: None)
    f = matching.load_filters(user_id)               # this user's keyword lists + thresholds
    boards = _available_boards(on_progress, sources)  # [(source, connector, token), …]
    if boards_limit is not None:
        boards = boards[:boards_limit]               # keep only the first N boards (list slice)

    filter_summary = {
        "salary_floor": f.salary_floor,
        "max_age": f.max_posting_age_days,
        "commute": list(f.commutable_locations),
        "title_similarity_pct": round(f.title_similarity * 100),
        "domain_kw": len(f.domain_keywords),
        "negative_kw": len(f.title_negative_keywords),
    }
    scanned_sources = source_names() if sources is None else sorted(sources)
    # The audit is OBSERVABILITY ONLY — it must never take down the scan it observes.
    # When disabled (the user declined to continue without it earlier is handled in app.py;
    # here `use_audit=False` means "run without it") or if the parent scans row can't be
    # created (DB hiccup), fall back to a no-op audit so the scan runs on.
    if use_audit:
        try:
            audit = ScanAudit(user_id, scanned_sources, filter_summary)
            on_progress(f"  (audit: scan #{audit.scan_id})")
        except Exception as e:
            audit = NullAudit()
            on_progress(f"  (audit disabled — couldn't open the scan record: {e})")
    else:
        audit = NullAudit()
        on_progress("  (audit disabled by request — scan runs without it)")

    # Build a set of job URLs we can skip outright: everything this user already tracks PLUS
    # everything their past deep screens already rejected (no point re-fetching those).
    seen_urls = tracked_job_urls(user_id)                # one indexed column read, not full rows
    with db.connect() as conn:
        rejected = db.fetch_rejected_urls(conn, user_id)
    if rejected:
        on_progress(f"  (skipping {len(rejected)} posting(s) rejected by earlier scans)")
        seen_urls |= rejected

    # Once-per-scan setup: the AI-scoring context (résumé text + already-scored ids)
    # and the meaning net's target vector. Both log ONE line and switch off cleanly
    # when they can't run (no OpenAI key / no base résumé).
    scoring = _prepare_fit_scoring(user_id, on_progress, fit_llm)  # None → store without AI scores
    target_vec = None
    if embeddings.available():
        try:
            target_vec = embeddings.target_roles_embedding(user_id)
        except embeddings.EmbeddingsError as e:
            on_progress(f"  (meaning net off: {e})")
    else:
        on_progress("  (no OpenAI key — meaning net off, no Match % embeddings)")

    matches: list[dict] = []
    processed: set[tuple] = set()                    # (company, title) done this scan
    state = {"boards_done": 0}
    lock = threading.Lock()                          # guards matches/processed/boards_done
    on_progress(f"Scanning {len(boards)} board(s) across {len(_SOURCES)} source(s), "
                f"{workers} at a time — screening, storing, and AI-scoring matches "
                "as they're found…")
    tick("scan", 0, len(boards), 0)

    def handle_board(board):
        """The full pipeline for ONE board (runs on a worker thread)."""
        source, connector, token = board
        try:
            jobs = connector.list_jobs(token)        # one cheap API call: all jobs on this board
        except ATSError as e:                        # catches GreenhouseError OR LeverError
            on_progress(f"  {source}:{token}: skipped ({e})")
            jobs = []

        # Sieve-aware split: sieve-passers are logged (and their list-stage outcome),
        # sieve-failers stay silent unless the meaning net admits them.
        candidates, borderline = _sieve_and_screen(jobs, f, seen_urls, audit)
        admitted = _meaning_net(_dedupe_company_title(borderline, f), target_vec, f,
                                on_progress, audit)
        candidates = _dedupe_company_title(candidates + admitted, f)

        # Take each candidate all the way to stored-and-scored before the next one.
        for job in candidates:
            # Cross-board guard: the same company+title seen earlier this scan (e.g. on
            # another board) was already handled — first board wins.
            key = (job.get("company", "").strip().lower(), job.get("title", "").strip().lower())
            with lock:
                is_dup = key in processed
                if not is_dup:
                    processed.add(key)
            if is_dup:                                    # audit call OUTSIDE handle_board's lock
                audit.stage(job, "dedup", "dropped")
                continue
            audit.stage(job, "dedup", "advanced")
            # Cheap pre-drop: a location that CONFIDENTLY names a non-US place fails the US
            # rule no matter what the description says — skip the fetch.
            if names_non_us_place(job.get("location", "")):
                reason = f"non-US location: {job.get('location', '')}"
                audit.stage(job, "pre_fetch_us", "dropped", [reason])
                on_progress(f"  dropped: {job.get('company', '?')} — {job.get('title', '?')} "
                            f"[{reason}]")
                with db.connect() as conn:
                    db.record_rejection(conn, user_id, job.get("url", ""), job.get("company", ""),
                                        job.get("title", ""), reason)
                continue
            audit.stage(job, "pre_fetch_us", "advanced")
            match = _screen_store_score(job, f, user_id, scoring, on_found, on_progress, audit)
            if match is not None:
                with lock:
                    matches.append(match)
                    tick("scan", state["boards_done"], len(boards), len(matches))
            if polite_delay:
                time.sleep(polite_delay)                  # per-thread pause — stays polite per host

        with lock:
            state["boards_done"] += 1
            tick("scan", state["boards_done"], len(boards), len(matches))

    # The work is nearly all network waiting (board APIs, OpenAI), which threads overlap
    # well — several boards in flight at once instead of one after another.
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(handle_board, boards))     # list() surfaces any worker exception
    finally:
        audit.close(len(matches))                    # footer even if a worker blew up

    matches.sort(key=lambda j: j["match"]["rank_score"], reverse=True)   # best fits first
    audit_suffix = f" Audit: scan #{audit.scan_id}" if audit.scan_id is not None else ""
    on_progress(f"Scan complete: {len(matches)} match(es) stored "
                f"across {len(boards)} board(s).{audit_suffix}")
    return matches


def generate_for_matches(user_id: int, matches: list[dict], max_resumes: int = None,
                         on_progress=_noop, **generate_kwargs) -> list[dict]:
    """For each match, fetch the full description and run Phase 1 generate().

    Returns a per-job result list with the produced file paths + meta (or an error).
    `max_resumes` caps how many to actually generate (protects token budget);
    extra `generate_kwargs` pass through to generate() (e.g. use_external_critic)."""
    # Take only the first max_resumes jobs (or all of them if max_resumes is None).
    todo = matches[:max_resumes] if max_resumes is not None else matches
    results = []

    for i, job in enumerate(todo, 1):
        title, company = job["title"], job["company"]
        on_progress(f"[{i}/{len(todo)}] Generating: {company} — {title}")
        try:
            # Reuse the description fetched during deep-screen; only re-fetch if absent,
            # using the right connector for this job's source.
            connector = _CONNECTORS.get(job.get("source"), greenhouse)
            full = job if job.get("description") else connector.get_job(
                job["board_token"], job["id"])
            html_path, pdf_path, meta = generate(
                user_id, company, title, source=full["url"], job_text=full["description"],
                **generate_kwargs,                   # forward any extra options to generate()
            )
            results.append({"job": job, "html": str(html_path), "pdf": str(pdf_path),
                            "meta": meta, "salary_text": full["salary_text"]})
        except Exception as e:
            on_progress(f"    failed: {e}")          # one job failing shouldn't stop the rest
            results.append({"job": job, "error": str(e)})

    return results


def search(user_id: int, generate_resumes: bool = False, boards_limit: int = None,
           max_resumes: int = None, on_progress=_noop, **generate_kwargs) -> dict:
    """Full Phase 2 run for one user: find their matches (each stored + embedded +
    AI-scored as it passes — see find_matches), optionally generate resumes for them.

    Returns {"matches": [...], "generated": [...]}. With generate_resumes=False no
    resumes are made (the scan itself still spends a few cents on OpenAI embedding +
    scoring calls per found job); set True to also produce tailored resumes."""
    matches = find_matches(user_id, boards_limit=boards_limit, on_progress=on_progress)
    generated = []
    if generate_resumes and matches:
        generated = generate_for_matches(user_id, matches, max_resumes=max_resumes,
                                         on_progress=on_progress, **generate_kwargs)
    return {"matches": matches, "generated": generated}


# This block runs only if you launch this file directly: `python -m src.job_search`.
if __name__ == "__main__":
    from src.current_user import current_user_id
    # Dry run by default: list matches, no resume generation, no tokens spent.
    result = search(current_user_id(), on_progress=print)  # log progress to the screen
    print(f"\n{len(result['matches'])} matching job(s):")
    for m in result["matches"]:
        flags = f"  [{'; '.join(m['match']['flags'])}]" if m["match"]["flags"] else ""
        print(f"  • {m['company']} — {m['title']} @ {m['location']}{flags}")
