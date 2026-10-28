# src/pipeline.py — the "conductor" for Phase 1 (turning your résumé + a job into a
# tailored résumé). Both the command-line tool (main.py) and the web app (app.py) call
# generate() below, so all the important logic lives here in ONE place.
#
# Read this file alongside docs/WORKFLOW.md ("Workflow 1") — the steps line up.

import hashlib                                          # short signature of the run's inputs (checkpointing)
import json                                             # read/write checkpoint files
import shutil                                           # remove the checkpoint folder on success
import time                                             # measure how long a generation takes
from pathlib import Path                               # filesystem paths for the checkpoint folder

# ── Imports: each line grabs a function from one of the small worker modules ──────────
from src.base_resume import resume_source_text, structured_profile  # base résumé: text + structured facts
from src.job_parser import parse_job_posting        # URL/file/text → clean job-posting text
from src.tailoring import tailor_resume, polish_resume, is_embedding_model  # AI: rewrite/fix résumé; embedding-model guard
from src.evidence import extract_evidence, format_evidence  # AI: map posting needs → real evidence (pre-pass)
from src.critic import score_resume                 # ask the AI to audit + score the draft
from src.external_critic import external_critique, has_key as external_has_key  # ChatGPT tips
from src.renderer import render_html, render_pdf, count_pdf_pages, OUTPUT_DIR, _slug  # render + path helpers
from src.tracker import log_application             # record the result in the jobs store (Postgres)

# Default settings. These are "module-level constants" (UPPER_CASE by convention). main.py
# imports them so its --help text and the pipeline agree on the same defaults.
DEFAULT_MAX_PAGES = 2       # aim for a résumé no longer than 2 pages
DEFAULT_MIN_PAGES = 2       # …and no shorter: 1 page for a long career history means content was dropped
DEFAULT_MAX_ATTEMPTS = 2    # try at most 2 times to get an accepted result (extra retries rarely helped and doubled cost)
DEFAULT_MIN_SCORE = 85      # the critic must score the résumé at least 85/100 to "accept" it
DEFAULT_MAX_POLISH = 2      # after the re-tailor loop, surgically fix residual flags up to N times


# ── Checkpointing: let a failed generation RESUME instead of restarting from scratch ──────
# Each expensive stage (the evidence map, the Pass-1 draft, the external suggestions, the latest
# Pass-2/polish draft) is saved to a per-résumé .checkpoints/<slug>/ folder AS SOON AS it's
# produced. If a later stage blows up (e.g. a truncated critic reply), the finished stages are
# already on disk, so a re-run with allow_resume=True reuses them and picks up where it stopped
# instead of re-paying for every earlier Claude call. A SUCCESSFUL generation clears the folder,
# so a later fresh regenerate always starts clean. All writes are best-effort — a checkpoint
# failure never breaks a generation. A checkpoint set is reused only when it's recent AND its
# SIGNATURE (per-stage LLMs, which optional stages ran, and a hash of the résumé + job text)
# matches the new run, so changing the LLM, the base résumé, or the posting forces a fresh run.
CKPT_MAX_AGE_SEC = 6 * 3600           # ignore checkpoints older than 6h (inputs may have moved on)


def _ckpt_dir(out_dir, company, role) -> Path:
    """The per-résumé checkpoint folder, co-located with its generated files."""
    base = Path(out_dir) if out_dir else OUTPUT_DIR
    return base / ".checkpoints" / _slug(company, role)


def _ckpt_save(cdir: Path, name: str, data) -> None:
    """Persist one stage's output as <name>.json (best-effort; never raises)."""
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _ckpt_load(cdir: Path, name: str):
    """Load one stage's saved output, or None if it's absent or unreadable."""
    try:
        f = cdir / f"{name}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except Exception:
        return None


def _ckpt_clear(cdir: Path) -> None:
    """Delete the whole checkpoint folder (on success, or before a fresh run)."""
    shutil.rmtree(cdir, ignore_errors=True)


def _ckpt_resumable(cdir: Path, signature: dict) -> bool:
    """True when a recent checkpoint set exists whose signature matches this run."""
    info = _ckpt_load(cdir, "meta")
    return bool(info and info.get("signature") == signature
                and time.time() - info.get("saved_at", 0) <= CKPT_MAX_AGE_SEC)


def generate(user_id: int, company: str, role: str, source: str,
             max_pages: int = DEFAULT_MAX_PAGES,
             min_pages: int = DEFAULT_MIN_PAGES,
             max_attempts: int = DEFAULT_MAX_ATTEMPTS,
             max_polish: int = DEFAULT_MAX_POLISH,
             min_score: int = DEFAULT_MIN_SCORE,
             use_external_critic: bool = True,
             use_evidence: bool = True,
             allow_resume: bool = False,
             job_text: str = None,
             job_id: str = None,
             llm: str = "claude",
             stage_llms: dict = None,
             out_dir=None,
             on_progress=None):
    """Two-pass Phase 1 pipeline with an external reviewer + internal guardrails.

    Each of the four LLM-driven stages can run on its own backend so you can, e.g., generate
    on Claude but check on the local model when you're low on ChatGPT credits. `stage_llms`
    is an optional dict with any of these keys (each a backend: "claude" | "chatgpt" | a local
    model key like "local_qwen", plus "off" for `external` to skip it):
        "generate"   — Stage 1, the initial draft (tailor).
        "external"   — Stage 2, the external fit-suggestions review.
        "check"      — Stage 3, the internal truthfulness critic.
        "regenerate" — Stage 4, the feedback-applying re-tailor loop.
    Any key not supplied falls back to `llm` (so callers that pass a single `llm=` — the CLI,
    the tests — behave exactly as before). `external` additionally falls back to ChatGPT when
    `use_external_critic` is on, preserving the historical default.

    PASS 1: tailor the résumé once (the baseline draft).
    EXTERNAL REVIEW: send the draft + job description to ChatGPT (OpenAI), which suggests
        truthful improvements to better fit the role.
    PASS 2: re-tailor applying ChatGPT's suggestions, then run the internal checks — the
        fidelity critic (catches any fabrication/inflation/posting-mirroring the suggestions
        introduced) and the page-length check. If not accepted, iterate using the internal
        critic's feedback, up to `max_attempts`, always keeping the best (truthfulness-first)
        attempt so the loop terminates.

    `source` may be a URL, a file path, or raw pasted job text.
    `job_text`: optional pre-fetched job description. When provided (e.g. Phase 2 already
        pulled the clean text from an ATS API), it's used as-is and `source` is NOT
        re-fetched — but `source` is still recorded in the log, so pass the real job URL
        there for a clickable, dedupe-able history entry.
    `out_dir`: optional folder to write the HTML/PDF/JSON into (default: the renderer's
        OUTPUT_DIR). Scheduled (cron) runs pass a dated sub-folder so each day's batch is
        grouped; the PDF is always written next to its HTML.
    `on_progress` is an optional callback(str) for status messages.
    Returns (html_path, pdf_path, meta).
    """
    # A tiny helper so the rest of the function can just call log("…") to report progress.
    # If the caller passed an on_progress function, forward the message to it; otherwise do
    # nothing. (Defining a function inside a function is normal in Python.)
    def log(msg):
        if on_progress:
            on_progress(msg)

    # Stopwatch for the whole generation — recorded in the tracker and shown on Stats.
    start_time = time.time()

    # A résumé can't be required to be both ≥3 and ≤2 pages — the ceiling wins.
    min_pages = min(min_pages, max_pages)

    # ── Resolve which backend runs each of the four LLM stages ────────────────────────
    # Each stage falls back to `llm` when not overridden in stage_llms. The external review's
    # fallback is ChatGPT (when enabled) to preserve the historical default cross-check.
    sl = stage_llms or {}
    gen_llm = sl.get("generate", llm)                       # Stage 1 — initial draft
    check_llm = sl.get("check", llm)                        # Stage 3 — truthfulness critic
    regen_llm = sl.get("regenerate", llm)                   # Stage 4 — feedback re-tailor
    external_default = "chatgpt" if use_external_critic else "off"
    external_llm = sl.get("external", external_default)     # Stage 2 — external fit review
    # "off" (or the master switch being off) means skip Stage 2 entirely.
    run_external = use_external_critic and external_llm not in ("off", None, "")

    # ── Guardrail: refuse an EMBEDDING model on any writing/critic stage ─────────────────
    # An embedding model (nomic-embed-text, mxbai-embed, …) only makes vectors — it can't draft
    # or audit a résumé. Picked for a stage it silently produces nothing, and the run ships an
    # UNAUDITED, fabrication-riddled résumé with a 0.0 score (this exact mistake put tools the
    # candidate never used into a live résumé). The stage pickers
    # now hide embedding models, but an old saved selection (localStorage / a stored <stage>_llm)
    # can still carry one, so we hard-fail FAST here — before any LLM cost — with a clear message.
    _stages_to_check = [("generate", gen_llm), ("check", check_llm), ("regenerate", regen_llm)]
    if run_external:
        _stages_to_check.append(("external", external_llm))
    for _stage, _llm in _stages_to_check:
        if is_embedding_model(_llm):
            raise ValueError(
                f"The '{_stage}' stage is set to '{_llm}', which is an embedding model — it can "
                f"only turn text into vectors, not read, write, or check a résumé. Pick a chat "
                f"model for this stage (embedding models are used only for the search/matching "
                f"backend, not résumé generation)."
            )

    # ── Gather the two inputs the AI needs: your real résumé, and the job posting ──────
    # Prefer this user's structured DB profile; fall back to their uploaded .docx.
    resume_text = resume_source_text(user_id)
    # The same profile in structured form — the tailor step locks the draft's FACTS (skills,
    # education, certs, contact, each role's company/title/dates) to it so a weak model can't
    # drop or alter them. None for a .docx-only user (nothing structured → locking is skipped).
    base_profile = structured_profile(user_id)
    # If the caller already has the job description text (Phase 2 does), use it directly;
    # otherwise fetch/parse it from `source`. "X if CONDITION else Y" is a one-line if.
    job_posting = job_text if job_text is not None else parse_job_posting(source)

    # ── CHECKPOINT SETUP: decide whether we're RESUMING an interrupted run ──────────────
    # The signature ties a checkpoint set to this exact run: the four stage backends, whether
    # the optional stages ran, and a hash of the résumé + posting text. If any of those changed
    # since the interrupted run, the old checkpoints are ignored and we start fresh.
    cdir = _ckpt_dir(out_dir, company, role)
    signature = {
        "generate": gen_llm, "external": external_llm if run_external else "off",
        "check": check_llm, "regenerate": regen_llm, "evidence": bool(use_evidence),
        "inputs": hashlib.sha256(
            ((resume_text or "") + "\x00" + (job_posting or "")).encode("utf-8")).hexdigest()[:16],
    }
    resuming = allow_resume and _ckpt_resumable(cdir, signature)
    if resuming:
        log("Resuming the previous run — reusing the steps it already finished…")
    else:
        _ckpt_clear(cdir)                                  # discard any stale/mismatched checkpoints
        _ckpt_save(cdir, "meta", {"saved_at": time.time(), "signature": signature})

    # ── EVIDENCE PRE-PASS: map the posting's needs to the candidate's REAL evidence ────
    # Before tailoring, split retrieval out from writing: one call reads the posting + base
    # résumé and returns a grounding map (each requirement → true matching evidence + a ranked
    # must/should/optional list). The tailor step then WRITES from that map instead of searching
    # and composing at once, so the strongest true matches surface near the top and honest gaps
    # are named. It adds no facts. Fully optional — any failure degrades to tailoring without it.
    evidence_text = _ckpt_load(cdir, "evidence") if resuming else None
    if evidence_text is not None:
        log("  Reusing the evidence map from the interrupted run.")
    elif use_evidence:
        try:
            log("Evidence pre-pass — mapping the posting's needs to your real experience…")
            evidence_text = format_evidence(
                extract_evidence(resume_text, job_posting, company, role, llm=gen_llm)) or None
            if evidence_text:
                log("  Evidence map built.")
                _ckpt_save(cdir, "evidence", evidence_text)  # resume point: skip this call next time
        except Exception as e:
            log(f"  Evidence pre-pass failed ({e}); continuing without it.")

    # ── PASS 1: produce a single baseline draft ───────────────────────────────────────
    pass1 = _ckpt_load(cdir, "pass1") if resuming else None
    if pass1 is not None:
        log("Pass 1 — reusing the saved draft from the interrupted run.")
    else:
        log("Pass 1 — initial tailoring…")
        pass1 = tailor_resume(resume_text, job_posting, company, role, llm=gen_llm,
                              base_profile=base_profile, evidence=evidence_text)  # AI → résumé JSON (facts locked to base)
        _ckpt_save(cdir, "pass1", pass1)                   # resume point
    p1_html = render_html(pass1, company, role, out_dir)    # JSON → an HTML file on disk
    p1_pages = count_pdf_pages(render_pdf(p1_html))         # HTML → PDF, then count its pages
    log(f"  Pass 1 draft: {p1_pages} page(s).")

    # ── EXTERNAL REVIEW (Stage 2): optional improvement suggestions ───────────────────
    # On resume, reuse the saved external result (the saved dict's presence means the stage
    # already ran — even if it produced no suggestions, so we don't re-call it).
    suggestions = None      # will hold the reviewer's text suggestions, or stay None if skipped
    ext_saved = _ckpt_load(cdir, "external") if resuming else None
    if ext_saved is not None:
        suggestions = ext_saved.get("suggestions")
        log("  Reusing the external review from the interrupted run.")
    elif run_external:
        # ChatGPT is the only backend that needs an API key; Claude/Local don't.
        if external_llm == "chatgpt" and not external_has_key():
            log("  No ChatGPT API key (CHATGPT_API_KEY) found; skipping external review.")
        else:
            # This may call an external service over the network, which can fail (no key,
            # timeout, rate limit). We wrap it in try/except so a failure here NEVER stops
            # the whole run — we just continue with the internal checks only.
            try:
                log(f"  Sending draft + job description to the external reviewer "
                    f"({external_llm}) for improvement suggestions…")
                suggestions = external_critique(pass1, job_posting, company, role,
                                                llm=external_llm)
                log("  External review suggestions received.")
            except Exception as e:
                log(f"  External review failed ({e}); continuing with internal checks only.")
        _ckpt_save(cdir, "external", {"suggestions": suggestions})  # resume point (marks stage done)

    # ── PASS 2: apply the suggestions, then enforce truthfulness + length in a loop ───
    # A single "score this draft" step shared by the re-tailor loop AND the polish loop below:
    # render it, count its pages, run the critic, and package it as a ranked attempt dict. The
    # ranking `key` is a tuple compared left-to-right (bigger is better). Truthfulness comes
    # first: accepted? then FEWER fidelity issues (negated count), then fewer AI-style issues,
    # then acceptable length, then higher score, then fewer pages. See WORKFLOW.md.
    def evaluate(tailored: dict) -> dict:
        html_path = render_html(tailored, company, role, out_dir)   # JSON → HTML on disk
        pages = count_pdf_pages(render_pdf(html_path))              # HTML → PDF, then count pages
        log("  Reviewing for fidelity + fit…")
        # The critic is a quality GATE, not the producer — the draft above is already a complete,
        # rendered résumé. A critic call that can't return parseable JSON (e.g. a reply truncated
        # at the token cap → "Unbalanced braces", even after call_llm_json's own retry) must NOT
        # throw away that finished draft. Treat this attempt as UNAUDITED: keep it as a viable
        # best-effort (it renders fine) but never let it be marked "accepted" (score 0, no
        # clean bill of health), so the loop still prefers any attempt the critic DID score, and
        # a total-critic-failure run ships the draft rather than failing outright.
        try:
            review = score_resume(tailored, job_posting, resume_text, company, role, llm=check_llm)
        except Exception as e:
            log(f"  ⚠ Critic could not score this draft ({e}); keeping it UNAUDITED.")
            review = {"overall_score": 0, "fidelity_issues": [], "overstatement_notes": [],
                      "style_issues": [], "revisions": [], "strengths": [], "unaudited": True}
        score = review["overall_score"]        # int 0–100
        fidelity = review["fidelity_issues"]   # Tier A: hard truthfulness problems (block accept)
        overstatement = review.get("overstatement_notes", [])  # Tier B: true-but-broad; do NOT block
        style = review["style_issues"]         # list; each reads AI-written / posting-echo
        # ACCEPT needs length in range (fits the ceiling AND the floor — a 1-page result for a
        # long career means real content was dropped) AND score ≥ floor AND zero fidelity/style.
        # Overstatement notes are legitimate tailoring (true claims stated broadly) — they are
        # surfaced but deliberately do NOT block acceptance, so the loop stops chasing them.
        fits_length = pages <= max_pages
        too_short = pages < min_pages
        length_ok = fits_length and not too_short
        accepted = length_ok and score >= min_score and not fidelity and not style
        log(f"  → {pages} page(s), score {score}/100, {len(fidelity)} fidelity + "
            f"{len(style)} AI-style + {len(overstatement)} overstatement note(s)"
            f"{' — ACCEPTED' if accepted else ''}")
        # Ranking (bigger better): truthfulness first, then style, then length/score; fewer
        # overstatement notes is only a weak late tiebreak so it never trades away real quality.
        key = (accepted, -len(fidelity), -len(style), length_ok, score, -len(overstatement), -pages)
        return {"tailored": tailored, "pages": pages, "review": review, "key": key,
                "fits_length": fits_length, "too_short": too_short, "accepted": accepted}

    # Build the instructions for the first pass-2 attempt: ChatGPT's tips + a length note.
    feedback = _pass2_feedback(suggestions, p1_pages, max_pages, min_pages)
    best = None             # will hold the best attempt we've seen so far (a dict)
    last_tailored = None    # the most recent attempt's data (used to avoid re-rendering)
    attempt = 0

    # Try up to max_attempts times. range(1, N+1) gives 1, 2, … N.
    for attempt in range(1, max_attempts + 1):
        # A friendly label for the log: the first attempt applies ChatGPT's tips; later ones
        # are "refining" based on the critic's feedback from the previous attempt.
        # On a resumed run, the first attempt reuses the draft the interrupted run had already
        # tailored (saved just before the critic — exactly where a truncated-critic failure hits),
        # so we re-run only the critic instead of paying for the tailor call again.
        saved_draft = _ckpt_load(cdir, "draft") if (resuming and attempt == 1) else None
        if saved_draft is not None:
            log(f"Pass 2 — reusing the saved draft from the interrupted run "
                f"(attempt {attempt}/{max_attempts})…")
            tailored = saved_draft
        else:
            # A friendly label for the log: the first attempt applies ChatGPT's tips; later ones
            # are "refining" based on the critic's feedback from the previous attempt.
            label = "applying ChatGPT improvements + checks" if attempt == 1 else "refining"
            log(f"Pass 2 — {label} (attempt {attempt}/{max_attempts})…")
            # Ask the AI to tailor again, this time WITH the feedback to apply, then score it.
            tailored = tailor_resume(resume_text, job_posting, company, role,
                                     revision_feedback=feedback, llm=regen_llm,
                                     base_profile=base_profile, evidence=evidence_text)
            _ckpt_save(cdir, "draft", tailored)            # resume point: skip re-tailoring if the critic fails
        last_tailored = tailored
        cand = evaluate(tailored)
        if best is None or cand["key"] > best["key"]:
            best = cand

        if cand["accepted"]:
            break       # perfect — stop trying

        # Not accepted: if we have attempts left, build feedback for the next one from the
        # critic's concrete fixes + any length problem.
        if attempt < max_attempts:
            feedback = _build_feedback(cand["pages"], max_pages, min_pages, cand["review"])

    # ── POLISH: surgically remove any residual truthfulness / AI-style flags ───────────
    # The re-tailor loop above rewrites the whole résumé each attempt and often can't fully
    # clear fidelity/style flags — a full rewrite keeps reintroducing posting-echo wording. If
    # the best attempt STILL has flags but its length is fine (polishing can't fix length, and
    # a fresh rewrite would risk the flags again), make targeted MINIMAL edits to fix exactly
    # those flags, then re-check. This is what makes detected problems actually disappear from
    # the final résumé instead of shipping in the "best effort".
    polish_round = 0
    while (max_polish > 0 and not best["accepted"]
           and best["fits_length"] and not best["too_short"]
           and (best["review"]["fidelity_issues"] or best["review"]["style_issues"])
           and polish_round < max_polish):
        polish_round += 1
        r = best["review"]
        log(f"Polish {polish_round}/{max_polish} — surgically fixing "
            f"{len(r['fidelity_issues'])} truthfulness + {len(r['style_issues'])} "
            f"AI-style issue(s) with minimal edits…")
        try:
            polished = polish_resume(best["tailored"], resume_text,
                                     r["fidelity_issues"], r["style_issues"],
                                     revisions=r["revisions"], llm=regen_llm,
                                     base_profile=base_profile)
        except Exception as e:
            log(f"  Polish step failed ({e}); keeping the best pre-polish attempt.")
            break
        _ckpt_save(cdir, "draft", polished)                # resume point: re-check this draft if the critic fails
        last_tailored = polished
        cand = evaluate(polished)
        # Only adopt the polished draft if it's genuinely better (fewer flags first) — a polish
        # that backfired is discarded, so this can never make the résumé worse.
        if cand["key"] > best["key"]:
            best = cand
        if best["accepted"]:
            break

    # ── After the loops: make the files on disk the BEST attempt, not whatever was rendered
    # last. Each attempt renders to the SAME path, so disk currently holds the most recent
    # attempt — which may not be the best. Re-render best unconditionally (a render is cheap,
    # no LLM call) so the saved HTML/PDF always match what we report.
    if best["tailored"] is not last_tailored:
        log(f"Keeping the best attempt ({best['pages']} pages, score {best['review']['overall_score']}).")
    html_path = render_html(best["tailored"], company, role, out_dir)
    pdf_path = render_pdf(html_path)

    # ── Record the result in the tracker (the jobs table in Postgres) ──────────────────
    review = best["review"]
    match_score = round(review["overall_score"] / 10, 1)   # convert the /100 score to a /10 "match"
    gen_seconds = round(time.time() - start_time)           # whole seconds this generation took
    # The backend used for each of the four stages (external is "off" when it was skipped),
    # recorded so the Stats page can show the model behind every part of the generation. The
    # row's single "LLM" is the Stage-4 re-tailor — the backend whose best attempt is on disk.
    stage_llms_used = {"generate": gen_llm, "external": external_llm if run_external else "off",
                       "check": check_llm, "regenerate": regen_llm}
    # The critic's full report is persisted with the résumé (resume_findings) so the UI
    # can always show WHY the score is what it is — not just the number.
    critic_findings = {"fidelity": review["fidelity_issues"],
                       "overstatement": review.get("overstatement_notes", []),
                       "style": review["style_issues"],
                       "revision": review["revisions"], "strength": review["strengths"]}
    log_application(user_id, company, role, source, html_path, pdf_path, job_id=job_id,
                    match_score=match_score, gen_seconds=gen_seconds,
                    stage_llms=stage_llms_used, critic_findings=critic_findings)

    # The generation finished and is saved — discard its checkpoints so a later fresh regenerate
    # starts clean (and disk doesn't accumulate). A FAILURE never reaches here, so its checkpoints
    # survive for the next allow_resume run to pick up.
    _ckpt_clear(cdir)

    # If we never reached "accepted", explain why in the log (so you know what to fix).
    if not best["accepted"]:
        why = []                                    # collect the reasons into a list
        if review.get("unaudited"):
            why.append("the critic couldn't score it — shipped UNAUDITED")
        if not best["fits_length"]:
            why.append(f"{best['pages']} > {max_pages} pages")
        if best["too_short"]:
            why.append(f"only {best['pages']} page(s) — under the {min_pages}-page floor")
        if review["overall_score"] < min_score:
            why.append(f"score {review['overall_score']} < {min_score}")
        if review["fidelity_issues"]:
            why.append(f"{len(review['fidelity_issues'])} fidelity issue(s)")
        if review["style_issues"]:
            why.append(f"{len(review['style_issues'])} AI-style issue(s)")
        polished_note = f" + {polish_round} polish round(s)" if polish_round else ""
        log(f"⚠ Not fully accepted after {attempt} pass-2 attempt(s){polished_note} "
            f"({'; '.join(why)}); kept the best.")

    # ── Bundle up the stats to hand back to the caller (main.py / app.py) ─────────────
    meta = {
        "pages": best["pages"],
        "score": review["overall_score"],
        "attempts": attempt,
        "polish_rounds": polish_round,                     # surgical-cleanup rounds run after pass 2
        "fit": best["fits_length"],
        "too_short": best["too_short"],
        "accepted": best["accepted"],
        "unaudited": review.get("unaudited", False),       # True if the critic couldn't score the final draft
        "fidelity_issues": review["fidelity_issues"],
        "overstatement_notes": review.get("overstatement_notes", []),
        "style_issues": review["style_issues"],
        "strengths": review["strengths"],
        "critic_findings": critic_findings,                # {kind: [finding,…]} for the score tooltip
        "evidence_used": evidence_text is not None,        # True if the evidence pre-pass ran + produced a map
        "resumed": resuming,                               # True if this run continued an interrupted one
        "external_review_used": suggestions is not None,   # True if the external review ran
        "gen_seconds": gen_seconds,                        # how long this run took (whole seconds)
        "llm": regen_llm,                                  # backend of the final saved résumé
        "stage_llms": stage_llms_used,                     # per-stage backends used
    }
    return html_path, pdf_path, meta


def _pass2_feedback(suggestions, pages, max_pages, min_pages) -> str:
    """Compose pass-2 instructions: ChatGPT's suggestions (truthfully filtered) + length.

    The leading underscore in the name (_pass2_feedback) is a Python convention meaning
    "private helper" — intended for use only inside this file, not imported elsewhere."""
    parts = []      # we'll join these pieces into one instruction string at the end
    if suggestions:
        # Wrap ChatGPT's raw suggestions with a strong reminder to apply them ONLY where the
        # base résumé can truthfully support them (ChatGPT sometimes over-reaches).
        parts.append(
            "An external reviewer (ChatGPT) suggested the following improvements to better fit "
            "the job. Apply each suggestion ONLY where it can be done truthfully using the base "
            "resume's real content. IGNORE any suggestion that would require inventing or "
            "exaggerating experience, claiming skills/audiences not in the base resume, or "
            "copying the posting's distinctive wording. Where a suggestion targets a genuine gap, "
            "leave it as an honest gap.\n\nSUGGESTIONS:\n" + suggestions
        )
    if pages > max_pages:
        parts.append(f"LENGTH: keep it within {max_pages} pages (the first draft was {pages}).")
    elif pages < min_pages:
        parts.append(_too_short_feedback(pages, min_pages, max_pages))
    # "\n\n".join(parts) glues the pieces with blank lines between them. If there are no
    # parts at all, return None (meaning "no feedback").
    return "\n\n".join(parts) if parts else None


def _too_short_feedback(pages, min_pages, max_pages) -> str:
    """The instruction for a draft that came out UNDER the page floor — real content was
    dropped, so tell the tailor to restore it (never to pad or invent)."""
    return (
        f"LENGTH: the previous draft filled only {pages} page(s) — far too thin; the finished "
        f"resume must fill {min_pages} full page(s) (up to {max_pages}). This means real "
        f"content was dropped. Restore it: keep 8-12 detailed bullets EACH for the two most "
        f"recent/relevant roles, include EVERY position from the base resume (older ones "
        f"condensed to 1-3 bullets, never omitted), and keep all real quantified achievements. "
        f"Expand ONLY with content that is actually in the base resume — never pad or invent."
    )


def _build_feedback(pages, max_pages, min_pages, review) -> str:
    """Compose the revision instructions for the NEXT attempt from length + critic output."""
    parts = []
    if pages > max_pages:
        parts.append(
            f"LENGTH: the previous draft rendered to {pages} pages but MUST fit within "
            f"{max_pages} pages. Condense older/less-relevant roles to fewer bullets, merge "
            f"overlapping bullets, and tighten wording — without dropping role-critical content."
        )
    elif pages < min_pages:
        parts.append(_too_short_feedback(pages, min_pages, max_pages))
    if review["fidelity_issues"]:
        # Turn the list of problems into one semicolon-separated instruction to fix them.
        parts.append(
            "TRUTHFULNESS — remove or correct each of these (do NOT embellish them): "
            + "; ".join(review["fidelity_issues"])
        )
    if review["style_issues"]:
        parts.append(
            "AI-STYLE — rewrite each of these so it reads like the candidate's own plain, "
            "specific wording (no résumé-bot clichés, no posting-echo phrasing): "
            + "; ".join(review["style_issues"])
        )
    if review["revisions"]:
        parts.append("IMPROVEMENTS (using only real base-resume content): "
                     + "; ".join(review["revisions"]))
    return "\n".join(parts)
