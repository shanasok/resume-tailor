# src/fit_verdict.py — ChatGPT reads job postings and scores each one 1-10.
#
# The embedding Match % (see src/embeddings.py) is cheap but it's just a number. This
# module adds the explanation layer: ChatGPT reads a posting side-by-side with the base
# résumé and writes a 1-10 fit score plus a one-line reason — stored on the job row
# (ai_fit_score / ai_fit_reason) and shown in the AI Fit column. The scan scores each
# job the moment it's stored (job_search._store_and_score calls verdict_for);
# run_stored_verdicts is the on-demand backfill for anything still unscored.
#
# Deliberately separate from external_critic.py: that module reviews résumé DRAFTS
# during generation; this one screens raw job POSTINGS. Different input, prompt, and
# moment in the workflow.
#
# Cost: one gpt-4o call per job at ~2-3k input tokens each — roughly $0.25-0.40 per
# 25 jobs, so a big first scan (500+ new jobs) runs a few dollars and adds real time
# (a few seconds per job). Later scans only score NEW jobs, so they're cheap again.
# Skips itself cleanly when no OpenAI key.
import time                                          # polite pause before retrying a board fetch

from src import db                                   # store verdicts; find unscored jobs
from src import embeddings                           # similarity ranking + backfill embeddings
from src.job_parser import fetch_job_description     # recover an OLD job's posting text
from src import base_resume                          # the candidate's base résumé (DB profile → text)
from src.tailoring import call_llm_json              # shared LLM plumbing (chatgpt backend)

# Keep the posting bounded so one giant description can't blow up the call cost.
_MAX_DESCRIPTION_CHARS = 15_000

_PROMPT = """You are screening job postings for one specific candidate. Rate how strong \
a fit the candidate is for THIS job on a 1-10 scale (10 = apply immediately, 1 = waste \
of time). Judge the posting's required experience against the resume, seniority match, \
domain overlap, and hard blockers (security clearance, licenses, relocation, unrelated \
specialties). Be blunt — an honest 4 beats a polite 7.
Reply as JSON only: {{"score": <integer 1-10>, "reason": "<one sentence, max 25 words>"}}

CANDIDATE RESUME:
{resume}

JOB POSTING ({company} — {title}):
{description}"""


def _noop(_msg):
    pass


def verdict_for(resume_text: str, company: str, title: str, description: str,
                llm: str = "chatgpt") -> tuple[int, str]:
    """One LLM call: (score 1-10, one-line reason) for this posting vs this résumé.
    `llm` picks the backend (chatgpt default — the historical scorer)."""
    reply = call_llm_json(_PROMPT.format(
        resume=resume_text,
        company=company or "?",
        title=title or "?",
        description=(description or "")[:_MAX_DESCRIPTION_CHARS],
    ), llm=llm)
    score = max(1, min(10, int(reply["score"])))     # clamp — the DB CHECK requires 1-10
    return score, str(reply.get("reason", "")).strip()


def _recover_description(row: dict, on_progress=_noop) -> str:
    """The posting text for a stored job: the stored copy when we have it (instant, no
    network), otherwise a polite re-fetch from the board — retried once after a pause,
    because boards rate-limit bursts and a single retry often gets through. A recovered
    text is stored so the next run never has to fetch it again."""
    if row.get("description"):
        return row["description"]
    for attempt in (1, 2):
        description, _url = fetch_job_description(row["job_id"], row["job_url"])
        if description:
            with db.connect() as conn:
                db.upsert_job_description(conn, row["job_id"], description)
            return description
        if attempt == 1:
            time.sleep(2.0)                          # back off, then try once more
    return ""


def run_stored_verdicts(user_id: int, top_n: int = None, on_progress=_noop,
                        llm: str = "chatgpt") -> int:
    """On-demand backfill (the "Score fits" button): AI-score EVERY job THIS user tracks
    that has no verdict yet (pass top_n to cap the embedded batch), and (re)embed any
    row whose vector is missing OR from a retired model (an EMBEDDING_BACKEND switch).
    Three kinds of row:
      1. Embedded (active model) but unverdicted — scored from their STORED description,
         best similarity first, falling back to a board re-fetch for older rows.
      2. No usable embedding — never embedded, or embedded by a retired model: recover
         text → store it → embed with the ACTIVE model → score if unscored.
      3. Already-scored rows with a stale vector — re-embedded only; verdicts judge
         TEXT, not vectors, so they survive a backend switch.
    Jobs whose text can't be recovered (pasted-text manual rows, dead postings) are
    skipped with a log line. Returns how many jobs were scored."""
    if not embeddings.available():
        return 0
    resume_vec = embeddings.cached_base_resume_embedding(user_id)
    if resume_vec is None:                           # first run OR the model just changed
        try:
            resume_vec = embeddings.base_resume_embedding(user_id)
        except (embeddings.EmbeddingsError, FileNotFoundError) as e:
            on_progress(f"(can't rank without the base résumé: {e})")
            return 0

    model = embeddings.active_model()
    with db.connect() as conn:                       # top_n=None → all unscored jobs
        embedded = db.fetch_top_unverdicted(conn, user_id, resume_vec, model, top_n)
        stale = db.fetch_unembedded_for_model(conn, user_id, model)
    if not embedded and not stale:
        on_progress("Every stored job already has an AI fit score.")
        return 0

    resume_text = base_resume.resume_source_text(user_id)
    scored = 0
    needs_embedding = {r["job_id"] for r in stale}
    on_progress(f"AI-scoring {len(embedded)} stored job(s)"
                + (f" + {len(stale)} job(s) needing a fresh embedding…"
                   if stale else "…"))
    for row in embedded + stale:
        description = _recover_description(row, on_progress)
        if not description:
            on_progress(f"  (skipped {row['company']} — {row['role']}: "
                        "posting text can't be recovered)")
            continue
        # A row without a usable vector gets one now (missing, or from a retired
        # model), so it (re)gains a Match % on the home page.
        if row["job_id"] in needs_embedding:
            try:
                vec = embeddings.embed_texts([f"{row['role']}\n{row['company']}\n\n"
                                              f"{description}"])[0]
                with db.connect() as conn:
                    db.upsert_job_embedding(conn, row["job_id"], vec, model)
            except embeddings.EmbeddingsError as e:
                on_progress(f"  (couldn't embed {row['company']} — {row['role']}: {e})")
        if row.get("ai_fit_score") is not None:      # stale-vector row, already scored —
            continue                                 # verdicts judge text; keep it
        try:
            score, reason = verdict_for(resume_text, row["company"], row["role"],
                                        description, llm=llm)
            with db.connect() as conn:
                db.set_fit_verdict(conn, user_id, row["job_id"], score, reason)
            scored += 1
            on_progress(f"  {score}/10 {row['company']} — {row['role']}")
        except Exception as e:
            on_progress(f"  (couldn't score {row['company']} — {row['role']}: {e})")
    return scored
