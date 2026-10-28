# src/external_critic.py — the "second opinion." It sends the tailored résumé draft + the
# job description to a reviewer LLM and asks for concrete, truthful suggestions to make the
# résumé a stronger fit. This is the EXTERNAL REVIEW step (Stage 2) in the Phase 1 pipeline.
#
# Which LLM? Historically this was always ChatGPT (OpenAI), chosen so a DIFFERENT model would
# cross-check the internal Claude critic (critic.py, the truthfulness gatekeeper). The backend
# is now selectable per run — pick ChatGPT to keep that cross-check, or Claude/Local if you
# don't have ChatGPT credits. The reviewer's output is prose, so it goes through call_llm_text.

from src.tailoring import call_llm_text, chatgpt_available  # shared LLM plumbing

# The system prompt: tells ChatGPT exactly how to behave — give truthful, concrete edit
# suggestions, and NEVER suggest inventing experience.
_SYSTEM = """You are an expert resume reviewer, ATS specialist, and former hiring manager. You will
be given a candidate's resume (already tailored to a job) and the target job description. Suggest
concrete, specific improvements that make the resume a STRONGER FIT for THIS job WHILE remaining
completely truthful.

CRITICAL TRUTHFULNESS RULE: Never suggest inventing or exaggerating experience, skills, employers,
audiences, metrics, or claims that are not already supported by the resume. Good suggestions
reframe, reorder, re-emphasize, or surface real content the candidate already has. If the role
wants something the resume doesn't show, say plainly that it's an honest gap to leave alone rather
than papering over it.

Give a concise, numbered list of the highest-impact, concrete edits (what to change and where).
Do NOT rewrite the whole resume — give actionable edit instructions a writer can apply."""


def has_key() -> bool:
    """True if a ChatGPT/OpenAI API key is configured. The pipeline checks this before
    running the external review WHEN the reviewer is ChatGPT (Claude/Local need no key)."""
    return chatgpt_available()


def _resume_to_text(d: dict) -> str:
    """Convert the résumé data dict into a readable plain-text version to show ChatGPT.
    `d` is the résumé; we build up a list of lines and join them at the end."""
    lines = [d.get("name", ""), ""]                  # start with the name and a blank line
    # For each section, only add it if it has content. "lines += [...]" appends several lines.
    if d.get("summary"):
        lines += ["PROFESSIONAL SUMMARY", d["summary"], ""]
    if d.get("competencies"):
        # " • ".join(list) glues the competency phrases together with " • " between them.
        lines += ["CORE COMPETENCIES", " • ".join(d["competencies"]), ""]
    lines.append("PROFESSIONAL EXPERIENCE")
    for job in d.get("experience", []):              # each job is a dict with company/title/bullets
        lines.append(f"{job.get('company','')} — {job.get('title','')} ({job.get('dates','')})")
        for b in job.get("bullets", []):
            lines.append(f"  - {b}")
        lines.append("")
    if d.get("education"):
        lines.append("EDUCATION")
        for e in d["education"]:
            # .rstrip() trims trailing spaces (in case 'detail' is empty).
            lines.append(f"  {e.get('institution','')} — {e.get('degree','')} "
                         f"({e.get('date','')}) {e.get('detail','')}".rstrip())
        lines.append("")
    if d.get("certifications"):
        lines += ["CERTIFICATIONS"] + [f"  - {c}" for c in d["certifications"]] + [""]
    if d.get("skills"):
        lines.append("TECHNICAL SKILLS")
        for s in d["skills"]:
            lines.append(f"  {s.get('category','')}: {s.get('items','')}")
    return "\n".join(lines)                           # join all lines into one string


def external_critique(resume_data: dict, job_posting: str, company: str, role: str,
                      llm: str = "chatgpt", timeout: int = 120) -> str:
    """Ask the reviewer LLM how to improve the tailored resume for this job while staying
    truthful, returning the suggestion text. `llm` selects the backend ("chatgpt", "claude",
    or a local model key like "local_qwen"). `timeout` is how many seconds to wait before
    giving up on the network call."""
    # Build the request: the role, the job description, and the current résumé as text. The
    # rules (_SYSTEM) are prepended to the prompt since call_llm_text takes a single string —
    # not every backend (the claude CLI) supports a separate system message.
    prompt = (
        f"{_SYSTEM}\n\n"
        f"TARGET ROLE: {role} at {company}\n\n"
        f"JOB DESCRIPTION:\n{job_posting}\n\n"
        f"---\n\nCURRENT TAILORED RESUME:\n{_resume_to_text(resume_data)}\n\n"
        f"Give your concrete, truthful improvement suggestions as a numbered list."
    )
    return call_llm_text(prompt, llm=llm, timeout=timeout)
