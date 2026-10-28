# Architecture Overview

This document explains **how the whole app is put together** — the big pieces, what
each one is responsible for, and how data flows between them. If you're new to the
code, read this first, then `WORKFLOW.md` for the step-by-step, then `PYTHON_PRIMER.md`
if any Python syntax is unfamiliar.

---

## What the app does (in one paragraph)

You give it a **base résumé** (a Word `.docx` file) and a **job posting**. It uses AI to
rewrite your résumé so it's tailored to that specific job — while staying strictly truthful
to your real experience — and saves the result as an HTML file (easy to read/edit) and a
PDF (for uploading to job sites). It also has a **job-search** half that scans company
career pages, filters for jobs that fit you, and tracks everything in one spreadsheet-like
file so you can generate a résumé for any job with one click.

The project is built in three phases:

- **Phase 1 — Tailor a résumé** for one job. ✅ Done.
- **Phase 2 — Find matching jobs** automatically, then run Phase 1 for each. 🚧 In progress.
- **Phase 3 — Auto-apply** to jobs. 🔜 Future.

---

## The two entry points

There are two ways to run the app. Both ultimately call the **same core function**,
`generate()` in `src/pipeline.py`, so the real logic lives in one place.

| File      | What it is                | How you use it                                   |
|-----------|---------------------------|--------------------------------------------------|
| `main.py` | Command-line tool (CLI)   | `python main.py --company "Acme" --role "..." …` |
| `app.py`  | Web app (Flask) + browser | `python app.py`, then open the local web page    |

> **Why two?** The CLI is handy for quick one-off runs and scripting. The web UI is nicer
> for everyday use (forms, buttons, a table of all your jobs). Keeping the shared logic in
> `pipeline.py` means a bug fix or improvement there benefits *both* automatically.

---

## The module map

Everything under `src/` is a **module** (a `.py` file you can import). Here's each one,
grouped by what part of the system it belongs to.

### Core résumé pipeline (Phase 1)

| Module                  | Responsibility (its ONE job)                                              |
|-------------------------|--------------------------------------------------------------------------|
| `src/pipeline.py`       | **The conductor.** Runs the whole tailor → review → render loop.          |
| `src/resume_parser.py`  | Read your base résumé `.docx` into plain text.                            |
| `src/job_parser.py`     | Turn a job URL / file / pasted text into clean job-posting text.          |
| `src/tailoring.py`      | Ask the AI to rewrite the résumé as structured data (JSON).              |
| `src/external_critic.py`| Ask a *second* AI (ChatGPT) for improvement suggestions.                  |
| `src/critic.py`         | Ask the AI to **audit** the draft for truthfulness + score it.           |
| `src/renderer.py`       | Turn the résumé data into an HTML file, then a PDF; count PDF pages.     |
| `src/tracker.py`        | Read/write `applications.csv` — the record of every job & résumé.        |
| `src/chat.py`           | A chat helper to capture *real* missing experience into your résumé.     |
| `src/resume_editor.py`  | Write that captured experience back into the base `.docx`.               |

### Job search (Phase 2)

| Module                          | Responsibility                                                    |
|---------------------------------|-------------------------------------------------------------------|
| `src/job_search.py`             | **The conductor for search.** Scan boards → filter → de-duplicate → embed.|
| `src/matching.py`               | The rules that decide if a job fits you (keywords + hard US/salary/workplace rules). |
| `src/filter_config.py`          | The editable lists/keywords/thresholds those rules use.           |
| `src/embeddings.py`             | Turns text into "meaning fingerprints" (OpenAI) — powers the title meaning net and the Match % column. |
| `src/fit_verdict.py`            | ChatGPT reads the top matches next to your résumé → 1-10 AI Fit score + reason. |
| `src/ats/greenhouse.py`         | Connector: talk to Greenhouse's public job-board API.             |
| `src/ats/lever.py`              | Connector: talk to Lever's public Postings API.                   |
| `src/ats/board_discovery.py`    | Shared engine: harvest + validate board tokens (any board).       |
| `src/ats/greenhouse_discovery.py`| Build the list of *which* Greenhouse boards exist (thin wrapper). |
| `src/ats/lever_discovery.py`    | Build the list of *which* Lever boards exist (thin wrapper).      |

### Supporting files (not code you import)

| Path                          | What it holds                                                       |
|-------------------------------|---------------------------------------------------------------------|
| `templates/resume.html`       | The visual layout of the generated résumé (a Jinja template).      |
| `templates/index.html`        | The web app's main page (the job table + forms).                   |
| `templates/settings.html`     | The web page for editing the search filters.                       |
| `data/filter_config.json`     | Your saved filter settings (overrides the defaults in code).        |
| `data/greenhouse_boards.json` | Cached directory of Greenhouse boards to scan.                      |
| `data/lever_boards.json`      | Cached directory of Lever boards to scan (built on demand).         |
| `resumes/…`                   | Your personal data: base résumé, outputs, the tracker CSV, backups.|

---

## How the pieces talk to each other

A key idea: **lower-level modules don't import higher-level ones.** The conductors
(`pipeline.py`, `job_search.py`, `app.py`) import the small workers; the workers don't
import the conductors. This keeps the dependencies a one-way tree (no loops) and makes
each worker easy to understand on its own.

```
            ┌──────────┐        ┌──────────┐
            │  app.py  │        │ main.py  │      ← entry points (you run these)
            └────┬─────┘        └────┬─────┘
                 │     both call      │
                 └─────────┬──────────┘
                           ▼
                 ┌───────────────────┐
                 │  src/pipeline.py  │              ← Phase 1 conductor
                 │   generate(...)   │
                 └───────┬───────────┘
       calls, in order:  │
   ┌───────────────┬─────┼──────────┬───────────────┬──────────────┐
   ▼               ▼     ▼          ▼               ▼              ▼
resume_parser  job_parser  tailoring  external_critic  critic   renderer → tracker
 (read .docx) (get JD text)(AI rewrite)(ChatGPT tips) (AI audit)(HTML+PDF) (log CSV)
```

And the search half:

```
   app.py  ─────►  src/job_search.py  (Phase 2 conductor)
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                            ▼
 ats/greenhouse.py    src/matching.py            src/tracker.py
 (fetch jobs from  )  (decide fit with     )     (save found jobs to
 (the public API   )  (pure-Python rules   )     ( applications.csv )
                          │
                          ▼
                 src/filter_config.py  ◄──  data/filter_config.json
                 (keyword lists the rules use; editable in the UI)
```

---

## The "one source of truth" files

A few files are deliberately the single place a piece of information lives:

- **Your real history:** `resumes/…Developer Enablement.docx` (the base résumé). The AI is
  told this is the *only* allowed source — it may re-emphasize what's here but never invent
  anything not in it. Extra experience captured via chat is merged in from
  `resumes/additional_experience.txt`.
- **Every job & résumé:** `resumes/applications.csv`. One row per job. A résumé is just
  data attached to a job's row. The job's "status" (Found → Resume Made → Applied) is
  *derived* from which columns are filled in, not stored separately.
- **Search filters:** the defaults live in `src/filter_config.py`; your edits are saved to
  `data/filter_config.json`, which overrides the defaults. The Settings web page edits the
  JSON.

---

## Where the AI comes in (and where it doesn't)

This matters for cost and trust:

- **Résumé generation** is the expensive part — three model calls per attempt: the
  *tailorer* (rewrites the résumé), the *external critic* (ChatGPT suggestions), and the
  internal *critic* (truthfulness audit + score).
- **Job search uses AI cheaply, for ranking — not for the hard rules.** The facts
  (US, salary, remote/hybrid/on-site, contract) are plain Python in `matching.py`, so
  scanning thousands of jobs stays fast and nearly free. AI enters in three small,
  skippable ways: the *meaning net* (keyword-less titles compared to your target-roles
  text — fractions of a cent), the *Match %* column (each stored job's description vs
  your résumé — about 2¢ per 1,000 jobs), and the *AI Fit* verdicts (ChatGPT reads the
  top ~25 per scan — a few cents). With no OpenAI key, search runs keywords-only.
- **Two different AIs cross-check each other.** ChatGPT optimizes for *fit* (and will
  sometimes over-reach); the Claude critic is the *truthfulness gatekeeper* that strips
  anything not backed by your real résumé. See `WORKFLOW.md` for how this loop runs.

### How the AI is accessed

The app talks to Claude through the **`claude` command-line tool**, which uses the owner's
Claude.ai Pro subscription (not a paid API account). There's also an optional **local AI**
backend (an OpenAI-compatible server on the home network) you can switch to. ChatGPT is
reached through OpenAI's web API with a key. All of this is wrapped in `src/tailoring.py`
and `src/external_critic.py`, so the rest of the code just calls a function and gets back
clean data.

---

## Quick glossary

- **ATS** — "Applicant Tracking System," the software employers use to scan résumés. The
  app optimizes the résumé so these systems parse it well.
- **Pipeline** — a series of processing steps where the output of one feeds the next.
- **Module** — one `.py` file. **Package** — a folder of modules (e.g. `src/ats/`).
- **JSON** — a simple text format for structured data (lists + labeled fields). The AI
  returns the résumé as JSON so the code can reliably place each piece into the template.
- **Jinja template** — an HTML file with `{{ placeholders }}` the code fills in with data.
- **Greenhouse** — a popular hiring platform; many companies' career pages run on it, and
  it has a free public API we read job listings from.
- **Connector** — a module whose job is to talk to one external service (e.g. Greenhouse).
```