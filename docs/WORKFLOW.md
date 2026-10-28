# Workflow Walkthrough

This document traces, **step by step**, what actually happens when you use the app — which
functions run, in what order, and what data passes between them.

There are two main workflows:

1. **Generate a tailored résumé** for one job (Phase 1).
2. **Find matching jobs** to generate résumés for (Phase 2).

---

## Workflow 1: Generate a tailored résumé (Phase 1)

This is the heart of the app. It runs whether you use the CLI (`main.py`) or the web app
(`app.py`) — both call `generate()` in `src/pipeline.py`.

### The inputs

- **`company`** and **`role`** — text, used for the output filename and the résumé header.
- **`source`** — *where the job posting is*: a URL, a path to a `.txt` file, or the posting
  text pasted directly.

### Step-by-step

```
generate(company, role, source)   ← src/pipeline.py
│
├─ 1. parse_resume()              ← src/resume_parser.py
│       Reads your base .docx into plain text. Also merges in any extra
│       experience you captured via chat (additional_experience.txt).
│
├─ 2. parse_job_posting(source)   ← src/job_parser.py
│       Turns the URL/file/text into clean job-posting text. For a URL it
│       fetches the page safely (see "Safety" below) and extracts just the
│       job description, throwing away menus/footers/ads.
│
├─ 3. PASS 1 — first draft
│       tailor_resume(resume, job, company, role)   ← src/tailoring.py
│         Sends your résumé + the job posting to the AI with detailed
│         instructions, and gets back the tailored résumé as JSON (structured
│         data: summary, competencies, experience bullets, etc.).
│       render_html(...) → render_pdf(...) → count_pdf_pages(...)  ← src/renderer.py
│         Lays the data into the HTML template, prints it to a PDF, counts pages.
│
├─ 4. EXTERNAL REVIEW (optional)
│       external_critique(draft, job, …)   ← src/external_critic.py
│         Sends the draft + job to ChatGPT, which suggests truthful ways to
│         make it a stronger fit. (Skipped if no ChatGPT key is configured.)
│
└─ 5. PASS 2 — refine in a loop (up to max_attempts, default 3)
        For each attempt:
          tailor_resume(..., revision_feedback=feedback)   ← apply the suggestions/fixes
          render_html → render_pdf → count_pdf_pages        ← make the files
          score_resume(...)   ← src/critic.py               ← AI audits + scores it
        Decide: is this attempt ACCEPTED?
          ACCEPTED  =  fits within max_pages  AND  fills at least min_pages (a 1-page
                       result for a long career history means real content was dropped)
                       AND  score ≥ min_score  AND  zero truthfulness ("fidelity")
                       problems  AND  zero "reads as AI-written" style problems.
        If accepted → stop. If not → feed the critic's exact fixes into the next attempt.
        Always remember the BEST attempt so the loop ends even if nothing is perfect.

   Finally:
     • Re-render the BEST attempt's files (so what's on disk is the best one).
     • log_application(...)   ← src/tracker.py   (write the resumes row in Postgres,
       including the critic's full findings — the resume_findings table — which the
       home page shows when you hover the Match score)
     • return (html_path, pdf_path, meta)
```

### Why two passes and two different AIs?

This is the clever part. The job posting is "untrusted" in the sense that it's tempting to
just copy its language to look like a perfect match — but that produces a fake-sounding
résumé. So the design makes two AIs play different roles:

- **ChatGPT (external critic)** optimizes for *fit*. It's eager and will sometimes suggest
  things your résumé can't honestly support (e.g. "say you owned the roadmap end-to-end").
- **Claude (internal critic)** is the *truthfulness gatekeeper*. It compares every line of
  the draft against your real base résumé and flags any fabrication, exaggeration, hedge
  phrase, or copied posting-language. Any such "fidelity issue" blocks acceptance and gets
  fed back as a concrete fix for the next attempt. It also flags wording that would read
  as AI-written (résumé-bot clichés, posting-echo phrasing, uniformly polished bullets) —
  those "style issues" block acceptance too, and it calls out real base-résumé content
  the draft dropped so the retry restores it.

The result is a résumé that's *as strong as the truth allows* — and no stronger.

### The "best attempt" idea

The loop might never produce a perfect résumé in 3 tries (some job postings are very
distinctive). Rather than fail, the pipeline always keeps the **best attempt so far**,
ranked in this priority order:

1. accepted (yes beats no)
2. fewer truthfulness problems
3. fewer "reads as AI-written" style problems
4. length in range (not over max_pages, not under min_pages)
5. higher quality score
6. fewer pages

So you always get a usable file, plus the critic's exact notes on anything still imperfect.

### Output

- `resumes/generated_resumes/<company>_<role>_resume.html` — readable/editable.
- `…_resume.pdf` — for uploading to job sites.
- `…_resume.json` — the structured data, cached so the files can be re-rendered later
  *without* spending another AI call (used by the `--render-only` option).
- A new/updated row in `resumes/applications.csv`.

### Safety note (web fetching)

When the job source is a URL, `job_parser.py` fetches it through `_safe_get()`, which
guards against **SSRF** (Server-Side Request Forgery) — a class of attack where a malicious
URL tricks the server into requesting private/internal addresses. It only allows
`http`/`https`, checks every redirect, blocks private/loopback addresses, and caps the
download size. This matters because the web app fetches URLs on the server's behalf.

---

## Workflow 2: Find matching jobs (Phase 2)

This half finds jobs worth applying to and adds them to your tracker. The keyword rules
are free; the AI parts (the meaning net, Match %, AI Fit) cost fractions of a cent to a
few cents per scan and are skipped automatically when no OpenAI key is configured.
You then click "Generate Resume" on any job to run Workflow 1 for it.

The conductor is `find_matches()` in `src/job_search.py`.

### Step-by-step

```
find_matches()   ← src/job_search.py
│
├─ load_directory()   ← src/ats/greenhouse_discovery.py
│     Loads data/greenhouse_boards.json — the cached list of company job
│     boards to scan. (This file is built separately; see "Discovery" below.)
│
├─ STAGE 1 — title pass (cheap, one API call per board)
│     For each board:
│       greenhouse.list_jobs(token)     ← src/ats/greenhouse.py   (all jobs, titles only)
│       filter_jobs(jobs)               ← src/matching.py         (keep the ones that fit)
│       keyword_only_rejects(jobs)      ← titles that ONLY failed the keyword sieve
│       drop any whose URL is already tracked (de-duplicate)
│
├─ MEANING NET — one batched AI call for the whole scan
│     _meaning_net(...)                 ← embed the keyword-less titles and re-admit
│                                         any whose MEANING is close to your "Target
│                                         roles" paragraph (Settings page)
│
├─ STAGE 2 — deep screen (only for title survivors)
│     For each surviving job:
│       greenhouse.get_job(token, id)   ← fetch the FULL description
│       match_job(full)                 ← re-check now that we can see salary,
│                                         remote/hybrid/on-site, US, gig phrases, etc.
│     Keep the jobs that still qualify, sorted best-fit first.
│
└─ EMBED — one batched AI call
      _embed_matches(...)               ← each kept job's description becomes a vector
                                          in the database, so the home page can show
                                          Match % (similarity to your base résumé)

Result: matching jobs → upsert_found_jobs(...) stores them in the database (as
"Found" rows, no résumé yet) → fit_verdict.run_scan_verdicts(...) has ChatGPT
read the top ~25 by Match % and write a 1-10 score + one-line reason (AI Fit).
```

### How a job is judged (the matcher)

`src/matching.py` decides "does this job fit me?". The net is deliberately WIDE — the
point is to store everything somewhat related and let the AI ranking sort it. A job
qualifies when ALL of these hold:

1. **Title sieve** — the title has any ONE domain keyword (learning, enablement,
   kubernetes, …) or qualifying phrase ("engineering manager", "senior engineer", …),
   and no *negative* keyword (e.g. `intern`, `research`). Generic words alone
   (`manager`, `director`, `software`) never qualify — "Account Manager" stays out.
   Some negatives are forgiven in the right context: "Sales **Enablement**" and
   "**Learning** Solutions Architect" pass. Titles with NO keyword at all get one more
   chance via the **meaning net** (embedding similarity to your target-roles text).
2. **Location** is confidently US (or the description confirms it).
3. **Workplace** — remote passes from anywhere; **hybrid** passes only when the
   location is within your configured commute zone (your commutable_locations);
   **on-site always drops**.
4. **Salary** — the description must state a starting salary at or above your floor;
   no stated salary is a drop too.
5. The description has no contract/part-time/gig phrases, and the posting isn't stale.

Each kept job also gets a **fit score (1–5)** and a **rank score**, and — with an
OpenAI key — a **Match %** (how similar its description is to your base résumé) and,
for the top ~25 per scan, an **AI Fit** score out of 10 with a one-line reason.
The keyword lists, target-roles paragraph, commute zone, and thresholds are all
editable on the Settings page (which writes `data/filter_config.json`);
`matching.reload_config()` re-reads them so edits take effect immediately.

### Why a two-stage scan?

Fetching every job's full description from every board would be slow and wasteful. So
stage 1 does a **cheap title-only filter** (one list call per board) to throw out the
obvious misses, and stage 2 fetches **full descriptions only for the survivors** to apply
the checks that need the description (salary, remote, gig phrases). Most jobs are rejected
in stage 1, for free.

### Discovery: where the board list comes from

Greenhouse has no "list every company" endpoint, so `greenhouse_discovery.py` builds the
directory itself: it harvests candidate board names from **Common Crawl** (a free, public
archive of the web), validates each one against Greenhouse's API, and caches the live ones
to `data/greenhouse_boards.json`. This is a slow, occasional build — everyday search just
reads the cached file.

---

## Workflow 3 (bonus): The résumé-enhancement chat

On any job that already has a résumé, you can open a chat (`src/chat.py`). It does two
things:

1. **Names the gaps** — concrete skills/experience the job wants that aren't in your résumé.
2. **Captures real experience** — when you tell it about something true you've done, it
   writes one polished bullet and saves it (via `src/resume_editor.py`) directly under the
   matching employer in your base `.docx` (with a timestamped backup), or to
   `additional_experience.txt` if it can't match an employer. Future résumés then draw on it.

It is strictly truthful: it only records what *you* state you actually did — it never
invents experience.

---

## Lifecycle of a job (the status you see in the UI)

A job moves through three derived stages (computed from which columns are filled in
`applications.csv`, not stored as a separate field):

```
   Found            →     Resume Made        →     Applied
(it's in your list)   (a résumé was generated)   (you applied)
 no HTML column          HTML column filled         Applied column filled
```

The web app's summary boxes count these: **Total** (all rows), **No Resume Yet** (Found
stage), **Resume Made**, and **Applied**.
```