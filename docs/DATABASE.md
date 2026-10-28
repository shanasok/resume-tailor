# Database Architecture

The app stores everything in one PostgreSQL database (`resume_handler`, running in
Docker with the pgvector extension). The design is a **hub and spokes**: one central
`jobs` table holds every job you're tracking, and small child tables hang off it, each
answering one question about that job. Three standalone tables sit outside the hub —
including the **rejection memory** that remembers jobs you deleted or that failed the
filters, so scans never pull them again.

## The diagram

Paste this into any Mermaid-aware tool (mermaid.live, GitHub, Notion, IntelliJ's
Markdown preview) to see it rendered.

```mermaid
erDiagram
    jobs ||--o{ job_flags : "0..many warning flags"
    jobs ||--o| job_descriptions : "0..1 stored posting text"
    jobs ||--o| job_embeddings : "0..1 Match % vector"
    jobs ||--o| resumes : "0..1 generated resume"
    resumes ||--o{ resume_findings : "0..many critic findings"
    jobs ||--o| applications : "0..1 application"
    jobs }o..o{ rejected_jobs : "no FK - deleting a job records its URL here"
    jobs }o..o{ generation_log : "no FK - audit rows outlive their job"

    jobs {
        TEXT job_id PK "source:board:id, e.g. greenhouse:stripe:4276"
        BIGSERIAL seq "insertion order tie-break"
        TEXT source "greenhouse | lever | rippling | manual"
        TEXT board_token "which company board"
        TEXT external_id "the job's id on that board"
        TEXT company
        TEXT role
        TEXT location
        TEXT job_url
        TEXT salary "free text from the posting"
        BOOLEAN technical "matcher verdict; NULL = never assessed"
        DATE date_found
        DATE date_posted
        SMALLINT ai_fit_score "ChatGPT 1-10 verdict (CHECK 1..10)"
        TEXT ai_fit_reason "one-line why"
        TIMESTAMPTZ ai_fit_at
    }

    job_flags {
        TEXT job_id PK, FK "part of composite PK"
        TEXT flag PK "one matcher warning, e.g. commutable hybrid"
        SMALLINT ord "original warning order"
    }

    job_descriptions {
        TEXT job_id PK, FK
        TEXT description "full posting text, saved at scan time"
        TIMESTAMPTZ fetched_at
    }

    job_embeddings {
        TEXT job_id PK, FK
        VECTOR_1536 embedding "description vector for Match %"
        TEXT model "text-embedding-3-small"
        TIMESTAMPTZ embedded_at
    }

    resumes {
        TEXT job_id PK, FK
        TEXT html_path
        TEXT pdf_path
        NUMERIC match_score "critic score out of 10"
        INTEGER gen_seconds
        TEXT generate_llm "backend per pipeline stage"
        TEXT external_llm
        TEXT check_llm
        TEXT regenerate_llm
    }

    resume_findings {
        TEXT job_id PK, FK "FK to resumes, part of composite PK"
        TEXT kind PK "fidelity | style | revision | strength"
        SMALLINT ord PK "the critic's original ordering"
        TEXT finding "one critic finding, e.g. an invented claim"
    }

    applications {
        TEXT job_id PK, FK
        TEXT applied_via "manual today; Phase 3 adds more"
        DATE applied_date
    }

    rejected_jobs {
        TEXT job_url PK "keyed by URL - these jobs have no jobs row"
        TEXT company
        TEXT role
        TEXT reason "deleted manually | no salary | non-US | ..."
        TIMESTAMPTZ rejected_at
    }

    generation_log {
        BIGSERIAL id PK
        TIMESTAMPTZ finished_at
        TEXT source "manual | scheduled"
        TEXT job_id "plain text on purpose"
        TEXT company
        TEXT role
        TEXT llm
        TEXT status "done | skipped | closed | error"
        NUMERIC match_score
        DOUBLE_PRECISION duration_sec
        TEXT message
    }

    profile_embeddings {
        TEXT name PK "base_resume | target_roles"
        TEXT content_hash "sha256 - text changed means re-embed"
        VECTOR_1536 embedding
        TEXT model
        TIMESTAMPTZ updated_at
    }

    schema_migrations {
        INTEGER version PK "applied migration steps, currently 1-7"
    }
```

## How to read it

**`jobs` is the hub.** One row per job you're tracking. Its primary key packs the
job's identity (`greenhouse:stripe:4276`), and the same identity also lives in three
real columns (`source` / `board_token` / `external_id`) with a CHECK constraint
guaranteeing the two forms can never disagree. The ChatGPT fit verdict
(`ai_fit_score`, 1–10, plus a one-line reason) lives directly on this table.

**The six spokes** all use the job's id as their own primary key (so each is strictly
0-or-1 per job, except `job_flags` and `resume_findings` which are 0-to-many), and all
say `ON DELETE CASCADE` — delete a job and every spoke row vanishes with it:

- **`job_flags`** — matcher warnings ("commutable hybrid", "visa unclear"), one per row.
- **`job_descriptions`** — the full posting text, saved the moment the scan fetches it.
  This is why scoring and resume generation never need to re-download a posting.
- **`job_embeddings`** — the description's AI vector, used for the Match % column
  (cosine similarity against your résumé's vector, computed in SQL). A separate table
  so the ~6 KB vectors stay out of ordinary page loads.
- **`resumes`** — row exists = a resume was generated. File paths, the critic's score,
  and which LLM ran each of the four pipeline stages.
- **`resume_findings`** — the critic's full report on that resume, one finding per row:
  truthfulness problems (`fidelity`), wording that reads AI-written or echoes the
  posting (`style`), suggested fixes (`revision`), and what the resume does well
  (`strength`). This hangs off `resumes` (not `jobs`) and is replaced wholesale every
  time the resume is regenerated or re-scored. The home page shows it when you hover
  a Match score — so a 6.8 always comes with its "why".
- **`applications`** — row exists = you applied. That's the whole design: presence is
  the fact, `applied_date` is the detail.

**`rejected_jobs` is the deletion/rejection memory** — the piece added for "don't
show me this again." It deliberately has **no** foreign key, because the whole point
is to remember jobs that *don't* have a `jobs` row: postings the deep screen dropped
(no salary, non-US, on-site, too old…), jobs **you deleted with the 🗑 button**
("deleted manually"), and postings that turned out to be dead ("posting no longer
active"). It's keyed by URL, and every scan loads the whole set up front and skips
those URLs before spending any API calls on them. Rows here are permanent on purpose.

**The other standalone tables:**

- **`generation_log`** — an append-only audit of every resume generation. Its
  `job_id` is plain text, not a foreign key, so the history survives job deletions.
- **`profile_embeddings`** — cached AI vectors for your base résumé and target-roles
  paragraph, keyed by a hash of the text: replace your résumé and it re-embeds (and
  clears all AI fit scores, since they were judged against the old résumé).
- **`schema_migrations`** — which numbered schema steps have been applied. The schema
  is append-only: new changes are new steps, applied ones are never edited.

**`job_overview`** (not in the diagram — it's a view, not a table) joins the hub with
`resumes`, `applications`, and the aggregated flags into one row per job. It's what
the home page reads. Descriptions and embeddings are intentionally *excluded* so
listing 1,000 jobs never drags megabytes of text and vectors along.

## Lifecycle of a job, in table terms

1. **Scan finds it** → row in `jobs` + `job_descriptions` + `job_embeddings`
   (+ `job_flags` if the matcher had warnings), and `ai_fit_*` filled in moments later.
2. **Scan drops it instead** → no `jobs` row at all; one row in `rejected_jobs`.
3. **You generate a resume** → a `resumes` row appears (audit row in `generation_log`).
4. **You apply** → an `applications` row appears.
5. **You delete it (🗑)** → the `jobs` row and all its spokes cascade away, and its URL
   is written to `rejected_jobs` so no future scan resurrects it.
