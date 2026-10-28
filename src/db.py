"""PostgreSQL persistence layer — ALL SQL lives in this module.

Layering: this module owns connections, the schema (as ordered migrations), and the
queries; src/tracker.py and src/run_log.py hold the domain logic and never write SQL
themselves. That keeps every statement in one reviewable place and lets the schema
evolve behind stable function signatures.

Schema (normalized, real types — see MIGRATIONS below for the authoritative DDL):

  jobs            one row per job POSTING (company, role, url, DATE columns, BOOLEAN…).
                  Board identity is real columns (source, board_token, external_id); the
                  packed job_id stays as the natural PK with a CHECK tying them together.
  job_flags       matcher warnings, one row per flag, FK → jobs, ON DELETE CASCADE.
  resumes         one row per generated RÉSUMÉ, FK → jobs, ON DELETE CASCADE. A job has
                  at most one résumé (job_id is the primary key); the backend that ran
                  each pipeline stage is its own column (<stage>_llm).
  resume_findings the critic's report on that résumé, one row per finding (kind =
                  fidelity | overstatement | style | revision | strength), FK → resumes, CASCADE.
                  Replaced wholesale on every (re)generation / re-score.
  applications    one row per APPLICATION you've made, FK → jobs, CASCADE. Existence of
                  the row IS the "applied" state; it records how and when.
  generation_log  append-only audit of every generation attempt (no FK — an audit trail
                  must survive the job being deleted). Timestamps are TIMESTAMPTZ.
  job_embeddings  1:1 child of jobs: the pgvector embedding of the posting's description
                  (drives the Match % ranking). FK → jobs, ON DELETE CASCADE. The
                  `embedding` column is dimensionless since v10 (different embedding
                  models produce different dimensions); reads filter on `model` instead.
  profile_embeddings  cached embeddings of the base résumé / target-roles profile,
                  keyed by a hash of the embedded text.
  board_directory the per-source list of live job boards discovery found (source, token,
                  job_count). SHARED across users (discovery is global), so no user_id.
                  The scan reads it to know which boards to walk; discovery replaces it.
  job_overview    a VIEW joining jobs ⟕ resumes ⟕ applications — one flat row per job.

Migrations: plain ordered SQL steps tracked in a `schema_migrations` table. A database
that predates the table (created by the first normalized version of this module) is
stamped as version 1 and picks up from there. A database still on the ORIGINAL flat
single-table layout is refused with a pointer to scripts/migrate_to_normalized_schema.py.

On the very first run against an EMPTY database, any legacy CSVs
(resumes/applications.csv, logs/generation_log.csv) are imported automatically and left
on disk untouched, as a backup.
"""
import csv                                           # read the legacy CSVs for the one-time import
import json                                          # vector values travel as their text form "[0.1,0.2,…]"
import os                                            # DATABASE_URL from the environment
import threading                                     # guard one-time schema init across threads
from datetime import date, datetime                  # typed values for DATE/TIMESTAMPTZ columns
from pathlib import Path                             # locate the legacy CSV files

import psycopg                                       # PostgreSQL driver (psycopg 3)
from psycopg.rows import dict_row                    # rows come back as {column: value} dicts
from psycopg.types.json import Jsonb                 # dict → JSONB for scans.filters

PROJECT_ROOT = Path(__file__).parent.parent

# Test override only — normally the URL is resolved from the environment AT CALL TIME
# (not at import time), so entry points that call load_dotenv() late still work.
DATABASE_URL = None
_DEFAULT_URL = "postgresql://resume:resume@localhost:5432/resume_handler"


def database_url() -> str:
    """The Postgres connection URL. The default matches docker-compose.yml's `db`
    service as seen FROM THE HOST; inside the app container, compose overrides
    DATABASE_URL with host `db` instead of localhost."""
    return DATABASE_URL or os.environ.get("DATABASE_URL", _DEFAULT_URL)


# The seeded default user. The app is multi-user at the data layer but has no login yet,
# so every request resolves to this user (see src/current_user.py). The v8 migration
# creates it and assigns all pre-existing data to it. Kept here because the SCHEMA owns
# the seeded row; current_user.DEFAULT_USER_ID mirrors this value for the app layer.
DEFAULT_USER_ID = 1


# The legacy flat-file stores, imported once into a fresh database.
APPLICATIONS_CSV = PROJECT_ROOT / "resumes" / "applications.csv"
GENERATION_LOG_CSV = PROJECT_ROOT / "logs" / "generation_log.csv"

# ── Schema, as ordered migrations ──────────────────────────────────────────────────────
# Append-only: NEVER edit an applied migration — add a new (version, sql) step instead.
# Each step runs in the same transaction as its schema_migrations bookkeeping row, so a
# half-applied migration can't be recorded as done.

_V1_BASELINE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,       -- "greenhouse:stripe:42" / "manual:acme_dev_resume"
    seq         BIGSERIAL,              -- insertion counter (stable tie-break when sorting by date)
    source      TEXT NOT NULL DEFAULT '',   -- manual | greenhouse | lever | …
    company     TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL DEFAULT '',
    location    TEXT NOT NULL DEFAULT '',
    job_url     TEXT NOT NULL DEFAULT '',
    flags       TEXT NOT NULL DEFAULT '',   -- matcher warnings, "; "-joined for display
    salary      TEXT NOT NULL DEFAULT '',   -- free text from the posting ("$150K–$200K + equity")
    technical   BOOLEAN,                    -- matcher's verdict; NULL = never assessed (manual rows)
    date_found  DATE,
    date_posted DATE
);

CREATE TABLE IF NOT EXISTS resumes (
    job_id      TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    html_path   TEXT NOT NULL,              -- generated files, relative to the project root
    pdf_path    TEXT NOT NULL DEFAULT '',
    match_score NUMERIC(4, 1),              -- critic's fit score out of 10
    gen_seconds INTEGER,                    -- how long the generation took
    llm         TEXT NOT NULL DEFAULT '',   -- backend that authored the final résumé
    stage_llms  TEXT NOT NULL DEFAULT ''    -- per-stage backends, "generate=claude;check=local;…"
);

CREATE TABLE IF NOT EXISTS applications (
    job_id       TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    applied_via  TEXT NOT NULL DEFAULT 'manual',  -- how you applied ("manual"; Phase 3 will add more)
    applied_date DATE
);

CREATE TABLE IF NOT EXISTS generation_log (
    id           BIGSERIAL PRIMARY KEY,
    finished_at  TIMESTAMP,
    source       TEXT NOT NULL DEFAULT '',  -- manual | scheduled
    job_id       TEXT NOT NULL DEFAULT '',  -- plain text on purpose: the log outlives the job
    company      TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL DEFAULT '',
    llm          TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT '',  -- done | skipped | closed | error
    match_score  NUMERIC(4, 1),
    duration_sec DOUBLE PRECISION,
    message      TEXT NOT NULL DEFAULT ''
);

CREATE OR REPLACE VIEW job_overview AS
    SELECT j.*,
           r.html_path, r.pdf_path, r.match_score, r.gen_seconds, r.llm, r.stage_llms,
           a.applied_via, a.applied_date
    FROM jobs j
    LEFT JOIN resumes r USING (job_id)
    LEFT JOIN applications a USING (job_id);
"""

_V2_TYPES_AND_INDEXES = """
-- Audit timestamps become timezone-aware: the host app and the container run in
-- different local timezones, and naive timestamps from both in one column are
-- meaningless. (Pre-existing naive rows are re-read as UTC — a one-time, cosmetic
-- few-hour shift on old audit rows.)
ALTER TABLE generation_log
    ALTER COLUMN finished_at TYPE TIMESTAMPTZ USING finished_at AT TIME ZONE 'UTC';

-- Every insert path sets date_found; make the schema say so.
ALTER TABLE jobs ALTER COLUMN date_found SET DEFAULT CURRENT_DATE;
ALTER TABLE jobs ALTER COLUMN date_found SET NOT NULL;

-- The hot paths: the newest-first listing, and the match-by-URL lookup on every
-- résumé save.
CREATE INDEX IF NOT EXISTS jobs_date_found_idx ON jobs (date_found DESC, seq);
CREATE INDEX IF NOT EXISTS jobs_job_url_idx ON jobs (job_url);
CREATE INDEX IF NOT EXISTS generation_log_finished_at_idx ON generation_log (finished_at);
"""

def _migrate_v3_full_normalization(conn) -> None:
    """v3 — remove the last normalization violations (a Python step: it needs to parse
    existing values, which SQL alone can't do cleanly):

    1. resumes.stage_llms ("generate=claude;check=local;…" — a map packed into TEXT,
       1NF violation) becomes four real columns, one per pipeline stage. resumes.llm was
       BY DEFINITION the regenerate-stage backend (a transitive dependency, 3NF
       violation), so it's folded into regenerate_llm and dropped. Pre-stage-tracking
       résumés (only llm recorded) backfill generate+regenerate — the same attribution
       the Stats page used to apply at display time.
    2. jobs.flags ("; "-joined list, 1NF violation) becomes the job_flags child table.
    3. jobs.job_id packs source:board_token:external_id while source is ALSO a column
       (redundancy). The parts become real columns; job_id stays the natural primary
       key (it's in URLs, forms, and the audit log), with a CHECK constraint enforcing
       the composition so the redundancy can never become an inconsistency."""
    conn.execute("DROP VIEW job_overview")

    # ── 1. Per-stage LLM columns ──
    conn.execute("""
        ALTER TABLE resumes
            ADD COLUMN generate_llm   TEXT NOT NULL DEFAULT '',
            ADD COLUMN external_llm   TEXT NOT NULL DEFAULT '',
            ADD COLUMN check_llm      TEXT NOT NULL DEFAULT '',
            ADD COLUMN regenerate_llm TEXT NOT NULL DEFAULT ''
    """)
    for row in conn.execute("SELECT job_id, llm, stage_llms FROM resumes").fetchall():
        conn.execute(
            "UPDATE resumes SET generate_llm = %(generate)s, external_llm = %(external)s, "
            "check_llm = %(check)s, regenerate_llm = %(regenerate)s WHERE job_id = %(job_id)s",
            {"job_id": row["job_id"],
             **_legacy_stage_columns(row["stage_llms"], row["llm"])},
        )
    conn.execute("ALTER TABLE resumes DROP COLUMN stage_llms, DROP COLUMN llm")

    # ── 2. Flags child table ──
    conn.execute("""
        CREATE TABLE job_flags (
            job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            ord    SMALLINT NOT NULL,        -- preserves the matcher's original order
            flag   TEXT NOT NULL,
            PRIMARY KEY (job_id, flag)
        )
    """)
    for row in conn.execute("SELECT job_id, flags FROM jobs WHERE flags <> ''").fetchall():
        _insert_flags(conn, row["job_id"], [f for f in row["flags"].split("; ") if f])
    conn.execute("ALTER TABLE jobs DROP COLUMN flags")

    # ── 3. Job identity columns ──
    conn.execute("ALTER TABLE jobs ADD COLUMN board_token TEXT, ADD COLUMN external_id TEXT")
    for row in conn.execute("SELECT job_id FROM jobs").fetchall():
        _, board_token, external_id = split_job_id(row["job_id"])
        conn.execute("UPDATE jobs SET board_token = %s, external_id = %s WHERE job_id = %s",
                     [board_token, external_id, row["job_id"]])
    conn.execute("""
        ALTER TABLE jobs
            ALTER COLUMN external_id SET NOT NULL,
            ADD CONSTRAINT jobs_identity_unique
                UNIQUE NULLS NOT DISTINCT (source, board_token, external_id),
            -- job_id is kept as the natural PK; this makes the packed form provably
            -- consistent with the identity columns (redundancy without anomaly)
            ADD CONSTRAINT jobs_job_id_composition CHECK (
                job_id = CASE WHEN board_token IS NULL
                              THEN source || ':' || external_id
                              ELSE source || ':' || board_token || ':' || external_id END)
    """)

    conn.execute("""
        CREATE VIEW job_overview AS
            SELECT j.*, f.flags,
                   r.html_path, r.pdf_path, r.match_score, r.gen_seconds,
                   r.generate_llm, r.external_llm, r.check_llm, r.regenerate_llm,
                   a.applied_via, a.applied_date
            FROM jobs j
            LEFT JOIN (SELECT job_id, array_agg(flag ORDER BY ord) AS flags
                       FROM job_flags GROUP BY job_id) f USING (job_id)
            LEFT JOIN resumes r USING (job_id)
            LEFT JOIN applications a USING (job_id)
    """)


def _migrate_v4_vectors(conn) -> None:
    """v4 — semantic job matching: pgvector embeddings + AI fit verdicts.

    1. The `vector` extension (needs the pgvector/pgvector:pg16 docker image).
    2. job_embeddings — 1:1 child of jobs (same pattern as resumes/applications: row
       existence = "this job has been embedded"). A child table, not a column on jobs,
       so the ~6 KB vectors stay out of job_overview / SELECT * page loads.
    3. profile_embeddings — cached vectors for the base résumé and the "target roles"
       profile, keyed by a hash of the embedded text so a changed résumé re-embeds.
    4. ai_fit_score / ai_fit_reason / ai_fit_at on jobs — ChatGPT's 1-10 verdict.
    5. job_overview recreated (a view freezes `j.*` at CREATE time, so the new jobs
       columns would otherwise never appear in reads).

    No vector index: at a few thousand rows a sequential scan is single-digit ms.
    Revisit (ivfflat/hnsw) only if the table ever approaches ~100k rows.

    This is a callable so the extension failure can carry a fix-it message."""
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.Error as e:
        raise RuntimeError(
            "The database can't load the 'vector' extension. Switch the db image in "
            "docker-compose.yml to pgvector/pgvector:pg16, restart it with "
            "`docker compose up -d db`, then start the app again."
        ) from e
    conn.execute("""
        CREATE TABLE job_embeddings (
            job_id      TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
            embedding   vector(1536) NOT NULL,   -- text-embedding-3-small dimension
            model       TEXT NOT NULL DEFAULT '',
            embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE profile_embeddings (
            name         TEXT PRIMARY KEY,       -- 'base_resume' | 'target_roles'
            content_hash TEXT NOT NULL,          -- sha256 of the embedded text
            model        TEXT NOT NULL DEFAULT '',
            embedding    vector(1536) NOT NULL,
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.execute("""
        ALTER TABLE jobs
            ADD COLUMN ai_fit_score  SMALLINT CHECK (ai_fit_score BETWEEN 1 AND 10),
            ADD COLUMN ai_fit_reason TEXT NOT NULL DEFAULT '',
            ADD COLUMN ai_fit_at     TIMESTAMPTZ
    """)
    conn.execute("DROP VIEW job_overview")
    conn.execute("""
        CREATE VIEW job_overview AS
            SELECT j.*, f.flags,
                   r.html_path, r.pdf_path, r.match_score, r.gen_seconds,
                   r.generate_llm, r.external_llm, r.check_llm, r.regenerate_llm,
                   a.applied_via, a.applied_date
            FROM jobs j
            LEFT JOIN (SELECT job_id, array_agg(flag ORDER BY ord) AS flags
                       FROM job_flags GROUP BY job_id) f USING (job_id)
            LEFT JOIN resumes r USING (job_id)
            LEFT JOIN applications a USING (job_id)
    """)


# v5 — store each job's posting text. Scoring/rewriting used to RE-FETCH descriptions
# from the boards on demand; at backfill volume (hundreds of rapid calls) the boards
# rate-limit and most jobs silently fail. ~10 KB of text per job is nothing to Postgres.
# A child table (same pattern as job_embeddings) so the text stays out of job_overview
# and SELECT * page loads.
_V5_JOB_DESCRIPTIONS = """
    CREATE TABLE job_descriptions (
        job_id      TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
        description TEXT NOT NULL,
        fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

# v6 — rejection memory. The deep screen fetches a candidate's full description only
# to drop ~95% of them (no salary, non-US, on-site…). Without memory, EVERY scan
# re-fetches and re-drops the same thousands of postings. Keyed by URL because a
# rejected job has no jobs-table row. Permanent by design: any posting old enough for
# its facts to have changed gets age-dropped anyway (max_posting_age_days).
_V6_REJECTED_JOBS = """
    CREATE TABLE rejected_jobs (
        job_url     TEXT PRIMARY KEY,
        company     TEXT NOT NULL DEFAULT '',
        role        TEXT NOT NULL DEFAULT '',
        reason      TEXT NOT NULL DEFAULT '',
        rejected_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

# v7 — persist the critic's findings. Every generation ends with a critic report
# (truthfulness problems, AI-style tells, suggested fixes, strengths), but only the
# COUNT used to survive (in the audit-log message) — the findings themselves vanished
# with the live progress feed, leaving nothing to explain a low match score later.
# One row per finding (the job_flags child-table pattern); FK → resumes, so deleting
# a job or its résumé takes the report with it. Replaced wholesale on each re-score.
_V7_RESUME_FINDINGS = """
    CREATE TABLE resume_findings (
        job_id  TEXT NOT NULL REFERENCES resumes(job_id) ON DELETE CASCADE,
        kind    TEXT NOT NULL,      -- fidelity | overstatement | style | revision | strength
        ord     SMALLINT NOT NULL,  -- the critic's original ordering within a kind
        finding TEXT NOT NULL,
        PRIMARY KEY (job_id, kind, ord)
    )
"""

def _migrate_v8_multi_tenant(conn) -> None:
    """v8 — multi-tenancy foundation (a Python step: it moves data onto the default user).

    Turns the single-user database into a per-user one WITHOUT real auth yet:
      * a `users` table, seeded with the default user (id 1) that owns everything today;
      * `jobs` becomes the SHARED posting catalog — its per-user columns (date_found,
        ai_fit_*) move into a new `user_jobs` overlay (one row per user × posting, holding
        that user's date_matched + AI-fit verdict);
      * the per-user children (resumes, applications, job_flags, resume_findings) are
        re-keyed to (user_id, job_id) and FK → user_jobs / resumes;
      * profile_embeddings, rejected_jobs, generation_log gain a user_id;
      * job_descriptions / job_embeddings STAY shared (posting-intrinsic, keyed by job_id);
      * per-user filter tables (user_settings, user_filter_terms) are created and seeded
        for user 1 from the machine-local snapshot if present (a fresh clone has no
        snapshot → no rows → filter_config falls back to the neutral defaults at runtime).
    Every pre-existing row is assigned to user 1, so behavior is unchanged for the owner.

    Board postings keep a globally-unique job_id (one shared row, many users point at it);
    manual postings become user-scoped by putting the user in the id going forward
    (tracker.manual_job_id → 'manual:{user_id}:{slug}') — existing 'manual:slug' rows stay
    valid and are owned by user 1.
    """
    uid = DEFAULT_USER_ID

    # ── users + the seeded default user ──
    conn.execute("""
        CREATE TABLE users (
            id           BIGSERIAL PRIMARY KEY,
            email        TEXT UNIQUE,
            display_name TEXT NOT NULL DEFAULT '',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.execute("INSERT INTO users (id, email, display_name) VALUES (%s, %s, %s)",
                 [uid, None, "Default user"])
    # We supplied the id explicitly, so the BIGSERIAL sequence didn't advance — bump it
    # past the seeded row so the next real signup doesn't collide on id.
    conn.execute("SELECT setval('users_id_seq', GREATEST((SELECT MAX(id) FROM users), 1))")

    # ── jobs → shared catalog: peel the per-user columns into a user_jobs overlay ──
    conn.execute("DROP VIEW job_overview")
    conn.execute("""
        CREATE TABLE user_jobs (
            user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id       TEXT   NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            date_matched DATE   NOT NULL DEFAULT CURRENT_DATE,
            ai_fit_score  SMALLINT CHECK (ai_fit_score BETWEEN 1 AND 10),
            ai_fit_reason TEXT NOT NULL DEFAULT '',
            ai_fit_at     TIMESTAMPTZ,
            PRIMARY KEY (user_id, job_id)
        )
    """)
    # Every existing posting becomes user 1's, carrying its found-date (now date_matched)
    # and AI-fit verdict.
    conn.execute(
        "INSERT INTO user_jobs (user_id, job_id, date_matched, ai_fit_score, ai_fit_reason, ai_fit_at) "
        "SELECT %s, job_id, date_found, ai_fit_score, ai_fit_reason, ai_fit_at FROM jobs",
        [uid])
    conn.execute("CREATE INDEX user_jobs_user_date_idx ON user_jobs (user_id, date_matched DESC, job_id)")
    conn.execute("DROP INDEX IF EXISTS jobs_date_found_idx")
    conn.execute("ALTER TABLE jobs DROP COLUMN date_found, DROP COLUMN ai_fit_score, "
                 "DROP COLUMN ai_fit_reason, DROP COLUMN ai_fit_at")

    # ── Re-key the per-user children to (user_id, job_id) ──
    # resumes / applications / job_flags: FK → user_jobs; resume_findings: FK → resumes.
    def _add_user_id(table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id BIGINT")
        conn.execute(f"UPDATE {table} SET user_id = %s", [uid])
        conn.execute(f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL")

    # Add user_id + drop the OLD FK to jobs on all four first. resume_findings' FK depends
    # on resumes' primary-key index, so every child FK must be gone before any old PK drops.
    for table in ("resumes", "applications", "job_flags", "resume_findings"):
        _add_user_id(table)
        conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_job_id_fkey")
    for table in ("resumes", "applications", "job_flags", "resume_findings"):
        conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_pkey")
    # New composite primary keys (resumes' must exist before resume_findings' FK targets it).
    conn.execute("ALTER TABLE resumes ADD PRIMARY KEY (user_id, job_id)")
    conn.execute("ALTER TABLE applications ADD PRIMARY KEY (user_id, job_id)")
    conn.execute("ALTER TABLE job_flags ADD PRIMARY KEY (user_id, job_id, flag)")
    conn.execute("ALTER TABLE resume_findings ADD PRIMARY KEY (user_id, job_id, kind, ord)")
    # New composite FKs → the overlay (or → resumes for findings).
    conn.execute("ALTER TABLE resumes ADD CONSTRAINT resumes_user_job_fkey "
                 "FOREIGN KEY (user_id, job_id) REFERENCES user_jobs (user_id, job_id) ON DELETE CASCADE")
    conn.execute("ALTER TABLE applications ADD CONSTRAINT applications_user_job_fkey "
                 "FOREIGN KEY (user_id, job_id) REFERENCES user_jobs (user_id, job_id) ON DELETE CASCADE")
    conn.execute("ALTER TABLE job_flags ADD CONSTRAINT job_flags_user_job_fkey "
                 "FOREIGN KEY (user_id, job_id) REFERENCES user_jobs (user_id, job_id) ON DELETE CASCADE")
    conn.execute("ALTER TABLE resume_findings ADD CONSTRAINT resume_findings_user_resume_fkey "
                 "FOREIGN KEY (user_id, job_id) REFERENCES resumes (user_id, job_id) ON DELETE CASCADE")

    # ── Add user_id to the remaining per-user stores ──
    for table in ("profile_embeddings", "rejected_jobs"):
        _add_user_id(table)
        conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_pkey")
        conn.execute(f"ALTER TABLE {table} ADD CONSTRAINT {table}_user_fkey "
                     f"FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE")
    conn.execute("ALTER TABLE profile_embeddings ADD PRIMARY KEY (user_id, name)")
    conn.execute("ALTER TABLE rejected_jobs ADD PRIMARY KEY (user_id, job_url)")
    # The audit log OUTLIVES a deleted user (like it outlives deleted jobs), so its user_id
    # is nullable and SET NULL on user delete.
    conn.execute("ALTER TABLE generation_log ADD COLUMN user_id BIGINT "
                 "REFERENCES users(id) ON DELETE SET NULL")
    conn.execute("UPDATE generation_log SET user_id = %s", [uid])
    conn.execute("CREATE INDEX generation_log_user_idx ON generation_log (user_id)")

    # ── Per-user filter config tables ──
    conn.execute("""
        CREATE TABLE user_settings (
            user_id                  BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            salary_floor             INTEGER,
            max_posting_age_days     INTEGER,
            title_similarity_percent INTEGER,
            target_roles_text        TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE user_filter_terms (
            user_id   BIGINT   NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            list_name TEXT     NOT NULL,     -- e.g. domain_keywords, title_negative_keywords
            ord       SMALLINT NOT NULL,     -- preserves the saved order within a list
            term      TEXT     NOT NULL,
            PRIMARY KEY (user_id, list_name, ord)
        )
    """)
    _seed_user_filters_from_snapshot(conn, uid)

    # ── Rebuild the read view around the overlay (it now carries user_id) ──
    conn.execute("""
        CREATE VIEW job_overview AS
            SELECT uj.user_id, j.*,
                   uj.date_matched, uj.ai_fit_score, uj.ai_fit_reason, uj.ai_fit_at,
                   f.flags,
                   r.html_path, r.pdf_path, r.match_score, r.gen_seconds,
                   r.generate_llm, r.external_llm, r.check_llm, r.regenerate_llm,
                   a.applied_via, a.applied_date
            FROM user_jobs uj
            JOIN jobs j USING (job_id)
            LEFT JOIN (SELECT user_id, job_id, array_agg(flag ORDER BY ord) AS flags
                       FROM job_flags GROUP BY user_id, job_id) f USING (user_id, job_id)
            LEFT JOIN resumes r USING (user_id, job_id)
            LEFT JOIN applications a USING (user_id, job_id)
    """)


# The machine-local snapshot of the owner's pre-migration filters (written before the
# committed DEFAULTS were neutralized; gitignored, so it never travels to another clone).
_USER1_FILTER_SEED = PROJECT_ROOT / "data" / "user1_filter_seed.json"


def _seed_user_filters_from_snapshot(conn, user_id: int) -> None:
    """Seed one user's filter rows from the machine-local snapshot, if it exists. Absent
    (a fresh clone / another engineer) → nothing written → the neutral committed defaults
    apply at runtime. This is what preserves the owner's exact tuning across the cutover
    without re-committing personal keywords to source."""
    if not _USER1_FILTER_SEED.exists():
        return
    try:
        cfg = json.loads(_USER1_FILTER_SEED.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    upsert_user_settings(conn, user_id, cfg)
    for list_name, terms in cfg.items():
        if isinstance(terms, list):
            replace_user_filter_terms(conn, user_id, list_name, terms)


# v9 — per-user BASE RÉSUMÉ content, structured. Until now the base résumé was a single
# global .docx flattened to text; these tables give each user their own résumé stored in
# the app's canonical shape (tailoring._SCHEMA): a 1:1 profile (contact + summary) plus
# ordered child rows per section. resume_profiles is the parent — the section tables FK to
# IT (not users), so clearing a user's profile takes all their sections with it; and the
# profile FKs to users, so deleting a user still cascades the whole résumé away.
_V9_RESUME_PROFILES = """
    CREATE TABLE resume_profiles (
        user_id    BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        full_name  TEXT NOT NULL DEFAULT '',
        location   TEXT NOT NULL DEFAULT '',
        phone      TEXT NOT NULL DEFAULT '',
        email      TEXT NOT NULL DEFAULT '',
        website    TEXT NOT NULL DEFAULT '',
        summary    TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE resume_experience (
        user_id  BIGINT   NOT NULL REFERENCES resume_profiles(user_id) ON DELETE CASCADE,
        ord      SMALLINT NOT NULL,               -- display order within the résumé
        company  TEXT     NOT NULL DEFAULT '',
        title    TEXT     NOT NULL DEFAULT '',
        dates    TEXT     NOT NULL DEFAULT '',     -- free text, e.g. "2019 – 2023"
        location TEXT     NOT NULL DEFAULT '',
        bullets  TEXT[]   NOT NULL DEFAULT '{}',   -- ordered achievement bullets
        PRIMARY KEY (user_id, ord)
    );
    CREATE TABLE resume_education (
        user_id     BIGINT   NOT NULL REFERENCES resume_profiles(user_id) ON DELETE CASCADE,
        ord         SMALLINT NOT NULL,
        institution TEXT     NOT NULL DEFAULT '',
        degree      TEXT     NOT NULL DEFAULT '',
        date_text   TEXT     NOT NULL DEFAULT '',  -- maps to _SCHEMA "date" (free text)
        detail      TEXT     NOT NULL DEFAULT '',
        PRIMARY KEY (user_id, ord)
    );
    CREATE TABLE resume_skills (
        user_id  BIGINT   NOT NULL REFERENCES resume_profiles(user_id) ON DELETE CASCADE,
        ord      SMALLINT NOT NULL,
        category TEXT     NOT NULL DEFAULT '',
        items    TEXT     NOT NULL DEFAULT '',     -- comma-separated list, per _SCHEMA
        PRIMARY KEY (user_id, ord)
    );
    CREATE TABLE resume_competencies (
        user_id BIGINT   NOT NULL REFERENCES resume_profiles(user_id) ON DELETE CASCADE,
        ord     SMALLINT NOT NULL,
        term    TEXT     NOT NULL,
        PRIMARY KEY (user_id, ord)
    );
    CREATE TABLE resume_certifications (
        user_id BIGINT   NOT NULL REFERENCES resume_profiles(user_id) ON DELETE CASCADE,
        ord     SMALLINT NOT NULL,
        text    TEXT     NOT NULL,
        PRIMARY KEY (user_id, ord)
    );
"""


_V10_DIMENSIONLESS_VECTORS = """
    -- Vectors from different embedding models have different dimensions (OpenAI 1536,
    -- local Ollama models 768/1024). Drop the hard-coded pin so any model's vectors
    -- fit; reads now filter on the `model` column instead — comparing mixed dimensions
    -- with <=> is a SQL error, so stale-model rows must be invisible to every query.
    ALTER TABLE job_embeddings ALTER COLUMN embedding TYPE vector;
    ALTER TABLE profile_embeddings ALTER COLUMN embedding TYPE vector;
"""


def _migrate_v11_board_directory(conn) -> None:
    """v11 — move the job-board DIRECTORY into the database (a Python step: it imports the
    existing on-disk directories).

    Discovery (Common Crawl harvest → validate) produces the per-source list of live boards
    the scan walks. It was the last app-owned data still cached OUTSIDE Postgres, as flat
    data/*_boards.json files. Bring it into the normalized schema like everything else.

    SHARED across users (discovery is global — it harvests the public web, not a user's data —
    exactly like job_embeddings / job_descriptions), so NO user_id. The one-time step carries
    over whatever the JSON files hold so the existing directories survive without re-harvesting;
    the files are left on disk untouched, as a backup (same courtesy as the legacy CSV import)."""
    conn.execute("""
        CREATE TABLE board_directory (
            source     TEXT    NOT NULL,   -- greenhouse | lever | rippling | workday | ashby
            token      TEXT    NOT NULL,   -- the board slug / packed board identity
            job_count  INTEGER NOT NULL DEFAULT 0,   -- live posting count at last validation
            first_seen DATE    NOT NULL DEFAULT CURRENT_DATE,
            last_seen  DATE    NOT NULL DEFAULT CURRENT_DATE,   -- refreshed on every rebuild
            PRIMARY KEY (source, token)
        )
    """)
    # The one hot read: all boards for one source, busiest first (mirrors the old JSON sort).
    conn.execute("CREATE INDEX board_directory_source_idx ON board_directory (source, job_count DESC)")

    # One-time carry-over of any existing on-disk directories (data/<source>_boards.json).
    for path in sorted((PROJECT_ROOT / "data").glob("*_boards.json")):
        source = path.name[: -len("_boards.json")]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                                  # unreadable/corrupt file — skip, don't fail the migration
        seen_on = parse_date(data.get("generated_at")) or date.today()
        for board in data.get("boards", []):
            token = board.get("token")
            if not token:
                continue
            conn.execute(
                "INSERT INTO board_directory (source, token, job_count, first_seen, last_seen) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (source, token) DO NOTHING",
                [source, token, int(board.get("job_count") or 0), seen_on, seen_on],
            )


# v12 — résumé version history. Every regenerate (the "From JSON" re-render AND the full
# "With LLM" rewrite) overwrites the current résumé's files + resumes row in place, so a
# rewrite that comes out worse than the original used to be unrecoverable. Before each
# overwrite we snapshot the CURRENT résumé here: its files (copied into a versions/ sub-folder
# next to the résumé), its match score, the critic report, and which LLMs made it. Append-only;
# the highest `version` per (user, résumé) is the most recent snapshot. findings/stage_llms are
# stored as packed JSON text — an immutable point-in-time snapshot doesn't need the live
# résumé's normalized resume_findings child table. FK → resumes so a deleted résumé's history
# cascades away with it.
_V12_RESUME_VERSIONS = """
    CREATE TABLE resume_versions (
        user_id     INTEGER NOT NULL,
        job_id      TEXT    NOT NULL,
        version     INTEGER NOT NULL,              -- 1, 2, 3… per (user_id, job_id)
        html_path   TEXT    NOT NULL DEFAULT '',   -- archived files, relative to generated_resumes
        pdf_path    TEXT    NOT NULL DEFAULT '',
        json_path   TEXT    NOT NULL DEFAULT '',   -- the résumé data — enough to re-render/restore
        match_score NUMERIC(4, 1),                 -- the critic's fit score this version had
        findings    TEXT    NOT NULL DEFAULT '',   -- JSON {kind: [finding, …]} snapshot
        stage_llms  TEXT    NOT NULL DEFAULT '',   -- JSON {stage: backend} that made this version
        note        TEXT    NOT NULL DEFAULT '',   -- why it was archived ("before AI rewrite", …)
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, job_id, version),
        FOREIGN KEY (user_id, job_id) REFERENCES resumes (user_id, job_id) ON DELETE CASCADE
    )
"""


# v13 — scan-audit tables. Moves the job-scan drop-funnel audit trail out of a flat .log file
# and into Postgres: `scans` is one row per scan run (who ran it, when, with what filters);
# `scan_events` is one row per drop-funnel decision within that scan (sieve reject, deep-screen
# drop, stored+scored, …), FK'd to its parent scan so deleting a scan cleans up its events.
_V13_SCAN_AUDIT = """
    CREATE TABLE scans (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        ended_at   TIMESTAMPTZ,
        status     TEXT NOT NULL DEFAULT 'running',   -- 'running' | 'complete'
        sources    TEXT[],
        filters    JSONB
    );
    CREATE TABLE scan_events (
        id            BIGSERIAL PRIMARY KEY,
        scan_id       INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
        seq           INTEGER NOT NULL,
        company       TEXT,
        title         TEXT,
        source        TEXT,
        board_token   TEXT,
        location      TEXT,
        stage         TEXT NOT NULL,
        outcome       TEXT NOT NULL,
        detail        TEXT,
        reason_bucket TEXT
    );
    CREATE INDEX scan_events_scan_seq ON scan_events (scan_id, seq);
"""

# v14 — the two identity fields a dropped job needs to be re-fetched from its board later
# (for the Scan Audit "Track this job" rescue button). Nullable so pre-v14 rows are fine.
_V14_SCAN_EVENT_IDENTITY = """
    ALTER TABLE scan_events ADD COLUMN external_id TEXT;
    ALTER TABLE scan_events ADD COLUMN url TEXT;
"""

# v15 — the "Under Qualified" AI-Fit threshold moves out of the browser. It used to live
# in localStorage, which meant only the browser knew where a job belonged; the jobs table
# therefore had to render EVERY row and let JavaScript sort them into tabs. Storing it
# per user lets the page ask SQL for one tab's rows. NULL = the user never set one (→ the
# filter_config default).
_V15_UNDERQUALIFIED_THRESHOLD = """
    ALTER TABLE user_settings ADD COLUMN underqualified_at_or_below SMALLINT;
"""


# Each step is either a SQL string or a callable(conn) for data-dependent changes.
MIGRATIONS: list[tuple[int, object]] = [
    (1, _V1_BASELINE),
    (2, _V2_TYPES_AND_INDEXES),
    (3, _migrate_v3_full_normalization),
    (4, _migrate_v4_vectors),
    (5, _V5_JOB_DESCRIPTIONS),
    (6, _V6_REJECTED_JOBS),
    (7, _V7_RESUME_FINDINGS),
    (8, _migrate_v8_multi_tenant),
    (9, _V9_RESUME_PROFILES),
    (10, _V10_DIMENSIONLESS_VECTORS),
    (11, _migrate_v11_board_directory),
    (12, _V12_RESUME_VERSIONS),
    (13, _V13_SCAN_AUDIT),
    (14, _V14_SCAN_EVENT_IDENTITY),
    (15, _V15_UNDERQUALIFIED_THRESHOLD),
]

# One-time-per-process schema init. The lock stops two threads (e.g. a web request and
# the scheduler) from both running the init; the flag makes later calls free.
_init_lock = threading.Lock()
_schema_ready = False


def connect():
    """Open a database connection (schema guaranteed to be current). Use it as a context
    manager — `with db.connect() as conn:` — which commits on success, rolls back on
    error, and closes the connection either way. Rows come back as dicts."""
    _ensure_schema()
    return _connect_raw()


def _connect_raw():
    """A plain connection, without the schema check (used BY the schema check)."""
    try:
        return psycopg.connect(database_url(), row_factory=dict_row)
    except psycopg.OperationalError as e:
        raise RuntimeError(
            f"Can't reach the database at {database_url()}. "
            "Is it running? Start it with: docker compose up -d db"
        ) from e


def _ensure_schema():
    """Bring the schema up to date on first use, then import any legacy CSV data if the
    tables are empty. Runs at most once per process."""
    global _schema_ready
    if _schema_ready:
        return
    with _init_lock:
        if _schema_ready:                            # another thread won the race
            return
        with _connect_raw() as conn:
            if _has_old_flat_schema(conn):
                raise RuntimeError(
                    "This database still has the old single-flat-table schema. "
                    "Migrate it once with: python scripts/migrate_to_normalized_schema.py"
                )
            apply_migrations(conn)
            ensure_default_user(conn)                 # always have a user to attribute to
            _import_legacy_csvs(conn)
        _schema_ready = True


def apply_migrations(conn) -> list[int]:
    """Run every not-yet-applied migration, in order, recording each in
    schema_migrations. Returns the versions applied this call (usually [])."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  version    INTEGER PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    done = {r["version"] for r in
            conn.execute("SELECT version FROM schema_migrations").fetchall()}
    if not done and _table_exists(conn, "jobs"):
        # The DB predates schema_migrations but already has the normalized tables
        # (created by the first version of this module) → stamp it as the baseline.
        conn.execute("INSERT INTO schema_migrations (version) VALUES (1)")
        done = {1}
    applied = []
    for version, step in MIGRATIONS:
        if version not in done:
            if callable(step):                       # a Python step (data-dependent changes)
                step(conn)
            else:                                    # a plain SQL step
                conn.execute(step)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", [version])
            applied.append(version)
    return applied


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s", [name]
    ).fetchone() is not None


def _has_old_flat_schema(conn) -> bool:
    """True when the jobs table exists in its pre-normalization shape (résumé columns
    like `html` folded into it). The normalized jobs table has no `html` column."""
    return conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'jobs' AND column_name = 'html'"
    ).fetchone() is not None


# The four LLM-driven pipeline stages, in display order — each is a column on resumes
# (<stage>_llm). tracker.STAGE_ORDER aliases this; defined here because the schema owns it.
STAGES = ["generate", "external", "check", "regenerate"]


def split_job_id(job_id: str) -> tuple[str, str | None, str]:
    """A packed job id → its (source, board_token, external_id) parts.
    "greenhouse:stripe:42" → ("greenhouse", "stripe", "42"); a two-part manual id
    "manual:acme_dev_resume" → ("manual", None, "acme_dev_resume")."""
    parts = job_id.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return parts[0], None, parts[1] if len(parts) == 2 else job_id


def _legacy_stage_columns(stage_llms_raw: str, llm_raw: str) -> dict:
    """Decode the LEGACY per-stage encoding ("generate=claude;check=local;…") into the
    four <stage>_llm column values. A pre-stage-tracking row (only `llm` recorded) is
    attributed to the two tailoring stages — the same fallback the Stats page used to
    apply at display time; now it's fixed in the data instead of patched on every read."""
    decoded = {}
    for part in (stage_llms_raw or "").split(";"):
        if "=" in part:
            key, _, val = part.partition("=")
            if key.strip() and val.strip():
                decoded[key.strip()] = val.strip()
    if not decoded and llm_raw:
        decoded = {"generate": llm_raw, "regenerate": llm_raw}
    return {stage: decoded.get(stage, "") for stage in STAGES}


def _insert_flags(conn, job_id: str, flags: list[str]) -> None:
    # PRE-v8 inserter (job_flags had no user_id yet) — used ONLY by the v3 migration, which
    # must keep operating on the schema as it was at v3 time. Runtime code uses
    # _insert_user_flags. Do not "modernize" this: it would break v3 on a fresh install.
    for i, flag in enumerate(flags):
        conn.execute("INSERT INTO job_flags (job_id, ord, flag) VALUES (%s, %s, %s) "
                     "ON CONFLICT (job_id, flag) DO NOTHING",
                     [job_id, i, flag])


# ── Typed-value parsing (legacy strings → real column types) ──────────────────────────

def parse_date(raw) -> date | None:
    """'2026-06-25' (or a longer ISO timestamp) → date; ''/None/garbage → None."""
    if isinstance(raw, date):
        return raw
    raw = (str(raw or "")).strip()
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def parse_float(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_int(raw) -> int | None:
    f = parse_float(raw)
    return None if f is None else int(f)


def parse_bool(raw) -> bool | None:
    """'yes'/'no' (the matcher's Technical verdict) → real boolean; anything else → NULL."""
    return {"yes": True, "no": False}.get(str(raw or "").strip().lower())


# ── Reads (all user-scoped: a read only ever returns the given user's data) ──────────────

def fetch_overview(conn, user_id: int, newest_first: bool = False) -> list[dict]:
    """This USER's tracked jobs from the join view. Default order: overlay insertion
    (`seq`). newest_first sorts by Date Matched descending, `seq` as the stable tie-break."""
    order = "date_matched DESC, seq" if newest_first else "seq"
    return conn.execute(
        f"SELECT * FROM job_overview WHERE user_id = %s ORDER BY {order}", [user_id]
    ).fetchall()


# ── The home page's display filters ──────────────────────────────────────────────────
#
# The jobs table used to render EVERY tracked row (3,500+ for a long-running search) and
# let the browser hide most of them with CSS. These helpers move that work into SQL so the
# page only ever renders what it shows.

# A job stays in the display list when ANY of these hold. The age cutoff alone would
# bury real work: an applied job is history the user still needs to see, and a generated
# résumé cost real LLM money. An undated posting isn't old — its age is simply unknowable
# (the same reason the scan-time filter in matching.match_job keeps them).
_AGE_VISIBLE = """(
    o.applied_via IS NOT NULL
    OR (o.html_path IS NOT NULL AND o.html_path <> '')
    OR o.date_posted IS NULL
    OR o.date_posted >= %(cutoff)s
)"""


def _display_where(user_id: int, posted_on_or_after) -> tuple[str, dict]:
    """The WHERE fragment + params shared by the page read and the tab counts: this
    user's rows, narrowed to what the posting-age cutoff leaves visible.

    `posted_on_or_after` is a real date computed by the CALLER in Python, never SQL's
    CURRENT_DATE — `date_posted` is written with the app's local date while the docker
    Postgres runs UTC, so an in-database "today" is a day ahead on US evenings."""
    where = ["o.user_id = %(uid)s"]
    params = {"uid": user_id}
    if posted_on_or_after is not None:
        # job_overview already carries the résumé/application columns, so the visibility
        # rule reads them straight off the row rather than re-joining.
        where.append(_AGE_VISIBLE)
        params["cutoff"] = posted_on_or_after
    return " AND ".join(where), params


# The four tabs, as SQL predicates. These mirror exactly what the browser used to decide
# per row: "under qualified" siphons low-scoring jobs OUT of To Apply, an unscored job
# stays in To Apply (no verdict is not a bad verdict), and Manual is a source filter that
# deliberately overlaps the others.
_TAB_SQL = {
    "applied": "o.applied_via IS NOT NULL",
    "manual": "o.source = 'manual'",
    "underqualified": ("o.applied_via IS NULL AND o.ai_fit_score IS NOT NULL "
                       "AND o.ai_fit_score <= %(uq)s"),
    "pending": ("o.applied_via IS NULL AND (o.ai_fit_score IS NULL "
                "OR o.ai_fit_score > %(uq)s)"),
}
JOB_TABS = tuple(_TAB_SQL)


def _display_where(user_id: int, posted_on_or_after, tab: str | None = None,
                   underqualified_at_or_below: int = 5,
                   company: str | None = None) -> tuple[str, dict]:
    """The WHERE fragment + params shared by the page read and the tab counts: this
    user's rows, narrowed to what the posting-age cutoff leaves visible, then to one tab.

    `posted_on_or_after` is a real date computed by the CALLER in Python, never SQL's
    CURRENT_DATE — `date_posted` is written with the app's local date while the docker
    Postgres runs UTC, so an in-database "today" is a day ahead on US evenings."""
    where = ["o.user_id = %(uid)s"]
    params = {"uid": user_id, "uq": underqualified_at_or_below}
    if posted_on_or_after is not None:
        # job_overview already carries the résumé/application columns, so the visibility
        # rule reads them straight off the row rather than re-joining.
        where.append(_AGE_VISIBLE)
        params["cutoff"] = posted_on_or_after
    if tab is not None:
        if tab not in _TAB_SQL:
            raise ValueError(f"unknown jobs tab: {tab!r}")
        where.append(f"({_TAB_SQL[tab]})")
    if company:
        where.append("o.company ILIKE %(company)s")
        params["company"] = f"%{company}%"
    return " AND ".join(where), params


# The sortable columns, as the SQL expression each one orders by. NULLIF folds the ""
# that joined-but-absent text columns read as into NULL, so "no value" is a single case
# the NULLS LAST below can bury — the table has always sorted blanks to the bottom
# regardless of direction, and a blank floating to the top of a descending sort would
# just be noise the user has to scroll past.
_SORT_SQL = {
    "ai_fit": "o.ai_fit_score",
    "source": "NULLIF(o.source, '')",
    "date_matched": "o.date_matched",
    "date_posted": "o.date_posted",
    "company": "NULLIF(o.company, '')",
    "role": "NULLIF(o.role, '')",
    "location": "NULLIF(o.location, '')",
    "applied": "o.applied_date",
}
SORT_COLUMNS = tuple(_SORT_SQL)

# Each tab's natural order when the user hasn't clicked a column header — the orders the
# browser used to impose after the fact. To Apply leads with the freshest postings,
# Manual with whatever you just typed in, and the rest with the best AI Fit.
_TAB_DEFAULT_SORT = {
    "pending": ("date_posted", True),
    "manual": ("date_matched", True),
    "underqualified": ("ai_fit", True),
    "applied": ("ai_fit", True),
}


def _order_by(tab: str | None, sort: str | None, descending: bool) -> str:
    """The ORDER BY clause for one page read. An explicit `sort` wins; otherwise the tab's
    natural order applies. `seq` (overlay insertion) is always the final tie-break so
    paging can never drop or repeat a row between two equal-keyed pages."""
    if sort is None:
        sort, descending = _TAB_DEFAULT_SORT.get(tab, ("ai_fit", True))
    if sort not in _SORT_SQL:
        raise ValueError(f"unknown sort column: {sort!r}")
    direction = "DESC" if descending else "ASC"
    # Newest-added first also means newest `seq` first, so the tie-break follows suit.
    tie = "o.seq DESC" if (sort == "date_matched" and descending) else "o.seq"
    return f"{_SORT_SQL[sort]} {direction} NULLS LAST, {tie}"


def fetch_overview_page(conn, user_id: int, *, posted_on_or_after=None,
                        tab: str | None = None,
                        underqualified_at_or_below: int = 5,
                        company: str | None = None,
                        sort: str | None = None, descending: bool = False,
                        limit: int | None = None, offset: int = 0) -> list[dict]:
    """One rendered page of the jobs table: this user's tracked jobs after the
    posting-age cutoff, narrowed to one tab and the company search, ordered, sliced.

    `posted_on_or_after` None disables the age filter; `tab` None skips the tab split;
    `limit` None returns every matching row."""
    where, params = _display_where(user_id, posted_on_or_after, tab,
                                   underqualified_at_or_below, company)
    sql = (f"SELECT o.* FROM job_overview o WHERE {where} "
           f"ORDER BY {_order_by(tab, sort, descending)}")
    if limit is not None:
        sql += " LIMIT %(limit)s OFFSET %(offset)s"
        params |= {"limit": limit, "offset": offset}
    return conn.execute(sql, params).fetchall()


def count_overview_tabs(conn, user_id: int, *, posted_on_or_after=None,
                        underqualified_at_or_below: int = 5,
                        company: str | None = None) -> dict[str, int]:
    """{tab: row count} for all four tabs in ONE query — the badge numbers the page used
    to compute by walking every rendered row. Counts the same rows each tab renders (the
    company search included), so a badge can never disagree with its tab."""
    where, params = _display_where(user_id, posted_on_or_after, None,
                                   underqualified_at_or_below, company)
    tallies = ", ".join(f"count(*) FILTER (WHERE {sql}) AS {tab}"
                        for tab, sql in _TAB_SQL.items())
    row = conn.execute(
        f"SELECT {tallies} FROM job_overview o WHERE {where}", params).fetchone()
    return {tab: row[tab] for tab in _TAB_SQL}


def count_overview_summary(conn, user_id: int, *, posted_on_or_after=None,
                           company: str | None = None) -> dict[str, int]:
    """The home page's summary-bar tallies, over the rows the page can actually show.
    Counted in SQL for the same reason the tabs are: the page no longer loads every row,
    so it can't add them up itself."""
    where, params = _display_where(user_id, posted_on_or_after, None, 5, company)
    row = conn.execute(f"""
        SELECT count(*) AS total,
               count(*) FILTER (WHERE o.applied_via IS NOT NULL) AS applied,
               count(*) FILTER (WHERE o.html_path IS NOT NULL AND o.html_path <> '')
                   AS generated
        FROM job_overview o WHERE {where}""", params).fetchone()
    return {"total": row["total"], "applied": row["applied"],
            "generated": row["generated"],
            "no_resume_yet": row["total"] - row["generated"],
            "not_applied": row["total"] - row["applied"]}


def fetch_overview_row(conn, user_id: int, job_id: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM job_overview WHERE user_id = %s AND job_id = %s",
        [user_id, job_id]).fetchone()


def fetch_job_urls(conn, user_id: int) -> set[str]:
    """Every URL THIS user already tracks (cheap dedupe during their board scan)."""
    return {r["job_url"] for r in conn.execute(
        "SELECT j.job_url FROM jobs j JOIN user_jobs uj USING (job_id) WHERE uj.user_id = %s",
        [user_id]).fetchall()}


def job_exists(conn, job_id: str) -> bool:
    """True if the posting exists in the SHARED catalog (any user). Use user_job_exists
    to ask whether a specific user tracks it."""
    return conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", [job_id]).fetchone() is not None


def user_job_exists(conn, user_id: int, job_id: str) -> bool:
    """True if THIS user tracks this posting (has a user_jobs overlay row)."""
    return conn.execute("SELECT 1 FROM user_jobs WHERE user_id = %s AND job_id = %s",
                        [user_id, job_id]).fetchone() is not None


def fetch_job_identity(conn, job_id: str) -> dict | None:
    """A posting's board identity — {source, board_token, external_id} — or None. Catalog
    level (identity is shared across users), so no user scoping needed."""
    return conn.execute(
        "SELECT source, board_token, external_id FROM jobs WHERE job_id = %s", [job_id]
    ).fetchone()


def find_job_for_resume(conn, user_id: int, job_id: str | None, url: str | None,
                        resume_name: str, manual_id: str) -> str | None:
    """Which of THIS user's tracked jobs does a freshly generated résumé belong to?

    Priority: the explicit `job_id` (the per-row button knows exactly which job was
    clicked) → the earliest-tracked posting matching the URL OR already owning a résumé
    with this filename → the derived manual id. None = no match (a new manual job should
    be created). Only ever considers postings in this user's overlay."""
    if job_id and user_job_exists(conn, user_id, job_id):
        return job_id
    row = conn.execute(
        """
        SELECT j.job_id FROM user_jobs uj
        JOIN jobs j USING (job_id)
        LEFT JOIN resumes r ON r.user_id = uj.user_id AND r.job_id = uj.job_id
        WHERE uj.user_id = %(uid)s
          AND ((%(url)s::text IS NOT NULL AND j.job_url = %(url)s)
               OR r.html_path = %(name)s
               OR r.html_path LIKE %(name_pat)s)
        ORDER BY j.seq LIMIT 1
        """,
        {"uid": user_id, "url": url, "name": resume_name, "name_pat": f"%/{resume_name}"},
    ).fetchone()
    if row:
        return row["job_id"]
    if user_job_exists(conn, user_id, manual_id):
        return manual_id
    return None


# ── Users & per-user filter config ───────────────────────────────────────────────────────

def ensure_default_user(conn) -> None:
    """Guarantee the seeded default user exists (idempotent). Cheap belt-and-suspenders on
    top of the v8 migration, so the single-user app always has a user to attribute to."""
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (%s, NULL, %s) "
        "ON CONFLICT (id) DO NOTHING", [DEFAULT_USER_ID, "Default user"])


# The scalar/text filter settings (the keyword LISTS live in user_filter_terms).
SETTINGS_COLS = ("salary_floor", "max_posting_age_days", "title_similarity_percent",
                 "target_roles_text", "underqualified_at_or_below")


def upsert_user_settings(conn, user_id: int, cfg: dict) -> None:
    """Create/update a user's scalar+text filter settings from a config dict (only the
    SETTINGS_COLS keys present are written)."""
    vals = {k: cfg[k] for k in SETTINGS_COLS if k in cfg}
    if not vals:
        conn.execute("INSERT INTO user_settings (user_id) VALUES (%s) "
                     "ON CONFLICT (user_id) DO NOTHING", [user_id])
        return
    cols = list(vals)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    conn.execute(
        f"INSERT INTO user_settings (user_id, {', '.join(cols)}) "
        f"VALUES (%s, {', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT (user_id) DO UPDATE SET {updates}",
        [user_id, *vals.values()])


def fetch_user_settings(conn, user_id: int) -> dict | None:
    """A user's scalar/text settings row, or None when they have none (→ neutral defaults)."""
    return conn.execute(
        "SELECT salary_floor, max_posting_age_days, title_similarity_percent, "
        "target_roles_text, underqualified_at_or_below "
        "FROM user_settings WHERE user_id = %s", [user_id]).fetchone()


def replace_user_filter_terms(conn, user_id: int, list_name: str, terms: list[str]) -> None:
    """Replace one keyword list for a user (delete + re-insert in order)."""
    conn.execute("DELETE FROM user_filter_terms WHERE user_id = %s AND list_name = %s",
                 [user_id, list_name])
    for i, term in enumerate(terms):
        conn.execute("INSERT INTO user_filter_terms (user_id, list_name, ord, term) "
                     "VALUES (%s, %s, %s, %s)", [user_id, list_name, i, term])


def fetch_user_filter_terms(conn, user_id: int) -> dict[str, list[str]]:
    """A user's keyword lists as {list_name: [term, …]} (only lists they've saved)."""
    out: dict[str, list[str]] = {}
    for r in conn.execute(
            "SELECT list_name, term FROM user_filter_terms WHERE user_id = %s "
            "ORDER BY list_name, ord", [user_id]).fetchall():
        out.setdefault(r["list_name"], []).append(r["term"])
    return out


def user_has_settings(conn, user_id: int) -> bool:
    """True once a user has a saved settings row (so callers know whether to fall back to
    the neutral defaults)."""
    return conn.execute("SELECT 1 FROM user_settings WHERE user_id = %s",
                        [user_id]).fetchone() is not None


# ── Per-user base résumé (structured, the tailoring._SCHEMA shape) ───────────────────────
# Stored across resume_profiles (1:1) + one ordered child table per section, and round-trips
# as the same dict shape the tailoring/render pipeline already uses.

def upsert_resume_profile(conn, user_id: int, profile: dict) -> None:
    """Store a user's whole base résumé (tailoring._SCHEMA shape): the 1:1 profile row plus
    every section, each replaced in order (delete + re-insert, like replace_user_filter_terms).
    Missing sections/fields are treated as empty. Wrap in `with db.connect()` — one transaction."""
    contact = profile.get("contact") or {}
    conn.execute(
        """
        INSERT INTO resume_profiles (user_id, full_name, location, phone, email, website,
                                     summary, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE SET
            full_name = EXCLUDED.full_name, location = EXCLUDED.location,
            phone = EXCLUDED.phone, email = EXCLUDED.email, website = EXCLUDED.website,
            summary = EXCLUDED.summary, updated_at = now()
        """,
        [user_id, profile.get("name", ""), contact.get("location", ""),
         contact.get("phone", ""), contact.get("email", ""), contact.get("website", ""),
         profile.get("summary", "")])

    # Each section: wipe this user's rows, then re-insert in list order.
    conn.execute("DELETE FROM resume_experience WHERE user_id = %s", [user_id])
    for i, e in enumerate(profile.get("experience") or []):
        conn.execute(
            "INSERT INTO resume_experience (user_id, ord, company, title, dates, location, "
            "bullets) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [user_id, i, e.get("company", ""), e.get("title", ""), e.get("dates", ""),
             e.get("location", ""), list(e.get("bullets") or [])])

    conn.execute("DELETE FROM resume_education WHERE user_id = %s", [user_id])
    for i, ed in enumerate(profile.get("education") or []):
        conn.execute(
            "INSERT INTO resume_education (user_id, ord, institution, degree, date_text, "
            "detail) VALUES (%s, %s, %s, %s, %s, %s)",
            [user_id, i, ed.get("institution", ""), ed.get("degree", ""),
             ed.get("date", ""), ed.get("detail", "")])

    conn.execute("DELETE FROM resume_skills WHERE user_id = %s", [user_id])
    for i, s in enumerate(profile.get("skills") or []):
        conn.execute(
            "INSERT INTO resume_skills (user_id, ord, category, items) VALUES (%s, %s, %s, %s)",
            [user_id, i, s.get("category", ""), s.get("items", "")])

    conn.execute("DELETE FROM resume_competencies WHERE user_id = %s", [user_id])
    for i, term in enumerate(profile.get("competencies") or []):
        conn.execute("INSERT INTO resume_competencies (user_id, ord, term) VALUES (%s, %s, %s)",
                     [user_id, i, term])

    conn.execute("DELETE FROM resume_certifications WHERE user_id = %s", [user_id])
    for i, cert in enumerate(profile.get("certifications") or []):
        conn.execute("INSERT INTO resume_certifications (user_id, ord, text) VALUES (%s, %s, %s)",
                     [user_id, i, cert])


def fetch_resume_profile(conn, user_id: int) -> dict | None:
    """A user's whole base résumé as the tailoring._SCHEMA dict, or None when they have no
    profile row yet. Sections come back in stored order; empty sections are []."""
    prof = conn.execute(
        "SELECT full_name, location, phone, email, website, summary "
        "FROM resume_profiles WHERE user_id = %s", [user_id]).fetchone()
    if prof is None:
        return None
    experience = [
        {"company": r["company"], "title": r["title"], "dates": r["dates"],
         "location": r["location"], "bullets": list(r["bullets"] or [])}
        for r in conn.execute(
            "SELECT company, title, dates, location, bullets FROM resume_experience "
            "WHERE user_id = %s ORDER BY ord", [user_id]).fetchall()]
    education = [
        {"institution": r["institution"], "degree": r["degree"], "date": r["date_text"],
         "detail": r["detail"]}
        for r in conn.execute(
            "SELECT institution, degree, date_text, detail FROM resume_education "
            "WHERE user_id = %s ORDER BY ord", [user_id]).fetchall()]
    skills = [
        {"category": r["category"], "items": r["items"]}
        for r in conn.execute(
            "SELECT category, items FROM resume_skills WHERE user_id = %s ORDER BY ord",
            [user_id]).fetchall()]
    competencies = [r["term"] for r in conn.execute(
        "SELECT term FROM resume_competencies WHERE user_id = %s ORDER BY ord",
        [user_id]).fetchall()]
    certifications = [r["text"] for r in conn.execute(
        "SELECT text FROM resume_certifications WHERE user_id = %s ORDER BY ord",
        [user_id]).fetchall()]
    return {
        "name": prof["full_name"],
        "contact": {"location": prof["location"], "phone": prof["phone"],
                    "email": prof["email"], "website": prof["website"]},
        "summary": prof["summary"],
        "competencies": competencies,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "skills": skills,
    }


def resume_profile_exists(conn, user_id: int) -> bool:
    """True once a user has a stored base résumé profile row."""
    return conn.execute("SELECT 1 FROM resume_profiles WHERE user_id = %s",
                        [user_id]).fetchone() is not None


# ── Embeddings (pgvector) ──────────────────────────────────────────────────────────────
# Vectors cross the driver as their text form ("[0.1,0.2,…]" — which is also valid JSON)
# with an explicit ::vector cast, so no pgvector Python adapter dependency is needed.
# Similarity math happens IN SQL (`<=>` is cosine distance; 1 - distance = similarity),
# so raw vectors never ship to Python on the read side.

def vector_literal(embedding: list[float]) -> str:
    """A Python vector → pgvector's input text form."""
    return json.dumps(list(embedding))


def upsert_job_embedding(conn, job_id: str, embedding: list[float], model: str = "") -> None:
    """Store (or refresh) the description embedding for one job."""
    conn.execute(
        """
        INSERT INTO job_embeddings (job_id, embedding, model)
        VALUES (%s, %s::vector, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            embedding = EXCLUDED.embedding, model = EXCLUDED.model, embedded_at = now()
        """,
        [job_id, vector_literal(embedding), model],
    )


def fetch_profile_embedding(conn, user_id: int, name: str) -> dict | None:
    """A user's cached profile vector — {content_hash, model, embedding: list[float]} — or
    None. Per-user because the base-résumé/target-roles text differs by user."""
    row = conn.execute(
        "SELECT content_hash, model, embedding::text AS embedding "
        "FROM profile_embeddings WHERE user_id = %s AND name = %s", [user_id, name]).fetchone()
    if row:
        row["embedding"] = json.loads(row["embedding"])
    return row


def upsert_profile_embedding(conn, user_id: int, name: str, content_hash: str,
                             embedding: list[float], model: str = "") -> None:
    conn.execute(
        """
        INSERT INTO profile_embeddings (user_id, name, content_hash, model, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        ON CONFLICT (user_id, name) DO UPDATE SET
            content_hash = EXCLUDED.content_hash, model = EXCLUDED.model,
            embedding = EXCLUDED.embedding, updated_at = now()
        """,
        [user_id, name, content_hash, model, vector_literal(embedding)],
    )


def record_rejection(conn, user_id: int, job_url: str, company: str, role: str,
                     reason: str) -> None:
    """Remember that THIS user screened and dropped this posting, so their future scans
    don't waste a description fetch on it. Per-user: one user's rejection must not hide a
    posting from another."""
    if not job_url:
        return
    conn.execute(
        """
        INSERT INTO rejected_jobs (user_id, job_url, company, role, reason)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, job_url) DO UPDATE SET
            reason = EXCLUDED.reason, rejected_at = now()
        """,
        [user_id, job_url, company, role, reason],
    )


def clear_rejection(conn, user_id: int, job_url: str) -> None:
    """Forget a prior deep-screen rejection for this posting, so a manual add is consistent
    with this user's future scans. No-op when the url is blank or was never rejected."""
    if not job_url:
        return
    conn.execute("DELETE FROM rejected_jobs WHERE user_id = %s AND job_url = %s",
                 [user_id, job_url])


def fetch_rejected_urls(conn, user_id: int) -> set[str]:
    """Every URL THIS user's deep screen has ever rejected (their scans skip these up front)."""
    return {r["job_url"] for r in conn.execute(
        "SELECT job_url FROM rejected_jobs WHERE user_id = %s", [user_id]).fetchall()}


def upsert_job_description(conn, job_id: str, description: str) -> None:
    """Store (or refresh) the posting text for one job."""
    conn.execute(
        """
        INSERT INTO job_descriptions (job_id, description)
        VALUES (%s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            description = EXCLUDED.description, fetched_at = now()
        """,
        [job_id, description],
    )


def fetch_job_description_text(conn, job_id: str) -> str | None:
    """The stored posting text for one job, or None if it was never stored."""
    row = conn.execute(
        "SELECT description FROM job_descriptions WHERE job_id = %s", [job_id]).fetchone()
    return row["description"] if row else None


def fetch_board_directory(conn, source: str) -> list[dict]:
    """Every live board for one source, busiest first (mirrors the old JSON sort). Each row
    is {"token", "job_count"} — the shape the scan already consumed from the JSON files.
    Board directories are SHARED (discovery is global), so this takes no user_id."""
    return conn.execute(
        "SELECT token, job_count FROM board_directory WHERE source = %s "
        "ORDER BY job_count DESC, token",
        [source]).fetchall()


def replace_board_directory(conn, source: str, boards: list[dict]) -> int:
    """Replace one source's directory with the freshly-validated `boards` (each {"token",
    "job_count"}): upsert every survivor (refreshing job_count + last_seen, PRESERVING the
    original first_seen) and prune boards that are no longer live. Mirrors the old build's
    "grow the set or drop dead boards" semantics. Returns the live board count."""
    tokens = [b["token"] for b in boards if b.get("token")]
    for board in boards:
        token = board.get("token")
        if not token:
            continue
        conn.execute(
            "INSERT INTO board_directory (source, token, job_count) VALUES (%s, %s, %s) "
            "ON CONFLICT (source, token) DO UPDATE SET "
            "    job_count = EXCLUDED.job_count, last_seen = CURRENT_DATE",
            [source, token, int(board.get("job_count") or 0)],
        )
    # Prune boards that were in this source's directory before but didn't survive validation.
    if tokens:
        conn.execute("DELETE FROM board_directory WHERE source = %s AND token <> ALL(%s)",
                     [source, tokens])
    else:
        conn.execute("DELETE FROM board_directory WHERE source = %s", [source])
    return len(tokens)


def fetch_similarity_map(conn, user_id: int, embedding: list[float],
                         model: str) -> dict[str, float]:
    """{job_id: cosine similarity to `embedding`} for every posting THIS user tracks
    that was embedded by `model`. The embedding is shared (posting-intrinsic); the row
    set is the user's. Only same-model vectors are compared — pgvector ERRORS on mixed
    dimensions, so stale-model rows must be invisible, not merely deprioritized."""
    return {r["job_id"]: r["sim"] for r in conn.execute(
        "SELECT uj.job_id, 1 - (e.embedding <=> %s::vector) AS sim "
        "FROM job_embeddings e JOIN user_jobs uj USING (job_id) "
        "WHERE uj.user_id = %s AND e.model = %s",
        [vector_literal(embedding), user_id, model]).fetchall()}


def fetch_top_unverdicted(conn, user_id: int, embedding: list[float], model: str,
                          limit: int | None = None) -> list[dict]:
    """This user's tracked jobs embedded by `model` that have no AI fit verdict yet,
    most-similar first (for backfill), each with its stored posting text ('' when never
    stored — the caller re-fetches). limit=None returns ALL of them."""
    return conn.execute(
        """
        SELECT uj.job_id, j.company, j.role, j.job_url,
               COALESCE(d.description, '') AS description,
               1 - (e.embedding <=> %s::vector) AS sim
        FROM user_jobs uj
        JOIN jobs j USING (job_id)
        JOIN job_embeddings e USING (job_id)
        LEFT JOIN job_descriptions d USING (job_id)
        WHERE uj.user_id = %s AND uj.ai_fit_score IS NULL AND e.model = %s
        ORDER BY sim DESC
        LIMIT %s
        """,
        [vector_literal(embedding), user_id, model, limit],
    ).fetchall()


def fetch_unembedded_for_model(conn, user_id: int, model: str) -> list[dict]:
    """This user's tracked jobs with NO embedding from `model` — never-embedded rows
    (pre-vector feature) plus rows embedded by a RETIRED model (an EMBEDDING_BACKEND
    switch). The backfill re-embeds these from their stored text; ai_fit_score rides
    along so already-scored jobs skip the verdict call (verdicts judge text, not
    vectors)."""
    return conn.execute(
        """
        SELECT uj.job_id, j.company, j.role, j.job_url, uj.ai_fit_score,
               COALESCE(d.description, '') AS description
        FROM user_jobs uj
        JOIN jobs j USING (job_id)
        LEFT JOIN job_embeddings e USING (job_id)
        LEFT JOIN job_descriptions d USING (job_id)
        WHERE uj.user_id = %s AND (e.job_id IS NULL OR e.model <> %s)
        ORDER BY uj.date_matched DESC NULLS LAST
        """,
        [user_id, model],
    ).fetchall()


def fetch_verdicted_ids(conn, user_id: int) -> set[str]:
    """Every job_id THIS user has already AI-scored (so their scans don't re-score them)."""
    return {r["job_id"] for r in conn.execute(
        "SELECT job_id FROM user_jobs WHERE user_id = %s AND ai_fit_score IS NOT NULL",
        [user_id]).fetchall()}


def set_fit_verdict(conn, user_id: int, job_id: str, score: int, reason: str) -> None:
    conn.execute(
        "UPDATE user_jobs SET ai_fit_score = %s, ai_fit_reason = %s, ai_fit_at = now() "
        "WHERE user_id = %s AND job_id = %s", [score, reason, user_id, job_id])


def clear_fit_verdicts(conn, user_id: int) -> int:
    """Wipe THIS user's AI fit verdicts (used when their base résumé changes — old verdicts
    were judged against the old résumé). Returns how many were cleared."""
    return conn.execute(
        "UPDATE user_jobs SET ai_fit_score = NULL, ai_fit_reason = '', ai_fit_at = NULL "
        "WHERE user_id = %s AND ai_fit_score IS NOT NULL", [user_id]).rowcount


# ── Writes ─────────────────────────────────────────────────────────────────────────────
# Column whitelists: a typo'd key fails loudly instead of becoming SQL.
# jobs is the SHARED posting catalog — the per-user columns (date_found→date_matched,
# ai_fit_*) moved to user_jobs in v8, so they're gone from here.
JOBS_COLS = {"job_id", "source", "board_token", "external_id", "company", "role",
             "location", "job_url", "salary", "technical", "date_posted"}
RESUMES_COLS = {"html_path", "pdf_path", "match_score", "gen_seconds",
                *(f"{stage}_llm" for stage in STAGES)}
GENLOG_COLS = {"user_id", "finished_at", "source", "job_id", "company", "role", "llm",
               "status", "match_score", "duration_sec", "message"}
SCAN_COLS = {"user_id", "started_at", "ended_at", "status", "sources", "filters"}
SCAN_EVENT_COLS = ["scan_id", "seq", "company", "title", "source", "board_token",
                   "location", "external_id", "url", "stage", "outcome", "detail",
                   "reason_bucket"]


def _insert_user_flags(conn, user_id: int, job_id: str, flags: list[str]) -> None:
    """Insert a user's matcher warnings for a posting (the post-v8, user-scoped inserter;
    _insert_flags is the pre-v8 variant kept only for the v3 migration)."""
    for i, flag in enumerate(flags):
        conn.execute("INSERT INTO job_flags (user_id, job_id, ord, flag) VALUES (%s, %s, %s, %s) "
                     "ON CONFLICT (user_id, job_id, flag) DO NOTHING",
                     [user_id, job_id, i, flag])


def upsert_user_job(conn, user_id: int, job_id: str, date_matched: date | None = None) -> bool:
    """Ensure this user tracks this posting (create the overlay row). Idempotent; sets
    date_matched only on first insert (a re-scan never moves an existing match date).
    Returns True when the overlay row was newly created for this user."""
    row = conn.execute(
        "INSERT INTO user_jobs (user_id, job_id, date_matched) "
        "VALUES (%s, %s, COALESCE(%s, CURRENT_DATE)) "
        "ON CONFLICT (user_id, job_id) DO NOTHING RETURNING 1 AS inserted",
        [user_id, job_id, date_matched]).fetchone()
    return row is not None


def _insert(conn, table: str, allowed: set, values: dict) -> None:
    bad = set(values) - allowed
    if bad:
        raise ValueError(f"unknown column(s) for {table}: {bad}")
    cols = list(values)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(['%s'] * len(cols))})",
        list(values.values()),
    )


def _update(conn, table: str, allowed: set, job_id: str, values: dict) -> None:
    bad = set(values) - allowed
    if bad:
        raise ValueError(f"unknown column(s) for {table}: {bad}")
    if not values:
        return
    assignments = ", ".join(f"{col} = %s" for col in values)
    conn.execute(f"UPDATE {table} SET {assignments} WHERE job_id = %s",
                 list(values.values()) + [job_id])


def insert_job(conn, values: dict) -> None:
    """INSERT one SHARED catalog row from {column: typed value}."""
    _insert(conn, "jobs", JOBS_COLS, values)


def upsert_posting(conn, values: dict) -> None:
    """Ensure a SHARED catalog row exists (INSERT … ON CONFLICT DO NOTHING). Used for
    manual jobs, where the catalog row may already linger (e.g. another user, or an
    orphan left after this user deleted their overlay)."""
    bad = set(values) - JOBS_COLS
    if bad:
        raise ValueError(f"unknown column(s) for jobs: {bad}")
    cols = list(values)
    conn.execute(
        f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT (job_id) DO NOTHING",
        list(values.values()))


def update_job(conn, job_id: str, values: dict) -> None:
    """UPDATE the given SHARED catalog columns on one posting."""
    _update(conn, "jobs", JOBS_COLS, job_id, values)


def upsert_discovered_job(conn, user_id: int, values: dict, flags: list[str]) -> bool:
    """Record a board-discovered job for a user against the SHARED catalog + their overlay:
      1. upsert the shared `jobs` row (refresh display fields; preserve identity/date_posted);
      2. ensure this user's `user_jobs` overlay row (carrying their date_matched);
      3. replace this user's flags for the posting.
    Each is one atomic statement, safe across concurrent scans. Returns True when the
    posting was newly added to THIS user's list (a new overlay row), False when it was
    already theirs (a refresh)."""
    conn.execute(
        """
        INSERT INTO jobs (job_id, source, board_token, external_id,
                          date_posted, company, role, location, job_url, salary, technical)
        VALUES (%(job_id)s, %(source)s, %(board_token)s, %(external_id)s,
                %(date_posted)s, %(company)s, %(role)s, %(location)s, %(job_url)s,
                %(salary)s, %(technical)s)
        ON CONFLICT (job_id) DO UPDATE SET
            company     = EXCLUDED.company,
            role        = EXCLUDED.role,
            location    = EXCLUDED.location,
            job_url     = EXCLUDED.job_url,
            salary      = EXCLUDED.salary,
            technical   = EXCLUDED.technical,
            -- a re-scan may omit the posting date; never erase one we already have
            date_posted = COALESCE(EXCLUDED.date_posted, jobs.date_posted)
        """,
        values,
    )
    inserted = upsert_user_job(conn, user_id, values["job_id"], values.get("date_found"))
    conn.execute("DELETE FROM job_flags WHERE user_id = %s AND job_id = %s",
                 [user_id, values["job_id"]])
    _insert_user_flags(conn, user_id, values["job_id"], flags)
    return inserted


def upsert_resume(conn, user_id: int, job_id: str, values: dict) -> None:
    """Create or update the (single) résumé row a user has for a posting. Only the columns
    in `values` are touched on update, so a re-render that passes no timing/LLM data won't
    wipe what's already recorded. The user's user_jobs overlay row must already exist."""
    bad = set(values) - RESUMES_COLS
    if bad:
        raise ValueError(f"unknown column(s) for resumes: {bad}")
    exists = conn.execute("SELECT 1 FROM resumes WHERE user_id = %s AND job_id = %s",
                          [user_id, job_id]).fetchone()
    if exists:
        if not values:
            return
        assignments = ", ".join(f"{col} = %s" for col in values)
        conn.execute(f"UPDATE resumes SET {assignments} WHERE user_id = %s AND job_id = %s",
                     [*values.values(), user_id, job_id])
    else:
        cols = list(values)
        conn.execute(
            f"INSERT INTO resumes (user_id, job_id, {', '.join(cols)}) "
            f"VALUES (%s, %s, {', '.join(['%s'] * len(cols))})",
            [user_id, job_id, *values.values()])


def replace_resume_findings(conn, user_id: int, job_id: str,
                            findings: dict[str, list[str]]) -> None:
    """Replace this user's critic report ({kind: [finding, …]}) for a résumé in one
    transaction. The report describes ONE generation, so stale findings from the previous
    run are deleted rather than merged."""
    conn.execute("DELETE FROM resume_findings WHERE user_id = %s AND job_id = %s",
                 [user_id, job_id])
    for kind, items in findings.items():
        for ord_, finding in enumerate(items):
            conn.execute(
                "INSERT INTO resume_findings (user_id, job_id, kind, ord, finding) "
                "VALUES (%s, %s, %s, %s, %s)", [user_id, job_id, kind, ord_, finding])


def fetch_resume_findings(conn, user_id: int, job_id: str) -> dict[str, list[str]]:
    """One résumé's critic report as {kind: [finding, …]} ({} when none stored)."""
    out: dict[str, list[str]] = {}
    for r in conn.execute(
            "SELECT kind, finding FROM resume_findings WHERE user_id = %s AND job_id = %s "
            "ORDER BY kind, ord", [user_id, job_id]).fetchall():
        out.setdefault(r["kind"], []).append(r["finding"])
    return out


def fetch_findings_map(conn, user_id: int) -> dict[str, dict[str, list[str]]]:
    """{job_id: {kind: [finding, …]}} for every one of THIS user's résumés that has a
    critic report — one query for the whole home page (the fetch_similarity_map pattern)."""
    out: dict[str, dict[str, list[str]]] = {}
    for r in conn.execute(
            "SELECT job_id, kind, finding FROM resume_findings WHERE user_id = %s "
            "ORDER BY job_id, kind, ord", [user_id]).fetchall():
        out.setdefault(r["job_id"], {}).setdefault(r["kind"], []).append(r["finding"])
    return out


def next_resume_version(conn, user_id: int, job_id: str) -> int:
    """The version number the NEXT snapshot of this résumé should get (1 higher than the
    current max, or 1 for the first). Call inside the same transaction as insert_resume_version
    so the number you name the files with matches the row that gets written."""
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS n FROM resume_versions "
        "WHERE user_id = %s AND job_id = %s", [user_id, job_id]).fetchone()
    return row["n"]


def insert_resume_version(conn, user_id: int, job_id: str, version: int, values: dict) -> None:
    """Record one archived résumé snapshot. `version` comes from next_resume_version (the
    caller names the copied files with it). findings/stage_llms are dicts, stored as JSON text."""
    conn.execute(
        "INSERT INTO resume_versions (user_id, job_id, version, html_path, pdf_path, "
        "json_path, match_score, findings, stage_llms, note) "
        "VALUES (%(user_id)s, %(job_id)s, %(version)s, %(html_path)s, %(pdf_path)s, "
        "%(json_path)s, %(match_score)s, %(findings)s, %(stage_llms)s, %(note)s)",
        {"user_id": user_id, "job_id": job_id, "version": version,
         "html_path": values.get("html_path", ""), "pdf_path": values.get("pdf_path", ""),
         "json_path": values.get("json_path", ""), "match_score": values.get("match_score"),
         "findings": json.dumps(values.get("findings") or {}),
         "stage_llms": json.dumps(values.get("stage_llms") or {}),
         "note": values.get("note", "")})


def _row_to_version(r: dict) -> dict:
    """A resume_versions row → a plain dict with findings/stage_llms parsed back from JSON."""
    return {
        "version": r["version"],
        "html_path": r["html_path"],
        "pdf_path": r["pdf_path"],
        "json_path": r["json_path"],
        "match_score": None if r["match_score"] is None else float(r["match_score"]),
        "findings": json.loads(r["findings"]) if r["findings"] else {},
        "stage_llms": json.loads(r["stage_llms"]) if r["stage_llms"] else {},
        "note": r["note"],
        "created_at": r["created_at"],
    }


def list_resume_versions(conn, user_id: int, job_id: str) -> list[dict]:
    """Every saved snapshot of one résumé, newest (highest version) first."""
    return [_row_to_version(r) for r in conn.execute(
        "SELECT * FROM resume_versions WHERE user_id = %s AND job_id = %s "
        "ORDER BY version DESC", [user_id, job_id]).fetchall()]


def get_resume_version(conn, user_id: int, job_id: str, version: int) -> dict | None:
    """One saved snapshot by its version number ({} → None if it doesn't exist)."""
    r = conn.execute(
        "SELECT * FROM resume_versions WHERE user_id = %s AND job_id = %s AND version = %s",
        [user_id, job_id, version]).fetchone()
    return _row_to_version(r) if r else None


def upsert_application(conn, user_id: int, job_id: str, applied_via: str,
                       applied_date: date | None) -> None:
    """Record that this user applied to this job (idempotent — re-marking updates the row)."""
    conn.execute(
        "INSERT INTO applications (user_id, job_id, applied_via, applied_date) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (user_id, job_id) DO UPDATE SET applied_via = EXCLUDED.applied_via, "
        "applied_date = EXCLUDED.applied_date",
        [user_id, job_id, applied_via, applied_date],
    )


def delete_job_row(conn, user_id: int, job_id: str) -> None:
    """Remove one posting from THIS user's list — delete their user_jobs overlay row; the
    user's résumé/application/flags/findings go with it (ON DELETE CASCADE). The SHARED
    catalog row (and its description/embedding) stays for other users."""
    conn.execute("DELETE FROM user_jobs WHERE user_id = %s AND job_id = %s",
                 [user_id, job_id])


def reset_scan_state(conn, user_id: int) -> dict:
    """Return THIS user's "Retrieve Jobs" scan to a fresh state, in one transaction:
    delete every scan-discovered job they haven't acted on — jobs.source <> 'manual', with
    NO résumé and NO application (deleting the user_jobs overlay cascades that user's
    job_flags / findings) — and clear their permanent rejection memory so the next scan
    re-discovers everything. The SHARED jobs catalog, descriptions, and embeddings are left
    intact (reused by the next scan). Returns {"jobs_deleted", "rejections_cleared"}."""
    jobs_deleted = conn.execute(
        """
        DELETE FROM user_jobs uj USING jobs j
        WHERE uj.job_id = j.job_id AND uj.user_id = %s
          AND j.source <> 'manual'
          AND NOT EXISTS (SELECT 1 FROM resumes r
                          WHERE r.user_id = uj.user_id AND r.job_id = uj.job_id)
          AND NOT EXISTS (SELECT 1 FROM applications a
                          WHERE a.user_id = uj.user_id AND a.job_id = uj.job_id)
        """,
        [user_id],
    ).rowcount
    rejections_cleared = conn.execute(
        "DELETE FROM rejected_jobs WHERE user_id = %s", [user_id]
    ).rowcount
    return {"jobs_deleted": jobs_deleted, "rejections_cleared": rejections_cleared}


def insert_generation_log(conn, values: dict) -> None:
    """INSERT one audit row (typed: finished_at datetime, duration_sec float, …)."""
    _insert(conn, "generation_log", GENLOG_COLS, values)


def insert_scan(conn, values: dict) -> int:
    """Insert the parent scan row (status='running') and return its new id.
    `values`: user_id (int), sources (list[str]), filters (dict → JSONB)."""
    bad = set(values) - SCAN_COLS
    if bad:
        raise ValueError(f"unknown column(s) for scans: {bad}")
    row = conn.execute(
        "INSERT INTO scans (user_id, sources, filters) VALUES (%s, %s, %s) RETURNING id",
        [values["user_id"], values.get("sources"), Jsonb(values.get("filters") or {})],
    ).fetchone()
    return row["id"]


def insert_scan_events(conn, rows: list[dict]) -> None:
    """Batched multi-row INSERT of scan_events. Each row is keyed by SCAN_EVENT_COLS.
    Empty list is a no-op."""
    if not rows:
        return
    placeholders = "(" + ", ".join(["%s"] * len(SCAN_EVENT_COLS)) + ")"
    params = [[r.get(c) for c in SCAN_EVENT_COLS] for r in rows]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO scan_events ({', '.join(SCAN_EVENT_COLS)}) VALUES {placeholders}",
            params,
        )


def append_scan_event(conn, scan_id: int, event: dict) -> int:
    """Append ONE event to an existing (possibly already-finished) scan, assigning the next
    seq for that scan. Used for post-scan events like a manual 'Track this job' add. `event`
    is keyed by SCAN_EVENT_COLS (minus scan_id/seq, which are filled in here). Returns the
    seq used."""
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM scan_events WHERE scan_id = %s",
        [scan_id]).fetchone()["n"]
    insert_scan_events(conn, [{**event, "scan_id": scan_id, "seq": seq}])
    return seq


def finish_scan(conn, scan_id: int, ended_at) -> None:
    """Mark a scan complete: set ended_at + status='complete'."""
    conn.execute("UPDATE scans SET ended_at = %s, status = 'complete' WHERE id = %s",
                 [ended_at, scan_id])


def fetch_scans(conn, user_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
    """Scan list for a user, newest first, each joined to its derived tallies
    (the summary footer computed on read — never stored, so it can't drift)."""
    return conn.execute(
        """
        SELECT s.id, s.started_at, s.ended_at, s.status, s.sources, s.filters,
               COALESCE(t.total_events, 0) AS total_events,
               COALESCE(t.sieve_passes, 0) AS sieve_passes,
               COALESCE(t.meaning_net, 0)  AS meaning_net,
               COALESCE(t.stored, 0)       AS stored
        FROM scans s
        LEFT JOIN (
            SELECT scan_id,
                   COUNT(*)                                                   AS total_events,
                   COUNT(*) FILTER (WHERE stage='sieve_pass' AND outcome='keyword')     AS sieve_passes,
                   COUNT(*) FILTER (WHERE stage='sieve_pass' AND outcome='meaning-net') AS meaning_net,
                   COUNT(*) FILTER (WHERE stage='store' AND outcome='stored')           AS stored
            FROM scan_events
            GROUP BY scan_id
        ) t ON t.scan_id = s.id
        WHERE s.user_id = %s
        ORDER BY s.started_at DESC
        LIMIT %s OFFSET %s
        """,
        [user_id, limit, offset],
    ).fetchall()


def fetch_scan_job_events(conn, user_id: int, *, scan_id=None, stage=None, outcome=None,
                          reason_bucket=None, company=None, source=None,
                          limit: int = 40, offset: int = 0) -> tuple[list[dict], int]:
    """A page of JOBS (not events) with every matching event for each, so a job's full
    stage trace is never split across a page boundary. A job is a
    (scan_id, company, title, source, board_token) group; jobs are ordered newest-scan-first
    then by the job's earliest seq (the order it entered the funnel) — EXCEPT when the salary
    reason filter is active (reason_bucket='salary'), where jobs are ordered by the posting's
    top salary (parsed from the reject detail) highest-first, no-salary last. `limit`/`offset`
    count JOBS. Returns (event_rows, total_jobs); event_rows carry the scan's started_at and are
    ordered job_rank ASC, seq ASC (so the caller can group them). Always user-scoped via the
    scans join; scan_id=None → cross-scan. Filters narrow the EVENTS, and a job is present
    iff it has a matching event."""
    wheres = ["s.user_id = %s"]
    params: list = [user_id]
    if scan_id is not None:
        wheres.append("e.scan_id = %s"); params.append(scan_id)
    if stage:
        wheres.append("e.stage = %s"); params.append(stage)
    if outcome:
        wheres.append("e.outcome = %s"); params.append(outcome)
    if reason_bucket:
        wheres.append("e.reason_bucket = %s"); params.append(reason_bucket)
    if company:
        wheres.append("e.company ILIKE %s"); params.append(f"%{company}%")
    if source:
        wheres.append("e.source = %s"); params.append(source)
    where_sql = " AND ".join(wheres)
    base_from = "FROM scan_events e JOIN scans s ON s.id = e.scan_id WHERE " + where_sql
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM (SELECT 1 {base_from} "
        "GROUP BY e.scan_id, e.company, e.title, e.source, e.board_token) t",
        params).fetchone()["n"]
    # Rank each distinct job (dense, 1-based) over the matching events, then return every
    # event whose job falls in the requested rank window. PARTITION/DENSE_RANK keep a job's
    # events together regardless of how seq interleaves them with other jobs. When the salary
    # reason filter is active we rank by the posting's top salary (parsed out of the reject
    # detail, e.g. "salary tops out at $220,000 …"): highest first, unparseable ("no salary
    # stated") last via NULLS LAST. Every other filter keeps the default scan/seq order.
    if reason_bucket == "salary":
        rank_order = ("salary_key DESC NULLS LAST, scan_id DESC, job_first_seq ASC, "
                      "company, title, source, board_token")
    else:
        rank_order = "scan_id DESC, job_first_seq ASC, company, title, source, board_token"
    rows = conn.execute(
        f"""
        WITH matched AS (
            SELECT e.*, s.started_at,
                   MIN(e.seq) OVER (PARTITION BY e.scan_id, e.company, e.title,
                                                 e.source, e.board_token) AS job_first_seq,
                   MAX(replace(substring(e.detail from '\\$([0-9,]+)'), ',', '')::numeric)
                       OVER (PARTITION BY e.scan_id, e.company, e.title,
                                          e.source, e.board_token) AS salary_key
            {base_from}
        ),
        ranked AS (
            SELECT matched.*,
                   DENSE_RANK() OVER (ORDER BY {rank_order}) AS job_rank
            FROM matched
        )
        SELECT * FROM ranked WHERE job_rank > %s AND job_rank <= %s
        ORDER BY job_rank ASC, seq ASC
        """,
        params + [offset, offset + limit],
    ).fetchall()
    return rows, total


# ── Legacy import: one flat CSV/table row → the normalized tables ─────────────────────

def insert_flat_row(conn, flat: dict, user_id: int = DEFAULT_USER_ID) -> None:
    """Split one legacy flat row (keyed by the old CSV headers: "Job ID", "Company",
    "HTML", "Applied", …) into the SHARED catalog + the given user's overlay (jobs +
    user_jobs + job_flags + resumes + applications), converting the string values to real
    types. Shared by the first-run CSV import and the one-time old-schema migration script;
    both attribute the legacy data to the default user."""
    job_id = flat.get("Job ID", "")
    source, board_token, external_id = split_job_id(job_id)
    upsert_posting(conn, {
        "job_id": job_id,
        "source": source,                            # from the id — the CSV's Source column
        "board_token": board_token,                  # is redundant with it by construction
        "external_id": external_id,
        "company": flat.get("Company", ""),
        "role": flat.get("Role", ""),
        "location": flat.get("Location", ""),
        "job_url": flat.get("Job URL", ""),
        "salary": flat.get("Salary", ""),
        "technical": parse_bool(flat.get("Technical")),
        "date_posted": parse_date(flat.get("Date Posted")),
    })
    upsert_user_job(conn, user_id, job_id, parse_date(flat.get("Date Found")) or date.today())
    _insert_user_flags(conn, user_id, job_id,
                       [f for f in (flat.get("Flags", "") or "").split("; ") if f])
    if flat.get("HTML"):                             # a résumé exists for this job
        stage_cols = _legacy_stage_columns(flat.get("Stage LLMs", ""), flat.get("LLM", ""))
        upsert_resume(conn, user_id, job_id, {
            "html_path": flat["HTML"],
            "pdf_path": flat.get("PDF", ""),
            "match_score": parse_float(flat.get("Match Score")),
            "gen_seconds": parse_int(flat.get("Gen Seconds")),
            **{f"{stage}_llm": llm for stage, llm in stage_cols.items()},
        })
    if flat.get("Applied"):                          # an application was recorded
        upsert_application(conn, user_id, job_id, flat["Applied"],
                           parse_date(flat.get("Applied Date")))


def insert_genlog_flat_row(conn, flat: dict) -> None:
    """One legacy generation-log row (old CSV headers) → a typed generation_log row."""
    raw_ts = str(flat.get("timestamp", "") or "").strip()
    try:
        finished_at = datetime.fromisoformat(raw_ts)
    except ValueError:
        finished_at = None
    insert_generation_log(conn, {
        "user_id": DEFAULT_USER_ID,                   # legacy audit rows → the default user
        "finished_at": finished_at,
        "source": flat.get("source", ""),
        "job_id": flat.get("job_id", ""),
        "company": flat.get("company", ""),
        "role": flat.get("role", ""),
        "llm": flat.get("llm", ""),
        "status": flat.get("status", ""),
        "match_score": parse_float(flat.get("match_score")),
        "duration_sec": parse_float(flat.get("duration_sec")),
        "message": flat.get("message", ""),
    })


def _import_legacy_csvs(conn) -> None:
    """One-time import of the old CSV stores into an EMPTY database. Skipped entirely
    once the tables have any rows, so it never duplicates or overwrites DB data. The
    CSV files themselves are left untouched (they're your backup)."""
    if APPLICATIONS_CSV.exists():
        count = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        if count == 0:
            with APPLICATIONS_CSV.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                insert_flat_row(conn, row)
            if rows:
                print(f"[db] Imported {len(rows)} job(s) from {APPLICATIONS_CSV.name} "
                      "(the CSV is kept as a backup; the database is now the live store).")

    if GENERATION_LOG_CSV.exists():
        count = conn.execute("SELECT COUNT(*) AS n FROM generation_log").fetchone()["n"]
        if count == 0:
            with GENERATION_LOG_CSV.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                insert_genlog_flat_row(conn, row)
            if rows:
                print(f"[db] Imported {len(rows)} row(s) from {GENERATION_LOG_CSV.name}.")
