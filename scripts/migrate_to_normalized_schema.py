"""One-time migration: the old single-flat-table schema → the normalized schema.

The first Postgres version of this app used one wide `jobs` table that mirrored the
old CSV (every column TEXT, résumé + application data folded in) and a TEXT-typed
`generation_log`. This script splits/retypes that data into the normalized schema
in src/db.py (jobs / resumes / applications + a typed generation_log).

Safe to run any time:
  - Everything happens in ONE transaction — any error rolls the whole thing back.
  - The old tables aren't dropped; they're renamed to *_flat_legacy and kept as an
    in-database backup. Drop them whenever you're satisfied (command printed at the end).
  - If the database already has the new schema, it says so and exits.

Run it (with the db container up):
    docker compose up -d db
    .venv/bin/python scripts/migrate_to_normalized_schema.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # make `src` importable when run as a script

from src import db

# The old flat jobs table's columns, in order, with the display names insert_flat_row
# expects. (This shape is frozen history — it only exists here, for the migration.)
OLD_JOBS_COLUMNS = [
    ("job_id", "Job ID"), ("source", "Source"), ("date_found", "Date Found"),
    ("date_posted", "Date Posted"), ("company", "Company"), ("role", "Role"),
    ("location", "Location"), ("job_url", "Job URL"), ("flags", "Flags"),
    ("salary", "Salary"), ("technical", "Technical"), ("match_score", "Match Score"),
    ("html", "HTML"), ("pdf", "PDF"), ("applied", "Applied"),
    ("applied_date", "Applied Date"), ("gen_seconds", "Gen Seconds"),
    ("llm", "LLM"), ("stage_llms", "Stage LLMs"),
]


def main() -> None:
    with db._connect_raw() as conn:                  # one connection = one transaction
        if not db._has_old_flat_schema(conn):
            print("Nothing to migrate — this database already has the normalized schema.")
            return

        # ── 1. Move the old tables out of the way (kept as backups) ──
        conn.execute("ALTER TABLE jobs RENAME TO jobs_flat_legacy")
        old_genlog = conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'generation_log' AND column_name = 'timestamp'"
        ).fetchone() is not None
        if old_genlog:
            conn.execute("ALTER TABLE generation_log RENAME TO generation_log_flat_legacy")

        # ── 2. Create the normalized schema (runs ALL migrations, so a fresh conversion
        #       lands on the newest version directly) ──
        db.apply_migrations(conn)

        # ── 3. Copy + convert the jobs (splitting out resumes/applications, typing values) ──
        old_rows = conn.execute("SELECT * FROM jobs_flat_legacy ORDER BY seq").fetchall()
        for old in old_rows:
            db.insert_flat_row(conn, {field: old[col] for col, field in OLD_JOBS_COLUMNS})

        # ── 4. Copy + convert the generation log ──
        log_n = 0
        if old_genlog:
            log_rows = conn.execute(
                "SELECT * FROM generation_log_flat_legacy ORDER BY id").fetchall()
            for old in log_rows:
                db.insert_genlog_flat_row(conn, old)   # old columns already match the CSV headers
            log_n = len(log_rows)

        # ── 5. Report what landed where (before the transaction commits) ──
        counts = {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
                  for t in ("jobs", "resumes", "applications", "generation_log")}
        print(f"Migrated {len(old_rows)} flat job row(s) → "
              f"{counts['jobs']} jobs, {counts['resumes']} resumes, "
              f"{counts['applications']} applications.")
        print(f"Migrated {log_n} generation_log row(s) → {counts['generation_log']} typed rows.")
        print("\nOld tables kept as backups: jobs_flat_legacy"
              + (", generation_log_flat_legacy" if old_genlog else "") + ".")
        print("When you're satisfied, drop them with:")
        print('  docker compose exec db psql -U resume resume_handler '
              '-c "DROP TABLE jobs_flat_legacy'
              + (", generation_log_flat_legacy" if old_genlog else "") + ';"')


if __name__ == "__main__":
    main()
