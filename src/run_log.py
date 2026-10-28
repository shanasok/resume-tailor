"""Append-only database log of every résumé-generation attempt (manual + scheduled).

This is the flat audit trail that answers "what ran, when, and why did it fail?" —
especially useful for overnight scheduled batch runs, where the per-run JSON in
data/scheduled_runs.json gets overwritten each run and is awkward to scan.

It used to be a CSV (logs/generation_log.csv); rows now go into the `generation_log`
table in Postgres (see src/db.py, which also imports the old CSV on first run).
To browse it:  docker compose exec db psql -U resume resume_handler
               -c "SELECT * FROM generation_log ORDER BY id DESC LIMIT 20;"

One row is inserted per job generation, the moment it finishes (success OR failure),
by `log()`.

Columns:
  finished_at   when the attempt finished (TIMESTAMP)
  source        "manual" (per-row button) or "scheduled" (batch run)
  job_id        the tracked job's ID (e.g. greenhouse:anthropic:5110511008) — plain
                text, no foreign key: the audit trail must survive the job's deletion
  company, role the posting
  llm           "claude" | "chatgpt" | a local model key like "local_qwen"
  status        done | skipped | closed | error
  match_score   fit score out of 10 (NULL unless status=done)
  duration_sec  wall-clock seconds the attempt took
  message       the human-readable outcome / error text (the WHY)
"""
import sys                                              # report logging failures on stderr
from datetime import datetime, timezone                 # timezone-AWARE timestamps

from src import db                                      # the PostgreSQL store


def _status_of(result: dict) -> str:
    """Collapse a _generate_for_job result dict into one status word."""
    if result.get("ok"):
        return "done"
    if result.get("skipped"):
        return "skipped"
    if result.get("closed"):
        return "closed"
    return "error"


def log(user_id: int, source: str, job_id: str, llm: str, result: dict,
        duration_sec: float) -> None:
    """Insert one row describing a finished generation attempt (attributed to `user_id`).
    Never raises — a logging failure must never break (or fail) the generation it records."""
    try:
        row = {
            "user_id": user_id,
            # Aware UTC timestamp: the host app and the container run in different local
            # timezones, so a naive now() would mix meanings in one column.
            "finished_at": datetime.now(timezone.utc),
            "source": source,
            "job_id": job_id,
            "company": result.get("company", ""),
            "role": result.get("role", ""),
            "llm": llm,
            "status": _status_of(result),
            "match_score": db.parse_float(result.get("match_score")),
            "duration_sec": round(duration_sec, 1),
            "message": result.get("message", ""),
        }
        with db.connect() as conn:
            db.insert_generation_log(conn, row)
    except Exception as e:                              # logging must never break generation…
        print(f"[run_log] failed to record generation attempt: {e}", file=sys.stderr)
