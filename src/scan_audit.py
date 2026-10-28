"""Per-scan drop-funnel audit for the Phase 2 job scan, persisted to Postgres.

One ScanAudit is created per find_matches() call (per press of "Retrieve Jobs").
It inserts a parent `scans` row, then records every job that clears the title
sieve as it advances or is dropped — one `scan_events` row per stage transition
(see src/db.py). Events are buffered in memory and flushed in batches, so the
scan's 6 board threads aren't each paying a DB round-trip per line. Thread-safe:
every public method takes the instance lock. Observability only — a DB failure
is logged and swallowed, never breaking the scan.
"""
import sys
import threading
from datetime import datetime, timezone

from src import db

FLUSH_EVERY = 200          # events buffered before a batched INSERT

# Raw match_job reject substrings (see src/matching.py:519-556) → footer bucket key.
# Order matters only in that the first matching substring wins.
_BUCKET_RULES = [
    ("salary tops out at", "salary"),
    ("no salary stated", "salary"),
    ("location is non-us", "non_us"),
    ("us not confirmed", "non_us"),
    ("on-site role", "onsite"),
    ("hybrid outside", "hybrid_zone"),
    ("no 'remote' found", "no_remote"),
    ("days ago", "stale"),
    ("gig/seniority term", "negative_term"),
    ("non-full-time term", "non_fulltime"),
    ("employment/gig phrase", "contract_phrase"),
]

# Display order for the footer (only nonzero buckets print).
BUCKET_ORDER = ["salary", "non_us", "onsite", "hybrid_zone", "no_remote",
                "stale", "negative_term", "non_fulltime", "contract_phrase", "other"]


def categorize(reason: str) -> str:
    """Map one raw match_job reject string to a footer bucket key (see BUCKET_ORDER).
    Unrecognized reasons fall into 'other' so nothing is silently uncounted."""
    low = reason.lower()
    for needle, bucket in _BUCKET_RULES:
        if needle in low:
            return bucket
    return "other"


class NullAudit:
    """No-op stand-in with ScanAudit's interface (used when the DB is unreachable
    and the user chose to run the scan anyway). scan_id is None."""
    scan_id = None

    def sieve_pass(self, job, how):
        pass

    def stage(self, job, stage, outcome, reasons=None):
        pass

    def close(self, match_count):
        pass


class ScanAudit:
    """One per find_matches() call. Thread-safe: every public method takes the lock.
    Events are buffered in memory and flushed to Postgres in batches of FLUSH_EVERY,
    and always on close()."""

    def __init__(self, user_id, sources, filter_summary):
        self._lock = threading.Lock()
        self._seq = 0
        self._buffer = []
        with db.connect() as conn:
            self.scan_id = db.insert_scan(conn, {
                "user_id": user_id,
                "sources": list(sources),
                "filters": dict(filter_summary or {}),
            })

    # ── internal (lock held) ────────────────────────────────────────────────
    def _append(self, job, stage, outcome, detail="", reason_bucket=None):
        self._seq += 1
        self._buffer.append({
            "scan_id": self.scan_id, "seq": self._seq,
            "company": job.get("company"), "title": job.get("title"),
            "source": job.get("source"), "board_token": job.get("board_token"),
            "location": job.get("location"),
            "external_id": job.get("id"), "url": job.get("url"),
            "stage": stage, "outcome": outcome,
            "detail": detail, "reason_bucket": reason_bucket,
        })
        if len(self._buffer) >= FLUSH_EVERY:
            self._flush()

    def _flush(self):
        """Batched INSERT of the buffer. Swallows DB errors — the scan must go on."""
        if not self._buffer:
            return
        batch, self._buffer = self._buffer, []
        try:
            with db.connect() as conn:
                db.insert_scan_events(conn, batch)
        except Exception as e:
            print(f"[scan_audit] flush of {len(batch)} event(s) failed: {e}",
                  file=sys.stderr)

    # ── public, thread-safe ─────────────────────────────────────────────────
    def sieve_pass(self, job, how):
        with self._lock:
            outcome = "meaning-net" if how.startswith("meaning-net") else "keyword"
            self._append(job, "sieve_pass", outcome, detail=how)

    def stage(self, job, stage, outcome, reasons=None):
        with self._lock:
            detail = "; ".join(reasons) if reasons else ""
            bucket = categorize(reasons[0]) if reasons else None
            self._append(job, stage, outcome, detail=detail, reason_bucket=bucket)

    def close(self, match_count):
        with self._lock:
            self._flush()
            try:
                with db.connect() as conn:
                    db.finish_scan(conn, self.scan_id, datetime.now(timezone.utc))
            except Exception as e:
                print(f"[scan_audit] finish_scan failed: {e}", file=sys.stderr)
