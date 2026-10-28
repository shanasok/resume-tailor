"""One-off: AI-score every stored-but-unscored job for a user (the "Score fits" button).

OpenAI enforces a per-minute request/token rate limit; firing hundreds of gpt-4o calls
back-to-back trips it and most return HTTP 429. The stock scorer doesn't retry, so a
rate limit silently drops the rest of the run. This wrapper paces each call and, on a
429, backs off exponentially and retries — so the whole backlog gets scored, slowly but
completely. run_stored_verdicts only touches still-unscored jobs, so it's resumable:
re-running after a partial run just continues.

Usage:
  python scripts/backfill_fit_verdicts.py [user_id] [top_n]

top_n caps how many get scored (handy for a small test run). Omit it to score all.
"""
import sys
import time
from dotenv import load_dotenv

load_dotenv()  # must precede imports that read API keys / DATABASE_URL

from src import fit_verdict

BASE_DELAY = 3.0          # seconds between calls — stay under the per-minute token cap
MAX_RETRIES = 8           # per job, on 429 only
BACKOFFS = [10, 20, 40, 60, 90, 120, 120, 120]  # seconds to wait after each 429


def _paced_verdict(*args, **kwargs):
    """verdict_for with pacing + 429 backoff. Non-rate-limit errors propagate immediately
    (so run_stored_verdicts logs+skips them as before)."""
    time.sleep(BASE_DELAY)
    for attempt in range(MAX_RETRIES):
        try:
            return _orig_verdict(*args, **kwargs)
        except Exception as e:            # noqa: BLE001 — inspect the message for 429
            if "429" not in str(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFFS[min(attempt, len(BACKOFFS) - 1)]
            print(f"    (rate limited; waiting {wait}s then retrying…)")
            time.sleep(wait)


_orig_verdict = fit_verdict.verdict_for
fit_verdict.verdict_for = _paced_verdict


def main():
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scored = fit_verdict.run_stored_verdicts(user_id, top_n=top_n, on_progress=print)
    print(f"\nDone. Scored {scored} job(s) this run.")


if __name__ == "__main__":
    main()
