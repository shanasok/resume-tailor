"""In-process scheduler for BATCH resume generation.

The web UI lets you tick several tracked jobs and schedule a "run" — generate a
tailored resume for each of them at a chosen Eastern-Time moment, with a chosen LLM.
This module owns that feature's persistence and timing.

How it works:
  • Runs are stored as JSON in data/scheduled_runs.json so they survive an app restart.
  • A single background thread (started once by app.py) wakes every CHECK_INTERVAL
    seconds, finds any PENDING run whose fire time has arrived (compared in Eastern
    Time), and generates a resume for each job in it by calling the `generate_cb`
    callback the app supplies — the SAME single-job generation the per-row button uses.

Important: the app (`python app.py`) must be running for a run to fire. A run whose
time passed while the app was down simply fires the next time the app starts (it's
still "due"). Nothing is lost; it just runs late.

A run/job lifecycle:
  run:  pending → running → done   (or cancelled while still pending)
  job:  pending → generating → done | error | skipped
"""
import atexit                                          # release the wake-lock when the app exits
import json                                            # read/write the runs store
import os                                              # our PID (for caffeinate's -w backstop)
import subprocess                                      # run macOS `caffeinate` to prevent sleep
import sys                                             # detect macOS (caffeinate is macOS-only)
import threading                                       # the background timer thread + a lock
import time                                            # sleep between due-checks
import uuid                                            # unique run IDs
from datetime import datetime                          # parse/compare fire times
from pathlib import Path                               # file paths

# Eastern Time (handles EST/EDT automatically). zoneinfo ships with Python 3.9+.
try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:                                      # pragma: no cover - very old Python
    EASTERN = None                                     # fall back to naive local time below

PROJECT_ROOT = Path(__file__).parent.parent
STORE_PATH = PROJECT_ROOT / "data" / "scheduled_runs.json"

CHECK_INTERVAL = 30                                    # seconds between "anything due?" checks

# Guards every read-modify-write of the JSON store so the background thread and web
# requests (create/cancel) can't clobber each other. RLock = reentrant (same thread ok).
_lock = threading.RLock()
_started = False                                       # ensures we only launch the thread once
_caffeinate = None                                     # the `caffeinate` process keeping macOS awake


# ── Persistence ───────────────────────────────────────────────────────────────
def _load() -> dict:
    """Read the runs store, returning {'runs': [...]}. Missing/corrupt file → empty."""
    if not STORE_PATH.exists():
        return {"runs": []}
    try:
        with STORE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"runs": []}
    data.setdefault("runs", [])
    return data


def _save(data: dict) -> None:
    """Overwrite the runs store with `data`."""
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Time helpers ────────────────────────────────────────────────────────────────
def _now():
    """The current moment in Eastern Time (or naive local if zoneinfo is unavailable)."""
    return datetime.now(EASTERN) if EASTERN else datetime.now()


def _parse_fire(fire_at: str):
    """Turn a stored 'YYYY-MM-DDTHH:MM' string (entered as Eastern Time) into a
    comparable datetime. Returns None if it can't be parsed."""
    try:
        dt = datetime.fromisoformat(fire_at)
    except (ValueError, TypeError):
        return None
    if EASTERN and dt.tzinfo is None:                  # the user typed an Eastern wall-clock time
        dt = dt.replace(tzinfo=EASTERN)
    return dt


def is_valid_fire(fire_at: str) -> bool:
    """True if `fire_at` is a parseable schedule time (used to validate UI input)."""
    return _parse_fire(fire_at) is not None


# ── Keep-awake (macOS) ───────────────────────────────────────────────────────────
# While any run is pending or still generating, hold the Mac awake with the built-in
# `caffeinate` tool so a scheduled run actually fires; release it the moment nothing is
# left so the Mac can sleep normally again. NOTE: caffeinate cannot stop sleep caused by
# CLOSING THE LID — keep the laptop open. On battery, `-s` is ignored but `-i`/`-m` still
# prevent idle sleep.
def _set_keep_awake(active: bool) -> None:
    """Start/stop the wake-lock to match `active`. No-op on non-macOS or if caffeinate
    isn't available. Safe to call repeatedly (idempotent)."""
    global _caffeinate
    if sys.platform != "darwin":
        return
    with _lock:
        alive = _caffeinate is not None and _caffeinate.poll() is None
        if active and not alive:
            try:
                # -i no idle sleep · -m no disk sleep · -s no system sleep (AC only) ·
                # -w <pid>: caffeinate also exits on its own if THIS app process dies.
                _caffeinate = subprocess.Popen(
                    ["caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, ValueError):
                _caffeinate = None                     # caffeinate missing → silently skip
        elif not active and alive:
            try:
                _caffeinate.terminate()
            except OSError:
                pass
            _caffeinate = None


def is_keeping_awake() -> bool:
    """True if we're currently holding the Mac awake for a scheduled run (UI indicator)."""
    return _caffeinate is not None and _caffeinate.poll() is None


def _has_active_runs() -> bool:
    """True if any run is still pending or running (i.e. we should stay awake)."""
    return any(r.get("status") in ("pending", "running") for r in _load()["runs"])


# ── Public API used by the web routes ───────────────────────────────────────────
def list_runs() -> list:
    """All scheduled runs, newest first (by creation time)."""
    runs = _load()["runs"]
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def create_run(fire_at: str, llms, jobs: list) -> dict:
    """Create and persist a new pending run. `jobs` is a list of {job_id, company, role}
    dicts. `llms` is the per-stage backend dict (generate/external/check/regenerate) that the
    run passes straight to the generate callback; a plain backend string is also accepted for
    backward compatibility. Returns the stored run."""
    run = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fire_at": fire_at,                            # Eastern wall-clock, e.g. "2026-06-27T19:30"
        "llms": llms,                                  # per-stage backend dict (or a string)
        "status": "pending",
        "jobs": [{"job_id": j["job_id"], "company": j.get("company", ""),
                  "role": j.get("role", ""), "status": "pending",
                  "message": "", "match_score": None} for j in jobs],
    }
    with _lock:
        data = _load()
        data["runs"].append(run)
        _save(data)
    _set_keep_awake(True)                               # hold the Mac awake until it fires
    return run


def cancel_run(run_id: str) -> bool:
    """Cancel a still-pending run. Returns True if it was found and cancelled (a run
    that's already running/done can't be cancelled)."""
    with _lock:
        data = _load()
        for r in data["runs"]:
            if r["id"] == run_id and r["status"] == "pending":
                r["status"] = "cancelled"
                _save(data)
                _set_keep_awake(_has_active_runs())    # maybe release the wake-lock now
                return True
    return False


def delete_run(run_id: str) -> bool:
    """Remove a run from the store entirely (used to clear out finished/cancelled runs)."""
    with _lock:
        data = _load()
        kept = [r for r in data["runs"] if r["id"] != run_id]
        if len(kept) == len(data["runs"]):
            return False
        data["runs"] = kept
        _save(data)
    return True


def add_jobs(run_id: str, jobs: list) -> dict:
    """Add jobs (list of {job_id, company, role}) to a STILL-PENDING run, de-duped by
    job_id. Returns the updated run, or None if the run isn't pending (can't edit a run
    that's already firing). Re-arms the wake-lock since there's still work to do."""
    with _lock:
        data = _load()
        run = next((r for r in data["runs"] if r["id"] == run_id), None)
        if not run or run["status"] != "pending":
            return None
        have = {j["job_id"] for j in run["jobs"]}
        for j in jobs:
            if j["job_id"] not in have:
                run["jobs"].append({"job_id": j["job_id"], "company": j.get("company", ""),
                                    "role": j.get("role", ""), "status": "pending",
                                    "message": "", "match_score": None})
                have.add(j["job_id"])
        _save(data)
    _set_keep_awake(True)
    return run


def remove_job(run_id: str, job_id: str) -> str:
    """Remove one job from a STILL-PENDING run. Returns 'removed', or 'run_deleted' if that
    emptied the run (so the empty run is dropped), or None if the run isn't pending or the
    job wasn't in it."""
    with _lock:
        data = _load()
        run = next((r for r in data["runs"] if r["id"] == run_id), None)
        if not run or run["status"] != "pending":
            return None
        kept = [j for j in run["jobs"] if j["job_id"] != job_id]
        if len(kept) == len(run["jobs"]):
            return None                                # job wasn't part of this run
        run["jobs"] = kept
        result = "removed"
        if not kept:                                   # last job removed → drop the empty run
            data["runs"] = [r for r in data["runs"] if r["id"] != run_id]
            result = "run_deleted"
        _save(data)
    _set_keep_awake(_has_active_runs())                # maybe release the wake-lock now
    return result


# ── Firing (background thread) ───────────────────────────────────────────────────
def _claim(run_id: str):
    """Atomically flip a run from pending → running and return it, so two checks can
    never process the same run twice (and a cancel that lands first wins). Returns the
    claimed run, or None if it wasn't pending anymore."""
    with _lock:
        data = _load()
        for r in data["runs"]:
            if r["id"] == run_id and r["status"] == "pending":
                r["status"] = "running"
                r["started_at"] = datetime.now().isoformat(timespec="seconds")
                _save(data)
                return r
    return None


def _persist_run(run: dict) -> None:
    """Write the in-progress `run` back to the store (called after each job updates),
    unless it was cancelled out from under us — then leave the cancellation alone."""
    with _lock:
        data = _load()
        for i, r in enumerate(data["runs"]):
            if r["id"] == run["id"]:
                if r.get("status") == "cancelled":
                    return
                data["runs"][i] = run
                _save(data)
                return


def _process_run(run: dict, generate_cb) -> None:
    """Generate a resume for each job in a claimed (running) run, updating status as it
    goes. `generate_cb(job_id, llms)` returns a dict like the per-row generate route:
    {'ok': bool, 'message': str, 'match_score': float|None, 'skipped': bool,
    'closed': bool}."""
    # Runs created before per-stage LLMs stored a single "llm" string; newer runs store an
    # "llms" dict. Pass whichever exists — the callback handles both.
    run_llms = run.get("llms", run.get("llm"))
    for job in run["jobs"]:
        if job.get("status") not in (None, "", "pending"):
            continue                                   # already handled (e.g. on a re-run)
        job["status"] = "generating"
        _persist_run(run)
        try:
            result = generate_cb(job["job_id"], run_llms)
        except Exception as e:                         # a callback blow-up must not kill the thread
            result = {"ok": False, "message": f"Generation failed: {e}"}
        if result.get("ok"):
            job["status"] = "done"
            job["match_score"] = result.get("match_score")
        elif result.get("skipped"):                    # already had a resume → nothing to do
            job["status"] = "skipped"
        elif result.get("closed"):                     # posting closed since discovery → don't generate
            job["status"] = "closed"
        else:
            job["status"] = "error"
        job["message"] = result.get("message", "")
        _persist_run(run)
    run["status"] = "done"
    run["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _persist_run(run)


def _loop(generate_cb) -> None:
    """The background thread body: every CHECK_INTERVAL seconds, fire any due run and keep
    the Mac awake while runs remain (releasing it once they're all done)."""
    while True:
        try:
            for run in _load()["runs"]:
                if run.get("status") == "pending":
                    fire = _parse_fire(run.get("fire_at"))
                    if fire is not None and _now() >= fire:
                        claimed = _claim(run["id"])     # only one claim wins
                        if claimed:
                            _process_run(claimed, generate_cb)
            # Stay awake while anything is pending/running; let the Mac sleep once idle.
            _set_keep_awake(_has_active_runs())
        except Exception:
            pass                                       # never let a transient error stop the loop
        time.sleep(CHECK_INTERVAL)


def start(generate_cb) -> None:
    """Launch the scheduler thread once. `generate_cb(job_id, llm)` is the app's
    single-job generation helper. Safe to call more than once (later calls are no-ops)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    atexit.register(_set_keep_awake, False)            # always release the wake-lock on exit
    threading.Thread(target=_loop, args=(generate_cb,), daemon=True).start()
