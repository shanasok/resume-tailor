import threading

import psycopg
import pytest

from src import db
from src.scan_audit import categorize, NullAudit, ScanAudit
from src.current_user import DEFAULT_USER_ID as UID


_TEST_DB = "resume_handler_test"
_SERVER_URL, _, _ = db.database_url().rpartition("/")
TEST_DATABASE_URL = f"{_SERVER_URL}/{_TEST_DB}"


def _server_available() -> bool:
    try:
        with psycopg.connect(f"{_SERVER_URL}/postgres", connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="Postgres not reachable — start it with: docker compose up -d db",
)


@pytest.fixture(autouse=True)
def test_database(monkeypatch, tmp_path):
    """Point src.db at the throwaway database, schema dropped + rebuilt fresh per test."""
    with psycopg.connect(f"{_SERVER_URL}/postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                            [_TEST_DB]).fetchone():
            conn.execute(f'CREATE DATABASE "{_TEST_DB}"')

    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "APPLICATIONS_CSV", db.APPLICATIONS_CSV.with_name("nope.csv"))
    monkeypatch.setattr(db, "GENERATION_LOG_CSV", db.GENERATION_LOG_CSV.with_name("nope.csv"))

    with db._connect_raw() as conn:
        conn.execute("DROP VIEW IF EXISTS job_overview")
        conn.execute("DROP TABLE IF EXISTS jobs, job_flags, resumes, resume_findings, "
                     "resume_versions, applications, generation_log, job_embeddings, "
                     "profile_embeddings, job_descriptions, rejected_jobs, users, user_jobs, "
                     "user_settings, user_filter_terms, resume_profiles, resume_experience, "
                     "resume_education, resume_skills, resume_competencies, "
                     "resume_certifications, board_directory, scans, scan_events, "
                     "schema_migrations CASCADE")
    with db.connect() as conn:
        pass
    yield
    db._schema_ready = False


@pytest.mark.parametrize("reason,bucket", [
    ("salary tops out at $150,000 (< $200,000)", "salary"),
    ("no salary stated in description", "salary"),
    ("location is non-US", "non_us"),
    ("US not confirmed in location or description", "non_us"),
    ("on-site role", "onsite"),
    ("hybrid outside your commute zone", "hybrid_zone"),
    ("no 'remote' found in location or description", "no_remote"),
    ("posted 200 days ago (> 90-day limit)", "stale"),
    ("title gig/seniority term(s): junior", "negative_term"),
    ("title non-full-time term(s): contract", "non_fulltime"),
    ("description employment/gig phrase(s): 1099", "contract_phrase"),
    ("some brand new reason we never saw", "other"),
])
def test_categorize_maps_reject_reason_to_bucket(reason, bucket):
    assert categorize(reason) == bucket


def _job(**kw):
    base = {"company": "Acme", "title": "Engineering Manager",
            "source": "greenhouse", "board_token": "acme", "location": "Remote - US"}
    base.update(kw)
    return base


FS = {"salary_floor": 200000, "max_age": 90, "commute": ["denver"],
      "title_similarity_pct": 45, "domain_kw": 12, "negative_kw": 8}


def _events(scan_id):
    with db.connect() as conn:
        return conn.execute("SELECT * FROM scan_events WHERE scan_id = %s "
                            "ORDER BY seq", [scan_id]).fetchall()


def test_scan_row_created_with_header_context():
    a = ScanAudit(UID, ["greenhouse"], FS)
    a.close(0)
    with db.connect() as conn:
        s = conn.execute("SELECT * FROM scans WHERE id = %s", [a.scan_id]).fetchone()
    assert s["status"] == "complete"
    assert s["sources"] == ["greenhouse"]
    assert s["filters"]["salary_floor"] == 200000
    assert s["filters"]["domain_kw"] == 12


def test_events_recorded_with_seq_and_bucket():
    a = ScanAudit(UID, ["greenhouse"], {})
    a.sieve_pass(_job(), "keyword")
    a.stage(_job(), "list_screen", "dropped", ["on-site role"])
    a.sieve_pass(_job(), "meaning-net (48%)")
    a.stage(_job(), "deep_screen", "dropped", ["salary tops out at $150,000 (< $200,000)"])
    a.stage(_job(), "store", "stored", ["ai_fit 8/10"])
    a.close(1)
    evs = _events(a.scan_id)
    assert [e["seq"] for e in evs] == [1, 2, 3, 4, 5]
    # sieve_pass outcome is normalized; the % detail is preserved
    mn = next(e for e in evs if e["stage"] == "sieve_pass" and e["outcome"] == "meaning-net")
    assert "48%" in mn["detail"]
    deep = next(e for e in evs if e["stage"] == "deep_screen")
    assert deep["reason_bucket"] == "salary"
    assert deep["detail"] == "salary tops out at $150,000 (< $200,000)"


def test_events_carry_external_id_and_url():
    a = ScanAudit(UID, ["greenhouse"], {})
    a.sieve_pass(_job(id="42", url="https://boards.greenhouse.io/acme/jobs/42"), "keyword")
    a.close(0)
    ev = _events(a.scan_id)[0]
    assert ev["external_id"] == "42"
    assert ev["url"] == "https://boards.greenhouse.io/acme/jobs/42"


def test_buffer_flushes_at_threshold_before_close(monkeypatch):
    import src.scan_audit as sa
    monkeypatch.setattr(sa, "FLUSH_EVERY", 3)
    a = ScanAudit(UID, ["greenhouse"], {})
    for _ in range(3):
        a.sieve_pass(_job(), "keyword")     # hits the threshold → flush
    # Rows are visible BEFORE close() because the threshold flushed them.
    assert len(_events(a.scan_id)) == 3
    a.close(0)


def test_thread_safe_row_count_and_unique_seq():
    a = ScanAudit(UID, ["greenhouse"], {})
    def worker():
        for _ in range(100):
            a.sieve_pass(_job(), "keyword")
            a.stage(_job(), "deep_screen", "dropped", ["on-site role"])
    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    a.close(0)
    evs = _events(a.scan_id)
    assert len(evs) == 1200                       # 6 × 100 × 2
    assert len({e["seq"] for e in evs}) == 1200   # no duplicate/lost seq


def test_close_stamps_ended_even_after_exception():
    a = ScanAudit(UID, ["greenhouse"], {})
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        a.close(0)                                # the find_matches finally-path
    with db.connect() as conn:
        s = conn.execute("SELECT * FROM scans WHERE id = %s", [a.scan_id]).fetchone()
    assert s["ended_at"] is not None and s["status"] == "complete"


def test_null_audit_is_a_silent_noop():
    a = NullAudit()
    assert a.scan_id is None
    a.sieve_pass({"title": "x"}, "keyword")
    a.stage({"title": "x"}, "deep_screen", "dropped", ["on-site role"])
    a.close(0)


def test_flush_failure_is_swallowed(monkeypatch):
    # A DB write blowing up mid-scan must not propagate into the caller.
    a = ScanAudit(UID, ["greenhouse"], {})
    monkeypatch.setattr("src.scan_audit.db.insert_scan_events",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    a.sieve_pass(_job(), "keyword")   # buffered
    a.close(0)                        # triggers a flush that raises internally — must not raise
