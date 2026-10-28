# tests/test_scan_audit_db.py — migration v13's scan-audit tables: `scans` (one row per
# job-search run) and `scan_events` (one row per drop-funnel decision within that scan).
# Same throwaway-database pattern as test_resume_versions_db.py: needs a running Postgres
# (docker compose up -d db) and skips cleanly when none is reachable.
import uuid

import psycopg
import pytest

from src import db, tracker
from src.current_user import DEFAULT_USER_ID as UID   # the seeded default user (id 1)


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


def test_v13_creates_scan_tables():
    with db.connect() as conn:
        cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'scan_events'").fetchall()}
        assert {"scan_id", "seq", "company", "title", "source", "board_token",
                "location", "stage", "outcome", "detail", "reason_bucket"} <= cols
        # scans table exists with a JSONB filters column
        ftype = conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'scans' AND column_name = 'filters'").fetchone()
        assert ftype["data_type"] == "jsonb"


def test_v14_adds_external_id_and_url_columns():
    with db.connect() as conn:
        cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'scan_events'").fetchall()}
        assert {"external_id", "url"} <= cols


def test_deleting_a_scan_cascades_its_events():
    with db.connect() as conn:
        sid = conn.execute(
            "INSERT INTO scans (user_id, sources, filters) "
            "VALUES (%s, %s, %s) RETURNING id",
            [UID, ["greenhouse"], None]).fetchone()["id"]
        conn.execute(
            "INSERT INTO scan_events (scan_id, seq, stage, outcome) "
            "VALUES (%s, 1, 'sieve_pass', 'keyword')", [sid])
    with db.connect() as conn:
        conn.execute("DELETE FROM scans WHERE id = %s", [sid])
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM scan_events "
                         "WHERE scan_id = %s", [sid]).fetchone()["n"]
        assert n == 0


from datetime import datetime, timezone

def test_insert_scan_and_events_roundtrip():
    with db.connect() as conn:
        sid = db.insert_scan(conn, {
            "user_id": UID,
            "sources": ["greenhouse", "lever"],
            "filters": {"salary_floor": 200000, "max_age": 90},
        })
        db.insert_scan_events(conn, [
            {"scan_id": sid, "seq": 1, "company": "Acme", "title": "Eng Mgr",
             "source": "greenhouse", "board_token": "acme", "location": "Remote - US",
             "stage": "sieve_pass", "outcome": "keyword", "detail": "", "reason_bucket": None},
            {"scan_id": sid, "seq": 2, "company": "Acme", "title": "Eng Mgr",
             "source": "greenhouse", "board_token": "acme", "location": "Remote - US",
             "stage": "deep_screen", "outcome": "dropped",
             "detail": "on-site role", "reason_bucket": "onsite"},
        ])
        db.finish_scan(conn, sid, datetime.now(timezone.utc))
    with db.connect() as conn:
        scan = conn.execute("SELECT * FROM scans WHERE id = %s", [sid]).fetchone()
        assert scan["status"] == "complete"
        assert scan["ended_at"] is not None
        assert scan["sources"] == ["greenhouse", "lever"]
        assert scan["filters"]["salary_floor"] == 200000     # JSONB → dict
        evs = conn.execute("SELECT * FROM scan_events WHERE scan_id = %s "
                           "ORDER BY seq", [sid]).fetchall()
        assert [e["seq"] for e in evs] == [1, 2]
        assert evs[1]["reason_bucket"] == "onsite"

def test_insert_scan_events_empty_is_noop():
    with db.connect() as conn:
        db.insert_scan_events(conn, [])   # must not raise


def test_insert_scan_rejects_unknown_key():
    with db.connect() as conn:
        with pytest.raises(ValueError):
            db.insert_scan(conn, {"user_id": UID, "filtres": {}})   # typo'd key


def _seed_scan(conn, events):
    sid = db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
    rows = []
    for i, (stage, outcome, bucket, company) in enumerate(events, start=1):
        rows.append({"scan_id": sid, "seq": i, "company": company, "title": "T",
                     "source": "greenhouse", "board_token": "b", "location": "US",
                     "stage": stage, "outcome": outcome, "detail": "", "reason_bucket": bucket})
    db.insert_scan_events(conn, rows)
    return sid


def test_fetch_scans_derives_tallies():
    with db.connect() as conn:
        sid = _seed_scan(conn, [
            ("sieve_pass", "keyword", None, "A"),
            ("sieve_pass", "keyword", None, "B"),
            ("sieve_pass", "meaning-net", None, "C"),
            ("deep_screen", "dropped", "salary", "A"),
            ("store", "stored", None, "B"),
        ])
        db.finish_scan(conn, sid, datetime.now(timezone.utc))
    with db.connect() as conn:
        scans = db.fetch_scans(conn, UID)
        row = next(s for s in scans if s["id"] == sid)
        assert row["sieve_passes"] == 2
        assert row["meaning_net"] == 1
        assert row["stored"] == 1
        assert row["total_events"] == 5

def test_fetch_scan_job_events_filters_and_paginates():
    with db.connect() as conn:
        sid = _seed_scan(conn, [           # Acme = 2 events (seq 1,2); Beta = 1 (seq 3)
            ("sieve_pass", "keyword", None, "Acme"),
            ("deep_screen", "dropped", "salary", "Acme"),
            ("deep_screen", "dropped", "non_us", "Beta"),
        ])
    with db.connect() as conn:
        # filter by reason_bucket → the one matching job (Acme)
        rows, total = db.fetch_scan_job_events(conn, UID, scan_id=sid, reason_bucket="salary")
        assert total == 1 and rows[0]["company"] == "Acme"
        # company ILIKE, cross-scan (no scan_id)
        rows, total = db.fetch_scan_job_events(conn, UID, company="bet")
        assert total == 1 and rows[0]["reason_bucket"] == "non_us"
        # limit counts JOBS: total is the job count, and one job's page holds all its events
        rows, total = db.fetch_scan_job_events(conn, UID, scan_id=sid, limit=1)
        assert total == 2                                     # two distinct jobs
        assert {r["company"] for r in rows} == {"Acme"}       # one whole job (both events)
        assert len(rows) == 2
        # each row carries its scan's start time (for the grouped card view)
        assert "started_at" in rows[0]

def test_fetch_scan_job_events_keeps_a_job_whole_across_pages():
    # The point of job pagination: even when a job's events interleave with others' in seq,
    # a page holds whole jobs — no job is split across a page boundary.
    with db.connect() as conn:
        sid = _seed_scan(conn, [           # first_seq: Aco=1, Bco=2, Cco=3
            ("sieve_pass", "keyword", None, "Aco"),
            ("sieve_pass", "keyword", None, "Bco"),
            ("sieve_pass", "keyword", None, "Cco"),
            ("list_screen", "dropped", "onsite", "Aco"),
            ("list_screen", "dropped", "non_us", "Bco"),
            ("store", "stored", None, "Cco"),
        ])
    with db.connect() as conn:
        # page 1 = first 2 jobs (Aco, Bco), each with BOTH its interleaved events
        rows, total = db.fetch_scan_job_events(conn, UID, scan_id=sid, limit=2, offset=0)
        assert total == 3
        assert {r["company"] for r in rows} == {"Aco", "Bco"}
        assert len([r for r in rows if r["company"] == "Aco"]) == 2
        assert "Cco" not in {r["company"] for r in rows}
        # page 2 = the remaining job (Cco), whole
        rows, total = db.fetch_scan_job_events(conn, UID, scan_id=sid, limit=2, offset=2)
        assert {r["company"] for r in rows} == {"Cco"}
        assert len(rows) == 2

def test_fetch_scan_job_events_scoped_to_user():
    # A scan owned by a different user must not leak into UID's results.
    other = 999
    with db.connect() as conn:
        conn.execute("INSERT INTO users (id, email) VALUES (%s, %s) "
                     "ON CONFLICT DO NOTHING", [other, "o@x.com"])
        osid = db.insert_scan(conn, {"user_id": other, "sources": [], "filters": {}})
        db.insert_scan_events(conn, [{"scan_id": osid, "seq": 1, "stage": "store",
                                      "outcome": "stored", "company": "Secret"}])
    with db.connect() as conn:
        rows, total = db.fetch_scan_job_events(conn, UID, company="Secret")
        assert total == 0


def test_clear_rejection_forgets_a_rejected_url():
    url = "https://boards.greenhouse.io/acme/jobs/42"
    with db.connect() as conn:
        db.record_rejection(conn, UID, url, "Acme", "Eng Mgr", "on-site role")
        assert url in db.fetch_rejected_urls(conn, UID)
        db.clear_rejection(conn, UID, url)
        assert url not in db.fetch_rejected_urls(conn, UID)
    with db.connect() as conn:
        db.clear_rejection(conn, UID, "")          # blank url is a silent no-op


def test_append_scan_event_assigns_next_seq():
    with db.connect() as conn:
        sid = db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
        db.insert_scan_events(conn, [
            {"scan_id": sid, "seq": 1, "stage": "sieve_pass", "outcome": "keyword"},
            {"scan_id": sid, "seq": 2, "stage": "deep_screen", "outcome": "dropped"},
        ])
        seq = db.append_scan_event(conn, sid, {
            "company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
            "board_token": "acme", "location": "US", "external_id": "42",
            "url": "https://x/42", "stage": "manual_add", "outcome": "stored",
            "detail": "added manually", "reason_bucket": None})
    assert seq == 3
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM scan_events WHERE scan_id = %s AND seq = 3",
                           [sid]).fetchone()
    assert row["stage"] == "manual_add" and row["outcome"] == "stored"
    assert row["detail"] == "added manually" and row["external_id"] == "42"
