# tests/test_manual_add_job.py — job_search.add_job_manually: re-fetch a scan-dropped
# posting via its board connector, store it, clear its rejection, and append a manual_add
# audit event. Throwaway-DB pattern (needs docker Postgres; skips cleanly without it).
import types

import psycopg
import pytest

from src import db, job_search, tracker
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
    # Keep tests offline: never make a real embedding call.
    monkeypatch.setattr(job_search.embeddings, "available", lambda: False)
    yield
    db._schema_ready = False


def _fake_connector(**overrides):
    job = {"source": "greenhouse", "board_token": "acme", "id": "42",
           "title": "Engineering Manager", "company": "Acme", "location": "Remote - US",
           "url": "https://boards.greenhouse.io/acme/jobs/42",
           "description": "Lead the platform team.", "salary_text": "$220k",
           "date_posted": "2026-07-01"}
    job.update(overrides)
    return types.SimpleNamespace(get_job=lambda token, ext: dict(job))


def _new_scan():
    with db.connect() as conn:
        return db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})


def test_add_job_manually_stores_and_marks(monkeypatch):
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse", _fake_connector())
    sid = _new_scan()
    # Pre-existing rejection for this posting — the rescue must clear it.
    with db.connect() as conn:
        db.record_rejection(conn, UID, "https://boards.greenhouse.io/acme/jobs/42",
                            "Acme", "Engineering Manager", "salary tops out at $150,000")

    result = job_search.add_job_manually(UID, sid, "greenhouse", "acme", "42",
                                         fit_llm="chatgpt")

    assert result["company"] == "Acme" and result["role"] == "Engineering Manager"
    assert result["stored"] in ("added", "refreshed")
    # Job row + description stored.
    row = tracker.get_job(UID, "greenhouse:acme:42")
    assert row is not None
    # Rejection cleared.
    with db.connect() as conn:
        assert "https://boards.greenhouse.io/acme/jobs/42" not in db.fetch_rejected_urls(conn, UID)
        ev = conn.execute("SELECT * FROM scan_events WHERE scan_id = %s AND stage = 'manual_add'",
                          [sid]).fetchone()
    assert ev is not None and ev["outcome"] == "stored" and ev["detail"] == "added manually"
    assert ev["company"] == "Acme" and ev["external_id"] == "42"


def test_manual_add_clears_rejection_by_audited_url(monkeypatch):
    # The scan records a deep/pre-fetch rejection under the url it had at screen time (stored on
    # the scan_event = the card's audited url). The re-fetched DETAIL url can differ, so clearing
    # only the re-fetched url would miss. Passing the audited url must clear the original rejection.
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse",
                        _fake_connector(url="https://boards.greenhouse.io/acme/jobs/42?gh_src=x"))
    sid = _new_scan()
    list_url = "https://boards.greenhouse.io/acme/jobs/42"      # what the scan rejected under
    with db.connect() as conn:
        db.record_rejection(conn, UID, list_url, "Acme", "Engineering Manager", "on-site role")

    job_search.add_job_manually(UID, sid, "greenhouse", "acme", "42", audit_url=list_url)

    with db.connect() as conn:
        assert list_url not in db.fetch_rejected_urls(conn, UID)   # audited url cleared


def test_add_job_manually_no_connector_raises(monkeypatch):
    sid = _new_scan()
    with pytest.raises(ValueError):
        job_search.add_job_manually(UID, sid, "nosuchsource", "acme", "42")
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM scan_events WHERE scan_id = %s",
                            [sid]).fetchone()["n"] == 0


def test_add_job_manually_fetch_error_propagates(monkeypatch):
    def boom(token, ext):
        raise RuntimeError("posting no longer live")
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse",
                        types.SimpleNamespace(get_job=boom))
    sid = _new_scan()
    with pytest.raises(RuntimeError):
        job_search.add_job_manually(UID, sid, "greenhouse", "acme", "42")
    assert tracker.get_job(UID, "greenhouse:acme:42") is None


def test_manual_add_event_uses_audited_company_title_for_grouping(monkeypatch):
    # A board's DETAIL fetch can return a different title than the LIST item the funnel
    # recorded. The manual_add event must carry the AUDITED (list) company/title so it groups
    # with the job's existing dropped events — not the re-fetched values.
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse",
                        _fake_connector(title="Engineering Manager (Platform)", company="Acme Inc"))
    sid = _new_scan()
    job_search.add_job_manually(UID, sid, "greenhouse", "acme", "42",
                                audit_company="Acme", audit_title="Engineering Manager")
    with db.connect() as conn:
        ev = conn.execute("SELECT * FROM scan_events WHERE scan_id = %s AND stage='manual_add'",
                          [sid]).fetchone()
    assert ev["company"] == "Acme" and ev["title"] == "Engineering Manager"
    # The STORED job still uses the connector's real (detail) values.
    assert tracker.get_job(UID, "greenhouse:acme:42") is not None
