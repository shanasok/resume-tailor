# tests/test_resume_versions_db.py — the résumé version-history table (v12) and its db.py
# helpers: next/insert/list/get, JSON round-trips for findings/stage_llms, and the FK cascade
# from resumes. Same throwaway-database pattern as test_tracker_db.py: needs a running
# Postgres (docker compose up -d db) and skips cleanly when none is reachable.
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


def _seed_resume():
    """Create a job + its résumé row (the FK parent of resume_versions) and return its job id."""
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": "Acme", "title": "Director of Engineering",
           "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
           "match": {"flags": [], "technical": True}}
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)
    tracker.log_application(UID, "Acme", "Director of Engineering", job["url"],
                            "resumes/generated_resumes/acme_director_resume.html",
                            "resumes/generated_resumes/acme_director_resume.pdf",
                            job_id=jid, match_score=8.5)
    return jid


def test_next_version_increments_per_resume():
    jid = _seed_resume()
    with db.connect() as conn:
        assert db.next_resume_version(conn, UID, jid) == 1
        db.insert_resume_version(conn, UID, jid, 1, {"note": "first"})
        assert db.next_resume_version(conn, UID, jid) == 2
        db.insert_resume_version(conn, UID, jid, 2, {"note": "second"})
        assert db.next_resume_version(conn, UID, jid) == 3


def test_insert_list_get_round_trip_with_json_fields():
    jid = _seed_resume()
    findings = {"fidelity": ["invented team size"], "style": ["posting echo"]}
    stage_llms = {"generate": "claude", "check": "local_qwen"}
    with db.connect() as conn:
        db.insert_resume_version(conn, UID, jid, 1, {
            "html_path": "versions/acme_director_resume_v1.html",
            "pdf_path": "versions/acme_director_resume_v1.pdf",
            "json_path": "versions/acme_director_resume_v1.json",
            "match_score": 6.8, "findings": findings, "stage_llms": stage_llms,
            "note": "before AI rewrite"})

    with db.connect() as conn:
        versions = db.list_resume_versions(conn, UID, jid)
    assert len(versions) == 1
    v = versions[0]
    assert v["version"] == 1
    assert v["match_score"] == 6.8                        # NUMERIC → float
    assert v["findings"] == findings                     # JSON text → dict
    assert v["stage_llms"] == stage_llms
    assert v["note"] == "before AI rewrite"
    assert v["html_path"] == "versions/acme_director_resume_v1.html"
    assert v["created_at"] is not None

    with db.connect() as conn:
        assert db.get_resume_version(conn, UID, jid, 1)["match_score"] == 6.8
        assert db.get_resume_version(conn, UID, jid, 99) is None


def test_list_is_newest_first():
    jid = _seed_resume()
    with db.connect() as conn:
        for n in (1, 2, 3):
            db.insert_resume_version(conn, UID, jid, n, {"note": f"v{n}"})
    with db.connect() as conn:
        versions = db.list_resume_versions(conn, UID, jid)
    assert [v["version"] for v in versions] == [3, 2, 1]


def test_empty_json_fields_default_to_empty_dicts():
    jid = _seed_resume()
    with db.connect() as conn:
        db.insert_resume_version(conn, UID, jid, 1, {"note": "bare"})   # no findings/stage_llms
    with db.connect() as conn:
        v = db.list_resume_versions(conn, UID, jid)[0]
    assert v["findings"] == {}
    assert v["stage_llms"] == {}
    assert v["match_score"] is None


def test_versions_cascade_when_resume_is_deleted():
    jid = _seed_resume()
    with db.connect() as conn:
        db.insert_resume_version(conn, UID, jid, 1, {"note": "v1"})
        db.insert_resume_version(conn, UID, jid, 2, {"note": "v2"})

    tracker.delete_job(UID, jid)                          # drops the user's résumé row

    with db.connect() as conn:
        assert db.list_resume_versions(conn, UID, jid) == []
