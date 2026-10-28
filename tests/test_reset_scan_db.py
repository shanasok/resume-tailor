# tests/test_reset_scan_db.py — db.reset_scan_state: wipes a user's unacted scan jobs +
# rejection memory, leaving the shared catalog and any acted-on rows (résumé/application)
# and manual jobs intact.
#
# These tests need a running Postgres (docker compose up -d db). They use their own
# throwaway database (resume_handler_test) so they NEVER touch your real tracked jobs,
# and they skip cleanly when no server is reachable (e.g. CI without docker).
import uuid
from datetime import date

import psycopg
import pytest

from src import db, tracker
from src.current_user import DEFAULT_USER_ID as UID   # the seeded default user (id 1)


# Derive the test-database URL from the configured one by swapping the database name.
_TEST_DB = "resume_handler_test"
_SERVER_URL, _, _ = db.database_url().rpartition("/")
TEST_DATABASE_URL = f"{_SERVER_URL}/{_TEST_DB}"


def _server_available() -> bool:
    try:
        # Connect to the default maintenance DB just to prove the server is up.
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
    """Point src.db at a dedicated throwaway database for the duration of each test.
    Created once if missing; the schema is dropped and rebuilt before every test for
    full isolation (also wipes any stale schema left by older versions of the code)."""
    with psycopg.connect(f"{_SERVER_URL}/postgres", autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", [_TEST_DB]
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{_TEST_DB}"')

    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)  # v11 board import is a no-op in tests
    monkeypatch.setattr(db, "_schema_ready", False)      # force re-init against the test DB
    # Don't import the real legacy CSVs into the test DB.
    monkeypatch.setattr(db, "APPLICATIONS_CSV", db.APPLICATIONS_CSV.with_name("nope.csv"))
    monkeypatch.setattr(db, "GENERATION_LOG_CSV", db.GENERATION_LOG_CSV.with_name("nope.csv"))

    with db._connect_raw() as conn:
        conn.execute("DROP VIEW IF EXISTS job_overview")
        # Includes the v8 multi-tenancy tables (users, user_jobs, user_settings,
        # user_filter_terms) — CASCADE clears their dependents too.
        conn.execute("DROP TABLE IF EXISTS jobs, job_flags, resumes, resume_findings, resume_versions, "
                     "applications, generation_log, job_embeddings, profile_embeddings, "
                     "job_descriptions, rejected_jobs, users, user_jobs, user_settings, "
                     "user_filter_terms, resume_profiles, resume_experience, "
                     "resume_education, resume_skills, resume_competencies, "
                     "resume_certifications, board_directory, scans, scan_events, "
                     "schema_migrations CASCADE")
    with db.connect() as conn:                           # recreates the schema fresh
        pass
    yield
    # monkeypatch restores DATABASE_URL; reset the flag so later non-test use re-inits.
    db._schema_ready = False


def _job(**overrides):
    base = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
            "company": "Acme", "title": "Director of Engineering",
            "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
            "match": {"flags": [], "technical": True}}
    base.update(overrides)
    return base


def _add_bare_manual_job(conn, user_id):
    """A manual job with NO résumé/application, inserted directly (the UI never makes this
    state, but it proves the source<>'manual' clause independently of the résumé clause).
    Manual job ids pack the user_id into board_token (see tracker.manual_job_id), so
    board_token must be str(user_id) here too — NULL would violate jobs_job_id_composition."""
    jid = f"manual:{user_id}:{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO jobs (job_id, source, board_token, external_id, company, role) "
        "VALUES (%s, 'manual', %s, %s, 'ManualCo', 'Hand-Entered Role')",
        [jid, str(user_id), jid.split(':')[-1]])
    conn.execute("INSERT INTO user_jobs (user_id, job_id) VALUES (%s, %s)", [user_id, jid])
    return jid


def test_reset_deletes_unacted_scan_jobs_and_clears_rejections():
    # A plain scan job with no résumé and no application → should be deleted.
    bare = _job()
    tracker.upsert_found_jobs(UID, [bare])
    # A rejection memory entry → should be cleared.
    with db.connect() as conn:
        db.record_rejection(conn, UID, "https://boards.acme.com/rejected-1",
                            "RejCo", "Rejected Role", "salary below floor")

    with db.connect() as conn:
        counts = db.reset_scan_state(conn, UID)

    assert counts == {"jobs_deleted": 1, "rejections_cleared": 1}
    assert tracker.read_applications(UID) == []
    with db.connect() as conn:
        assert db.fetch_rejected_urls(conn, UID) == set()


def test_reset_keeps_acted_on_and_manual():
    # (a) scan job WITH a résumé → kept
    resumed = _job()
    tracker.upsert_found_jobs(UID, [resumed])
    tracker.log_application(UID, "Acme", "Director of Engineering", resumed["url"],
                            "resumes/generated_resumes/acme_resume.html",
                            "resumes/generated_resumes/acme_resume.pdf",
                            job_id=tracker.make_job_id(resumed), match_score=8.0)
    # (b) scan job that was APPLIED to → kept
    applied = _job()
    tracker.upsert_found_jobs(UID, [applied])
    tracker.mark_applied(UID, tracker.make_job_id(applied))
    # (c) bare manual job → kept (source clause)
    with db.connect() as conn:
        manual_jid = _add_bare_manual_job(conn, UID)
    # (d) a plain scan job → the only thing that should go
    bare = _job()
    tracker.upsert_found_jobs(UID, [bare])

    with db.connect() as conn:
        counts = db.reset_scan_state(conn, UID)

    assert counts["jobs_deleted"] == 1
    kept_ids = {r["Job ID"] for r in tracker.read_applications(UID)}
    assert tracker.make_job_id(resumed) in kept_ids
    assert tracker.make_job_id(applied) in kept_ids
    assert manual_jid in kept_ids
    assert tracker.make_job_id(bare) not in kept_ids


def test_reset_leaves_other_users_untouched():
    # Second user, seeded independently of UID (id 1).
    with db.connect() as conn:
        conn.execute("INSERT INTO users (id, display_name) VALUES (2, 'Second user')")

    u2_job = _job()
    tracker.upsert_found_jobs(2, [u2_job])
    with db.connect() as conn:
        db.record_rejection(conn, 2, "https://boards.acme.com/u2-reject",
                            "U2Co", "U2 Role", "salary below floor")

    u1_job = _job()
    tracker.upsert_found_jobs(UID, [u1_job])

    with db.connect() as conn:
        db.reset_scan_state(conn, UID)

    assert tracker.read_applications(UID) == []
    u2_ids = {r["Job ID"] for r in tracker.read_applications(2)}
    assert tracker.make_job_id(u2_job) in u2_ids
    with db.connect() as conn:
        assert "https://boards.acme.com/u2-reject" in db.fetch_rejected_urls(conn, 2)


def test_reset_leaves_shared_catalog_and_descriptions_intact():
    job = _job(description="Full posting text kept for reuse.")
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)

    with db.connect() as conn:
        db.reset_scan_state(conn, UID)

    with db.connect() as conn:  # overlay gone, but the shared catalog row + text remain
        assert conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", [jid]).fetchone() is not None
        assert db.fetch_job_description_text(conn, jid) == "Full posting text kept for reuse."


def test_tracker_reset_scan_wraps_the_db_call():
    bare = _job()
    tracker.upsert_found_jobs(UID, [bare])
    counts = tracker.reset_scan(UID)
    assert counts == {"jobs_deleted": 1, "rejections_cleared": 0}
    assert tracker.read_applications(UID) == []
