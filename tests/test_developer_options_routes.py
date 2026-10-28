# tests/test_developer_options_routes.py — the Tools → Developer Options button is gated by
# the DEVELOPER_OPTIONS env flag: hidden by default, shown (with a working link) when the flag
# is on. Uses the throwaway test DB (see test_resume_versions_routes.py for the pattern this
# fixture is copied from).
import uuid

import psycopg
import pytest

import app as webapp
from src import db, tracker
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


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Throwaway DB, rebuilt fresh.
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

    webapp.app.config["TESTING"] = True
    yield webapp.app.test_client()
    db._schema_ready = False


def test_tools_button_hidden_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    html = client.get("/").get_data(as_text=True)
    assert "Developer Options" not in html


def test_tools_button_shown_when_flag_on(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    html = client.get("/").get_data(as_text=True)
    assert "Developer Options" in html
    assert "/developer" in html


def test_developer_page_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    assert client.get("/developer").status_code == 404


def test_developer_page_200_and_has_reset_form_when_on(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    resp = client.get("/developer")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Reset Retrieve Jobs" in html
    assert 'action="/developer/reset-scan"' in html
    assert "RESET" in html   # the type-to-confirm keyword is shown


def test_reset_route_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    assert client.post("/developer/reset-scan", data={"confirm": "RESET"}).status_code == 404


def test_reset_route_deletes_unacted_jobs_and_flashes(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": "Acme", "title": "Director of Engineering",
           "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
           "match": {"flags": [], "technical": True}}
    tracker.upsert_found_jobs(UID, [job])

    resp = client.post("/developer/reset-scan", data={"confirm": "RESET"}, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "fresh" in html.lower()            # the summary flash rendered
    assert tracker.read_applications(UID) == []   # the unacted job is gone


def test_reset_route_wrong_confirm_deletes_nothing(client, monkeypatch):
    # The real anti-accidental-delete gate is server-side — the client-side JS check is
    # bypassable, so a stray/forged POST with a blank (or wrong) confirm value must no-op.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": "Acme", "title": "Director of Engineering",
           "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
           "match": {"flags": [], "technical": True}}
    tracker.upsert_found_jobs(UID, [job])

    resp = client.post("/developer/reset-scan", data={"confirm": ""}, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "confirm" in html.lower()          # the not-confirmed flash rendered
    kept_ids = {r["Job ID"] for r in tracker.read_applications(UID)}
    assert tracker.make_job_id(job) in kept_ids   # nothing was deleted


def test_reset_route_db_error_flashes_not_500(client, monkeypatch):
    # The spec requires a DB failure to roll back and flash, not bubble up as a 500.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(webapp, "reset_scan", _boom)

    resp = client.post("/developer/reset-scan", data={"confirm": "RESET"}, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "failed" in html.lower()
