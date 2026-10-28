# tests/test_job_list_routes.py — the home page's paged jobs table, driven through Flask.
#
# The page used to render every tracked job and let the browser hide most of them. These
# tests pin the server-side behaviour that replaced that: the posting-age window, the tab
# split, the company search, sorting, and paging.
#
# Needs a running Postgres (docker compose up -d db); uses the throwaway test database.
import uuid
from datetime import date, timedelta

import psycopg
import pytest

import app as webapp
from src import db, filter_config, tracker
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
    with webapp.app.test_client() as c:
        yield c
    db._schema_ready = False


def _add(days_old=None, company="Acme", **overrides):
    posted = None if days_old is None else (date.today() - timedelta(days=days_old)).isoformat()
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": company, "title": "Director of Engineering", "date_posted": posted,
           "location": "Remote — USA", "url": f"https://b.co/{uuid.uuid4().hex[:8]}",
           "match": {"flags": [], "technical": True}}
    job.update(overrides)
    tracker.upsert_found_jobs(UID, [job])
    return tracker.make_job_id(job)


def _set_age_window(days):
    cfg = filter_config.load(UID)
    cfg["max_posting_age_days"] = days
    filter_config.save(UID, cfg)


def test_home_page_hides_postings_older_than_the_window(client):
    _set_age_window(64)
    _add(days_old=5, company="FreshCo")
    _add(days_old=300, company="StaleCo")
    body = client.get("/").get_data(as_text=True)
    assert "FreshCo" in body
    assert "StaleCo" not in body


def test_home_page_keeps_an_old_job_you_applied_to(client):
    _set_age_window(64)
    old = _add(days_old=300, company="StaleCo")
    tracker.mark_applied(UID, old)
    body = client.get("/?tab=applied").get_data(as_text=True)
    assert "StaleCo" in body


def test_home_page_renders_only_the_requested_tab(client):
    _set_age_window(64)
    pending = _add(days_old=5, company="PendingCo")
    applied = _add(days_old=5, company="AppliedCo")
    tracker.mark_applied(UID, applied)

    to_apply = client.get("/?tab=pending").get_data(as_text=True)
    assert "PendingCo" in to_apply
    assert "AppliedCo" not in to_apply, "the To Apply tab must not ship applied rows"

    applied_tab = client.get("/?tab=applied").get_data(as_text=True)
    assert "AppliedCo" in applied_tab
    assert "PendingCo" not in applied_tab


def test_home_page_pages_the_jobs_table(client):
    _set_age_window(64)
    monkey = [_add(days_old=5, company=f"Co{n:02d}") for n in range(6)]
    webapp.app.config["JOBS_PER_PAGE"] = 2
    first = client.get("/?tab=pending&sort=company&page=1").get_data(as_text=True)
    second = client.get("/?tab=pending&sort=company&page=2").get_data(as_text=True)
    assert "Co00" in first and "Co01" in first and "Co02" not in first
    assert "Co02" in second and "Co00" not in second


def test_company_search_narrows_the_page(client):
    _set_age_window(64)
    _add(days_old=5, company="Acme Corp")
    _add(days_old=5, company="Beta Inc")
    body = client.get("/?tab=pending&q=acme").get_data(as_text=True)
    assert "Acme Corp" in body
    assert "Beta Inc" not in body


def test_unknown_tab_falls_back_to_to_apply(client):
    _set_age_window(64)
    _add(days_old=5, company="PendingCo")
    body = client.get("/?tab=nonsense").get_data(as_text=True)
    assert body.count("PendingCo") >= 1, "a bad tab must not 500 — fall back to To Apply"


def test_saving_the_under_qualified_threshold_moves_a_job_between_tabs(client):
    _set_age_window(64)
    job = _add(days_old=5, company="WeakCo")
    with db.connect() as conn:
        db.set_fit_verdict(conn, UID, job, 4, "thin match")

    assert "WeakCo" in client.get("/?tab=underqualified").get_data(as_text=True)

    resp = client.post("/underqualified-threshold", data={"threshold": "3"})
    assert resp.status_code in (200, 302)
    assert filter_config.load(UID)["underqualified_at_or_below"] == 3
    assert "WeakCo" in client.get("/?tab=pending").get_data(as_text=True), \
        "a 4 is no longer under-qualified at a threshold of 3"


def test_threshold_route_rejects_a_value_outside_one_to_ten(client):
    before = filter_config.load(UID)["underqualified_at_or_below"]
    client.post("/underqualified-threshold", data={"threshold": "99"})
    assert filter_config.load(UID)["underqualified_at_or_below"] == before
