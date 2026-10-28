# tests/test_resume_profile_db.py — the per-user base-résumé store (v9): profile + section
# child tables, round-tripping in the tailoring._SCHEMA shape.
#
# Same throwaway-database pattern as test_tracker_db.py (needs docker compose up -d db;
# skips cleanly when no server is reachable).
import psycopg
import pytest

from src import db
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
    """Fresh schema per test against the throwaway database."""
    with psycopg.connect(f"{_SERVER_URL}/postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                            [_TEST_DB]).fetchone():
            conn.execute(f'CREATE DATABASE "{_TEST_DB}"')
    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)  # v11 board import is a no-op in tests
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "APPLICATIONS_CSV", db.APPLICATIONS_CSV.with_name("nope.csv"))
    monkeypatch.setattr(db, "GENERATION_LOG_CSV", db.GENERATION_LOG_CSV.with_name("nope.csv"))
    with db._connect_raw() as conn:
        conn.execute("DROP VIEW IF EXISTS job_overview")
        conn.execute("DROP TABLE IF EXISTS jobs, job_flags, resumes, resume_findings, resume_versions, "
                     "applications, generation_log, job_embeddings, profile_embeddings, "
                     "job_descriptions, rejected_jobs, users, user_jobs, user_settings, "
                     "user_filter_terms, resume_profiles, resume_experience, "
                     "resume_education, resume_skills, resume_competencies, "
                     "resume_certifications, board_directory, scans, scan_events, "
                     "schema_migrations CASCADE")
    with db.connect() as conn:                       # applies migrations 1-9 fresh
        pass
    yield
    db._schema_ready = False


def _full_profile():
    """A complete résumé in the tailoring._SCHEMA shape."""
    return {
        "name": "Jane Doe",
        "contact": {"location": "Denver, CO", "phone": "555-0100",
                    "email": "jane@example.com", "website": "janedoe.dev"},
        "summary": "Learning & enablement leader with 15 years in developer education.",
        "competencies": ["L&D strategy", "Curriculum design", "Developer enablement"],
        "experience": [
            {"company": "Acme", "title": "Director of Enablement", "dates": "2019 – 2024",
             "location": "Remote", "bullets": ["Built the L&D org from 0 to 20.",
                                               "Cut onboarding time 40%."]},
            {"company": "Beta", "title": "Training Manager", "dates": "2015 – 2019",
             "location": "Denver, CO", "bullets": ["Ran the technical curriculum."]},
        ],
        "education": [
            {"institution": "State U", "degree": "B.A. English", "date": "2011",
             "detail": "cum laude"},
        ],
        "certifications": ["Certified Scrum Master", "AWS Cloud Practitioner"],
        "skills": [
            {"category": "Tools", "items": "Jira, Figma, SQL"},
            {"category": "Languages", "items": "Python, JavaScript"},
        ],
    }


def test_no_profile_returns_none():
    with db.connect() as conn:
        assert db.fetch_resume_profile(conn, UID) is None
        assert db.resume_profile_exists(conn, UID) is False


def test_roundtrip_full_profile():
    profile = _full_profile()
    with db.connect() as conn:
        db.upsert_resume_profile(conn, UID, profile)
    with db.connect() as conn:
        assert db.resume_profile_exists(conn, UID) is True
        got = db.fetch_resume_profile(conn, UID)
    # Round-trips exactly, including nested bullets and section order.
    assert got == profile


def test_upsert_replaces_all_sections():
    with db.connect() as conn:
        db.upsert_resume_profile(conn, UID, _full_profile())     # 2 experience, 2 skills, …
    # Re-save a slimmer résumé — old child rows must not linger.
    slim = {"name": "Jane Doe", "contact": {}, "summary": "Shorter.",
            "competencies": ["One thing"], "experience": [
                {"company": "Gamma", "title": "Consultant", "dates": "2024 –",
                 "location": "", "bullets": []}],
            "education": [], "certifications": [], "skills": []}
    with db.connect() as conn:
        db.upsert_resume_profile(conn, UID, slim)
    with db.connect() as conn:
        got = db.fetch_resume_profile(conn, UID)
        counts = {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t} WHERE user_id = %s",
                                  [UID]).fetchone()["n"]
                  for t in ("resume_experience", "resume_education", "resume_skills",
                            "resume_competencies", "resume_certifications")}
    assert got["summary"] == "Shorter."
    assert [e["company"] for e in got["experience"]] == ["Gamma"]
    assert counts == {"resume_experience": 1, "resume_education": 0, "resume_skills": 0,
                      "resume_competencies": 1, "resume_certifications": 0}


def test_empty_sections_come_back_as_lists():
    bare = {"name": "No Sections", "contact": {"email": "x@y.z"}, "summary": "",
            "competencies": [], "experience": [], "education": [],
            "certifications": [], "skills": []}
    with db.connect() as conn:
        db.upsert_resume_profile(conn, UID, bare)
        got = db.fetch_resume_profile(conn, UID)
    assert got["name"] == "No Sections"
    assert got["contact"]["email"] == "x@y.z"
    assert got["contact"]["phone"] == ""          # missing contact field → empty string
    for section in ("competencies", "experience", "education", "certifications", "skills"):
        assert got[section] == []


def test_two_users_have_separate_resumes():
    with db.connect() as conn:
        conn.execute("INSERT INTO users (id, display_name) VALUES (2, 'Second user')")
        db.upsert_resume_profile(conn, UID, _full_profile())
        db.upsert_resume_profile(conn, 2, {"name": "Bob Roe", "contact": {},
                                           "summary": "Different person.",
                                           "competencies": [], "experience": [],
                                           "education": [], "certifications": [], "skills": []})
    with db.connect() as conn:
        assert db.fetch_resume_profile(conn, UID)["name"] == "Jane Doe"
        assert db.fetch_resume_profile(conn, 2)["name"] == "Bob Roe"


def test_clearing_the_profile_row_cascades_to_all_sections():
    """resume_profiles is the parent: deleting just the profile row takes every section
    with it (no orphan experience/skills/etc. left behind)."""
    with db.connect() as conn:
        db.upsert_resume_profile(conn, UID, _full_profile())
        conn.execute("DELETE FROM resume_profiles WHERE user_id = %s", [UID])
        for t in ("resume_experience", "resume_education", "resume_skills",
                  "resume_competencies", "resume_certifications"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {t} WHERE user_id = %s",
                            [UID]).fetchone()["n"]
            assert n == 0, f"{t} row survived clearing the profile"
        assert db.fetch_resume_profile(conn, UID) is None


def test_profile_cascades_on_user_delete():
    with db.connect() as conn:
        conn.execute("INSERT INTO users (id, display_name) VALUES (2, 'Doomed user')")
        db.upsert_resume_profile(conn, 2, _full_profile())
        conn.execute("DELETE FROM users WHERE id = 2")
        for t in ("resume_profiles", "resume_experience", "resume_education",
                  "resume_skills", "resume_competencies", "resume_certifications"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {t} WHERE user_id = 2").fetchone()["n"]
            assert n == 0, f"{t} row survived the user delete"
