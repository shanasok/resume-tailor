# tests/test_tracker_db.py — the Postgres-backed tracker: upsert, dedupe, applied, delete.
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
    """A minimal discovered job, unique per call unless an id is supplied."""
    base = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
            "company": "Acme", "title": "Director of Engineering",
            "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
            "match": {"flags": [], "technical": True}}
    base.update(overrides)
    return base


def test_upsert_found_jobs_inserts_then_refreshes():
    job = _job(match={"flags": ["visa unclear", "salary below floor"], "technical": True})
    assert tracker.upsert_found_jobs(UID, [job]) == {"added": 1, "refreshed": 0}
    # Same job again → refreshed, not duplicated.
    job["title"] = "VP of Engineering"
    assert tracker.upsert_found_jobs(UID, [job]) == {"added": 0, "refreshed": 1}

    rows = tracker.read_applications(UID)
    assert len(rows) == 1
    assert rows[0]["Role"] == "VP of Engineering"
    assert rows[0]["Status"] == "Found"                  # no résumé yet
    assert rows[0]["Date Matched"] == date.today()       # a real date, not a string (per-user)
    assert rows[0]["Technical"] is True                  # a real boolean
    assert rows[0]["Flags"] == ["visa unclear", "salary below floor"]  # ordered child rows
    with db.connect() as conn:                           # identity parts are real columns
        ident = db.fetch_job_identity(conn, rows[0]["Job ID"])
    assert ident == {"source": "greenhouse", "board_token": job["board_token"],
                     "external_id": str(job["id"])}


def test_upsert_found_jobs_stores_the_description():
    job = _job(description="We need a leader for developer education.\nSalary $200k.")
    tracker.upsert_found_jobs(UID, [job])
    with db.connect() as conn:
        stored = db.fetch_job_description_text(conn, tracker.make_job_id(job))
    assert stored == job["description"]

    # A re-discovered job with fresher text refreshes the stored copy.
    job["description"] = "Updated posting text."
    tracker.upsert_found_jobs(UID, [job])
    with db.connect() as conn:
        assert db.fetch_job_description_text(conn, tracker.make_job_id(job)) == \
            "Updated posting text."
    # A job with no description stored (e.g. fetch failed) reads back as None.
    bare = _job()
    tracker.upsert_found_jobs(UID, [bare])
    with db.connect() as conn:
        assert db.fetch_job_description_text(conn, tracker.make_job_id(bare)) is None


def test_log_application_attaches_resume_to_found_job():
    job = _job()
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)

    tracker.log_application(UID, "Acme", "Director of Engineering", job["url"],
                            "resumes/generated_resumes/acme_director_resume.html",
                            "resumes/generated_resumes/acme_director_resume.pdf",
                            job_id=jid, match_score=8.5, gen_seconds=42,
                            stage_llms={"generate": "claude", "check": "local"})

    rows = tracker.read_applications(UID)
    assert len(rows) == 1                                # attached, not duplicated
    row = rows[0]
    assert row["Status"] == "Generated"
    assert row["Match Score"] == 8.5                     # a real number (NUMERIC in the DB)
    assert row["Gen Seconds"] == 42                      # a real int
    assert row["Stage LLMs"] == {"generate": "claude", "check": "local"}


def test_log_application_stores_and_replaces_critic_findings():
    job = _job()
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)
    html = "resumes/generated_resumes/acme_director_resume.html"
    pdf = "resumes/generated_resumes/acme_director_resume.pdf"

    findings = {"fidelity": ["invented team management"],
                "style": ["posting-echo phrasing", "buzzword bullet"],
                "revision": ["restore the AmEx role"],
                "strength": ["metrics preserved"]}
    tracker.log_application(UID, "Acme", "Director of Engineering", job["url"], html, pdf,
                            job_id=jid, match_score=6.8, critic_findings=findings)
    assert tracker.get_job(UID, jid)["Critic Findings"] == findings
    assert tracker.read_applications(UID)[0]["Critic Findings"] == findings

    # A re-score REPLACES the report (it describes one generation, no merging)…
    tracker.log_application(UID, "Acme", "Director of Engineering", job["url"], html, pdf,
                            job_id=jid, match_score=9.1,
                            critic_findings={"strength": ["clean now"]})
    assert tracker.get_job(UID, jid)["Critic Findings"] == {"strength": ["clean now"]}

    # …while a plain re-render (no critic_findings passed) keeps the existing report.
    tracker.log_application(UID, "Acme", "Director of Engineering", job["url"], html, pdf,
                            job_id=jid)
    assert tracker.get_job(UID, jid)["Critic Findings"] == {"strength": ["clean now"]}

    # Deleting the job takes the findings with it (FK cascade).
    tracker.delete_job(UID, jid)
    with db.connect() as conn:
        assert db.fetch_findings_map(conn, UID) == {}


def test_log_application_manual_insert_and_rerun_updates_in_place():
    html = "resumes/generated_resumes/acme_dev_resume.html"
    tracker.log_application(UID, "Acme", "Dev", "pasted job text\nline two", html,
                            "resumes/generated_resumes/acme_dev_resume.pdf")
    tracker.log_application(UID, "Acme", "Dev", "pasted job text\nline two", html,
                            "resumes/generated_resumes/acme_dev_resume.pdf", gen_seconds=7)

    rows = tracker.read_applications(UID)
    assert len(rows) == 1                                # second run updated, didn't duplicate
    assert rows[0]["Job ID"] == f"manual:{UID}:acme_dev_resume"  # per-user manual id
    assert rows[0]["Job URL"] == "(pasted text)"
    assert rows[0]["Gen Seconds"] == 7


def test_mark_applied_and_delete():
    job = _job()
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)

    assert tracker.mark_applied(UID, jid) is True
    assert tracker.mark_applied(UID, "no-such-id") is False
    row = tracker.read_applications(UID)[0]
    assert row["Status"] == "Applied"
    assert row["Applied"] == "manual"
    assert row["Applied Date"] == date.today()           # a real date

    deleted = tracker.delete_job(UID, jid)
    assert deleted["Job ID"] == jid                      # returns the row for file cleanup
    assert tracker.delete_job(UID, jid) is None          # already gone
    assert tracker.read_applications(UID) == []


def test_delete_remembers_the_job_so_scans_skip_it():
    """Deleting a job means "don't show me this again" — the URL joins the rejection
    memory, which find_matches folds into its skip-list, so no scan re-pulls it."""
    job = _job()
    tracker.upsert_found_jobs(UID, [job])
    tracker.delete_job(UID, tracker.make_job_id(job))
    with db.connect() as conn:
        assert job["url"] in db.fetch_rejected_urls(conn, UID)
        row = conn.execute("SELECT reason FROM rejected_jobs WHERE user_id = %s "
                           "AND job_url = %s", [UID, job["url"]]).fetchone()
    assert row["reason"] == "deleted manually"

    # Pasted-text manual rows have no real URL — deleting them records nothing.
    tracker.log_application(UID, "Acme", "Dev", "pasted job text",
                            "resumes/generated_resumes/acme_dev_resume.html",
                            "resumes/generated_resumes/acme_dev_resume.pdf")
    assert tracker.delete_job(UID, f"manual:{UID}:acme_dev_resume") is not None
    with db.connect() as conn:
        urls = db.fetch_rejected_urls(conn, UID)
    assert not any(u.startswith("(") for u in urls)


def test_delete_cascades_to_resume_and_application_rows():
    job = _job()
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)
    tracker.log_application(UID, "Acme", "Dev", job["url"],
                            "resumes/generated_resumes/acme_dev2_resume.html",
                            "resumes/generated_resumes/acme_dev2_resume.pdf", job_id=jid)
    tracker.mark_applied(UID, jid)

    tracker.delete_job(UID, jid)
    with db.connect() as conn:
        # The user's overlay + résumé + application rows go with the delete…
        for table in ("user_jobs", "resumes", "applications"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            assert n == 0, f"{table} row survived the cascade"
        # …but the SHARED catalog posting stays for other users.
        assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1


def test_run_log_writes_a_row():
    from src import run_log
    run_log.log(UID, "manual", "greenhouse:acme:1", "claude",
                {"ok": True, "company": "Acme", "role": "Dev", "match_score": 9.0},
                duration_sec=12.34)
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM generation_log").fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["duration_sec"] == 12.3               # DOUBLE PRECISION — a real number now
    assert float(rows[0]["match_score"]) == 9.0
    assert rows[0]["finished_at"].tzinfo is not None     # TIMESTAMPTZ — timezone-aware


def test_read_applications_orders_newest_first_with_stable_ties():
    with db.connect() as conn:
        for jid, found in [("t:a:1", date(2026, 6, 1)), ("t:a:2", date(2026, 7, 1)),
                           ("t:a:3", date(2026, 7, 1))]:  # 2 and 3 tie on the date
            source, board_token, external_id = db.split_job_id(jid)
            db.insert_job(conn, {"job_id": jid, "source": source, "board_token": board_token,
                                 "external_id": external_id})
            # date_found now lives on the per-user overlay (user_jobs.date_matched).
            db.upsert_user_job(conn, UID, jid, found)
    ids = [r["Job ID"] for r in tracker.read_applications(UID)]
    # Newest date first; within the tie, insertion order (seq) — 2 before 3.
    assert ids == ["t:a:2", "t:a:3", "t:a:1"]


def test_first_run_imports_legacy_csvs(tmp_path, monkeypatch):
    """A fresh, empty database bootstraps itself from the old CSV files — with values
    landing TYPED in the normalized tables (attributed to the default user)."""
    import csv as csv_mod
    # The OLD flat-file header (including the retired LLM / joined-string columns) —
    # frozen history, exactly what a pre-database applications.csv contains.
    legacy_fields = ["Job ID", "Source", "Date Found", "Date Posted", "Company", "Role",
                     "Location", "Job URL", "Flags", "Salary", "Technical", "Match Score",
                     "HTML", "PDF", "Applied", "Applied Date", "Gen Seconds", "LLM",
                     "Stage LLMs"]
    apps_csv = tmp_path / "applications.csv"
    with apps_csv.open("w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=legacy_fields)
        w.writeheader()
        w.writerow({"Job ID": "greenhouse:acme:9", "Source": "greenhouse",
                    "Date Found": "2026-06-01", "Company": "Acme", "Role": "Dev",
                    "Technical": "yes", "Match Score": "8.3", "Gen Seconds": "120",
                    "HTML": "resumes/generated_resumes/acme_dev_resume.html",
                    "PDF": "resumes/generated_resumes/acme_dev_resume.pdf",
                    "Applied": "manual", "Applied Date": "2026-06-02", "LLM": "claude"})
        w.writerow({"Job ID": "greenhouse:beta:1", "Source": "greenhouse",
                    "Date Found": "2026-06-03", "Company": "Beta", "Role": "PM"})
    gen_csv = tmp_path / "generation_log.csv"
    gen_csv.write_text("timestamp,source,job_id,company,role,llm,status,match_score,duration_sec,message\n"
                       "2026-06-01T10:00:00,manual,greenhouse:acme:9,Acme,Dev,claude,done,8.3,95.5,ok\n")

    monkeypatch.setattr(db, "APPLICATIONS_CSV", apps_csv)
    monkeypatch.setattr(db, "GENERATION_LOG_CSV", gen_csv)
    db._schema_ready = False                             # force the bootstrap path to re-run

    rows = tracker.read_applications(UID)                # first connect triggers the import
    assert [r["Job ID"] for r in rows] == ["greenhouse:beta:1", "greenhouse:acme:9"]
    acme = rows[1]
    assert acme["Date Matched"] == date(2026, 6, 1)      # typed on the way in
    assert acme["Match Score"] == 8.3
    assert acme["Gen Seconds"] == 120
    assert acme["Technical"] is True
    assert acme["Status"] == "Applied"
    # Pre-stage-tracking row (only LLM=claude in the CSV) → backfilled to the two
    # tailoring stages, the same attribution the UI used to compute at display time.
    assert acme["Stage LLMs"] == {"generate": "claude", "regenerate": "claude"}
    with db.connect() as conn:
        log = conn.execute("SELECT * FROM generation_log").fetchall()
    assert len(log) == 1 and log[0]["duration_sec"] == 95.5
