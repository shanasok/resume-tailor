# tests/test_board_directory_db.py — the board_directory table (v11): the db.py read/write
# pair, and board_discovery's DB-backed load_directory / build_directory (harvest + validate
# FAKED, only the DB write is real). Same throwaway-database pattern as test_tracker_db.py:
# needs a running Postgres, skips cleanly when none is reachable.
import json
from datetime import date

import psycopg
import pytest

from src import db
from src.ats import board_discovery


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
    """Point src.db at a throwaway database, schema dropped + rebuilt fresh per test.
    PROJECT_ROOT is redirected to an empty temp dir so the v11 migration's one-time import
    of data/*_boards.json is a no-op — tests own their board_directory rows (and don't pay to
    import the real 6k-row directories on every rebuild)."""
    with psycopg.connect(f"{_SERVER_URL}/postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                            [_TEST_DB]).fetchone():
            conn.execute(f'CREATE DATABASE "{_TEST_DB}"')

    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)
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
    with db.connect() as conn:                           # recreates the schema fresh (runs v11)
        pass
    yield
    db._schema_ready = False


# ── db.py read/write pair ─────────────────────────────────────────────────────

def test_replace_and_fetch_orders_busiest_first():
    with db.connect() as conn:
        db.replace_board_directory(conn, "lever", [
            {"token": "small", "job_count": 3},
            {"token": "big", "job_count": 200},
            {"token": "mid", "job_count": 50},
        ])
    with db.connect() as conn:
        rows = db.fetch_board_directory(conn, "lever")
    assert [r["token"] for r in rows] == ["big", "mid", "small"]   # job_count DESC
    assert rows[0]["job_count"] == 200


def test_fetch_is_scoped_to_source():
    with db.connect() as conn:
        db.replace_board_directory(conn, "lever", [{"token": "a", "job_count": 1}])
        db.replace_board_directory(conn, "ashby", [{"token": "b", "job_count": 1}])
    with db.connect() as conn:
        assert [r["token"] for r in db.fetch_board_directory(conn, "lever")] == ["a"]
        assert [r["token"] for r in db.fetch_board_directory(conn, "ashby")] == ["b"]


def test_replace_prunes_dead_boards_and_updates_counts():
    with db.connect() as conn:
        db.replace_board_directory(conn, "lever", [
            {"token": "keep", "job_count": 10},
            {"token": "dies", "job_count": 5},
        ])
    # A rebuild that no longer finds "dies" and re-counts "keep".
    with db.connect() as conn:
        n = db.replace_board_directory(conn, "lever", [{"token": "keep", "job_count": 42}])
    assert n == 1
    with db.connect() as conn:
        rows = db.fetch_board_directory(conn, "lever")
    assert [(r["token"], r["job_count"]) for r in rows] == [("keep", 42)]


def test_replace_with_empty_list_clears_the_source():
    with db.connect() as conn:
        db.replace_board_directory(conn, "lever", [{"token": "a", "job_count": 1}])
        assert db.replace_board_directory(conn, "lever", []) == 0
    with db.connect() as conn:
        assert db.fetch_board_directory(conn, "lever") == []


def test_replace_preserves_first_seen():
    with db.connect() as conn:
        db.replace_board_directory(conn, "lever", [{"token": "a", "job_count": 1}])
        conn.execute("UPDATE board_directory SET first_seen = %s WHERE token = 'a'",
                     [date(2020, 1, 1)])
    with db.connect() as conn:                           # a later rebuild re-sees "a"
        db.replace_board_directory(conn, "lever", [{"token": "a", "job_count": 9}])
        row = conn.execute("SELECT first_seen, last_seen FROM board_directory "
                           "WHERE token = 'a'").fetchone()
        # last_seen is written with SQL CURRENT_DATE, so compare against the DB's clock (UTC in
        # the docker Postgres), not the app's local date.today() — on US evenings they differ.
        db_today = conn.execute("SELECT CURRENT_DATE AS d").fetchone()["d"]
    assert row["first_seen"] == date(2020, 1, 1)         # original first_seen kept
    assert row["last_seen"] == db_today                  # last_seen refreshed (DB clock)


# ── board_discovery, DB-backed ────────────────────────────────────────────────

def test_load_directory_raises_when_source_empty():
    with pytest.raises(FileNotFoundError, match="No boards stored for 'lever'"):
        board_discovery.load_directory("lever")


def test_load_directory_returns_stored_boards():
    with db.connect() as conn:
        db.replace_board_directory(conn, "ashby", [{"token": "ramp", "job_count": 129}])
    directory = board_discovery.load_directory("ashby")
    assert directory["board_count"] == 1
    assert directory["boards"][0]["token"] == "ramp"


def test_build_directory_writes_to_db(monkeypatch):
    # Fake the two network stages: harvest returns tokens, validate returns a job count.
    monkeypatch.setattr(board_discovery, "harvest_tokens",
                        lambda *a, **k: {"live-a", "live-b", "dead"})
    counts = {"live-a": 12, "live-b": 40, "dead": None}
    directory = board_discovery.build_directory(
        "greenhouse", ["boards.greenhouse.io/*"], lambda t: counts[t],
        merge_existing=False)
    assert directory["board_count"] == 2                 # "dead" (None) dropped
    with db.connect() as conn:
        rows = db.fetch_board_directory(conn, "greenhouse")
    assert [r["token"] for r in rows] == ["live-b", "live-a"]   # busiest first, persisted


def test_build_directory_merges_existing_from_db(monkeypatch):
    # Pre-seed a board the new harvest won't find; merge_existing should re-validate it.
    with db.connect() as conn:
        db.replace_board_directory(conn, "greenhouse", [{"token": "old", "job_count": 1}])
    monkeypatch.setattr(board_discovery, "harvest_tokens", lambda *a, **k: {"new"})
    board_discovery.build_directory(
        "greenhouse", ["boards.greenhouse.io/*"], lambda t: {"old": 7, "new": 3}[t],
        merge_existing=True)
    with db.connect() as conn:
        tokens = {r["token"] for r in db.fetch_board_directory(conn, "greenhouse")}
    assert tokens == {"old", "new"}                      # existing "old" merged in, not lost


# ── The v11 one-time carry-over of the on-disk directories ────────────────────

def test_migration_imports_existing_json_files(tmp_path):
    """v11 seeds board_directory from any data/<source>_boards.json on disk, so the boards
    discovered before the DB migration survive without re-harvesting."""
    data_dir = tmp_path / "data"                         # tmp_path is db.PROJECT_ROOT (per the fixture)
    data_dir.mkdir()
    (data_dir / "greenhouse_boards.json").write_text(json.dumps({
        "generated_at": "2025-01-15", "board_count": 2,
        "boards": [{"token": "stripe", "job_count": 90}, {"token": "acme", "job_count": 5}]}))

    # Re-run just the v11 step: drop the table + its migration bookkeeping, re-init the schema.
    with db.connect() as conn:
        conn.execute("DROP TABLE board_directory")
        conn.execute("DELETE FROM schema_migrations WHERE version = 11")
    db._schema_ready = False                             # force _ensure_schema to re-apply v11
    with db.connect() as conn:
        rows = db.fetch_board_directory(conn, "greenhouse")
        first_seen = conn.execute("SELECT first_seen FROM board_directory "
                                  "WHERE token = 'stripe'").fetchone()["first_seen"]
    assert {(r["token"], r["job_count"]) for r in rows} == {("stripe", 90), ("acme", 5)}
    assert first_seen == date(2025, 1, 15)               # generated_at becomes first_seen
