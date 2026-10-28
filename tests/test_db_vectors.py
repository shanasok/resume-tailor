# tests/test_db_vectors.py — the pgvector layer: migration v4, embedding storage,
# SQL similarity ranking, the profile-embedding cache, and AI fit verdicts.
#
# Needs a running Postgres WITH the vector extension (the pgvector/pgvector:pg16
# image — docker compose up -d db). Uses the same throwaway resume_handler_test
# database as test_tracker_db.py and skips cleanly when the server (or the
# extension) isn't available.

import psycopg
import pytest

from src import db, embeddings
from src.current_user import DEFAULT_USER_ID as UID   # the seeded default user (id 1)

_TEST_DB = "resume_handler_test"
_SERVER_URL, _, _ = db.database_url().rpartition("/")
TEST_DATABASE_URL = f"{_SERVER_URL}/{_TEST_DB}"


def _server_has_vector() -> bool:
    try:
        with psycopg.connect(f"{_SERVER_URL}/postgres", connect_timeout=2) as conn:
            return conn.execute(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
            ).fetchone() is not None
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_has_vector(),
    reason="Postgres with pgvector not reachable — start it with: docker compose up -d db "
           "(db image must be pgvector/pgvector:pg16)",
)


@pytest.fixture(autouse=True)
def test_database(monkeypatch, tmp_path):
    """Same throwaway-database pattern as test_tracker_db.py: fresh schema per test."""
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
        # Includes the v8 multi-tenancy tables — CASCADE clears their dependents too.
        conn.execute("DROP TABLE IF EXISTS jobs, job_flags, resumes, resume_findings, resume_versions, "
                     "applications, generation_log, job_embeddings, profile_embeddings, "
                     "job_descriptions, rejected_jobs, users, user_jobs, user_settings, "
                     "user_filter_terms, resume_profiles, resume_experience, "
                     "resume_education, resume_skills, resume_competencies, "
                     "resume_certifications, board_directory, scans, scan_events, "
                     "schema_migrations CASCADE")
    with db.connect() as conn:                       # applies migrations 1-8 fresh
        pass
    yield
    db._schema_ready = False


def _vec(direction: int) -> list[float]:
    """A unit vector along one axis of the 1536-dim space — cosine math is exact."""
    v = [0.0] * db_dim()
    v[direction] = 1.0
    return v


def db_dim() -> int:
    return embeddings.EMBEDDING_DIM


def _insert_job(conn, jid: str) -> None:
    """Insert a SHARED catalog row AND the default user's overlay, so per-user queries
    (verdicts, similarity, the overview view) have a row to find."""
    source, board_token, external_id = db.split_job_id(jid)
    db.insert_job(conn, {"job_id": jid, "source": source, "board_token": board_token,
                         "external_id": external_id, "company": "Acme", "role": "Dev"})
    db.upsert_user_job(conn, UID, jid)


def test_migration_v8_moved_verdicts_to_the_user_overlay():
    with db.connect() as conn:
        assert db._table_exists(conn, "job_embeddings")
        assert db._table_exists(conn, "profile_embeddings")
        # The AI-fit verdict is per-user now — it lives on user_jobs, not the shared jobs.
        uj_cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='user_jobs'")}
        assert {"ai_fit_score", "ai_fit_reason", "ai_fit_at"} <= uj_cols
        jobs_cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='jobs'")}
        assert not ({"ai_fit_score", "ai_fit_reason", "ai_fit_at"} & jobs_cols)
        # …and the recreated view still exposes the verdict (joined from the overlay).
        view_cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='job_overview'")}
        assert "ai_fit_score" in view_cols


def test_job_embedding_upsert_and_similarity_ranking():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        _insert_job(conn, "t:a:2")
        db.upsert_job_embedding(conn, "t:a:1", _vec(0), "test-model")
        db.upsert_job_embedding(conn, "t:a:2", _vec(1), "test-model")
        sims = db.fetch_similarity_map(conn, UID, _vec(0), "test-model")
    assert sims["t:a:1"] == pytest.approx(1.0)       # same direction as the query
    assert sims["t:a:2"] == pytest.approx(0.0)       # orthogonal
    # Upsert replaces (no duplicate-key error, new vector wins).
    with db.connect() as conn:
        db.upsert_job_embedding(conn, "t:a:2", _vec(0), "test-model")
        assert db.fetch_similarity_map(conn, UID, _vec(0), "test-model")["t:a:2"] == pytest.approx(1.0)


def test_embedding_cascades_with_the_shared_job():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        db.upsert_job_embedding(conn, "t:a:1", _vec(0))
        # The embedding is a child of the SHARED posting (not a user's overlay), so it
        # cascades when the catalog row goes — deleting a user's overlay would leave it.
        conn.execute("DELETE FROM jobs WHERE job_id = %s", ["t:a:1"])
        n = conn.execute("SELECT COUNT(*) AS n FROM job_embeddings").fetchone()["n"]
    assert n == 0


def test_profile_embedding_roundtrip():
    with db.connect() as conn:
        assert db.fetch_profile_embedding(conn, UID, "base_resume") is None
        db.upsert_profile_embedding(conn, UID, "base_resume", "hash1", _vec(0), "test-model")
        row = db.fetch_profile_embedding(conn, UID, "base_resume")
        assert row["content_hash"] == "hash1" and row["model"] == "test-model"
        assert row["embedding"][0] == 1.0 and len(row["embedding"]) == db_dim()
        db.upsert_profile_embedding(conn, UID, "base_resume", "hash2", _vec(1))
        assert db.fetch_profile_embedding(conn, UID, "base_resume")["content_hash"] == "hash2"


def test_get_profile_embedding_caches_by_hash_and_clears_verdicts_on_change(monkeypatch):
    calls = []

    def fake_embed(text, timeout=60):
        calls.append(text)
        return _vec(len(calls) - 1)

    monkeypatch.setattr(embeddings, "embed_text", fake_embed)
    with db.connect() as conn:                       # a verdicted job, to prove the wipe
        _insert_job(conn, "t:a:1")
        db.set_fit_verdict(conn, UID, "t:a:1", 8, "solid match")

    assert embeddings.get_profile_embedding(UID, "base_resume", "resume v1")[0] == 1.0
    assert embeddings.get_profile_embedding(UID, "base_resume", "resume v1")[0] == 1.0
    assert len(calls) == 1                           # second call hit the cache
    with db.connect() as conn:                       # verdict untouched so far
        assert db.fetch_verdicted_ids(conn, UID) == {"t:a:1"}

    embeddings.get_profile_embedding(UID, "base_resume", "resume v2 — I learned new things")
    assert len(calls) == 2                           # changed text → re-embedded…
    with db.connect() as conn:                       # …and the stale verdicts were cleared
        assert db.fetch_verdicted_ids(conn, UID) == set()


def test_fit_verdict_roundtrip_and_backfill_query():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        _insert_job(conn, "t:a:2")
        db.upsert_job_embedding(conn, "t:a:1", _vec(0), "test-model")
        db.upsert_job_embedding(conn, "t:a:2", _vec(1), "test-model")
        db.set_fit_verdict(conn, UID, "t:a:1", 9, "near-perfect overlap")

        row = db.fetch_overview_row(conn, UID, "t:a:1")   # verdict flows through the view
        assert row["ai_fit_score"] == 9 and row["ai_fit_reason"] == "near-perfect overlap"
        assert row["ai_fit_at"] is not None

        # The backfill query returns only UNverdicted embedded jobs, best match first,
        # carrying the stored posting text ('' when none was ever stored).
        db.upsert_job_description(conn, "t:a:2", "posting text for job 2")
        todo = db.fetch_top_unverdicted(conn, UID, _vec(1), "test-model", limit=10)
        assert [r["job_id"] for r in todo] == ["t:a:2"]
        assert todo[0]["description"] == "posting text for job 2"

        # No limit (the default) returns ALL unverdicted jobs.
        db.clear_fit_verdicts(conn, UID)
        todo = db.fetch_top_unverdicted(conn, UID, _vec(1), "test-model")
        assert {r["job_id"] for r in todo} == {"t:a:1", "t:a:2"}
        assert next(r for r in todo if r["job_id"] == "t:a:1")["description"] == ""
        db.set_fit_verdict(conn, UID, "t:a:1", 9, "near-perfect overlap")

        assert db.clear_fit_verdicts(conn, UID) == 1
        assert db.fetch_verdicted_ids(conn, UID) == set()


def test_unembedded_for_model_finds_pre_vector_and_stale_rows():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")                   # never embedded → candidate
        _insert_job(conn, "t:a:2")                   # embedded by the ACTIVE model → not one
        _insert_job(conn, "t:a:3")                   # embedded by a RETIRED model → candidate
        db.upsert_job_embedding(conn, "t:a:2", _vec(0), "new-model")
        db.upsert_job_embedding(conn, "t:a:3", [1.0, 0.0, 0.0, 0.0], "old-model")
        db.set_fit_verdict(conn, UID, "t:a:3", 5, "middling")   # scored rides along
        db.upsert_job_description(conn, "t:a:1", "old but stored text")

        rows = db.fetch_unembedded_for_model(conn, UID, "new-model")
        by_id = {r["job_id"]: r for r in rows}
    assert set(by_id) == {"t:a:1", "t:a:3"}
    assert by_id["t:a:1"]["description"] == "old but stored text"
    assert by_id["t:a:1"]["ai_fit_score"] is None    # unscored → the backfill scores it
    assert by_id["t:a:3"]["ai_fit_score"] == 5       # scored → re-embed only


def test_v10_allows_any_dimension_and_model_filter_hides_stale_rows():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        _insert_job(conn, "t:a:2")
        db.upsert_job_embedding(conn, "t:a:1", _vec(0), "new-model")
        # 4-dim vector: rejected by vector(1536), accepted after v10 unpins the column.
        db.upsert_job_embedding(conn, "t:a:2", [1.0, 0.0, 0.0, 0.0], "old-model")
        # The stale-model row is INVISIBLE (not an error) to model-scoped similarity.
        sims = db.fetch_similarity_map(conn, UID, _vec(0), "new-model")
    assert set(sims) == {"t:a:1"}
    assert sims["t:a:1"] == pytest.approx(1.0)


def test_job_description_roundtrip_and_cascade():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        assert db.fetch_job_description_text(conn, "t:a:1") is None
        db.upsert_job_description(conn, "t:a:1", "v1 text")
        db.upsert_job_description(conn, "t:a:1", "v2 text")   # refresh, not duplicate
        assert db.fetch_job_description_text(conn, "t:a:1") == "v2 text"

        # The text is a child of the SHARED posting — it cascades with the catalog row.
        conn.execute("DELETE FROM jobs WHERE job_id = %s", ["t:a:1"])
        assert db.fetch_job_description_text(conn, "t:a:1") is None


def test_rejection_memory_roundtrip():
    with db.connect() as conn:
        db.record_rejection(conn, UID, "https://x.co/1", "Acme", "Sales Manager", "no salary")
        db.record_rejection(conn, UID, "https://x.co/2", "Beta", "Staff Engineer", "IC level")
        db.record_rejection(conn, UID, "", "NoUrl", "Role", "ignored")   # empty URL → no row
        assert db.fetch_rejected_urls(conn, UID) == {"https://x.co/1", "https://x.co/2"}

        # Re-rejecting the same URL updates the reason instead of erroring.
        db.record_rejection(conn, UID, "https://x.co/1", "Acme", "Sales Manager", "on-site")
        row = conn.execute("SELECT reason FROM rejected_jobs WHERE user_id = %s "
                           "AND job_url = %s", [UID, "https://x.co/1"]).fetchone()
        assert row["reason"] == "on-site"


def test_fit_score_check_constraint():
    with db.connect() as conn:
        _insert_job(conn, "t:a:1")
        with pytest.raises(psycopg.errors.CheckViolation):
            db.set_fit_verdict(conn, UID, "t:a:1", 11, "impossible score")
