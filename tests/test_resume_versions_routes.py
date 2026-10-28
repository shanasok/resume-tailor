# tests/test_resume_versions_routes.py — end-to-end drive of the résumé-history feature
# through the Flask routes: a regenerate archives the current résumé's files + row snapshot,
# /resume-versions lists them, and /restore-resume-version swaps a saved version back in and
# saves the current one to history first. Uses the throwaway test DB + a temp generated-files
# dir + stubbed renderers (no Playwright/LLM), so it exercises app.py's own wiring for real.
import uuid
from pathlib import Path

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


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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

    # Point the app's generated-files dir at a temp folder and stub the renderers so we don't
    # need Playwright — they just write the files restore/regenerate would produce.
    gen = tmp_path / "generated"
    gen.mkdir()
    monkeypatch.setattr(webapp, "GENERATED_DIR", gen.resolve())

    def fake_render_html(data, company, role, out_dir=None):
        out = Path(out_dir) if out_dir else gen
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{company}_{role}".lower().replace(" ", "_") + "_resume"
        (out / f"{stem}.json").write_text(webapp.json.dumps(data), encoding="utf-8")
        html = out / f"{stem}.html"
        html.write_text(f"<h1>{data.get('marker', '')}</h1>", encoding="utf-8")
        return html

    def fake_render_pdf(html_path):
        pdf = Path(html_path).with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 " + Path(html_path).read_bytes())
        return pdf

    monkeypatch.setattr(webapp, "render_html", fake_render_html)
    monkeypatch.setattr(webapp, "render_pdf", fake_render_pdf)

    webapp.app.config["TESTING"] = True
    yield webapp.app.test_client(), gen
    db._schema_ready = False


def _seed_current_resume(gen: Path, marker: str, score: float):
    """Create a job + résumé row whose files really exist under the temp generated dir."""
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": "Acme", "title": "Director",
           "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
           "match": {"flags": [], "technical": True}}
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)
    stem = "acme_director_resume"
    _write(gen / f"{stem}.html", f"<h1>{marker}</h1>")
    _write(gen / f"{stem}.pdf", "pdf-" + marker)
    _write(gen / f"{stem}.json", webapp.json.dumps({"marker": marker}))
    tracker.log_application(UID, "Acme", "Director", job["url"],
                            str(gen / f"{stem}.html"), str(gen / f"{stem}.pdf"),
                            job_id=jid, match_score=score,
                            critic_findings={"strength": [f"v-{marker}"]})
    return jid, stem


def test_snapshot_list_and_restore_round_trip(client):
    cl, gen = client
    jid, stem = _seed_current_resume(gen, "ORIGINAL", 9.0)

    # Simulate what a regenerate does to the CURRENT résumé: archive it, then overwrite files.
    version = webapp._snapshot_current_resume(UID, jid, "before AI rewrite")
    assert version == 1
    # The archived copies exist under versions/ and preserve the ORIGINAL content.
    varch = gen / "versions" / f"{stem}_v1.html"
    assert varch.exists() and "ORIGINAL" in varch.read_text()
    assert (gen / "versions" / f"{stem}_v1.pdf").exists()
    assert (gen / "versions" / f"{stem}_v1.json").exists()
    # Now overwrite the live files to a WORSE version (what a bad rewrite would leave).
    _write(gen / f"{stem}.html", "<h1>WORSE</h1>")
    _write(gen / f"{stem}.pdf", "pdf-WORSE")
    _write(gen / f"{stem}.json", webapp.json.dumps({"marker": "WORSE"}))
    tracker.log_application(UID, "Acme", "Director", "https://boards.acme.com/x",
                            str(gen / f"{stem}.html"), str(gen / f"{stem}.pdf"),
                            job_id=jid, match_score=4.0,
                            critic_findings={"fidelity": ["invented scope"]})

    # /resume-versions leads with the CURRENT (live) résumé — so its LLM is always visible —
    # then the archived ORIGINAL (score 9.0, restorable from its JSON).
    resp = cl.get(f"/resume-versions?job_id={jid}")
    data = resp.get_json()
    assert data["ok"] and len(data["versions"]) == 2
    cur = data["versions"][0]
    assert cur["is_current"] is True and cur["version"] is None and cur["can_restore"] is False
    v = data["versions"][1]
    assert v["version"] == 1 and v["match_score"] == 9.0 and v["can_restore"] is True
    assert v["note"] == "before AI rewrite" and v["is_current"] is False

    # Restore version 1 → the current (WORSE) résumé is archived as v2, and the live files +
    # row go back to the ORIGINAL's content and score.
    resp = cl.post("/restore-resume-version", data={"job_id": jid, "version": "1"})
    rdata = resp.get_json()
    assert rdata["ok"], rdata
    assert rdata["match_score"] == 9.0
    assert "ORIGINAL" in (gen / f"{stem}.html").read_text()      # live files restored
    row = tracker.get_job(UID, jid)
    assert float(row["Match Score"]) == 9.0                      # row points back at v1's score
    assert row["Critic Findings"] == {"strength": ["v-ORIGINAL"]}

    # The WORSE version we replaced was saved to history first → current + two archived now.
    versions = cl.get(f"/resume-versions?job_id={jid}").get_json()["versions"]
    assert [v["version"] for v in versions] == [None, 2, 1]       # current, then archived v2, v1
    assert versions[0]["is_current"] is True                     # current = restored ORIGINAL
    assert versions[1]["match_score"] == 4.0                     # v2 is the archived WORSE one


def test_restore_rejects_unknown_version(client):
    cl, gen = client
    jid, _ = _seed_current_resume(gen, "ORIGINAL", 9.0)
    resp = cl.post("/restore-resume-version", data={"job_id": jid, "version": "99"})
    data = resp.get_json()
    assert data["ok"] is False and "no longer exists" in data["message"]


def test_snapshot_noops_without_a_current_resume(client):
    cl, gen = client
    # A job with no résumé row → nothing to archive.
    job = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
           "company": "Acme", "title": "Director",
           "location": "Remote — USA", "url": "https://boards.acme.com/none",
           "match": {"flags": [], "technical": True}}
    tracker.upsert_found_jobs(UID, [job])
    jid = tracker.make_job_id(job)
    assert webapp._snapshot_current_resume(UID, jid, "noop") is None
    assert cl.get(f"/resume-versions?job_id={jid}").get_json()["versions"] == []
