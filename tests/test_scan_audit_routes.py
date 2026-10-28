# tests/test_scan_audit_routes.py — the gated /developer scan-audit UI: a scan-list page
# (Task 7) and a filterable event-table page (Task 8) over the Postgres-backed audit trail
# written by src/scan_audit.py. Uses the throwaway test DB (fixture copied from
# tests/test_developer_options_routes.py).
import types
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


def test_scan_audits_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    assert client.get("/developer/scan-audits").status_code == 404


def test_scan_audits_lists_scans(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {"salary_floor": 200000})
    a.sieve_pass({"company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
                  "board_token": "acme", "location": "US"}, "keyword")
    a.stage({"company": "Acme", "title": "Eng Mgr"}, "store", "stored", ["ai_fit 8/10"])
    a.close(1)
    r = client.get("/developer/scan-audits")
    assert r.status_code == 200
    assert b"greenhouse" in r.data
    assert b"scan-audits/events" in r.data      # row links into the event view


def test_scan_events_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    assert client.get("/developer/scan-audits/events").status_code == 404


def test_scan_events_drill_in_and_filter(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "T", "source": "greenhouse",
             "board_token": "acme", "location": "US"}, "deep_screen", "dropped",
            ["salary tops out at $150,000 (< $200,000)"])
    a.stage({"company": "Beta", "title": "T", "source": "greenhouse",
             "board_token": "beta", "location": "France"}, "list_screen", "dropped",
            ["location is non-US"])
    a.close(0)
    # drill-in shows both
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert r.status_code == 200
    assert b"Acme" in r.data and b"Beta" in r.data
    # filter by reason_bucket=salary → only Acme
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}&reason_bucket=salary")
    assert b"Acme" in r.data and b"Beta" not in r.data
    # cross-scan company filter
    r = client.get("/developer/scan-audits/events?company=bet")
    assert b"Beta" in r.data and b"Acme" not in r.data


def test_scan_events_group_a_jobs_stages_into_one_card(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    job = {"company": "Acme", "title": "Eng Manager", "source": "greenhouse",
           "board_token": "acme", "location": "Hawthorne, CA"}
    a = ScanAudit(UID, ["greenhouse"], {})
    a.sieve_pass(job, "keyword")
    a.stage(job, "list_screen", "advanced")
    a.stage(job, "fetch", "ok")
    a.stage(job, "deep_screen", "dropped", ["salary tops out at $110,000 (< $200,000)"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert r.status_code == 200
    # One card for the job (title rendered exactly once), not one row per event.
    assert r.data.count(b"Eng Manager") == 1
    assert b"jobcard" in r.data
    # The full stage trace is present inside the (collapsed) card.
    for stage in (b"sieve_pass", b"list_screen", b"fetch", b"deep_screen"):
        assert stage in r.data
    # The collapsed summary names where + why it ended.
    assert b"deep_screen" in r.data and b"salary tops out at $110,000" in r.data
    # Grouped under a scan-time header.
    assert b"Scan #" in r.data


def test_scan_events_group_interleaved_jobs_each_into_one_card(client, monkeypatch):
    # Regression: seq is a single global counter across concurrent board threads, so a
    # job's stage events are interleaved with other jobs' events, not contiguous. Grouping
    # must still collapse each job into ONE card.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    alpha = {"company": "Alpha", "title": "Zeus Role", "source": "greenhouse",
             "board_token": "alpha", "location": "US"}
    bravo = {"company": "Bravo", "title": "Hera Role", "source": "greenhouse",
             "board_token": "bravo", "location": "US"}
    a = ScanAudit(UID, ["greenhouse"], {})
    a.sieve_pass(alpha, "keyword")                       # seq 1
    a.sieve_pass(bravo, "keyword")                       # seq 2  (interleaves alpha)
    a.stage(alpha, "list_screen", "advanced")           # seq 3
    a.stage(bravo, "list_screen", "dropped", ["location is non-US"])  # seq 4
    a.stage(alpha, "deep_screen", "dropped", ["salary tops out at $110,000"])  # seq 5
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert r.status_code == 200
    # Each job collapses to exactly ONE card despite the interleaving.
    assert r.data.count(b"Zeus Role") == 1
    assert r.data.count(b"Hera Role") == 1


def test_scan_events_per_page_selector(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "T", "source": "greenhouse",
             "board_token": "acme", "location": "US"}, "store", "stored", ["ai_fit 8/10"])
    a.close(1)
    # a valid per is honored and shown selected in the dropdown
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}&per=200")
    assert r.status_code == 200
    assert b'value="200" selected' in r.data
    # 'all' is a valid choice → renders the job with no pager, "All" selected
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}&per=all")
    assert r.status_code == 200
    assert b'value="all" selected' in r.data
    assert b"Acme" in r.data
    assert b"Next \xe2\x86\x92" not in r.data          # no next-page link under "All"
    # a junk per is rejected and falls back to the default (40)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}&per=7")
    assert b'value="40" selected' in r.data


def _fake_connector():
    job = {"source": "greenhouse", "board_token": "acme", "id": "42",
           "title": "Engineering Manager", "company": "Acme", "location": "Remote - US",
           "url": "https://boards.greenhouse.io/acme/jobs/42",
           "description": "Lead the platform team.", "salary_text": "$220k",
           "date_posted": "2026-07-01"}
    return types.SimpleNamespace(get_job=lambda token, ext: dict(job))


def test_add_job_route_404_when_flag_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", False)
    r = client.post("/developer/scan-audits/add-job",
                    data={"source": "greenhouse", "board_token": "acme",
                          "external_id": "42", "scan_id": "1"})
    assert r.status_code == 404


def test_add_job_route_missing_identity_400(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    r = client.post("/developer/scan-audits/add-job", data={"source": "greenhouse"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_add_job_route_happy_path_stores_and_marks(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src import job_search
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse", _fake_connector())
    monkeypatch.setattr(job_search.embeddings, "available", lambda: False)  # stay offline
    with db.connect() as conn:
        sid = db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
    r = client.post("/developer/scan-audits/add-job",
                    data={"source": "greenhouse", "board_token": "acme",
                          "external_id": "42", "scan_id": str(sid)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and "Acme" in body["message"]
    assert tracker.get_job(UID, "greenhouse:acme:42") is not None
    with db.connect() as conn:
        ev = conn.execute("SELECT * FROM scan_events WHERE scan_id = %s AND stage='manual_add'",
                          [sid]).fetchone()
    assert ev is not None and ev["detail"] == "added manually"


def test_add_job_route_reports_fetch_failure(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src import job_search
    def boom(token, ext):
        raise RuntimeError("posting no longer live")
    monkeypatch.setitem(job_search._CONNECTORS, "greenhouse",
                        types.SimpleNamespace(get_job=boom))
    with db.connect() as conn:
        sid = db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
    r = client.post("/developer/scan-audits/add-job",
                    data={"source": "greenhouse", "board_token": "acme",
                          "external_id": "42", "scan_id": str(sid)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False and "no longer live" in body["error"]


def test_track_button_renders_for_dropped_job_with_external_id(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
             "board_token": "acme", "location": "US", "id": "42",
             "url": "https://x/42"}, "deep_screen", "dropped",
            ["salary tops out at $150,000 (< $200,000)"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert b"track-btn" in r.data
    assert b'data-ext="42"' in r.data


def test_no_track_button_when_external_id_missing(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
             "board_token": "acme", "location": "US"},   # no id/url → external_id NULL
            "deep_screen", "dropped", ["on-site role"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert b"track-btn" not in r.data


def test_manual_add_event_renders_added_by_me_badge(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    with db.connect() as conn:
        sid = db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
        db.insert_scan_events(conn, [{"scan_id": sid, "seq": 1, "company": "Acme",
            "title": "Eng Mgr", "source": "greenhouse", "board_token": "acme",
            "location": "US", "external_id": "42", "url": "https://x/42",
            "stage": "deep_screen", "outcome": "dropped", "detail": "salary",
            "reason_bucket": "salary"}])
        db.append_scan_event(conn, sid, {"company": "Acme", "title": "Eng Mgr",
            "source": "greenhouse", "board_token": "acme", "location": "US",
            "external_id": "42", "url": "https://x/42", "stage": "manual_add",
            "outcome": "stored", "detail": "added manually", "reason_bucket": None})
    r = client.get(f"/developer/scan-audits/events?scan_id={sid}")
    assert b"added by me" in r.data
    # A manually-added job is no longer offered a Track button.
    assert r.data.count(b"track-btn") == 0


def test_stage_filter_offers_manual_add(client, monkeypatch):
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    with db.connect() as conn:
        db.insert_scan(conn, {"user_id": UID, "sources": ["greenhouse"], "filters": {}})
    r = client.get("/developer/scan-audits/events")
    assert b'value="manual_add"' in r.data


def test_job_posting_url_rendered_as_clickable_link(client, monkeypatch):
    # A card whose event captured a posting url shows a "View posting" link to the live posting,
    # so the developer can open it to decide whether to track it (and for debugging). Shown for
    # ANY card with a url, not only track-eligible ones.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
             "board_token": "acme", "location": "US", "id": "42",
             "url": "https://boards.greenhouse.io/acme/jobs/42"}, "deep_screen", "dropped",
            ["on-site role"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert b'href="https://boards.greenhouse.io/acme/jobs/42"' in r.data
    assert b"View posting" in r.data


def test_events_page_wires_fit_llm_from_localstorage(client, monkeypatch):
    # The Track POST forwards the home page's persisted fit-LLM selection (localStorage
    # 'resumeLlmFit', shared with index.html) so a rescue scores on the user's chosen backend
    # instead of always defaulting to ChatGPT.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Acme", "title": "Eng Mgr", "source": "greenhouse",
             "board_token": "acme", "location": "US", "id": "42", "url": "https://x/42"},
            "deep_screen", "dropped", ["on-site role"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert b"resumeLlmFit" in r.data


def test_no_posting_link_when_url_missing(client, monkeypatch):
    # Pre-v14 rows (and any event with no url) render no "View posting" link.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "Beta", "title": "Eng Mgr", "source": "greenhouse",
             "board_token": "beta", "location": "US"},   # no id/url
            "deep_screen", "dropped", ["on-site role"])
    a.close(0)
    r = client.get(f"/developer/scan-audits/events?scan_id={a.scan_id}")
    assert b"View posting" not in r.data


def test_salary_filter_orders_cards_by_posting_salary(client, monkeypatch):
    # With reason_bucket=salary active, cards sort by the posting's top salary
    # (highest first), and "no salary stated" jobs sink to the bottom. Jobs are
    # inserted mid/none/high so the result order can only come from the salary sort.
    monkeypatch.setattr(webapp, "DEVELOPER_OPTIONS", True)
    from src.scan_audit import ScanAudit
    a = ScanAudit(UID, ["greenhouse"], {})
    a.stage({"company": "MidCo", "title": "T", "source": "greenhouse",
             "board_token": "midco", "location": "US"}, "deep_screen", "dropped",
            ["salary tops out at $190,000 (< $200,000)"])
    a.stage({"company": "NoneCo", "title": "T", "source": "greenhouse",
             "board_token": "noneco", "location": "US"}, "deep_screen", "dropped",
            ["no salary stated in description"])
    a.stage({"company": "HighCo", "title": "T", "source": "greenhouse",
             "board_token": "highco", "location": "US"}, "deep_screen", "dropped",
            ["salary tops out at $220,000 (< $250,000)"])
    a.close(0)
    r = client.get(
        f"/developer/scan-audits/events?scan_id={a.scan_id}&reason_bucket=salary")
    assert r.status_code == 200
    body = r.data
    assert body.count(b"HighCo") == 1
    assert body.count(b"MidCo") == 1
    assert body.count(b"NoneCo") == 1
    # highest salary first, then next, then the no-salary job last
    assert body.index(b"HighCo") < body.index(b"MidCo") < body.index(b"NoneCo")
