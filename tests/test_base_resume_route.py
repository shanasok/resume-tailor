# tests/test_base_resume_route.py — the /base-resume enter/edit screen and the /import
# endpoint: form parsing, pre-fill, save wiring, and safe .docx import. The DB and the
# extractor are stubbed, so these run without Postgres or an LLM.
from contextlib import contextmanager
from io import BytesIO

import pytest

import app as app_module
from src import db, resume_parser, resume_extract


@pytest.fixture
def client(monkeypatch):
    """Flask test client with db.connect stubbed to a no-op (the route's fetch/upsert calls
    are stubbed per test), so no database is needed."""
    @contextmanager
    def fake_connect():
        yield object()
    monkeypatch.setattr(db, "connect", fake_connect)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_get_shows_entry_form_when_empty(client, monkeypatch):
    monkeypatch.setattr(db, "fetch_resume_profile", lambda conn, uid: None)
    resp = client.get("/base-resume")
    assert resp.status_code == 200
    assert b"Your Base R\xc3\xa9sum\xc3\xa9" in resp.data       # heading renders
    assert b"Enter your r\xc3\xa9sum\xc3\xa9 once" in resp.data  # first-time copy


def test_get_prefills_existing_profile(client, monkeypatch):
    profile = {"name": "Jane Doe",
               "contact": {"location": "", "phone": "", "email": "jane@x.co", "website": ""},
               "summary": "Sum.", "competencies": ["L&D"],
               "experience": [{"company": "Acme", "title": "Dir", "dates": "2020",
                               "location": "", "bullets": ["Did a thing"]}],
               "education": [], "certifications": [], "skills": []}
    monkeypatch.setattr(db, "fetch_resume_profile", lambda conn, uid: profile)
    resp = client.get("/base-resume")
    assert resp.status_code == 200
    assert b"Jane Doe" in resp.data
    assert b"jane@x.co" in resp.data
    assert b"Acme" in resp.data
    assert b"Did a thing" in resp.data
    assert b"Edit the r\xc3\xa9sum\xc3\xa9" in resp.data         # edit-mode copy


def test_post_parses_and_saves_profile(client, monkeypatch):
    saved = {}
    monkeypatch.setattr(db, "upsert_resume_profile",
                        lambda conn, uid, profile: saved.update(profile=profile, uid=uid))
    resp = client.post("/base-resume", data={
        "name": "Jane Doe", "email": "jane@x.co", "phone": "555",
        "location": "NJ", "website": "j.dev",
        "summary": "My summary.",
        "competencies": "L&D\nEnablement\n",
        # Third experience block is an empty leftover "Add" row → must be dropped.
        "exp_company": ["Acme", "Beta", ""],
        "exp_title": ["Director", "Manager", ""],
        "exp_dates": ["2020-2024", "2015-2020", ""],
        "exp_location": ["Remote", "Denver, CO", ""],
        "exp_bullets": ["Built the org\nCut time 40%", "Ran curriculum", ""],
        "edu_institution": ["State U"], "edu_degree": ["BA"],
        "edu_date": ["2011"], "edu_detail": [""],
        "skill_category": ["Tools", ""], "skill_items": ["Jira, SQL", ""],
        "certifications": "CSM\nAWS",
    }, follow_redirects=False)

    assert resp.status_code == 302                     # save → redirect back to the form
    assert saved["uid"] == 1                           # the default user
    p = saved["profile"]
    assert p["name"] == "Jane Doe"
    assert p["contact"] == {"location": "NJ", "phone": "555",
                            "email": "jane@x.co", "website": "j.dev"}
    assert p["summary"] == "My summary."
    assert p["competencies"] == ["L&D", "Enablement"]
    # Empty trailing block dropped; bullets split per line.
    assert [e["company"] for e in p["experience"]] == ["Acme", "Beta"]
    assert p["experience"][0]["bullets"] == ["Built the org", "Cut time 40%"]
    assert p["education"] == [{"institution": "State U", "degree": "BA",
                              "date": "2011", "detail": ""}]
    assert p["skills"] == [{"category": "Tools", "items": "Jira, SQL"}]
    assert p["certifications"] == ["CSM", "AWS"]


# ── /base-resume/import (safe .docx → structured profile) ────────────────────────────────

def _import(client, filename="resume.docx", data=b"fake-docx-bytes"):
    return client.post("/base-resume/import",
                       data={"resume_file": (BytesIO(data), filename)},
                       content_type="multipart/form-data", follow_redirects=False)


def test_import_saves_the_extracted_profile(client, monkeypatch):
    profile = {"name": "Jane", "contact": {}, "summary": "", "competencies": [],
               "experience": [{"company": "Acme", "title": "Dir", "dates": "",
                               "location": "", "bullets": []}],
               "education": [], "certifications": [], "skills": [{"category": "T", "items": "x"}]}
    monkeypatch.setattr(resume_parser, "extract_docx_text", lambda src: "some résumé text")
    monkeypatch.setattr(resume_extract, "extract_profile_from_text",
                        lambda text, **k: profile)
    saved = {}
    monkeypatch.setattr(db, "upsert_resume_profile",
                        lambda conn, uid, p: saved.update(uid=uid, p=p))
    resp = _import(client)
    assert resp.status_code == 302                    # → back to /base-resume for review
    assert saved["uid"] == 1 and saved["p"] == profile


def test_import_rejects_non_docx_before_touching_the_model(client, monkeypatch):
    reached = {"extract": False}
    monkeypatch.setattr(resume_extract, "extract_profile_from_text",
                        lambda *a, **k: reached.__setitem__("extract", True) or {})
    resp = _import(client, filename="resume.txt")
    assert resp.status_code == 302
    assert reached["extract"] is False


def test_import_rejects_oversized_file_before_the_model(client, monkeypatch):
    monkeypatch.setattr(app_module, "_RESUME_UPLOAD_MAX_BYTES", 10)
    reached = {"extract": False}
    monkeypatch.setattr(resume_extract, "extract_profile_from_text",
                        lambda *a, **k: reached.__setitem__("extract", True) or {})
    resp = _import(client, data=b"x" * 100)
    assert resp.status_code == 302
    assert reached["extract"] is False


def test_import_survives_extraction_failure_without_saving(client, monkeypatch):
    monkeypatch.setattr(resume_parser, "extract_docx_text", lambda src: "text")

    def boom(text, **k):
        raise resume_extract.ExtractionError("Couldn't read a résumé out of that document.")
    monkeypatch.setattr(resume_extract, "extract_profile_from_text", boom)
    saved = {"called": False}
    monkeypatch.setattr(db, "upsert_resume_profile",
                        lambda *a, **k: saved.__setitem__("called", True))
    resp = _import(client)
    assert resp.status_code == 302                    # graceful redirect, no crash
    assert saved["called"] is False                   # nothing stored on failure


def test_import_passes_the_chosen_llm_through(client, monkeypatch):
    monkeypatch.setattr(resume_parser, "extract_docx_text", lambda src: "text")
    seen = {}

    def fake_extract(text, llm="claude"):
        seen["llm"] = llm
        raise resume_extract.ExtractionError("stop after capturing the llm")
    monkeypatch.setattr(resume_extract, "extract_profile_from_text", fake_extract)
    resp = client.post("/base-resume/import",
                       data={"resume_file": (BytesIO(b"fake-docx-bytes"), "resume.docx"),
                             "import_llm": "chatgpt"},
                       content_type="multipart/form-data", follow_redirects=False)
    assert resp.status_code == 302
    assert seen["llm"] == "chatgpt"


def test_import_defaults_to_claude_when_no_llm_field(client, monkeypatch):
    monkeypatch.setattr(resume_parser, "extract_docx_text", lambda src: "text")
    seen = {}

    def fake_extract(text, llm=None):
        seen["llm"] = llm
        raise resume_extract.ExtractionError("stop after capturing the llm")
    monkeypatch.setattr(resume_extract, "extract_profile_from_text", fake_extract)
    _import(client)                                   # helper sends no import_llm field
    assert seen["llm"] == "claude"
