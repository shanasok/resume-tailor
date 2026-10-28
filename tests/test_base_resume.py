# tests/test_base_resume.py — the per-user base résumé the pipeline tailors from:
# rendering a structured profile to text, the "has content" test, and the DB-profile
# text source. No DB or files needed (db/resume_parser are stubbed).
from contextlib import contextmanager

import pytest

from src import base_resume
from src.base_resume import profile_to_text, profile_has_content


def _full():
    return {
        "name": "Jane Doe",
        "contact": {"location": "Denver, CO", "phone": "555-0100",
                    "email": "jane@x.co", "website": "janedoe.dev"},
        "summary": "L&D leader with 15 years in developer education.",
        "competencies": ["L&D strategy", "Curriculum design"],
        "experience": [
            {"company": "Acme", "title": "Director of Enablement", "dates": "2019 – 2024",
             "location": "Remote", "bullets": ["Built the L&D org.", "Cut onboarding 40%."]},
        ],
        "education": [{"institution": "State U", "degree": "B.A. English", "date": "2011",
                       "detail": "cum laude"}],
        "certifications": ["Certified Scrum Master"],
        "skills": [{"category": "Tools", "items": "Jira, SQL"}],
    }


def test_profile_to_text_renders_all_sections_in_order():
    text = profile_to_text(_full())
    # Section headers present, in résumé order.
    for header in ("SUMMARY", "CORE COMPETENCIES", "EXPERIENCE", "EDUCATION", "SKILLS",
                   "CERTIFICATIONS"):
        assert header in text
    assert text.index("EXPERIENCE") < text.index("EDUCATION") < text.index("SKILLS")
    # Contact line, experience header shape, and bullets.
    assert "Jane Doe" in text
    assert "Denver, CO | 555-0100 | jane@x.co | janedoe.dev" in text
    assert "Director of Enablement — Acme (Remote) | 2019 – 2024" in text
    assert "- Built the L&D org." in text
    assert "Tools: Jira, SQL" in text
    assert "- Certified Scrum Master" in text


def test_profile_to_text_omits_empty_sections():
    text = profile_to_text({"name": "Solo Name", "contact": {}, "summary": "",
                            "competencies": [], "experience": [], "education": [],
                            "certifications": [], "skills": []})
    assert text == "Solo Name"                       # just the name, no section headers
    assert "EXPERIENCE" not in text


def test_profile_to_text_handles_empty_and_none():
    assert profile_to_text({}) == ""
    assert profile_to_text(None) == ""


@pytest.mark.parametrize("profile,expected", [
    (None, False),
    ({}, False),
    ({"name": "", "summary": "", "experience": [], "competencies": []}, False),
    ({"name": "Jane"}, True),
    ({"summary": "Some summary"}, True),
    ({"experience": [{"company": "Acme"}]}, True),
    ({"competencies": ["x"]}, True),
    ({"certifications": ["   "]}, False),            # only-whitespace → not content
])
def test_profile_has_content(profile, expected):
    assert profile_has_content(profile) is expected


# ── resume_source_text / has_base_resume (db + resume_parser stubbed) ─────────────────────

@pytest.fixture
def stub_db(monkeypatch):
    """Stub base_resume's DB access; returns a setter for the profile fetch_resume_profile
    should return."""
    from src import db

    @contextmanager
    def fake_connect():
        yield object()
    monkeypatch.setattr(db, "connect", fake_connect)
    # No chat additions in tests → append_additions is a no-op.
    monkeypatch.setattr(base_resume.resume_parser, "read_additions", lambda: "")

    def set_profile(profile):
        monkeypatch.setattr(db, "fetch_resume_profile", lambda conn, uid: profile)
    return set_profile


def test_resume_source_text_uses_the_db_profile(stub_db):
    stub_db(_full())
    text = base_resume.resume_source_text(1)
    assert "Jane Doe" in text and "Director of Enablement — Acme" in text


def test_resume_source_text_raises_when_no_profile(stub_db):
    stub_db(None)
    with pytest.raises(FileNotFoundError) as exc:
        base_resume.resume_source_text(1)
    # The DB profile is the only source — the error points to the Base Résumé screen
    # and must NOT offer the retired .docx upload path.
    msg = str(exc.value)
    assert "/base-resume" in msg
    assert ".docx" not in msg


def test_has_base_resume(stub_db):
    stub_db(_full())
    assert base_resume.has_base_resume(1) is True
    stub_db(None)
    assert base_resume.has_base_resume(1) is False
