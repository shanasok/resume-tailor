# tests/test_resume_diff.py — the base-vs-generated résumé diff (src/resume_diff.py). Pure logic,
# no DB/LLM/network: builds two tailoring._SCHEMA dicts and asserts how each section is classified
# (same/changed/added/removed), that reworded bullets FUZZY-PAIR into one word-level "changed" row
# instead of a removed+added pair, and that inline word diffs are HTML-escaped.
from src import resume_diff as R


def _profile(**over):
    """A minimal valid base/generated profile; override any section per test."""
    base = {
        "name": "Test User",
        "contact": {"location": "NJ", "phone": "", "email": "", "website": ""},
        "summary": "Engineer with ten years building platforms.",
        "competencies": ["Python", "Leadership", "SQL"],
        "experience": [{"company": "Acme", "title": "Engineer", "dates": "2020-2024",
                        "bullets": ["Built the billing service handling 1M requests a day.",
                                    "Mentored three junior engineers on code review."]}],
        "education": [{"institution": "State U", "degree": "BS CS", "date": "2016", "detail": ""}],
        "certifications": ["AWS Cloud Practitioner"],
        "skills": [{"category": "Languages", "items": "Python, Go"}],
    }
    base.update(over)
    return base


# ── word-level inline diff ────────────────────────────────────────────────────
def test_inline_diff_marks_removed_and_added_words():
    left, right = R.inline_diff("the quick brown fox", "the slow brown fox")
    assert '<span class="wd-del">quick</span>' in left
    assert '<span class="wd-ins">slow</span>' in right
    assert "brown fox" in left and "brown fox" in right       # shared words stay plain


def test_inline_diff_escapes_html():
    left, right = R.inline_diff("a < b & c", "a < b & <b>d</b>")
    assert "&lt;" in left and "&amp;" in left
    assert "<b>d</b>" not in right and "&lt;b&gt;d&lt;/b&gt;" in right


# ── summary ───────────────────────────────────────────────────────────────────
def test_summary_changed_is_word_level():
    d = R.diff_resumes(_profile(), _profile(summary="Engineer with twelve years building platforms."))
    rows = next(s for s in d["sections"] if s["title"] == "Professional Summary")["rows"]
    assert [r["status"] for r in rows] == ["changed"]
    assert "wd-del" in rows[0]["left_html"] and "wd-ins" in rows[0]["right_html"]


def test_identical_summary_is_same():
    d = R.diff_resumes(_profile(), _profile())
    rows = next(s for s in d["sections"] if s["title"] == "Professional Summary")["rows"]
    assert [r["status"] for r in rows] == ["same"]


# ── competencies: discrete add/remove, NO fuzzy renames ───────────────────────
def test_competencies_add_and_remove():
    gen = _profile(competencies=["Python", "SQL", "Kubernetes"])   # drop Leadership, add Kubernetes
    d = R.diff_resumes(_profile(), gen)
    rows = next(s for s in d["sections"] if s["title"] == "Core Competencies")["rows"]
    by_status = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r["left"] or r["right"])
    assert "Leadership" in by_status["removed"]
    assert "Kubernetes" in by_status["added"]
    assert "changed" not in by_status                            # labels never fuzzy-pair


# ── experience bullets: reword pairs as one changed row ───────────────────────
def test_reworded_bullet_pairs_as_changed_not_add_remove():
    gen = _profile(experience=[{"company": "Acme", "title": "Engineer", "dates": "2020-2024",
        "bullets": ["Built the billing service handling 2M requests a day.",   # reworded 1M→2M
                    "Mentored three junior engineers on code review."]}])      # identical
    d = R.diff_resumes(_profile(), gen)
    block = next(s for s in d["sections"] if s["title"] == "Experience")["roles"][0]
    statuses = [r["status"] for r in block["rows"]]
    assert statuses.count("changed") == 1 and statuses.count("same") == 1
    assert "added" not in statuses and "removed" not in statuses
    changed = next(r for r in block["rows"] if r["status"] == "changed")
    assert "wd-del" in changed["left_html"] and "wd-ins" in changed["right_html"]


def test_dropped_bullet_is_removed():
    gen = _profile(experience=[{"company": "Acme", "title": "Engineer", "dates": "2020-2024",
        "bullets": ["Built the billing service handling 1M requests a day."]}])   # dropped mentoring
    d = R.diff_resumes(_profile(), gen)
    block = next(s for s in d["sections"] if s["title"] == "Experience")["roles"][0]
    statuses = [r["status"] for r in block["rows"]]
    assert statuses.count("removed") == 1 and statuses.count("same") == 1


def test_unmatched_role_is_whole_added_block():
    gen = _profile()
    gen["experience"] = gen["experience"] + [{"company": "NewCo", "title": "Lead", "dates": "2024-",
        "bullets": ["Led the platform team."]}]
    d = R.diff_resumes(_profile(), gen)
    roles = next(s for s in d["sections"] if s["title"] == "Experience")["roles"]
    added_block = next(b for b in roles if b["status"] == "added")
    assert added_block["left_header"] == "" and "NewCo" in added_block["right_header"]


# ── counts ────────────────────────────────────────────────────────────────────
def test_counts_tally_every_row():
    d = R.diff_resumes(_profile(), _profile(summary="Totally different summary text here."))
    total = sum(d["counts"].values())
    counted = 0
    for s in d["sections"]:
        counted += sum(len(b["rows"]) for b in s["roles"]) if s["kind"] == "experience" else len(s["rows"])
    assert total == counted
