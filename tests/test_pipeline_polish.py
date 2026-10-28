# tests/test_pipeline_polish.py — the surgical POLISH stage: after the re-tailor loop, the
# pipeline makes minimal edits to remove any residual truthfulness / AI-style flags so the
# FINAL résumé ships clean instead of shipping the "best effort" with known problems.
#
# These tests stub every LLM/render/DB boundary the pipeline imports, so nothing hits a real
# model or the database — they assert the CONTROL FLOW (when polish runs, that it converges,
# and that a backfired polish is discarded).
import pytest

from src import pipeline


@pytest.fixture
def stub_io(monkeypatch):
    """Stub the non-LLM boundaries: base résumé + job text, rendering, and the DB write."""
    monkeypatch.setattr(pipeline, "resume_source_text", lambda uid: "BASE RESUME TEXT")
    monkeypatch.setattr(pipeline, "structured_profile", lambda uid: None)  # skip fact-locking
    monkeypatch.setattr(pipeline, "parse_job_posting", lambda src: "JOB POSTING TEXT")
    monkeypatch.setattr(pipeline, "render_html", lambda data, c, r, out: "/tmp/x.html")
    monkeypatch.setattr(pipeline, "render_pdf", lambda html: "/tmp/x.pdf")
    monkeypatch.setattr(pipeline, "count_pdf_pages", lambda pdf: 2)        # length always fine
    logged = {}
    monkeypatch.setattr(pipeline, "log_application",
                        lambda *a, **k: logged.update(k))
    return logged


def _review(score, fidelity=(), style=()):
    return {"overall_score": score, "fidelity_issues": list(fidelity),
            "style_issues": list(style), "revisions": ["rename X to Y"], "strengths": ["real metrics kept"]}


def test_critic_failure_ships_draft_unaudited_not_a_hard_failure(monkeypatch, stub_io):
    """The Collectorsuniverse failure: a completed, valid draft existed but the critic call
    couldn't return parseable JSON (truncated → 'Unbalanced braces'), which killed the WHOLE
    generation. The critic is a quality GATE, not the producer — its failure must degrade to
    shipping the finished draft UNAUDITED (never marked accepted), not raise."""
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})

    def boom_score(*a, **k):
        raise ValueError("Claude API returned invalid JSON: Unbalanced braces …")
    monkeypatch.setattr(pipeline, "score_resume", boom_score)

    # No exception — the pipeline returns a rendered résumé instead of failing outright.
    html, pdf, meta = pipeline.generate(1, "Co", "Role", "src",
                                        max_attempts=1, use_external_critic=False)

    assert html and pdf                       # the finished draft was still rendered to disk
    assert meta["unaudited"] is True          # flagged so the UI/log can say the audit didn't run
    assert meta["accepted"] is False          # never falsely blessed without a clean audit
    assert meta["fidelity_issues"] == []      # no findings to show — the critic never ran
    assert stub_io["match_score"] == 0.0      # score 0 → the row reflects the unscored state


@pytest.mark.parametrize("stage", ["generate", "check", "regenerate", "external"])
def test_embedding_model_on_any_stage_is_refused_before_work(monkeypatch, stage):
    """An embedding model (vectors only) can't draft or critique a résumé. Picked for any
    writing/critic stage, generate() must fail FAST with a clear message — before any LLM,
    render, or DB call — rather than silently shipping an unaudited, fabricated résumé (the
    real fabricated-tools incident)."""
    # Trip a loud failure if the guard ever lets execution reach real work.
    for boundary in ("resume_source_text", "parse_job_posting", "tailor_resume", "score_resume"):
        monkeypatch.setattr(pipeline, boundary,
                            lambda *a, **k: pytest.fail("guard did not fire before doing work"))
    sl = {"generate": "claude", "check": "claude", "regenerate": "claude", "external": "chatgpt"}
    sl[stage] = "local:nomic-embed-text:latest"
    with pytest.raises(ValueError) as exc:
        pipeline.generate(1, "Co", "Role", "src", stage_llms=sl,
                          use_external_critic=(stage == "external"))
    assert stage in str(exc.value) and "embedding model" in str(exc.value)


def test_embedding_model_on_skipped_external_stage_is_allowed(monkeypatch, stub_io):
    """The external stage is optional. If it's turned OFF, an embedding value sitting in that
    slot must NOT block the run — only stages that actually execute are guarded."""
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})
    monkeypatch.setattr(pipeline, "score_resume", lambda *a, **k: _review(92))
    _, _, meta = pipeline.generate(
        1, "Co", "Role", "src", max_attempts=1, use_external_critic=False,
        stage_llms={"generate": "claude", "check": "claude", "regenerate": "claude",
                    "external": "local:nomic-embed-text:latest"})
    assert meta["accepted"] is True           # ran to completion; the unused embedding slot was ignored


def test_polish_clears_residual_flags_and_accepts(monkeypatch, stub_io):
    """Re-tailor keeps producing a draft with a truthfulness flag; ONE surgical polish removes
    it and the result is accepted — the flag does not ship in the final résumé."""
    # The re-tailor always yields the flawed draft (marker v=1); polish yields the clean one (v=2).
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})
    monkeypatch.setattr(pipeline, "polish_resume", lambda *a, **k: {"v": 2})

    # The critic flags the flawed draft and accepts the polished one.
    def fake_score(resume_data, *a, **k):
        if resume_data["v"] == 1:
            return _review(60, fidelity=["Vendor Governance inflates the base"])
        return _review(92)                    # polished draft: clean + high score
    monkeypatch.setattr(pipeline, "score_resume", fake_score)

    _, _, meta = pipeline.generate(1, "Co", "Role", "src",
                                   max_attempts=1, use_external_critic=False)

    assert meta["accepted"] is True
    assert meta["fidelity_issues"] == []
    assert meta["style_issues"] == []
    assert meta["polish_rounds"] == 1         # exactly one surgical round was needed
    assert stub_io["match_score"] == 9.2      # the polished score was the one persisted


def test_polish_is_discarded_when_it_backfires(monkeypatch, stub_io):
    """A polish that makes things WORSE (introduces more flags) must be thrown away — the
    pipeline keeps the better pre-polish draft, so polishing can never lower quality."""
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})
    monkeypatch.setattr(pipeline, "polish_resume", lambda *a, **k: {"v": 2})

    def fake_score(resume_data, *a, **k):
        if resume_data["v"] == 1:
            return _review(68, style=["echoes the posting"])          # 1 style flag
        return _review(40, fidelity=["new fabrication"], style=["a", "b"])  # worse
    monkeypatch.setattr(pipeline, "score_resume", fake_score)

    _, _, meta = pipeline.generate(1, "Co", "Role", "src",
                                   max_attempts=1, max_polish=2, use_external_critic=False)

    # Kept the pre-polish draft (score 6.8, its single style flag) rather than the worse polish.
    assert meta["score"] == 68
    assert meta["style_issues"] == ["echoes the posting"]
    assert meta["fidelity_issues"] == []


def test_no_polish_when_max_polish_zero(monkeypatch, stub_io):
    """max_polish=0 disables the stage entirely (opt-out), even with residual flags."""
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})
    def boom(*a, **k):
        raise AssertionError("polish_resume must not be called when max_polish=0")
    monkeypatch.setattr(pipeline, "polish_resume", boom)
    monkeypatch.setattr(pipeline, "score_resume",
                        lambda *a, **k: _review(60, fidelity=["some issue"]))

    _, _, meta = pipeline.generate(1, "Co", "Role", "src",
                                   max_attempts=1, max_polish=0, use_external_critic=False)
    assert meta["polish_rounds"] == 0
    assert meta["accepted"] is False


def test_no_polish_when_already_clean(monkeypatch, stub_io):
    """An accepted draft (no flags) never triggers polishing — no wasted LLM calls."""
    monkeypatch.setattr(pipeline, "tailor_resume", lambda *a, **k: {"v": 1})
    def boom(*a, **k):
        raise AssertionError("polish_resume must not be called for an already-clean draft")
    monkeypatch.setattr(pipeline, "polish_resume", boom)
    monkeypatch.setattr(pipeline, "score_resume", lambda *a, **k: _review(90))

    _, _, meta = pipeline.generate(1, "Co", "Role", "src",
                                   max_attempts=1, use_external_critic=False)
    assert meta["accepted"] is True
    assert meta["polish_rounds"] == 0
