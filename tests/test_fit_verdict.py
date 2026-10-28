# tests/test_fit_verdict.py — fit-verdict backend selection: verdict_for's llm kwarg,
# the scan's _prepare_fit_scoring gate, and the backfill. All LLM/DB calls are stubbed.
import pytest

from src import fit_verdict, job_search


def test_verdict_for_uses_the_chosen_backend(monkeypatch):
    seen = {}

    def fake_call(prompt, llm="claude"):
        seen["llm"] = llm
        return {"score": 7, "reason": "ok"}
    monkeypatch.setattr(fit_verdict, "call_llm_json", fake_call)
    score, reason = fit_verdict.verdict_for("resume", "Acme", "Dev", "desc",
                                            llm="local_qwen")
    assert score == 7 and reason == "ok"
    assert seen["llm"] == "local_qwen"


def test_verdict_for_defaults_to_chatgpt(monkeypatch):
    seen = {}

    def fake_call(prompt, llm="claude"):
        seen["llm"] = llm
        return {"score": 5, "reason": "meh"}
    monkeypatch.setattr(fit_verdict, "call_llm_json", fake_call)
    fit_verdict.verdict_for("resume", "Acme", "Dev", "desc")
    assert seen["llm"] == "chatgpt"


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_scan_db(monkeypatch):
    monkeypatch.setattr(job_search.base_resume, "resume_source_text",
                        lambda uid: "resume text")
    monkeypatch.setattr(job_search.db, "connect", lambda: _FakeConn())
    monkeypatch.setattr(job_search.db, "fetch_verdicted_ids", lambda conn, uid: set())


def test_prepare_fit_scoring_local_backend_needs_no_key(monkeypatch):
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _stub_scan_db(monkeypatch)
    scoring = job_search._prepare_fit_scoring(1, fit_llm="local_qwen")
    assert scoring is not None
    assert scoring["llm"] == "local_qwen"


def test_prepare_fit_scoring_chatgpt_without_key_skips(monkeypatch):
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _stub_scan_db(monkeypatch)
    assert job_search._prepare_fit_scoring(1, fit_llm="chatgpt") is None


def test_backfill_reembeds_scored_stale_rows_without_rescoring(monkeypatch):
    monkeypatch.setattr(fit_verdict.embeddings, "available", lambda: True)
    monkeypatch.setattr(fit_verdict.embeddings, "cached_base_resume_embedding",
                        lambda uid: [1.0])
    monkeypatch.setattr(fit_verdict.embeddings, "active_model", lambda: "ollama:test")
    monkeypatch.setattr(fit_verdict.embeddings, "embed_texts", lambda texts: [[0.5]])
    monkeypatch.setattr(fit_verdict.base_resume, "resume_source_text", lambda uid: "resume")
    monkeypatch.setattr(fit_verdict.db, "connect", lambda: _FakeConn())
    stale = [{"job_id": "t:a:1", "company": "A", "role": "R", "job_url": "u",
              "ai_fit_score": 8, "description": "text"}]
    monkeypatch.setattr(fit_verdict.db, "fetch_top_unverdicted", lambda *a, **k: [])
    monkeypatch.setattr(fit_verdict.db, "fetch_unembedded_for_model", lambda *a: stale)
    embedded, verdicts = [], []
    monkeypatch.setattr(fit_verdict.db, "upsert_job_embedding",
                        lambda conn, jid, vec, model: embedded.append((jid, model)))
    monkeypatch.setattr(fit_verdict, "verdict_for",
                        lambda *a, **k: verdicts.append(a) or (5, "x"))

    scored = fit_verdict.run_stored_verdicts(1)

    assert embedded == [("t:a:1", "ollama:test")]    # stale vector refreshed…
    assert verdicts == [] and scored == 0            # …but the existing verdict kept


def test_prepare_fit_scoring_reads_db_profile(monkeypatch):
    """Regression: the scan sources the base résumé from the DB profile (not a .docx)."""
    _stub_scan_db(monkeypatch)
    scoring = job_search._prepare_fit_scoring(1, fit_llm="local_qwen")
    assert scoring is not None
    assert scoring["resume_text"] == "resume text"
