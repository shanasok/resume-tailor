# tests/test_llm_selection.py — per-stage LLM backend validation, labels, and dropdown markup
# accept the 3 named local models.
import pytest

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with resume paths pointed at a temp folder and the tracker read
    stubbed out — matches tests/test_upload_route.py's pattern so no database is needed."""
    monkeypatch.setattr(app_module, "read_applications", lambda uid: [])
    monkeypatch.setattr(app_module, "available_local_models", lambda: [])
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_normalize_llm_accepts_all_three_local_model_keys():
    assert app_module._normalize_llm("local_qwen") == "local_qwen"
    assert app_module._normalize_llm("local_gptoss") == "local_gptoss"
    assert app_module._normalize_llm("local_mistral") == "local_mistral"


def test_normalize_llm_rejects_old_bare_local_key():
    # The old single "local" key is retired; an unrecognized value falls back to claude.
    assert app_module._normalize_llm("local") == "claude"


def test_normalize_llm_accepts_prefixed_local_tag():
    assert app_module._normalize_llm("local:qwen3.6:27b") == "local:qwen3.6:27b"
    assert app_module._normalize_llm("local:llama3.3:70b") == "local:llama3.3:70b"
    assert app_module._normalize_llm("local:") == "claude"   # empty tag → fallback


def test_normalize_llm_still_accepts_legacy_aliases():
    assert app_module._normalize_llm("local_qwen") == "local_qwen"


def test_llm_label_and_css_for_local():
    assert app_module.llm_label("local:qwen3.6:27b") == "Local — qwen3.6:27b"
    assert app_module.llm_label("local_qwen") == "Local — qwen3.6:27b"
    assert app_module.llm_label("claude") == "Claude"
    assert app_module.llm_label("off") == "off — skipped"
    assert app_module.llm_label(None) == "—"
    assert app_module.llm_css("local:foo:1b") == "llm-local"
    assert app_module.llm_css("local_qwen") == "llm-local"
    assert app_module.llm_css("chatgpt") == "llm-chatgpt"
    assert app_module.llm_css("off") == "llm-none"


def test_index_page_dropdowns_render_discovered_local_models(client, monkeypatch):
    monkeypatch.setattr(app_module, "available_local_models",
                        lambda: ["gpt-oss:20b", "qwen3.6:27b"])
    html = client.get("/").data.decode()
    for stage_id in ("llm-generate", "llm-external", "llm-check", "llm-regenerate",
                     "llm-import", "llm-chat", "llm-fit"):
        start = html.index(f'id="{stage_id}"')
        end = html.index("</select>", start)
        block = html[start:end]
        assert 'value="local:gpt-oss:20b"' in block
        assert 'value="local:qwen3.6:27b"' in block
        assert 'value="local_qwen"' not in block      # nicknames no longer offered


def test_index_page_renders_the_llm_selection_section(client):
    resp = client.get("/")
    html = resp.data.decode()
    assert "LLM selection" in html
    assert "Resume LLMs" not in html                  # old title fully gone
    assert "Resume stage 1 - Generate draft" in html
    assert "Resume stage 2 - External review (fit)" in html
    assert "Resume stage 3 - Truthfulness check" in html
    assert "Resume stage 4 - Regenerate w/ feedback" in html


def test_index_page_default_selection_is_cloud(client, monkeypatch):
    monkeypatch.setattr(app_module, "available_local_models", lambda: ["qwen3.6:27b"])
    html = client.get("/").data.decode()
    # Each select's first <option> is a cloud backend, never a local model.
    for stage_id, first in (("llm-generate", "claude"), ("llm-check", "claude"),
                            ("llm-regenerate", "claude"), ("llm-import", "claude"),
                            ("llm-chat", "claude"), ("llm-external", "chatgpt"),
                            ("llm-fit", "chatgpt")):
        start = html.index(f'id="{stage_id}"')
        first_opt = html.index("<option", start)
        assert f'value="{first}"' in html[first_opt:first_opt + 40]


def test_chat_route_uses_the_posted_llm(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_job",
                        lambda uid, jid: {"Job URL": "", "Company": "A", "Role": "R"})
    seen = {}
    monkeypatch.setattr(app_module, "chat_turn",
                        lambda *a, **k: seen.update(k) or {"reply": "hi"})
    resp = client.post("/chat", json={"job_id": "x", "messages": [], "llm": "local_qwen"})
    assert resp.get_json()["ok"] is True
    assert seen["llm"] == "local_qwen"


def test_chat_turn_reads_db_resume(monkeypatch):
    """chat_turn sources the résumé from the per-user DB profile."""
    from src import chat, base_resume
    monkeypatch.setattr(base_resume, "resume_source_text", lambda uid: f"RESUME-{uid}")
    monkeypatch.setattr(chat, "fetch_job_description", lambda jid, url: ("JD", url))
    seen = {}
    monkeypatch.setattr(chat, "call_llm_json",
                        lambda prompt, llm="claude": seen.update(prompt=prompt) or
                        {"reply": "hi", "addition": "", "company": ""})
    out = chat.chat_turn(9, "jid", "url", "Acme", "Dev", [], llm="claude")
    assert out["reply"] == "hi"
    assert "RESUME-9" in seen["prompt"]


def test_index_page_has_refresh_local_models_button(client):
    html = client.get("/").data.decode()
    assert 'id="refresh-local-models"' in html
    assert 'id="local-models-status"' in html


def test_fit_verdicts_route_passes_the_chosen_llm(client, monkeypatch):
    monkeypatch.setattr(app_module.embeddings, "available", lambda: True)
    seen = {}
    monkeypatch.setattr(app_module.fit_verdict, "run_stored_verdicts",
                        lambda uid, on_progress=None, llm="chatgpt":
                        seen.update(llm=llm) or 1)
    # Run the background work synchronously so the assertion can see it.
    monkeypatch.setattr(app_module, "_start_generation", lambda work: (work(), "tid")[1])
    resp = client.post("/fit-verdicts", data={"llm_fit": "local_qwen"})
    assert resp.get_json()["ok"] is True
    assert seen["llm"] == "local_qwen"
