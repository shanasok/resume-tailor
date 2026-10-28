# tests/test_health_checks.py — the local-LLM health-check registry + runner.
import pytest

from src import health_checks


class _FakeResp:
    """Minimal stand-in for a requests.Response."""
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def test_list_checks_derives_show_from_discovered_models(monkeypatch):
    monkeypatch.setattr(health_checks, "list_local_models", lambda: ["qwen3.6:27b", "phi4:latest"])
    ids = [c["id"] for c in health_checks.list_checks()]
    assert ids == ["tags", "ps", "show:qwen3.6:27b", "show:phi4:latest"]


def test_show_check_body_uses_tag(monkeypatch):
    monkeypatch.setattr(health_checks, "list_local_models", lambda: ["qwen3.6:27b"])
    show = next(c for c in health_checks.list_checks() if c["id"] == "show:qwen3.6:27b")
    assert show["body"] == {"model": "qwen3.6:27b"}
    assert show["method"] == "POST"


def test_list_checks_degrades_when_server_unreachable(monkeypatch):
    import requests
    def boom():
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(health_checks, "list_local_models", boom)
    ids = [c["id"] for c in health_checks.list_checks()]
    assert ids == ["tags", "ps"]           # no per-model rows, no crash


def test_tags_check_is_get_no_body():
    tags = next(c for c in health_checks.list_checks() if c["id"] == "tags")
    assert tags["method"] == "GET"
    assert tags["body"] is None
    assert tags["url"].endswith("/api/tags")


def test_run_check_success_2xx(monkeypatch):
    monkeypatch.setattr(health_checks.requests, "get",
                        lambda *a, **k: _FakeResp(200, {"models": []}))
    result = health_checks.run_check("tags")
    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["response"] == {"models": []}
    assert result["error"] is None
    assert isinstance(result["elapsed_ms"], int)


def test_run_check_non_2xx_preserves_code_and_body(monkeypatch):
    monkeypatch.setattr(health_checks, "list_local_models", lambda: ["qwen3.6:27b"])
    monkeypatch.setattr(health_checks.requests, "post",
                        lambda *a, **k: _FakeResp(404, {"error": "model not found"}))
    result = health_checks.run_check("show:qwen3.6:27b")
    assert result["ok"] is False
    assert result["status_code"] == 404
    assert result["response"] == {"error": "model not found"}


def test_run_check_transport_failure(monkeypatch):
    def boom(*a, **k):
        raise health_checks.requests.RequestException("connection refused")
    monkeypatch.setattr(health_checks.requests, "get", boom)
    result = health_checks.run_check("ps")
    assert result["ok"] is False
    assert result["status_code"] is None
    assert "connection refused" in result["error"]


def test_run_check_sends_auth_header_and_model_body(monkeypatch):
    captured = {}
    monkeypatch.setattr(health_checks, "list_local_models", lambda: ["phi4:latest"])

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(health_checks.requests, "post", fake_post)
    health_checks.run_check("show:phi4:latest")
    assert captured["url"].endswith("/api/show")
    assert captured["json"] == {"model": "phi4:latest"}
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_run_check_get_sends_auth_header(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured.update(url=url, headers=headers)
        return _FakeResp(200, {"models": []})

    monkeypatch.setattr(health_checks.requests, "get", fake_get)
    health_checks.run_check("tags")
    assert captured["url"].endswith("/api/tags")
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_run_check_non_json_body_returns_text(monkeypatch):
    monkeypatch.setattr(health_checks.requests, "get",
                        lambda *a, **k: _FakeResp(200, payload=None, text="plain not json"))
    result = health_checks.run_check("tags")
    assert result["response"] == "plain not json"


def test_run_check_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        health_checks.run_check("does-not-exist")


def test_chat_params_shape():
    names = [p["name"] for p in health_checks.CHAT_PARAMS]
    # required params are NOT in the optional catalog
    assert "model" not in names and "messages" not in names
    # a representative top-level and options param are present
    assert "stream" in names
    assert "temperature" in names
    for p in health_checks.CHAT_PARAMS:
        assert set(p) == {"name", "type", "group", "desc"}
        assert p["group"] in ("top", "options")
        assert p["desc"]  # non-empty description


def test_chat_models_returns_discovered(monkeypatch):
    monkeypatch.setattr(health_checks, "list_local_models", lambda: ["a:1b", "b:2b"])
    assert health_checks.chat_models() == ["a:1b", "b:2b"]


def test_run_chat_success(monkeypatch):
    monkeypatch.setattr(health_checks.requests, "post",
                        lambda *a, **k: _FakeResp(200, {"message": {"content": "hi"}}))
    result = health_checks.run_chat({"model": "m", "messages": []}, timeout=42)
    assert result["id"] == "chat"
    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["response"] == {"message": {"content": "hi"}}
    assert result["url"].endswith("/api/chat")
    assert isinstance(result["elapsed_ms"], int)


def test_run_chat_forwards_body_auth_and_timeout(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _FakeResp(200, {"ok": True})

    monkeypatch.setattr(health_checks.requests, "post", fake_post)
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hey"}],
            "options": {"temperature": 0.3}}
    health_checks.run_chat(body, timeout=77)
    assert captured["url"].endswith("/api/chat")
    assert captured["json"] == body
    assert captured["timeout"] == 77
    assert captured["headers"]["Authorization"].startswith("Bearer ")


def test_run_chat_non_2xx_preserves_code_and_body(monkeypatch):
    monkeypatch.setattr(health_checks.requests, "post",
                        lambda *a, **k: _FakeResp(500, {"error": "boom"}))
    result = health_checks.run_chat({"model": "m", "messages": []}, timeout=5)
    assert result["ok"] is False
    assert result["status_code"] == 500
    assert result["response"] == {"error": "boom"}


def test_run_chat_transport_failure(monkeypatch):
    def boom(*a, **k):
        raise health_checks.requests.RequestException("connection refused")
    monkeypatch.setattr(health_checks.requests, "post", boom)
    result = health_checks.run_chat({"model": "m", "messages": []}, timeout=5)
    assert result["ok"] is False
    assert result["status_code"] is None
    assert "connection refused" in result["error"]


def test_run_chat_non_json_returns_text(monkeypatch):
    # stream:true returns multiple concatenated JSON chunks — resp.json() raises,
    # so we fall back to the raw text.
    monkeypatch.setattr(health_checks.requests, "post",
                        lambda *a, **k: _FakeResp(200, payload=None,
                                                  text='{"a":1}\n{"a":2}'))
    result = health_checks.run_chat({"model": "m", "messages": []}, timeout=5)
    assert result["response"] == '{"a":1}\n{"a":2}'
