# tests/test_health_checks_routes.py — the /health-checks page + runner routes.
import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_page_renders_with_controls_and_rows(client):
    resp = client.get("/health-checks")
    assert resp.status_code == 200
    assert b"Test all" in resp.data
    assert b"Re-test failed" in resp.data
    assert b"tags" in resp.data                 # a check row is present


def test_run_route_returns_runner_json(client, monkeypatch):
    monkeypatch.setattr(app_module.health_checks, "run_check",
                        lambda cid: {"id": cid, "ok": True, "status_code": 200})
    resp = client.get("/health-checks/run/tags")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == "tags"
    assert body["ok"] is True


def test_run_route_passes_colon_id_through(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(app_module.health_checks, "list_local_models", lambda: ["qwen3.6:27b"])
    monkeypatch.setattr(app_module.health_checks, "run_check",
                        lambda cid: seen.setdefault("id", cid) or {"id": cid, "ok": True})
    resp = client.get("/health-checks/run/show:qwen3.6:27b")
    assert resp.status_code == 200
    assert seen["id"] == "show:qwen3.6:27b"


def test_run_route_unknown_id_404(client, monkeypatch):
    def boom(cid):
        raise KeyError(cid)
    monkeypatch.setattr(app_module.health_checks, "run_check", boom)
    resp = client.get("/health-checks/run/nope")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "unknown check"


def test_chat_route_returns_runner_json(client, monkeypatch):
    monkeypatch.setattr(app_module.health_checks, "run_chat",
                        lambda body, timeout: {"id": "chat", "ok": True,
                                               "status_code": 200, "response": body})
    resp = client.post("/health-checks/chat",
                       json={"body": {"model": "m", "messages": []}, "timeout": 60})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == "chat"
    assert body["ok"] is True


def test_chat_route_passes_body_and_timeout(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(app_module.health_checks, "run_chat",
                        lambda body, timeout: seen.update(body=body, timeout=timeout) or {"ok": True})
    client.post("/health-checks/chat",
                json={"body": {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                      "timeout": 90})
    assert seen["body"]["model"] == "qwen"
    assert seen["timeout"] == 90


def test_chat_route_bad_timeout_falls_back_to_default(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(app_module.health_checks, "run_chat",
                        lambda body, timeout: seen.update(timeout=timeout) or {"ok": True})
    # missing timeout
    client.post("/health-checks/chat", json={"body": {"model": "m", "messages": []}})
    assert seen["timeout"] == app_module.HEALTH_CHAT_TIMEOUT
    # non-positive timeout
    client.post("/health-checks/chat",
                json={"body": {"model": "m", "messages": []}, "timeout": 0})
    assert seen["timeout"] == app_module.HEALTH_CHAT_TIMEOUT
    # garbage timeout
    client.post("/health-checks/chat",
                json={"body": {"model": "m", "messages": []}, "timeout": "abc"})
    assert seen["timeout"] == app_module.HEALTH_CHAT_TIMEOUT


def test_page_renders_chat_card(client):
    resp = client.get("/health-checks")
    assert resp.status_code == 200
    assert b"/api/chat" in resp.data          # the chat card heading
    assert b"chat-send" in resp.data          # the Send button
    assert b"temperature" in resp.data        # a catalog param rendered
