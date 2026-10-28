# tests/test_local_models_route.py — /local-models proxies Ollama /api/tags via an
# in-process cache and degrades gracefully when the server is unreachable.
import pytest

import app as app_module
from src import tailoring


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "read_applications", lambda uid: [])
    app_module.app.config["TESTING"] = True
    app_module._reset_local_models_cache()          # isolate the module-level cache per test
    return app_module.app.test_client()


def test_local_models_returns_discovered(client, monkeypatch):
    monkeypatch.setattr(tailoring, "list_local_models", lambda timeout=5: ["a:1b", "b:2b"])
    resp = client.get("/local-models")
    assert resp.status_code == 200
    assert resp.get_json() == {"models": ["a:1b", "b:2b"], "error": None}


def test_local_models_degrades_on_error(client, monkeypatch):
    import requests
    def boom(timeout=5):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(tailoring, "list_local_models", boom)
    resp = client.get("/local-models")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["models"] == []
    assert "reach Ollama" in body["error"]


def test_refresh_busts_cache(client, monkeypatch):
    monkeypatch.setattr(tailoring, "list_local_models", lambda timeout=5: ["a:1b"])
    assert client.get("/local-models").get_json()["models"] == ["a:1b"]
    monkeypatch.setattr(tailoring, "list_local_models", lambda timeout=5: ["a:1b", "c:3b"])
    # Without refresh the cache still serves the old list...
    assert client.get("/local-models").get_json()["models"] == ["a:1b"]
    # ...refresh forces a re-fetch.
    assert client.get("/local-models?refresh=1").get_json()["models"] == ["a:1b", "c:3b"]
