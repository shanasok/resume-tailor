# tests/test_embeddings.py — the OpenAI embeddings client, with requests.post FAKED
# (no network, no key spent): batching, order preservation, truncation, and the
# clear-error paths. Plus the pure-math cosine().
import pytest
import requests

from src import embeddings


@pytest.fixture(autouse=True)
def _default_backend(monkeypatch):
    """Pin the default (openai) backend so a developer's .env can't flip these tests."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("LOCAL_EMBEDDING_MODEL", raising=False)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("CHATGPT_API_KEY", "test-key")


def _install_fake_post(monkeypatch, calls, item_for=None):
    """Fake the embeddings endpoint. Records each request body in `calls` and answers
    with one vector per input — returned OUT of order, to prove we re-sort by index."""
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        data = [{"index": i,
                 "embedding": (item_for(text) if item_for else [float(i), 1.0])}
                for i, text in enumerate(json["input"])]
        return _FakeResponse({"data": list(reversed(data))})
    monkeypatch.setattr(embeddings.requests, "post", fake_post)


def test_embed_texts_batches_and_preserves_order(fake_key, monkeypatch):
    calls = []
    _install_fake_post(monkeypatch, calls)
    texts = [f"text {i}" for i in range(250)]        # > 2 full batches of 100
    vectors = embeddings.embed_texts(texts)
    assert [len(c["input"]) for c in calls] == [100, 100, 50]
    assert len(vectors) == 250
    # Input order survives the per-request index sort: vector i is for input i.
    assert vectors[0][0] == 0.0 and vectors[99][0] == 99.0 and vectors[100][0] == 0.0


def test_embed_texts_truncates_and_fixes_empty_inputs(fake_key, monkeypatch):
    calls = []
    _install_fake_post(monkeypatch, calls)
    embeddings.embed_texts(["x" * 50_000, ""])       # way over the cap + an empty string
    sent = calls[0]["input"]
    assert len(sent[0]) == embeddings._MAX_CHARS     # truncated
    assert sent[1] == " "                            # the API rejects "" — padded
    assert calls[0]["model"] == embeddings.EMBEDDING_MODEL
    assert calls[0]["dimensions"] == embeddings.EMBEDDING_DIM


def test_embed_texts_without_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(embeddings.EmbeddingsError, match="No OpenAI key"):
        embeddings.embed_texts(["hello"])


def test_embed_texts_wraps_http_failures(fake_key, monkeypatch):
    def exploding_post(*a, **k):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(embeddings.requests, "post", exploding_post)
    with pytest.raises(embeddings.EmbeddingsError, match="request failed"):
        embeddings.embed_texts(["hello"])


def test_embed_texts_rejects_wrong_count(fake_key, monkeypatch):
    monkeypatch.setattr(embeddings.requests, "post",
                        lambda *a, **k: _FakeResponse({"data": []}))   # 0 vectors for 1 input
    with pytest.raises(embeddings.EmbeddingsError):
        embeddings.embed_texts(["hello"])


def test_cosine():
    assert embeddings.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert embeddings.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert embeddings.cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(0.7071, abs=1e-3)
    assert embeddings.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0   # zero-vector guard


# ── local (Ollama) backend ───────────────────────────────────────────────────────────

@pytest.fixture
def local_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _install_fake_ollama(monkeypatch, calls):
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        return _FakeResponse({"embeddings": [[0.1, 0.2] for _ in json["input"]]})
    monkeypatch.setattr(embeddings.requests, "post", fake_post)


def test_available_truth_table(monkeypatch):
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert embeddings.available() is False           # openai default, no key
    monkeypatch.setenv("CHATGPT_API_KEY", "k")
    assert embeddings.available() is True            # openai + key
    monkeypatch.delenv("CHATGPT_API_KEY")
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    assert embeddings.available() is True            # local needs no key


def test_active_model_per_backend(monkeypatch):
    assert embeddings.active_model() == embeddings.EMBEDDING_MODEL
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
    assert embeddings.active_model() == "ollama:nomic-embed-text"


def test_local_backend_embeds_without_a_key(local_backend, monkeypatch):
    calls = []
    _install_fake_ollama(monkeypatch, calls)
    vectors = embeddings.embed_texts(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert calls[0]["url"].endswith("/api/embed")
    assert calls[0]["json"]["model"] == "nomic-embed-text"
    assert "dimensions" not in calls[0]["json"]      # openai-only request field


def test_local_backend_batches_conservatively(local_backend, monkeypatch):
    calls = []
    _install_fake_ollama(monkeypatch, calls)
    embeddings.embed_texts([f"text {i}" for i in range(20)])
    assert [len(c["json"]["input"]) for c in calls] == [16, 4]


def test_local_backend_wraps_failures(local_backend, monkeypatch):
    def exploding_post(*a, **k):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(embeddings.requests, "post", exploding_post)
    with pytest.raises(embeddings.EmbeddingsError, match="request failed"):
        embeddings.embed_texts(["hello"])


def test_local_backend_rejects_wrong_count(local_backend, monkeypatch):
    monkeypatch.setattr(embeddings.requests, "post",
                        lambda *a, **k: _FakeResponse({"embeddings": []}))
    with pytest.raises(embeddings.EmbeddingsError):
        embeddings.embed_texts(["hello"])


def test_base_resume_embedding_reads_db_profile(monkeypatch):
    """base_resume_embedding sources its text from the per-user DB profile."""
    from src import base_resume
    calls = {}

    def fake_source(uid):
        calls["uid"] = uid
        return "PROFILE TEXT"

    monkeypatch.setattr(base_resume, "resume_source_text", fake_source)
    captured = {}

    def fake_get_profile_embedding(user_id, name, text):
        captured.update(user_id=user_id, name=name, text=text)
        return [0.42]

    monkeypatch.setattr(embeddings, "get_profile_embedding", fake_get_profile_embedding)
    vec = embeddings.base_resume_embedding(7)
    assert vec == [0.42]
    assert calls["uid"] == 7
    assert captured["text"] == "PROFILE TEXT"
    assert captured["name"] == embeddings.BASE_RESUME_PROFILE
