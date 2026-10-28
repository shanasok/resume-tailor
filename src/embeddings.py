# src/embeddings.py — turn text into "meaning fingerprints" (embeddings) via OpenAI or a
# local Ollama model.
#
# An embedding is a list of 1536 numbers that captures what a text MEANS, not which
# exact words it uses. Two texts about the same kind of thing get nearby vectors, so
# "how similar are these?" becomes simple math (cosine similarity). This module powers:
#   • the title "meaning net" during a job scan — a title with zero keyword hits still
#     qualifies when its meaning is close to the target-roles profile;
#   • the Match % column — every stored job's description compared to the base résumé.
#
# Uses the same CHATGPT_API_KEY / OPENAI_API_KEY as the external reviewer (see
# tailoring.chatgpt_key) and the same requests-based call style — no OpenAI SDK.
# When no key is configured everything degrades gracefully: callers check available()
# and simply skip the semantic features.

import hashlib                                       # hash the résumé text to detect changes
import os                                            # env overrides
import time                                          # backoff between rate-limit retries

import requests                                      # HTTP to the OpenAI embeddings endpoint

from src import db                                   # cached profile vectors live in Postgres
from src import filter_config                        # the editable "target roles" paragraph
from src.tailoring import chatgpt_key, local_base_url # shared OpenAI credential + Ollama root

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
# text-embedding-3-small: ~$0.02 per MILLION tokens. EMBEDDING_DIM pins the OpenAI
# request's `dimensions`; the DB columns are dimensionless since migration v10 —
# each vector's `model` column says what produced it, and reads filter on that.
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Defensive input limits. The API caps one input at 8,191 tokens (~32k characters) and
# one REQUEST at ~300k tokens total, so long texts are truncated and requests are
# flushed once they hold enough characters. chars/4 ≈ tokens is a safe rough estimate.
_MAX_CHARS = 24_000                                  # per input (~6k tokens)
_MAX_BATCH_INPUTS = 100                              # per request
_MAX_BATCH_CHARS = 600_000                           # per request (~150k tokens)

# Local batches are deliberately small: an Ollama embed call is one GPU pass per input,
# and a modest box handles 16 short texts far more predictably than 100 long ones.
_MAX_BATCH_INPUTS_LOCAL = 16
_MAX_BATCH_CHARS_LOCAL = 100_000

# The two cached profile vectors (rows in the profile_embeddings table).
BASE_RESUME_PROFILE = "base_resume"
TARGET_ROLES_PROFILE = "target_roles"


class EmbeddingsError(RuntimeError):
    """Anything that stops us getting embeddings (no key, HTTP failure, bad response).
    Callers catch THIS ONE type, log, and continue without semantic features."""


def backend() -> str:
    """Which embeddings backend is active: "openai" (default) or "local" (Ollama).
    Read from the environment at CALL time (like database_url) so .env load order
    and tests can't bite. This is GLOBAL, not per-user — job embeddings are shared."""
    return os.environ.get("EMBEDDING_BACKEND", "openai").strip().lower()


def local_model() -> str:
    """The Ollama embedding model tag used when EMBEDDING_BACKEND=local."""
    return os.environ.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text").strip()


def active_model() -> str:
    """The identity string stored in the `model` column with every vector. Vectors from
    different models are mathematically incomparable (and differently sized), so every
    read filters on this."""
    return f"ollama:{local_model()}" if backend() == "local" else EMBEDDING_MODEL


def available() -> bool:
    """True when the active backend can embed: local needs no key; openai needs one."""
    return backend() == "local" or bool(chatgpt_key())


def embed_texts(texts: list[str], timeout: int = 60) -> list[list[float]]:
    """Embed every text, preserving order, on the active backend (openai | local).
    Batches large lists into as few requests as the backend's size limits allow.
    Raises EmbeddingsError on any failure."""
    # Both backends reject/degenerate on empty strings; truncation keeps any input
    # under the token cap.
    cleaned = [(t or " ").strip()[:_MAX_CHARS] or " " for t in texts]
    if backend() == "local":
        def embed_batch(batch):
            return _embed_batch_local(batch, timeout)
        max_inputs, max_chars = _MAX_BATCH_INPUTS_LOCAL, _MAX_BATCH_CHARS_LOCAL
    else:
        key = chatgpt_key()
        if not key:
            raise EmbeddingsError("No OpenAI key configured (set CHATGPT_API_KEY or "
                                  "OPENAI_API_KEY in .env, or EMBEDDING_BACKEND=local).")

        def embed_batch(batch):
            return _embed_batch(batch, key, timeout)
        max_inputs, max_chars = _MAX_BATCH_INPUTS, _MAX_BATCH_CHARS
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in cleaned:
        if batch and (len(batch) >= max_inputs or batch_chars + len(text) > max_chars):
            vectors.extend(embed_batch(batch))
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        vectors.extend(embed_batch(batch))
    return vectors


def embed_text(text: str, timeout: int = 60) -> list[float]:
    """Embed one text."""
    return embed_texts([text], timeout=timeout)[0]


def _embed_batch(batch: list[str], key: str, timeout: int) -> list[list[float]]:
    """One POST to the embeddings endpoint. Retries once on a 429 (rate limit)."""
    body = {"model": EMBEDDING_MODEL, "input": batch, "dimensions": EMBEDDING_DIM}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in (1, 2):
        try:
            resp = requests.post(OPENAI_EMBEDDINGS_URL, headers=headers, json=body,
                                 timeout=timeout)
            if resp.status_code == 429 and attempt == 1:
                time.sleep(2)                        # brief backoff, then one retry
                continue
            resp.raise_for_status()
            data = resp.json()["data"]
            if len(data) != len(batch):
                raise EmbeddingsError(
                    f"OpenAI returned {len(data)} embeddings for {len(batch)} inputs.")
            # The API may return items out of order; "index" says which input each is for.
            ordered = sorted(data, key=lambda item: item["index"])
            return [item["embedding"] for item in ordered]
        except requests.RequestException as e:
            raise EmbeddingsError(f"OpenAI embeddings request failed: {e}") from e
        except (KeyError, ValueError, IndexError, TypeError) as e:
            raise EmbeddingsError(f"OpenAI embeddings response had an unexpected shape: {e}") from e
    raise EmbeddingsError("OpenAI embeddings request kept hitting the rate limit.")


def _embed_batch_local(batch: list[str], timeout: int) -> list[list[float]]:
    """One POST to Ollama's /api/embed (accepts a list input; vectors come back in
    order). Failures raise EmbeddingsError so every caller's catch-log-continue
    path works unchanged."""
    url = f"{local_base_url()}/api/embed"
    try:
        resp = requests.post(url, json={"model": local_model(), "input": batch},
                             timeout=timeout)
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]
        if len(vectors) != len(batch):
            raise EmbeddingsError(
                f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs.")
        return vectors
    except requests.RequestException as e:
        raise EmbeddingsError(f"Local embeddings request failed ({url}, "
                              f"{local_model()}): {e}") from e
    except (KeyError, ValueError, TypeError) as e:
        raise EmbeddingsError(
            f"Local embeddings response had an unexpected shape: {e}") from e


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors: 1.0 = same meaning, 0 = unrelated.
    (OpenAI vectors are unit-length, so this is essentially a dot product — the
    norms are computed anyway as a guard against a zero vector.)"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Cached profile vectors (base résumé + target-roles paragraph) ──────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_profile_embedding(user_id: int, name: str, text: str) -> list[float]:
    """The embedding for one user's named profile text, cached per user in Postgres.
    Re-embeds ONLY when the text changed (its hash differs from the cached one). When
    THAT user's BASE RÉSUMÉ changes, their stored AI fit verdicts are cleared too — those
    verdicts were judged against the old résumé, and the next scan re-scores the top."""
    content_hash = _sha256(text)
    with db.connect() as conn:
        cached = db.fetch_profile_embedding(conn, user_id, name)
        if cached and cached["content_hash"] == content_hash and cached["model"] == active_model():
            return cached["embedding"]
    vector = embed_text(text)                        # outside the connection: slow network call
    with db.connect() as conn:
        if name == BASE_RESUME_PROFILE and cached and cached["content_hash"] != content_hash:
            cleared = db.clear_fit_verdicts(conn, user_id)
            if cleared:
                print(f"[embeddings] Base résumé changed — cleared {cleared} stale AI fit "
                      "verdict(s); the next scan re-scores the top matches.")
        db.upsert_profile_embedding(conn, user_id, name, content_hash, vector, active_model())
    return vector


def base_resume_embedding(user_id: int) -> list[float]:
    """The (cached) embedding of the user's base résumé text. May call OpenAI."""
    from src import base_resume                       # local import avoids an import cycle
    return get_profile_embedding(user_id, BASE_RESUME_PROFILE,
                                 base_resume.resume_source_text(user_id))


def target_roles_embedding(user_id: int) -> list[float]:
    """The (cached) embedding of the user's editable target-roles paragraph (Settings)."""
    text = filter_config.load(user_id).get("target_roles_text", "").strip()
    if not text:
        raise EmbeddingsError("The 'target roles' description is empty — fill it in on "
                              "the Settings page to enable the title meaning net.")
    return get_profile_embedding(user_id, TARGET_ROLES_PROFILE, text)


def cached_base_resume_embedding(user_id: int) -> list[float] | None:
    """The user's stored base-résumé vector WITHOUT any embedding call (for page loads).
    None when it was never computed OR was computed by a different model/backend —
    a stale-model vector has the wrong dimension and a different space, so callers
    must recompute or skip ranking rather than compare against it."""
    with db.connect() as conn:
        cached = db.fetch_profile_embedding(conn, user_id, BASE_RESUME_PROFILE)
    if not cached or cached["model"] != active_model():
        return None
    return cached["embedding"]
