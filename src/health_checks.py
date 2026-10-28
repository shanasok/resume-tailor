# src/health_checks.py — probes the local Ollama server the app depends on.
#
# Each "check" is one HTTP call to a metadata endpoint (/api/tags, /api/ps,
# /api/show), run SERVER-SIDE with `requests` using the same base URL + auth
# header the tailoring pipeline uses — so a check exercises the real call path,
# not a browser fetch straight to Ollama. The Flask routes in app.py expose
# list_checks() / run_check(); no HTML or Flask lives here.
#
# The registry is EXTENSIBLE: two static checks (tags, ps) plus one `show:<tag>`
# check per LIVE discovered Ollama model. Adding a future probe (DB, Claude CLI,
# ChatGPT) is a one-entry change.

import time

import requests

from src.tailoring import LOCAL_LLM_KEY, local_base_url, list_local_models

# Metadata endpoints are quick; a modest timeout still surfaces a hung server.
_TIMEOUT_SEC = 30

# The /api/chat optional-parameter catalog (per https://docs.ollama.com/api/chat).
# Required params (model, messages) are handled specially by the template's required
# block and are intentionally NOT listed here. `group` decides which collapsible the
# field renders in AND whether the client nests it under the `options` object.
CHAT_PARAMS = [
    {"name": "stream", "type": "boolean", "group": "top",
     "desc": "Stream tokens as they generate (default true)."},
    {"name": "think", "type": "bool | string", "group": "top",
     "desc": "Return separate thinking output; high | medium | low."},
    {"name": "format", "type": "string | object", "group": "top",
     "desc": 'Response format: "json" or a JSON schema object.'},
    {"name": "keep_alive", "type": "string | number", "group": "top",
     "desc": "How long to keep the model loaded, e.g. 5m or 0."},
    {"name": "tools", "type": "array", "group": "top",
     "desc": "Function tools the model may call (JSON)."},
    {"name": "logprobs", "type": "boolean", "group": "top",
     "desc": "Include log-probabilities of output tokens."},
    {"name": "top_logprobs", "type": "integer", "group": "top",
     "desc": "How many top tokens to return per position."},
    {"name": "temperature", "type": "number", "group": "options",
     "desc": "Randomness: higher = more random."},
    {"name": "num_ctx", "type": "integer", "group": "options",
     "desc": "Context length in tokens."},
    {"name": "num_predict", "type": "integer", "group": "options",
     "desc": "Max tokens to generate."},
    {"name": "seed", "type": "integer", "group": "options",
     "desc": "Seed for reproducible outputs."},
    {"name": "top_k", "type": "integer", "group": "options",
     "desc": "Limit sampling to the K most likely tokens."},
    {"name": "top_p", "type": "number", "group": "options",
     "desc": "Nucleus-sampling cumulative-probability threshold."},
    {"name": "min_p", "type": "number", "group": "options",
     "desc": "Minimum probability threshold for a token."},
    {"name": "stop", "type": "string | array", "group": "options",
     "desc": "Stop sequence(s) that halt generation."},
]


def chat_models() -> list[str]:
    """The discovered Ollama model tags, for the chat card's model dropdown. Empty when
    the server is unreachable."""
    try:
        return list_local_models()
    except Exception:
        return []


def list_checks() -> list[dict]:
    """The ordered check definitions. `body` is None for GETs; for POSTs it's the
    JSON payload to send. The `show:<tag>` checks are derived from the LIVE
    discovered models so the set tracks whatever's actually on the Ollama server.
    NOTE: this issues a live /api/tags discovery fetch on every call (not free) —
    it degrades to just the two static probes if the server is unreachable."""
    base = local_base_url()
    checks = [
        {"id": "tags", "label": "list models — /api/tags",
         "method": "GET", "url": f"{base}/api/tags", "body": None},
        {"id": "ps", "label": "running models — /api/ps",
         "method": "GET", "url": f"{base}/api/ps", "body": None},
    ]
    try:
        tags = list_local_models()
    except Exception:
        tags = []                       # server down → just the two static probes, no crash
    for tag in tags:
        checks.append({
            "id": f"show:{tag}",
            "label": f"show — {tag}",
            "method": "POST",
            "url": f"{base}/api/show",
            "body": {"model": tag},
        })
    return checks


def _find_check(check_id: str) -> dict | None:
    """Return the check definition with this id, or None. The registry is rebuilt
    from `list_checks()` on each call ON PURPOSE — it's tiny (a handful of entries)
    and derived from the live discovered models, which can change between calls, so
    caching it would stale the `show:*` entries. Indexed by id for a clean lookup."""
    return {check["id"]: check for check in list_checks()}.get(check_id)


def _execute(result: dict, do_request) -> dict:
    """Run `do_request()` (a zero-arg callable returning a requests.Response),
    time it, and fold the outcome into `result`. Shared by run_check and run_chat
    so both obey one contract: a 2xx status => ok=True; a non-2xx keeps the real
    status_code AND body; a transport failure (RequestException) => error set,
    status_code stays None (no HTTP response happened). JSON body is parsed; a
    non-JSON body (e.g. stream:true's concatenated chunks) falls back to text."""
    start = time.monotonic()
    try:
        resp = do_request()
    except requests.RequestException as e:
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000)
        result["error"] = str(e)
        return result
    result["elapsed_ms"] = round((time.monotonic() - start) * 1000)
    result["status_code"] = resp.status_code
    result["ok"] = 200 <= resp.status_code < 300
    try:
        result["response"] = resp.json()          # parsed JSON when the body is JSON
    except ValueError:
        result["response"] = resp.text            # else raw text
    return result


def run_check(check_id: str) -> dict:
    """Run ONE check by id and return a result dict:

        {id, label, method, url, ok, status_code, response, error, elapsed_ms}

    `ok` is True only for a 2xx HTTP response. A transport failure (server down,
    timeout, DNS) → ok=False, status_code=None, error set (no HTTP happened).
    A non-2xx response keeps the real status_code AND the body. Unknown id →
    raises KeyError so the route can 404."""
    check = _find_check(check_id)
    if check is None:
        raise KeyError(check_id)

    result = {
        "id": check["id"], "label": check["label"], "method": check["method"],
        "url": check["url"], "ok": False, "status_code": None,
        "response": None, "error": None, "elapsed_ms": None,
    }
    headers = {"Authorization": f"Bearer {LOCAL_LLM_KEY}"}
    if check["method"] == "POST":
        do = lambda: requests.post(check["url"], headers=headers,
                                   json=check["body"], timeout=_TIMEOUT_SEC)
    else:
        do = lambda: requests.get(check["url"], headers=headers, timeout=_TIMEOUT_SEC)
    return _execute(result, do)


def run_chat(body: dict, timeout: int) -> dict:
    """POST a user-composed chat request to {base}/api/chat with the pipeline's
    auth header, synchronously, honoring `timeout` (seconds). Returns the standard
    result dict (id='chat'). Not part of list_checks() — it's driven by its own
    interactive card, not the Test-all batch."""
    url = f"{local_base_url()}/api/chat"
    result = {
        "id": "chat", "label": "chat — /api/chat", "method": "POST",
        "url": url, "ok": False, "status_code": None,
        "response": None, "error": None, "elapsed_ms": None,
    }
    headers = {"Authorization": f"Bearer {LOCAL_LLM_KEY}"}
    return _execute(result, lambda: requests.post(url, headers=headers,
                                                  json=body, timeout=timeout))
