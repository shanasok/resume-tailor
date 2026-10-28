# tests/test_tailoring.py — local-model dispatch routes each local_* key to its own Ollama model.
import pytest

from src import tailoring


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def no_real_claude_cli(monkeypatch):
    """Guard rail: fail loudly instead of shelling out to the user's real Claude
    subscription if dispatch ever falls through to the claude-CLI branch."""
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run (claude CLI) should not be called in this test")
    monkeypatch.setattr(tailoring.subprocess, "run", _boom)


@pytest.mark.parametrize("llm_key, expected_model", [
    ("local_qwen", "qwen3.6:27b"),
    ("local_gptoss", "gpt-oss:20b"),
    ("local_mistral", "mistral-small3.1:24b"),
])
def test_call_llm_text_routes_to_correct_local_model(monkeypatch, no_real_claude_cli,
                                                       llm_key, expected_model):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse({"message": {"content": "tailored output"}})

    monkeypatch.setattr(tailoring.requests, "post", fake_post)

    result = tailoring.call_llm_text("some prompt", llm=llm_key)

    assert result == "tailored output"
    assert captured["json"]["model"] == expected_model


def test_legacy_aliases_present():
    assert set(tailoring.LEGACY_LOCAL_ALIASES) == {"local_qwen", "local_gptoss", "local_mistral"}


def test_local_base_url_strips_trailing_slash_and_v1(monkeypatch):
    monkeypatch.setattr(tailoring, "LOCAL_LLM_URL", "http://host:11434/v1/")
    assert tailoring.local_base_url() == "http://host:11434"


def test_local_base_url_plain_root_unchanged(monkeypatch):
    monkeypatch.setattr(tailoring, "LOCAL_LLM_URL", "http://host:11434")
    assert tailoring.local_base_url() == "http://host:11434"


def test_local_native_url_appends_api_chat(monkeypatch):
    monkeypatch.setattr(tailoring, "LOCAL_LLM_URL", "http://host:11434")
    assert tailoring._local_native_url() == "http://host:11434/api/chat"


def test_list_local_models_parses_tags(monkeypatch):
    from src import tailoring

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"models": [{"name": "gpt-oss:20b"}, {"name": "qwen3.6:27b"},
                               {"name": ""}, {"other": "no-name"}]}

    captured = {}
    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        return FakeResp()
    monkeypatch.setattr(tailoring.requests, "get", fake_get)

    models = tailoring.list_local_models()
    assert models == ["gpt-oss:20b", "qwen3.6:27b"]   # sorted; blank/name-less dropped
    assert captured["url"].endswith("/api/tags")


def test_list_local_models_filters_embedding_models(monkeypatch):
    from src import tailoring

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"models": [
                {"name": "gpt-oss:20b", "details": {"family": "gptoss"}},
                {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},  # embed by name+family
                {"name": "mxbai-embed-large:latest"},                                     # embed by name only
                {"name": "mystery:1b", "details": {"family": "bert"}},                     # embed by family only
                {"name": "mistral-small3.1:24b", "details": {"family": "mistral"}},
            ]}
    monkeypatch.setattr(tailoring.requests, "get", lambda url, headers=None, timeout=None: FakeResp())
    # Only the two chat models survive — every embedding model is filtered out.
    assert tailoring.list_local_models() == ["gpt-oss:20b", "mistral-small3.1:24b"]


def test_is_embedding_model():
    from src import tailoring
    # By name (bare tag or "local:" selection), with or without a colon-version suffix.
    assert tailoring.is_embedding_model("nomic-embed-text:latest") is True
    assert tailoring.is_embedding_model("local:nomic-embed-text:latest") is True
    assert tailoring.is_embedding_model("mxbai-embed-large") is True
    assert tailoring.is_embedding_model("all-minilm") is True
    assert tailoring.is_embedding_model("bge-m3") is True
    # By family, when the name alone gives nothing away.
    assert tailoring.is_embedding_model("mystery:1b", family="bert") is True
    # Chat models and the cloud backends are NOT embedding models.
    assert tailoring.is_embedding_model("gpt-oss:20b", family="gptoss") is False
    assert tailoring.is_embedding_model("local:mistral-small3.1:24b") is False
    assert tailoring.is_embedding_model("claude") is False
    assert tailoring.is_embedding_model("chatgpt") is False


def test_list_local_models_propagates_transport_error(monkeypatch):
    from src import tailoring
    import requests as _rq
    def boom(*a, **k):
        raise _rq.ConnectionError("refused")
    monkeypatch.setattr(tailoring.requests, "get", boom)
    with pytest.raises(_rq.RequestException):
        tailoring.list_local_models()


# ── Prefix-based local dispatch (local:<tag>) + legacy alias back-compat ──────────────────
def test_local_tag_resolves_prefix_and_legacy():
    from src import tailoring
    assert tailoring.local_tag("local:qwen3.6:27b") == "qwen3.6:27b"
    assert tailoring.local_tag("local:llama3.3:70b") == "llama3.3:70b"
    assert tailoring.local_tag("local_qwen") == "qwen3.6:27b"   # legacy alias
    assert tailoring.local_tag("local:") is None                # empty tag
    assert tailoring.local_tag("claude") is None
    assert tailoring.local_tag(None) is None


def test_is_local_llm():
    from src import tailoring
    assert tailoring.is_local_llm("local:foo:1b") is True
    assert tailoring.is_local_llm("local_mistral") is True
    assert tailoring.is_local_llm("chatgpt") is False
    assert tailoring.is_local_llm(None) is False


def test_call_llm_text_routes_prefixed_tag_to_ollama(monkeypatch, no_real_claude_cli):
    from src import tailoring
    seen = {}
    monkeypatch.setattr(tailoring, "_call_local_text",
                        lambda prompt, model, timeout, **kw: seen.setdefault("model", model) or "ok")
    tailoring.call_llm_text("p", llm="local:phi4:latest")
    assert seen["model"] == "phi4:latest"


# ── Em-dash stripping (an "AI-written" tell) ──────────────────────────────────────────────
# Models love the "clause — interrupting aside — clause" em-dash rhythm and ignore prompts not
# to, so _normalize strips em-dashes from the model's prose deterministically. Only the em-dash
# (—) is touched; the en-dash (–) used in date ranges must survive, and digit—digit ranges too.

def test_de_emdash_converts_paired_interrupter_to_commas():
    # The exact shape the user flagged: "word — aside (…) — word".
    text = "Managed vendor relationships — including Acme (later acquired by Globex) — and set SLAs"
    assert tailoring._de_emdash(text) == (
        "Managed vendor relationships, including Acme (later acquired by Globex), and set SLAs")


def test_de_emdash_handles_trailing_dangling_dash():
    assert tailoring._de_emdash("Owned vendor governance — including Sana —") == "Owned vendor governance, including Sana"


def test_de_emdash_single_dash_becomes_comma():
    assert tailoring._de_emdash("Grew adoption — a first for the team") == "Grew adoption, a first for the team"


def test_de_emdash_leaves_en_dash_and_plain_hyphen_alone():
    # En-dash date range and a hyphenated compound must be untouched.
    assert tailoring._de_emdash("Full-stack work, March 2019 – June 2020") == "Full-stack work, March 2019 – June 2020"


def test_de_emdash_leaves_numeric_range_alone():
    # A digit—digit em-dash range is left as-is, not mangled into "10, 20".
    assert tailoring._de_emdash("Handled 10—20 vendors") == "Handled 10—20 vendors"


def test_de_emdash_noop_when_no_em_dash():
    assert tailoring._de_emdash("Plain bullet, nothing to do") == "Plain bullet, nothing to do"


def test_normalize_strips_em_dashes_from_summary_competencies_and_bullets():
    data = {
        "summary": "Engineer — hands-on — and pragmatic.",
        "competencies": ["Vendor Management — end to end"],
        "experience": [{"company": "Acme", "title": "Dev", "dates": "2019 – 2024",
                        "bullets": ["Built APIs — including a virtual-card API — at scale"]}],
        "skills": [], "education": [], "certifications": [],
    }
    out = tailoring._normalize(data)
    assert out["summary"] == "Engineer, hands-on, and pragmatic."
    assert out["competencies"] == ["Vendor Management, end to end"]
    assert out["experience"][0]["bullets"] == ["Built APIs, including a virtual-card API, at scale"]
    assert out["experience"][0]["dates"] == "2019 – 2024"    # en-dash date range untouched


# ── Fact-locking ────────────────────────────────────────────────────────────────────────
# The base profile is the source of truth; after the model tailors, its FACTS (name, contact,
# skills, education, certs, and each role's company/title/dates) are overwritten from the base
# so a weak model can't drop or alter them, while the model's WRITING (summary, competencies,
# bullet wording, role ordering) is kept.

def _base_profile():
    """A structured base résumé in tailoring._SCHEMA shape — the source of truth to lock to."""
    return {
        "name": "Jordan Rivera",
        "contact": {"location": "Denver, CO", "phone": "555-1212",
                    "email": "jordan@example.com", "website": "jordan.dev"},
        "summary": "Base summary.",
        "competencies": ["Leadership", "Strategy"],
        "experience": [
            {"company": "Acme Corp", "title": "Director of Engineering",
             "dates": "2018–2024", "location": "Remote",
             "bullets": ["Base bullet A", "Base bullet B"]},
            {"company": "Globex LLC", "title": "Engineering Manager",
             "dates": "2014–2018", "location": "Denver, CO",
             "bullets": ["Base bullet C"]},
        ],
        "education": [{"institution": "State U", "degree": "BS CS",
                       "date": "2010", "detail": ""}],
        "certifications": ["PMP", "AWS SA"],
        "skills": [{"category": "Languages", "items": "Python, Java"},
                   {"category": "Cloud", "items": "AWS, GCP"}],
    }


def test_lock_facts_overwrites_altered_location_name_and_contact():
    # The model moved the candidate's city to match the posting and tweaked the name/email.
    tailored = {
        "name": "Jordan R.",
        "contact": {"location": "San Francisco, CA", "phone": "999", "email": "x@y.z"},
        "summary": "Tailored summary the model wrote.",
        "experience": [], "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())

    assert locked["name"] == "Jordan Rivera"
    assert locked["contact"]["location"] == "Denver, CO"   # city restored
    assert locked["contact"]["email"] == "jordan@example.com"
    # The model's WRITING is untouched.
    assert locked["summary"] == "Tailored summary the model wrote."


def test_lock_facts_restores_dropped_skills_certs_and_education():
    # The model dropped skill groups, a cert, and education entirely.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "skills": [{"category": "Languages", "items": "Python"}],   # one group, lost the rest
        "certifications": [], "education": [], "experience": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())

    assert locked["skills"] == [{"category": "Languages", "items": "Python, Java"},
                                {"category": "Cloud", "items": "AWS, GCP"}]
    assert locked["certifications"] == ["PMP", "AWS SA"]
    assert locked["education"] == [{"institution": "State U", "degree": "BS CS",
                                    "date": "2010", "detail": ""}]


def test_lock_facts_forces_company_title_dates_but_keeps_tailored_bullets():
    # The model reworded the bullets (good) but also changed the title and dates (must revert).
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "Acme Corp", "title": "VP of Engineering",   # inflated title
             "dates": "2016–2024",                                   # altered dates
             "bullets": ["Reworded bullet the model improved"]},
            {"company": "Globex LLC", "title": "Engineering Manager",
             "dates": "2014–2018", "bullets": ["Reworded C"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())
    acme = locked["experience"][0]

    assert acme["title"] == "Director of Engineering"   # reverted
    assert acme["dates"] == "2018–2024"                 # reverted
    assert acme["location"] == "Remote"                 # from base
    assert acme["bullets"] == ["Reworded bullet the model improved"]   # model's wording kept


def test_lock_facts_matches_company_despite_punctuation_drift():
    # "Acme Corp." vs base "Acme Corp" should still lock to the base role's facts.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [{"company": "  ACME  CORP. ", "title": "Wrong Title",
                        "dates": "wrong", "bullets": ["kept bullet"]}],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())
    role = locked["experience"][0]

    assert role["company"] == "Acme Corp"
    assert role["title"] == "Director of Engineering"
    assert role["bullets"] == ["kept bullet"]


def test_lock_facts_preserves_model_role_ordering():
    # The model prioritized Globex first; that ordering (its judgment) must survive.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "Globex LLC", "title": "x", "dates": "y", "bullets": ["gb"]},
            {"company": "Acme Corp", "title": "x", "dates": "y", "bullets": ["ab"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())
    assert [r["company"] for r in locked["experience"]] == ["Globex LLC", "Acme Corp"]


def test_lock_facts_restores_role_the_model_dropped_with_base_bullets():
    # The model dropped Globex entirely — it must be restored (no real job vanishes).
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [{"company": "Acme Corp", "title": "x", "dates": "y",
                        "bullets": ["ab"]}],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())
    companies = [r["company"] for r in locked["experience"]]

    assert companies == ["Acme Corp", "Globex LLC"]        # restored, appended in base order
    globex = locked["experience"][1]
    assert globex["bullets"] == ["Base bullet C"]          # falls back to base bullets


def test_lock_facts_keeps_unmatched_chat_added_role_as_is():
    # A role with no base match (e.g. experience added via chat, not in the profile) is kept.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "Acme Corp", "title": "x", "dates": "y", "bullets": ["ab"]},
            {"company": "Startup XYZ", "title": "Advisor", "dates": "2024",
             "bullets": ["Advised on strategy"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile())
    extra = [r for r in locked["experience"] if r["company"] == "Startup XYZ"]

    assert len(extra) == 1
    assert extra[0]["title"] == "Advisor"                  # kept exactly as the model wrote it


# ── Two roles at the SAME employer (a promotion) ──────────────────────────────────────────
# Regression: the fact-lock once keyed experience by company alone, so a candidate with two
# roles at one employer (e.g. American Express: "Lead API Engineer" then "Program Manager")
# had the second role silently dropped — it was neither matched nor restored. Each base role
# must now survive exactly once, matched to the right tailored role by title/dates.

def _base_profile_two_roles_same_company():
    """A base résumé with a promotion: two American Express roles (older + newer)."""
    p = _base_profile()
    p["experience"] = [
        {"company": "American Express", "title": "Program Manager",
         "dates": "2020–2025", "location": "Denver, CO",
         "bullets": ["Ran the enablement program"]},
        {"company": "American Express", "title": "Lead API Engineer",
         "dates": "2019–2020", "location": "Denver, CO",
         "bullets": ["Built the virtual-card API"]},
    ]
    return p


def test_lock_facts_keeps_both_roles_at_same_company():
    # The model kept both AmEx roles (reworded bullets); BOTH must survive, distinctly.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "American Express", "title": "Program Manager",
             "dates": "2020–2025", "bullets": ["Reworded PM bullet"]},
            {"company": "American Express", "title": "Lead API Engineer",
             "dates": "2019–2020", "bullets": ["Reworded API bullet"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile_two_roles_same_company())
    titles = [r["title"] for r in locked["experience"]]

    assert titles == ["Program Manager", "Lead API Engineer"]     # both kept, matched by title
    assert locked["experience"][0]["bullets"] == ["Reworded PM bullet"]
    assert locked["experience"][1]["bullets"] == ["Reworded API bullet"]


def test_lock_facts_restores_second_same_company_role_the_model_dropped():
    # The model dropped the older AmEx role (the original bug) — it must be restored.
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "American Express", "title": "Program Manager",
             "dates": "2020–2025", "bullets": ["Reworded PM bullet"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile_two_roles_same_company())
    titles = [r["title"] for r in locked["experience"]]

    assert titles == ["Program Manager", "Lead API Engineer"]     # dropped role restored
    api = locked["experience"][1]
    assert api["dates"] == "2019–2020"                            # from base
    assert api["bullets"] == ["Built the virtual-card API"]       # base bullets (model dropped it)


def test_lock_facts_disambiguates_same_company_roles_by_dates_when_titles_drift():
    # The model reworded BOTH titles; dates still line each tailored role up with the right
    # base role (so bullets don't get swapped onto the wrong role).
    tailored = {
        "name": "Jordan Rivera", "contact": {},
        "experience": [
            {"company": "American Express", "title": "Engineering Program Lead",
             "dates": "2020–2025", "bullets": ["newer-role bullet"]},
            {"company": "American Express", "title": "Senior API Developer",
             "dates": "2019–2020", "bullets": ["older-role bullet"]},
        ],
        "skills": [], "education": [], "certifications": [],
    }
    locked = tailoring._lock_facts(tailored, _base_profile_two_roles_same_company())

    newer, older = locked["experience"][0], locked["experience"][1]
    assert newer["title"] == "Program Manager" and newer["bullets"] == ["newer-role bullet"]
    assert older["title"] == "Lead API Engineer" and older["bullets"] == ["older-role bullet"]


def test_lock_facts_no_base_profile_returns_draft_unchanged():
    # A .docx-only user has no structured profile — the draft passes through untouched.
    draft = {"name": "Whatever The Model Said", "contact": {"location": "Anywhere"}}
    assert tailoring._lock_facts(dict(draft), None) == draft
    assert tailoring._lock_facts(dict(draft), {}) == draft


def test_tailor_resume_applies_locking_when_base_profile_given(monkeypatch, no_real_claude_cli):
    # End-to-end: the model returns tampered facts; tailor_resume must lock them to the base.
    tampered = {
        "name": "Wrong Name",
        "contact": {"location": "San Francisco, CA"},
        "summary": "Model summary.",
        "experience": [{"company": "Acme Corp", "title": "CEO", "dates": "2000",
                        "bullets": ["improved bullet"]}],
        "skills": [], "education": [], "certifications": [],
    }
    monkeypatch.setattr(tailoring, "call_llm_json", lambda *a, **k: tampered)

    result = tailoring.tailor_resume("resume text", "job posting", "Co", "Role",
                                     llm="local_qwen", base_profile=_base_profile())

    assert result["name"] == "Jordan Rivera"
    assert result["contact"]["location"] == "Denver, CO"
    assert result["experience"][0]["title"] == "Director of Engineering"
    assert result["experience"][0]["bullets"] == ["improved bullet"]   # writing preserved
    assert result["summary"] == "Model summary."


def test_tailor_resume_without_base_profile_skips_locking(monkeypatch, no_real_claude_cli):
    # No base profile → whatever the model produced is returned (after _normalize only).
    draft = {"name": "Model Name", "contact": {"location": "Anywhere"},
             "experience": [], "skills": []}
    monkeypatch.setattr(tailoring, "call_llm_json", lambda *a, **k: dict(draft))

    result = tailoring.tailor_resume("resume text", "job posting", "Co", "Role",
                                     llm="local_qwen")
    assert result["name"] == "Model Name"
    assert result["contact"]["location"] == "Anywhere"


# ── "claude" backend routing: metered API when keyed, subscription CLI otherwise ─────────
def test_claude_backend_uses_api_when_key_present(monkeypatch):
    """With CLAUDE_CODE_API_KEY set, the 'claude' backend hits the metered API, not the CLI."""
    monkeypatch.setenv("CLAUDE_CODE_API_KEY", "sk-test")
    monkeypatch.setattr(tailoring, "_call_claude_api_json",
                        lambda p, t, system=None, cache_prefix=None: {"via": "api"})
    monkeypatch.setattr(tailoring, "call_claude_json", lambda p, t: {"via": "cli"})
    assert tailoring.call_llm_json("prompt", llm="claude") == {"via": "api"}

    monkeypatch.setattr(tailoring, "_call_claude_api_text",
                        lambda p, t, system=None, cache_prefix=None: "api")
    monkeypatch.setattr(tailoring, "_call_claude_text", lambda p, t: "cli")
    assert tailoring.call_llm_text("prompt", llm="claude") == "api"


def test_claude_backend_falls_back_to_cli_without_key(monkeypatch):
    """No CLAUDE_CODE_API_KEY → the 'claude' backend uses the subscription CLI (unchanged)."""
    monkeypatch.delenv("CLAUDE_CODE_API_KEY", raising=False)
    monkeypatch.setattr(tailoring, "_call_claude_api_json", lambda p, t: {"via": "api"})
    monkeypatch.setattr(tailoring, "call_claude_json", lambda p, t: {"via": "cli"})
    assert tailoring.call_llm_json("prompt", llm="claude") == {"via": "cli"}


def _fake_claude_client(stop_reason, text):
    """A stand-in for the anthropic client whose messages.create returns one text block with the
    given stop_reason — used to exercise _call_claude_api_text's truncation handling."""
    block = type("Block", (), {"type": "text", "text": text})()
    message = type("Msg", (), {"content": [block], "stop_reason": stop_reason})()

    class _Msgs:
        def create(self, **kwargs):
            return message

    class _Client:
        def with_options(self, **kwargs):
            return self

        messages = _Msgs()

    return _Client()


def test_api_text_raises_clear_error_on_max_tokens_truncation(monkeypatch):
    """A reply cut off at the token cap (stop_reason='max_tokens') must raise a clear, RETRYABLE
    ValueError naming CLAUDE_API_MAX_TOKENS — NOT return the partial text that later blows up as a
    cryptic 'Unbalanced braces' in _extract_json. This was the Collectorsuniverse failure mode."""
    monkeypatch.setattr(tailoring, "_claude_client",
                        lambda: _fake_claude_client("max_tokens", '{"overall_score": 8'))
    with pytest.raises(ValueError) as exc:
        tailoring._call_claude_api_text("prompt")
    assert "max_tokens" in str(exc.value)
    assert "CLAUDE_API_MAX_TOKENS" in str(exc.value)


def test_api_text_returns_text_on_normal_stop(monkeypatch):
    """A reply that finished normally (stop_reason='end_turn') returns its text unchanged."""
    monkeypatch.setattr(tailoring, "_claude_client",
                        lambda: _fake_claude_client("end_turn", '{"overall_score": 88}'))
    assert tailoring._call_claude_api_text("prompt") == '{"overall_score": 88}'


def test_claude_client_passes_the_metered_key_explicitly(monkeypatch):
    """The API client must be built with CLAUDE_CODE_API_KEY passed EXPLICITLY — never a bare
    Anthropic() — since the SDK doesn't auto-discover a key under this non-default env var."""
    import anthropic
    monkeypatch.setattr(tailoring, "_claude_api_client", None)          # reset the cached client
    monkeypatch.setenv("CLAUDE_CODE_API_KEY", "sk-metered")
    captured = {}

    class FakeAnthropic:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    tailoring._claude_client()
    assert captured["api_key"] == "sk-metered"
