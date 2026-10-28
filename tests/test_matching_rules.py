# tests/test_matching_rules.py — the wide-net matcher: title sieve, carve-outs,
# workplace (remote/hybrid/onsite) rules, hard drops, and the meaning-net hook.
# Pure Python — no database, no network, no AI calls (the meaning-net test fakes the
# embedding functions with tiny deterministic vectors).
import json
from pathlib import Path

import pytest

from src import matching

# A representative SAMPLE user's matching config (a fictional profile, frozen as a
# fixture) — so these pure-logic tests own their input and need
# neither the database nor any user's personal filters. Filters is what match_job runs on.
_CFG = json.loads((Path(__file__).parent / "fixtures" / "matcher_config.json").read_text())
_F = matching.Filters(_CFG)


# A description that passes every description-level screen (salary + US + remote),
# so tests can isolate the ONE rule they're about.
GOOD_DESC = ("We're hiring! Salary range $190,000 to $230,000 per year. "
             "This remote role is open across the United States.")


def _match(title="Engineering Manager", location="Remote - US", description="", **extra):
    return matching.match_job({"title": title, "location": location,
                               "description": description, **extra}, _F)


# ── The title sieve ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("title,should_pass", [
    # domain keyword alone qualifies
    ("Head of Kubernetes Training", True),
    ("Curriculum Developer", True),
    ("Director of AI Enablement", True),
    ("Machine Learning Engineer", False),         # ML roles are off-field (title_negative_keywords)
    ("Head of Technical Enablement", True),
    # qualifying phrases pass even though their words are generic
    ("Engineering Manager", True),
    ("Senior Engineer", True),
    ("General Manager", True),
    # staff/principal IC engineer levels are ABOVE the target level — excluded,
    # including compound titles that split the phrase ("Staff ML Engineer")
    ("Staff Engineer", False),
    ("Principal Engineer", False),
    ("Staff Software Engineer", False),
    ("Principal Software Engineer", False),
    ("Staff Machine Learning Engineer", False),
    ("Principal Backend Engineer", False),
    # …but staff/principal MANAGER-track and L&D titles are untouched
    ("Staff Engineering Manager", True),
    # generic words alone never qualify
    ("Account Manager", False),
    ("Director of Operations", False),
    ("Director of Finance", False),
    ("Customer Experience Lead", False),
    ("Head of People", False),
])
def test_title_sieve(title, should_pass):
    verdict = _match(title=title)
    assert verdict["qualifies"] is should_pass, verdict["reject_reasons"]


def test_generic_only_title_gets_the_sieve_reject_reason():
    verdict = _match(title="Director of Operations")
    assert verdict["reject_reasons"] == [matching.TITLE_SIEVE_REJECT]


# ── Negative-term carve-outs ─────────────────────────────────────────────────

@pytest.mark.parametrize("title,should_pass", [
    ("Sales Enablement Manager", True),           # 'sales' forgiven by 'enablement'
    ("Sales Training Specialist", True),          # 'sales' forgiven by 'training'
    ("Sales Manager", False),                     # bare 'sales' still drops
    ("Learning Solutions Architect", True),       # 'architect' forgiven by 'learning'
    ("Solutions Architect", False),               # bare 'architect' still drops
])
def test_negative_carveouts(title, should_pass):
    assert _match(title=title)["qualifies"] is should_pass


@pytest.mark.parametrize("title", [
    "Frontend Engineer", "Front End Developer", "Senior Front-End Engineer",
])
def test_front_end_titles_are_dropped(title):
    assert _match(title=title)["qualifies"] is False


# ── Workplace classification + the hybrid commute-zone rule ─────────────────

@pytest.mark.parametrize("location,description,expected", [
    ("Remote - US", "", "remote"),
    ("Denver, CO (Hybrid)", "", "hybrid"),
    ("Onsite - Austin, TX", "", "onsite"),
    ("Hybrid or Remote", "", "hybrid"),           # location: hybrid outranks remote
    ("Austin, TX", "", "unverified"),
    # location silent → the description decides, REMOTE first ("remote or hybrid
    # options" can be done remotely, so it shouldn't be held to the commute rule)
    ("Austin, TX", "You can work remote or hybrid from our Austin office.", "remote"),
    ("Austin, TX", "This is a hybrid schedule: 3 days in office.", "hybrid"),
])
def test_classify_workplace(location, description, expected):
    assert matching.classify_workplace(location, description) == expected


@pytest.mark.parametrize("location,commutable", [
    ("Hybrid - Boulder, CO", True),
    ("Denver, CO (Hybrid)", True),
    ("Aurora, CO - Hybrid", True),
    ("Hybrid - Englewood, CO", True),
    ("Hybrid - Seattle, WA", False),
    ("San Francisco, CA (Hybrid)", False),
])
def test_is_commutable(location, commutable):
    assert matching.is_commutable(location, _F) is commutable


def test_hybrid_rule_end_to_end():
    # Commutable hybrid passes (and is flagged so it's visibly not fully remote)…
    ok = _match(location="Denver, CO (Hybrid)", description=GOOD_DESC)
    assert ok["qualifies"], ok["reject_reasons"]
    assert any("hybrid — commutable" in f for f in ok["flags"])
    # …a far-away hybrid and any on-site role drop.
    far = _match(location="Hybrid - Seattle, WA", description=GOOD_DESC)
    assert "hybrid outside your commute zone" in far["reject_reasons"]
    onsite = _match(location="Onsite - Austin, TX", description=GOOD_DESC)
    assert "on-site role" in onsite["reject_reasons"]


# ── Salary hard drops (unchanged rules, re-pinned here) ──────────────────────

def test_salary_rules_once_description_exists():
    no_pay = _match(description="Great mission, no pay mentioned. Remote, United States.")
    assert "no salary stated in description" in no_pay["reject_reasons"]
    low_pay = _match(description="Pay: $150,000 per year. Remote, United States.")
    assert any(r.startswith("salary tops out at $150,000") for r in low_pay["reject_reasons"])
    good = _match(description=GOOD_DESC)
    assert good["qualifies"], good["reject_reasons"]


def test_range_top_clears_floor_even_when_low_is_below():
    # Fixture floor is 175000. Range 125k–250k: the top clears it → kept.
    desc = ("Compensation: $125,000 to $250,000 per year. "
            "Remote role across the United States.")
    assert _match(description=desc)["qualifies"]


def test_multi_zone_range_screens_on_highest_top():
    desc = ("Base pay varies by zone. Zone A: $120,000–$150,000. "
            "Zone B: $200,000 to $240,000. Remote, United States.")
    assert _match(description=desc)["qualifies"]


def test_range_below_floor_drops_even_with_higher_ote_single():
    # Highest base range only: the $300k OTE single is ignored; top of range is 150k < 175k.
    desc = ("Base salary: $120,000 to $150,000 per year. OTE up to $300,000. "
            "Remote, United States.")
    low = _match(description=desc)
    assert any(r.startswith("salary tops out at $150,000") for r in low["reject_reasons"])


def test_structured_pay_screens_on_salary_max():
    # Structured path: low 120k below floor but max 250k clears it → kept.
    kept = _match(description=GOOD_DESC, salary_min=120000, salary_max=250000)
    assert kept["qualifies"], kept["reject_reasons"]


def test_structured_pay_without_max_falls_back_to_min():
    # No salary_max supplied → max falls back to min (120k) → drops.
    dropped = _match(description=GOOD_DESC, salary_min=120000)
    assert any(r.startswith("salary tops out at $120,000") for r in dropped["reject_reasons"])


# ── parse_salary_range (range-aware; screens on the top) ─────────────────────

def test_parse_salary_range_spans_a_single_range():
    lo, hi, text = matching.parse_salary_range("Pay range: $125,000 to $250,000 per year.")
    assert (lo, hi) == (125000, 250000)
    assert "125,000" in text

def test_parse_salary_range_uses_highest_across_multiple_ranges():
    # Two zones/levels → lowest low, HIGHEST high.
    desc = "Denver: $150,000–$175,000. Other regions: $200,000 to $240,000 annually."
    lo, hi, _ = matching.parse_salary_range(desc)
    assert (lo, hi) == (150000, 240000)

def test_parse_salary_range_ignores_stray_single_when_a_range_exists():
    # Base RANGE present → the OTE single is ignored (highest base range only).
    desc = "Base salary: $120,000 to $150,000 per year. OTE up to $300,000."
    lo, hi, _ = matching.parse_salary_range(desc)
    assert (lo, hi) == (120000, 150000)

def test_parse_salary_range_single_amount_sets_min_equal_max():
    lo, hi, _ = matching.parse_salary_range("Salary: $150,000 per year.")
    assert (lo, hi) == (150000, 150000)

def test_parse_salary_range_none_when_no_pay():
    assert matching.parse_salary_range("Great mission, no pay mentioned.") == (None, None, "")


# ── semantic_pass (the meaning net's re-admission switch) ────────────────────

def test_semantic_pass_suppresses_only_the_title_sieve():
    job = {"title": "Head of Craft Excellence", "location": "Remote - US",
           "description": GOOD_DESC}
    assert matching.match_job(job, _F)["reject_reasons"] == [matching.TITLE_SIEVE_REJECT]
    admitted = matching.match_job(job, _F, semantic_pass=True)
    assert admitted["qualifies"] and admitted["semantic_pass"] is True
    # Every OTHER screen still applies under semantic_pass.
    onsite = matching.match_job({**job, "location": "Onsite - Denver"}, _F, semantic_pass=True)
    assert "on-site role" in onsite["reject_reasons"]


def test_keyword_only_rejects_finds_meaning_net_candidates():
    jobs = [
        {"title": "Head of Craft Excellence", "location": "Remote - US"},   # sieve-only reject
        {"title": "Engineering Manager", "location": "Remote - US"},        # qualifies outright
        {"title": "Sales Manager", "location": "Remote - US"},              # negative term too
    ]
    candidates = matching.keyword_only_rejects(jobs, _F)
    assert [j["title"] for j in candidates] == ["Head of Craft Excellence"]


# ── The meaning net itself (fake embeddings, real wiring) ────────────────────

def test_meaning_net_admits_similar_titles_only(monkeypatch):
    from src import embeddings, job_search
    # Titles about "enablement-y" work point the same way as the target vector;
    # everything else is orthogonal (similarity 0). The target vector is computed
    # once per scan by find_matches and passed in.
    monkeypatch.setattr(embeddings, "embed_texts",
                        lambda texts: [[1.0, 0.0] if "Craft" in t else [0.0, 1.0]
                                       for t in texts])
    borderline = [
        {"title": "Head of Craft Excellence", "company": "A", "location": "Remote - US"},
        {"title": "Chief Fun Officer", "company": "B", "location": "Remote - US"},
    ]
    admitted = job_search._meaning_net(borderline, [1.0, 0.0], _F)
    assert [j["title"] for j in admitted] == ["Head of Craft Excellence"]
    assert admitted[0]["match"]["semantic_pass"] is True


def test_meaning_net_off_without_a_target_vector():
    from src import job_search
    # target_vec=None means the net is off (e.g. no OpenAI key) — silent empty result;
    # find_matches logs the one "meaning net off" line, not this helper.
    admitted = job_search._meaning_net(
        [{"title": "Head of Craft Excellence", "location": "Remote - US"}],
        None, _F)
    assert admitted == []
