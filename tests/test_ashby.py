# tests/test_ashby.py — the Ashby connector + discovery validator, with all HTTP FAKED
# (no network): slug→company name, structured-salary parsing (annual USD only), location
# assembly, description fallback, list_jobs isListed filtering, get_job by-id lookup, and
# error wrapping. Mirrors the style of test_workday.py.
import pytest
import requests

from src.ats import ashby, ashby_discovery


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


TOKEN = "ramp"


def _job(**overrides):
    """A minimal realistic Ashby posting; override any field per-test."""
    job = {
        "id": "abc-123",
        "title": "Staff Engineer",
        "location": "Remote (US)",
        "workplaceType": "Remote",
        "isListed": True,
        "isRemote": True,
        "publishedAt": "2026-07-07T20:47:09.753+00:00",
        "jobUrl": "https://jobs.ashbyhq.com/ramp/abc-123",
        "address": {"postalAddress": {"addressCountry": "United States"}},
        "secondaryLocations": [],
        "descriptionPlain": "Build cool things.",
        "compensation": {},
    }
    job.update(overrides)
    return job


# ── Company name ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("token,expected", [
    ("ramp", "Ramp"),
    ("some-co", "Some Co"),
    ("acme_corp", "Acme Corp"),
])
def test_company_name_from_slug(token, expected):
    assert ashby._company_name(token) == expected


# ── Structured salary parsing ─────────────────────────────────────────────────

def test_salary_reads_annual_usd_from_summary_components():
    comp = {"scrapeableCompensationSalarySummary": "$151K - $231K",
            "summaryComponents": [
                {"compensationType": "EquityCashValue", "minValue": None, "interval": "1 YEAR"},
                {"compensationType": "Salary", "minValue": 151000, "maxValue": 231000,
                 "currencyCode": "USD", "interval": "1 YEAR"}]}
    salary_min, salary_max, text = ashby._salary(_job(compensation=comp))
    assert salary_min == 151000
    assert salary_max == 231000
    assert text == "$151K - $231K"                       # prefers the clean scrapeable summary


def test_salary_max_falls_back_to_min_when_absent():
    comp = {"summaryComponents": [
        {"compensationType": "Salary", "minValue": 200000, "currencyCode": "USD",
         "interval": "1 YEAR"}]}
    salary_min, salary_max, _ = ashby._salary(_job(compensation=comp))
    assert (salary_min, salary_max) == (200000, 200000)  # no maxValue → max == min


def test_salary_ignores_hourly_and_equity_components():
    comp = {"summaryComponents": [
        {"compensationType": "Salary", "minValue": 55, "interval": "1 HOUR", "currencyCode": "USD"},
        {"compensationType": "EquityCashValue", "minValue": 100000, "interval": "1 YEAR"}]}
    salary_min, salary_max, _ = ashby._salary(_job(compensation=comp))
    assert salary_min is None                             # hourly + equity are not an annual salary
    assert salary_max is None


def test_salary_ignores_non_usd():
    comp = {"summaryComponents": [
        {"compensationType": "Salary", "minValue": 90000, "interval": "1 YEAR", "currencyCode": "EUR"}]}
    assert ashby._salary(_job(compensation=comp))[0] is None


def test_salary_falls_back_to_tiers_and_picks_lowest():
    comp = {"compensationTierSummary": "$168K – $231K • Multiple Ranges",
            "compensationTiers": [
                {"components": [{"compensationType": "Salary", "minValue": 200000, "maxValue": 231000,
                                 "interval": "1 YEAR", "currencyCode": "USD"}]},
                {"components": [{"compensationType": "Salary", "minValue": 168000, "maxValue": 210000,
                                 "interval": "1 YEAR", "currencyCode": "USD"}]}]}
    salary_min, salary_max, text = ashby._salary(_job(compensation=comp))
    assert salary_min == 168000                           # lowest annual across tiers
    assert salary_max == 231000                           # HIGHEST annual across tiers
    assert text == "$168K – $231K • Multiple Ranges"      # tier summary when no scrapeable


def test_salary_absent_is_none():
    assert ashby._salary(_job(compensation={})) == (None, None, "")
    assert ashby._salary(_job(compensation=None)) == (None, None, "")


# ── Location assembly ─────────────────────────────────────────────────────────

def test_location_folds_signals_and_dedupes():
    job = _job(location="Remote (US)", workplaceType="Remote",
               secondaryLocations=[{"location": "Austin, TX"}, {"location": "Remote (US)"}])
    loc = ashby._location(job)
    assert loc == "Remote (US) · Remote · United States · Austin, TX"   # dup 'Remote (US)' dropped


def test_location_maps_usa_country_spelling():
    job = _job(location="Austin, TX", workplaceType="OnSite",
               address={"postalAddress": {"addressCountry": "USA"}}, secondaryLocations=[])
    assert "United States" in ashby._location(job)


# ── Description ───────────────────────────────────────────────────────────────

def test_description_prefers_plain_then_strips_html():
    assert ashby._description(_job(descriptionPlain="Plain text.")) == "Plain text."
    job = _job(descriptionPlain="", descriptionHtml="<p>Hello &amp; welcome.</p>")
    assert ashby._description(job) == "Hello & welcome."


# ── list_jobs ─────────────────────────────────────────────────────────────────

def test_list_jobs_normalizes_and_skips_unlisted(monkeypatch):
    payload = {"jobs": [_job(id="a", isListed=True),
                        _job(id="b", isListed=False),          # hidden/internal — skipped
                        _job(id="c", isListed=True)]}
    monkeypatch.setattr(ashby.requests, "get",
                        lambda url, params=None, headers=None, timeout=None: _FakeResponse(payload))
    jobs = ashby.list_jobs(TOKEN)
    assert [j["id"] for j in jobs] == ["a", "c"]
    j = jobs[0]
    assert j["source"] == "ashby" and j["board_token"] == TOKEN
    assert j["company"] == "Ramp"
    assert j["url"] == "https://jobs.ashbyhq.com/ramp/abc-123"
    assert j["date_posted"] == "2026-07-07"                    # publishedAt truncated to a date
    assert j["description"] == "Build cool things."


def test_list_jobs_wraps_network_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(ashby.requests, "get", boom)
    with pytest.raises(ashby.AshbyError, match="Could not list"):
        ashby.list_jobs(TOKEN)


def test_list_jobs_rejects_bad_shape(monkeypatch):
    monkeypatch.setattr(ashby.requests, "get",
                        lambda *a, **k: _FakeResponse({"not_jobs": []}))
    with pytest.raises(ashby.AshbyError, match="Unexpected response shape"):
        ashby.list_jobs(TOKEN)


# ── get_job ───────────────────────────────────────────────────────────────────

def test_get_job_finds_by_id_including_unlisted(monkeypatch):
    payload = {"jobs": [_job(id="a"), _job(id="target", isListed=False, title="Found It")]}
    monkeypatch.setattr(ashby.requests, "get",
                        lambda *a, **k: _FakeResponse(payload))
    job = ashby.get_job(TOKEN, "target")
    assert job["title"] == "Found It"                         # re-fetchable even if since-unlisted


def test_get_job_raises_when_missing(monkeypatch):
    monkeypatch.setattr(ashby.requests, "get",
                        lambda *a, **k: _FakeResponse({"jobs": [_job(id="a")]}))
    with pytest.raises(ashby.AshbyError, match="not found"):
        ashby.get_job(TOKEN, "nope")


# ── Discovery validation ──────────────────────────────────────────────────────

def test_discovery_validate_token(monkeypatch):
    monkeypatch.setattr(ashby_discovery.requests, "get",
                        lambda *a, **k: _FakeResponse({"jobs": [_job(), _job()]}))
    assert ashby_discovery.validate_token(TOKEN) == 2
    monkeypatch.setattr(ashby_discovery.requests, "get",
                        lambda *a, **k: _FakeResponse({}, status_code=404))
    assert ashby_discovery.validate_token("no-such-board") is None
