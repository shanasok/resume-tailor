# tests/test_workday.py — the Workday connector + discovery extractor, with all HTTP
# FAKED (no network): token packing, relative dates, pagination + the dynamic US facet,
# detail mapping, and URL→token extraction. Mirrors the style of test_embeddings.py.
from datetime import date, timedelta

import pytest
import requests

from src.ats import workday, workday_discovery


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


TOKEN = "nvidia.wd5@NVIDIAExternalCareerSite"


# ── Token packing ─────────────────────────────────────────────────────────────

def test_parse_token_roundtrip():
    assert workday._parse_token(TOKEN) == ("nvidia", "wd5", "NVIDIAExternalCareerSite")
    # tenant/dc lowercase, site case preserved
    assert workday._parse_token("Acme.WD12@My_Site") == ("acme", "wd12", "My_Site")


@pytest.mark.parametrize("bad", ["", "nvidia", "nvidia@Site", "nvidia.wd5", "@Site"])
def test_parse_token_rejects_bad_shapes(bad):
    with pytest.raises(workday.WorkdayError, match="Bad Workday token"):
        workday._parse_token(bad)


# ── Relative posting dates ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_days_ago", [
    ("Posted Today", 0),
    ("Just Posted", 0),
    ("Posted Yesterday", 1),
    ("Posted 12 Days Ago", 12),
    ("posted 1 day ago", 1),
])
def test_relative_date_parses(text, expected_days_ago):
    today = date(2026, 7, 6)
    got = workday._relative_date(text, today=today)
    assert got == (today - timedelta(days=expected_days_ago)).isoformat()


@pytest.mark.parametrize("text", ["Posted 30+ Days Ago", "", "sometime", None])
def test_relative_date_returns_empty_when_unknown(text):
    # Open-ended/garbage → "" so the 90-day age filter KEEPS the job (age unknown).
    assert workday._relative_date(text) == ""


# ── The dynamic US facet ─────────────────────────────────────────────────────

_FACETS = [
    {"facetParameter": "timeType", "values": [{"descriptor": "Full time", "id": "t1"}]},
    {"facetParameter": "locationHierarchy1", "values": [
        {"descriptor": "Canada", "id": "guid-ca"},
        {"descriptor": "United States of America", "id": "guid-us"},
    ]},
]


def test_us_facet_finds_per_tenant_guid():
    assert workday._us_facet(_FACETS) == {"locationHierarchy1": ["guid-us"]}


def test_us_facet_prefers_country_parameter():
    facets = _FACETS + [{"facetParameter": "locationCountry", "values": [
        {"descriptor": "United States", "id": "guid-country"}]}]
    assert workday._us_facet(facets) == {"locationCountry": ["guid-country"]}


def test_us_facet_degrades_to_empty():
    assert workday._us_facet([]) == {}
    assert workday._us_facet([{"facetParameter": "x", "values": [
        {"descriptor": "Germany", "id": "g"}]}]) == {}
    assert workday._us_facet("not-even-a-list") == {}   # never raises


# ── list_jobs pagination ─────────────────────────────────────────────────────

def _posting(n):
    return {"title": f"Engineer {n}", "externalPath": f"/job/US-Remote/Engineer_{n}",
            "locationsText": "Remote - US", "postedOn": "Posted Today"}


def test_list_jobs_paginates_and_applies_us_facet(monkeypatch):
    calls = []                                          # (offset, facets) per request

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((json["offset"], json["appliedFacets"]))
        assert json["limit"] == workday.LIST_LIMIT      # never exceed the hard cap
        if not json["appliedFacets"]:                   # the unfiltered first probe
            return _FakeResponse({"total": 999, "facets": _FACETS,
                                  "jobPostings": [_posting(0)]})
        offset = json["offset"]                         # filtered walk: 45 US jobs → 3 pages
        page = [_posting(offset + i) for i in range(min(20, 45 - offset))]
        return _FakeResponse({"total": 45, "jobPostings": page})

    monkeypatch.setattr(workday.requests, "post", fake_post)
    monkeypatch.setattr(workday.time, "sleep", lambda s: None)   # no real pauses in tests

    jobs = workday.list_jobs(TOKEN)
    # Probe (offset 0, no facet) → filtered restart (0) → offsets 20, 40.
    assert calls == [(0, {}), (0, {"locationHierarchy1": ["guid-us"]}),
                     (20, {"locationHierarchy1": ["guid-us"]}),
                     (40, {"locationHierarchy1": ["guid-us"]})]
    assert len(jobs) == 45
    job = jobs[0]
    assert job["source"] == "workday" and job["board_token"] == TOKEN
    assert job["id"] == "job/US-Remote/Engineer_0"      # externalPath minus leading slash
    assert ":" not in job["id"]                          # reserved by the packed job_id
    assert job["url"] == ("https://nvidia.wd5.myworkdayjobs.com/"
                          "NVIDIAExternalCareerSite/job/US-Remote/Engineer_0")
    assert job["company"] == "Nvidia"
    assert job["description"] == ""                      # deep screen fetches it later


def test_list_jobs_early_stops_when_total_lies(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        if json["offset"] == 0:
            return _FakeResponse({"total": 2000, "facets": [],   # display-cap style total
                                  "jobPostings": [_posting(i) for i in range(20)]})
        return _FakeResponse({"total": 2000, "jobPostings": []})  # …but page 2 is empty

    monkeypatch.setattr(workday.requests, "post", fake_post)
    monkeypatch.setattr(workday.time, "sleep", lambda s: None)
    assert len(workday.list_jobs(TOKEN)) == 20           # stopped, didn't walk to 2000


def test_list_jobs_wraps_network_errors(monkeypatch):
    def exploding_post(*a, **k):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(workday.requests, "post", exploding_post)
    with pytest.raises(workday.WorkdayError, match="Could not list"):
        workday.list_jobs(TOKEN)


# ── get_job detail mapping ───────────────────────────────────────────────────

_DETAIL = {"jobPostingInfo": {
    "title": "Senior Production Engineer",
    "jobDescription": "<p>Build &amp; run AI infra.</p><p>Salary: $184,000 - $287,500.</p>",
    "location": "US, CA, Remote",
    "additionalLocations": ["US, NC, Remote", "US, TX, Remote"],
    "startDate": "2026-07-01",
    "postedOn": "Posted 5 Days Ago",
    "externalUrl": "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/x",
}}


def test_get_job_maps_detail_fields(monkeypatch):
    monkeypatch.setattr(workday.requests, "get",
                        lambda url, headers=None, timeout=None: _FakeResponse(_DETAIL))
    job = workday.get_job(TOKEN, "job/US-CA-Remote/Senior-Production-Engineer_JR1")
    assert job["title"] == "Senior Production Engineer"
    assert "Build & run AI infra." in job["description"]      # HTML unescaped + stripped
    assert "<p>" not in job["description"]
    assert job["location"] == "US, CA, Remote, US, NC, Remote, US, TX, Remote"
    assert job["date_posted"] == "2026-07-01"                 # real startDate wins
    assert job["salary_min"] is None                          # never structured on Workday
    assert job["salary_max"] is None                          # never structured on Workday
    assert job["salary_text"].startswith("$184,000")          # snippet found in the text
    assert job["url"].endswith("/job/x")                      # externalUrl preferred


def test_get_job_falls_back_to_relative_date(monkeypatch):
    detail = {"jobPostingInfo": {**_DETAIL["jobPostingInfo"], "startDate": None,
                                 "postedOn": "Posted Today"}}
    monkeypatch.setattr(workday.requests, "get",
                        lambda url, headers=None, timeout=None: _FakeResponse(detail))
    job = workday.get_job(TOKEN, "job/x/y")
    assert job["date_posted"] == date.today().isoformat()


# ── Discovery: URL → token extraction ────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US/Eng_JR1",
     "nvidia.wd5@NVIDIAExternalCareerSite"),
    # locale-prefixed public URL — the en-US segment must be skipped
    ("https://salesforce.wd12.myworkdayjobs.com/en-US/External_Career_Site/job/x",
     "salesforce.wd12@External_Career_Site"),
    ("https://acme-corp.wd1.myworkdayjobs.com/Careers", "acme-corp.wd1@Careers"),
    # API paths, bare hosts, and non-board hosts are junk
    ("https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/Site/jobs", None),
    ("https://nvidia.wd5.myworkdayjobs.com/", None),
    ("https://www.myworkdayjobs.com/whatever", None),
    ("https://nvidia.example.com/Careers", None),
    ("https://nvidia.wd5.myworkdayjobs.com/bad%20site/job/x", None),   # bad charset
])
def test_discovery_token_from_url(url, expected):
    assert workday_discovery.token_from_url(url) == expected


def test_discovery_validate_token(monkeypatch):
    monkeypatch.setattr(workday_discovery.requests, "post",
                        lambda *a, **k: _FakeResponse({"total": 1343}))
    assert workday_discovery.validate_token(TOKEN) == 1343
    monkeypatch.setattr(workday_discovery.requests, "post",
                        lambda *a, **k: _FakeResponse({}, status_code=422))
    assert workday_discovery.validate_token(TOKEN) is None
    assert workday_discovery.validate_token("garbage") is None   # unparseable token
