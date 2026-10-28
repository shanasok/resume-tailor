# tests/test_smartrecruiters.py — the SmartRecruiters connector + discovery, with all HTTP
# FAKED (no network): summary normalization, paginated US-filtered listing, detail mapping
# (joined description sections + structured compensation), and the discovery validator.
# Mirrors the style of test_workday.py / test_ashby.py.
import pytest
import requests

from src.ats import smartrecruiters, smartrecruiters_discovery

TOKEN = "BoschGroup"


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


# ── Location + salary + date helpers ─────────────────────────────────────────

def test_location_uses_full_location_and_flags():
    posting = {"location": {"fullLocation": "Bethlehem, PA, United States",
                            "remote": False, "hybrid": True}}
    assert smartrecruiters._location(posting) == "Bethlehem, PA, United States · Hybrid"


def test_location_assembles_from_pieces_when_no_full_location():
    posting = {"location": {"city": "Austin", "region": "TX", "country": "us", "remote": True}}
    assert smartrecruiters._location(posting) == "Austin, TX, US · Remote"


def test_salary_reads_annual_usd_range():
    posting = {"compensation": {"min": 100000, "max": 135000, "currency": "USD", "period": "YEARLY"}}
    assert smartrecruiters._salary(posting) == (100000, 135000, "$100,000 - $135,000")


def test_salary_max_falls_back_to_min():
    posting = {"compensation": {"min": 90000, "currency": "USD", "period": "YEARLY"}}
    assert smartrecruiters._salary(posting) == (90000, 90000, "$90,000 - $90,000")


@pytest.mark.parametrize("comp", [
    None,
    {"min": 55, "max": 70, "currency": "USD", "period": "HOURLY"},   # hourly ≠ yearly floor
    {"min": 100000, "max": 120000, "currency": "EUR", "period": "YEARLY"},  # non-USD
    {"max": 120000, "currency": "USD", "period": "YEARLY"},          # no min
])
def test_salary_ignores_unusable_compensation(comp):
    assert smartrecruiters._salary({"compensation": comp}) == (None, None, "")


def test_posted_date_trims_iso():
    assert smartrecruiters._posted_date({"releasedDate": "2026-07-13T03:01:25.836Z"}) == "2026-07-13"
    assert smartrecruiters._posted_date({}) == ""


# ── list_jobs: pagination + US filter + summary shape ────────────────────────

def _summary(n):
    return {"id": str(1000 + n), "name": f"Engineer {n}",
            "company": {"identifier": TOKEN, "name": "Bosch Group"},
            "location": {"fullLocation": f"City {n}, PA, United States"},
            "releasedDate": "2026-07-10T00:00:00.000Z",
            "postingUrl": f"https://jobs.smartrecruiters.com/{TOKEN}/{1000 + n}-engineer-{n}"}


def test_list_jobs_paginates_and_filters_us(monkeypatch):
    calls = []                                           # (offset, country, limit) per request

    def fake_get(url, params=None, headers=None, timeout=None):
        params = params or {}
        calls.append((params.get("offset"), params.get("country"), params.get("limit")))
        offset = params["offset"]                        # 250 US jobs → 3 pages of 100/100/50
        page = [_summary(offset + i) for i in range(min(smartrecruiters.LIST_LIMIT, 250 - offset))]
        return _FakeResponse({"totalFound": 250, "limit": 100, "offset": offset, "content": page})

    monkeypatch.setattr(smartrecruiters.requests, "get", fake_get)
    monkeypatch.setattr(smartrecruiters.time, "sleep", lambda s: None)   # no real pauses in tests

    jobs = smartrecruiters.list_jobs(TOKEN)
    assert calls == [(0, "us", 100), (100, "us", 100), (200, "us", 100)]  # US-filtered walk
    assert len(jobs) == 250
    job = jobs[0]
    assert job["source"] == "smartrecruiters" and job["board_token"] == TOKEN
    assert job["id"] == "1000" and job["title"] == "Engineer 0"
    assert job["company"] == "Bosch Group"              # from company.name
    assert job["location"] == "City 0, PA, United States"
    assert job["url"].endswith("/1000-engineer-0")
    assert job["date_posted"] == "2026-07-10"
    assert job["description"] == ""                      # deep screen fetches it later


def test_list_jobs_early_stops_when_total_lies(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if params["offset"] == 0:
            return _FakeResponse({"totalFound": 5000,    # over-reported total
                                  "content": [_summary(i) for i in range(100)]})
        return _FakeResponse({"totalFound": 5000, "content": []})   # …page 2 is empty
    monkeypatch.setattr(smartrecruiters.requests, "get", fake_get)
    monkeypatch.setattr(smartrecruiters.time, "sleep", lambda s: None)
    assert len(smartrecruiters.list_jobs(TOKEN)) == 100   # stopped, didn't chase 5000


def test_list_jobs_skips_ids_with_reserved_colon(monkeypatch):
    bad = {**_summary(1), "id": "12:34"}                 # ':' collides with the packed job_id
    monkeypatch.setattr(smartrecruiters.requests, "get",
                        lambda *a, **k: _FakeResponse({"totalFound": 1, "content": [bad]}))
    assert smartrecruiters.list_jobs(TOKEN) == []


def test_list_jobs_wraps_network_errors(monkeypatch):
    def exploding_get(*a, **k):
        raise requests.ConnectionError("boom")
    monkeypatch.setattr(smartrecruiters.requests, "get", exploding_get)
    with pytest.raises(smartrecruiters.SmartRecruitersError, match="Could not list"):
        smartrecruiters.list_jobs(TOKEN)


# ── get_job: detail mapping ──────────────────────────────────────────────────

_DETAIL = {
    "id": "744000137359929",
    "name": "Senior Web Developer",
    "company": {"identifier": TOKEN, "name": "Bosch Group"},
    "location": {"fullLocation": "Bethlehem, PA, United States", "remote": False, "hybrid": True},
    "releasedDate": "2026-07-01T12:00:00.000Z",
    "compensation": {"min": 120000, "max": 160000, "currency": "USD", "period": "YEARLY"},
    "postingUrl": "https://jobs.smartrecruiters.com/BoschGroup/744000137359929-senior-web-developer",
    "jobAd": {"sections": {
        "jobDescription": {"title": "Job Description", "text": "<ul><li>Build &amp; run web apps.</li></ul>"},
        "qualifications": {"title": "Qualifications", "text": "<p>5+ years experience.</p>"},
        "additionalInformation": {"title": "Additional Information", "text": "<p>Hybrid role.</p>"},
        "companyDescription": {"title": "Company Description", "text": "<p>Bosch is a global company.</p>"},
    }},
}


def test_get_job_maps_detail_fields(monkeypatch):
    monkeypatch.setattr(smartrecruiters.requests, "get",
                        lambda url, headers=None, timeout=None: _FakeResponse(_DETAIL))
    job = smartrecruiters.get_job(TOKEN, "744000137359929")
    assert job["title"] == "Senior Web Developer"
    assert job["company"] == "Bosch Group"
    assert job["location"] == "Bethlehem, PA, United States · Hybrid"
    assert job["url"].endswith("-senior-web-developer")
    assert job["date_posted"] == "2026-07-01"
    # structured pay preferred (matcher hard-drops jobs with no stated salary)
    assert job["salary_min"] == 120000 and job["salary_max"] == 160000
    assert job["salary_text"] == "$120,000 - $160,000"
    # description = all sections, HTML unescaped + stripped, job-relevant first
    desc = job["description"]
    assert "Build & run web apps." in desc and "<ul>" not in desc
    assert "5+ years experience." in desc and "Hybrid role." in desc
    assert desc.index("Build & run web apps.") < desc.index("Bosch is a global company.")


def test_get_job_falls_back_to_description_salary_snippet(monkeypatch):
    detail = {**_DETAIL, "compensation": None,
              "jobAd": {"sections": {"jobDescription": {"text": "Pay is $185,000 per year."}}}}
    monkeypatch.setattr(smartrecruiters.requests, "get",
                        lambda url, headers=None, timeout=None: _FakeResponse(detail))
    job = smartrecruiters.get_job(TOKEN, "1")
    assert job["salary_min"] is None                    # no structured comp
    assert job["salary_text"].startswith("$185,000")    # snippet scraped from the text


def test_get_job_wraps_errors(monkeypatch):
    monkeypatch.setattr(smartrecruiters.requests, "get",
                        lambda *a, **k: _FakeResponse({}, status_code=404))
    with pytest.raises(smartrecruiters.SmartRecruitersError, match="Could not fetch"):
        smartrecruiters.get_job(TOKEN, "missing")


# ── Discovery: validate_token ────────────────────────────────────────────────

def test_discovery_validate_token(monkeypatch):
    monkeypatch.setattr(smartrecruiters_discovery.requests, "get",
                        lambda *a, **k: _FakeResponse({"totalFound": 272}))
    assert smartrecruiters_discovery.validate_token(TOKEN) == 272
    # an unknown identifier still returns 200 but 0 US jobs → not stored
    monkeypatch.setattr(smartrecruiters_discovery.requests, "get",
                        lambda *a, **k: _FakeResponse({"totalFound": 0}))
    assert smartrecruiters_discovery.validate_token("nope") is None
    # non-200 → None
    monkeypatch.setattr(smartrecruiters_discovery.requests, "get",
                        lambda *a, **k: _FakeResponse({}, status_code=500))
    assert smartrecruiters_discovery.validate_token(TOKEN) is None
