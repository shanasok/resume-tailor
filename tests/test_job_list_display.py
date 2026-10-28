# tests/test_job_list_display.py — the HOME PAGE's display filters: the posting-age
# cutoff, tab splitting, company search, sorting and paging. These decide what the jobs
# table actually renders, and they are the reason the page stopped loading 3,500+ rows.
#
# Like the other DB-backed tests these need a running Postgres (docker compose up -d db),
# use the throwaway `resume_handler_test` database, and skip cleanly when none is up.
import uuid
from datetime import date, timedelta

import psycopg
import pytest

from src import db, tracker
from src.current_user import DEFAULT_USER_ID as UID


_TEST_DB = "resume_handler_test"
_SERVER_URL, _, _ = db.database_url().rpartition("/")
TEST_DATABASE_URL = f"{_SERVER_URL}/{_TEST_DB}"


def _server_available() -> bool:
    try:
        with psycopg.connect(f"{_SERVER_URL}/postgres", connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason="Postgres not reachable — start it with: docker compose up -d db",
)


@pytest.fixture(autouse=True)
def test_database(monkeypatch, tmp_path):
    """A dedicated throwaway database per test (same isolation as test_tracker_db)."""
    with psycopg.connect(f"{_SERVER_URL}/postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                            [_TEST_DB]).fetchone():
            conn.execute(f'CREATE DATABASE "{_TEST_DB}"')

    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setattr(db, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "APPLICATIONS_CSV", db.APPLICATIONS_CSV.with_name("nope.csv"))
    monkeypatch.setattr(db, "GENERATION_LOG_CSV", db.GENERATION_LOG_CSV.with_name("nope.csv"))

    with db._connect_raw() as conn:
        conn.execute("DROP VIEW IF EXISTS job_overview")
        conn.execute("DROP TABLE IF EXISTS jobs, job_flags, resumes, resume_findings, "
                     "resume_versions, applications, generation_log, job_embeddings, "
                     "profile_embeddings, job_descriptions, rejected_jobs, users, "
                     "user_jobs, user_settings, user_filter_terms, resume_profiles, "
                     "resume_experience, resume_education, resume_skills, "
                     "resume_competencies, resume_certifications, board_directory, "
                     "scans, scan_events, schema_migrations CASCADE")
    with db.connect() as conn:
        pass
    yield
    db._schema_ready = False


def _job(**overrides):
    """A minimal discovered job, unique per call unless an id is supplied."""
    base = {"source": "greenhouse", "board_token": "acme", "id": uuid.uuid4().hex[:8],
            "company": "Acme", "title": "Director of Engineering",
            "location": "Remote — USA", "url": f"https://boards.acme.com/{uuid.uuid4().hex[:8]}",
            "match": {"flags": [], "technical": True}}
    base.update(overrides)
    return base


def _add(days_old=None, **overrides):
    """Track one job posted `days_old` days ago (None = no posting date). Returns its id."""
    posted = None if days_old is None else (date.today() - timedelta(days=days_old)).isoformat()
    job = _job(date_posted=posted, **overrides)
    tracker.upsert_found_jobs(UID, [job])
    return tracker.make_job_id(job)


CUTOFF_64 = 64          # a sample window: "posted in the last 64 days"


def _visible_ids(cutoff_days=CUTOFF_64, **kwargs):
    """The job ids the display filter keeps, as a set."""
    with db.connect() as conn:
        rows = db.fetch_overview_page(
            conn, UID, posted_on_or_after=date.today() - timedelta(days=cutoff_days),
            tab=None, **kwargs)
    return {r["job_id"] for r in rows}


# ── The posting-age cutoff ────────────────────────────────────────────────────────────

def test_recent_job_is_visible_and_old_job_is_hidden():
    fresh = _add(days_old=10)
    stale = _add(days_old=200)
    assert _visible_ids() == {fresh}, "only the in-window posting should render"


def test_old_job_stays_visible_once_applied():
    stale = _add(days_old=200)
    tracker.mark_applied(UID, stale)
    assert stale in _visible_ids(), "applied jobs are history — never age out of the list"


def test_old_job_stays_visible_once_a_resume_exists():
    stale = _add(days_old=200)
    tracker.log_application(UID, "Acme", "Director of Engineering", "https://x/1",
                            "resumes/generated_resumes/acme_resume.html",
                            "resumes/generated_resumes/acme_resume.pdf", job_id=stale)
    assert stale in _visible_ids(), "a generated résumé cost real money — keep it findable"


def test_job_with_no_posting_date_stays_visible():
    undated = _add(days_old=None)
    assert undated in _visible_ids(), "no date means unknown age, not old"


def test_no_cutoff_keeps_everything():
    fresh, stale = _add(days_old=10), _add(days_old=200)
    with db.connect() as conn:
        rows = db.fetch_overview_page(conn, UID, posted_on_or_after=None, tab=None)
    assert {r["job_id"] for r in rows} == {fresh, stale}


# ── Tabs: the split the page used to do in the browser, now done in SQL ───────────────

def _fit(job_id, score):
    """Give a tracked job an AI Fit verdict (what splits To Apply from Under Qualified)."""
    with db.connect() as conn:
        db.set_fit_verdict(conn, UID, job_id, score, "because")


def _tab_ids(tab, threshold=5, cutoff_days=CUTOFF_64, **kwargs):
    with db.connect() as conn:
        rows = db.fetch_overview_page(
            conn, UID, posted_on_or_after=date.today() - timedelta(days=cutoff_days),
            tab=tab, underqualified_at_or_below=threshold, **kwargs)
    return {r["job_id"] for r in rows}


def test_applied_tab_holds_only_applied_jobs():
    applied, pending = _add(days_old=5), _add(days_old=5)
    tracker.mark_applied(UID, applied)
    assert _tab_ids("applied") == {applied}


def test_under_qualified_tab_takes_low_fit_jobs_out_of_to_apply():
    weak, strong = _add(days_old=5), _add(days_old=5)
    _fit(weak, 3)
    _fit(strong, 9)
    assert _tab_ids("underqualified") == {weak}
    assert _tab_ids("pending") == {strong}


def test_unscored_jobs_stay_in_to_apply():
    unscored = _add(days_old=5)                          # no AI Fit verdict at all
    assert _tab_ids("pending") == {unscored}
    assert _tab_ids("underqualified") == set()


def test_applied_job_is_never_under_qualified():
    weak = _add(days_old=5)
    _fit(weak, 2)
    tracker.mark_applied(UID, weak)
    assert _tab_ids("underqualified") == set()
    assert _tab_ids("applied") == {weak}


def test_threshold_decides_where_a_job_lands():
    job = _add(days_old=5)
    _fit(job, 5)
    assert _tab_ids("underqualified", threshold=5) == {job}   # "5 and under"
    assert _tab_ids("underqualified", threshold=4) == set()
    assert _tab_ids("pending", threshold=4) == {job}


def test_manual_tab_holds_manually_added_jobs():
    tracker.add_manual_job(UID, "Acme", "Head of Learning", "https://acme.com/jobs/1")
    _add(days_old=5)                                     # a scanned job, not manual
    assert len(_tab_ids("manual")) == 1


def test_tab_counts_match_the_rows_each_tab_returns():
    strong, weak, applied = _add(days_old=5), _add(days_old=5), _add(days_old=5)
    _add(days_old=300)                                   # aged out — counted nowhere
    _fit(strong, 9)
    _fit(weak, 2)
    tracker.mark_applied(UID, applied)
    with db.connect() as conn:
        counts = db.count_overview_tabs(
            conn, UID, posted_on_or_after=date.today() - timedelta(days=CUTOFF_64),
            underqualified_at_or_below=5)
    assert counts == {"pending": 1, "underqualified": 1, "applied": 1, "manual": 0}


# ── Sorting, company search and paging ────────────────────────────────────────────────

def _page(**kwargs):
    """Job ids in the order the page would render them (list, not set — order matters)."""
    with db.connect() as conn:
        rows = db.fetch_overview_page(
            conn, UID, posted_on_or_after=date.today() - timedelta(days=CUTOFF_64),
            **kwargs)
    return [r["job_id"] for r in rows]


def test_to_apply_defaults_to_most_recently_posted_first():
    older, newer = _add(days_old=30), _add(days_old=2)
    assert _page(tab="pending") == [newer, older]


def test_manual_tab_defaults_to_newest_added_first():
    first = tracker.add_manual_job(UID, "Acme", "Head of Learning", "https://a/1")["job_id"]
    second = tracker.add_manual_job(UID, "Beta", "Learning Director", "https://b/2")["job_id"]
    assert _page(tab="manual") == [second, first]


def test_other_tabs_default_to_best_ai_fit_first():
    low, high = _add(days_old=5), _add(days_old=5)
    _fit(low, 2)
    _fit(high, 4)
    assert _page(tab="underqualified") == [high, low]


def test_explicit_sort_orders_by_that_column():
    beta = _add(days_old=5, company="Beta")
    acme = _add(days_old=5, company="Acme")
    assert _page(tab="pending", sort="company") == [acme, beta]
    assert _page(tab="pending", sort="company", descending=True) == [beta, acme]


def test_missing_values_sort_last_in_both_directions():
    dated = _add(days_old=5)
    undated = _add(days_old=None)
    assert _page(tab="pending", sort="date_posted") == [dated, undated]
    assert _page(tab="pending", sort="date_posted", descending=True) == [dated, undated]


def test_company_search_narrows_to_matching_companies():
    acme = _add(days_old=5, company="Acme Corp")
    _add(days_old=5, company="Beta Inc")
    assert _page(tab="pending", company="acme") == [acme]      # case-insensitive substring
    assert _page(tab="pending", company="cme cor") == [acme]


def test_paging_returns_successive_slices():
    ids = [_add(days_old=n, company=f"C{n:02d}") for n in range(1, 6)]
    first = _page(tab="pending", sort="company", limit=2, offset=0)
    second = _page(tab="pending", sort="company", limit=2, offset=2)
    third = _page(tab="pending", sort="company", limit=2, offset=4)
    assert first + second + third == ids                      # no gaps, no repeats
    assert len(first) == len(second) == 2 and len(third) == 1


def test_counts_respect_the_company_search():
    _add(days_old=5, company="Acme Corp")
    _add(days_old=5, company="Beta Inc")
    with db.connect() as conn:
        counts = db.count_overview_tabs(
            conn, UID, posted_on_or_after=date.today() - timedelta(days=CUTOFF_64),
            company="acme")
    assert counts["pending"] == 1


# ── The tracker boundary: the rolling window the Settings page configures ──────────────

def test_read_job_page_applies_the_max_posting_age_setting():
    fresh, stale = _add(days_old=10), _add(days_old=200)
    rows = tracker.read_job_page(UID, tab="pending", max_age_days=64)
    assert [r["Job ID"] for r in rows] == [fresh]


def test_read_job_page_with_no_age_limit_keeps_everything():
    fresh, stale = _add(days_old=10), _add(days_old=200)
    rows = tracker.read_job_page(UID, tab="pending", max_age_days=0)
    assert {r["Job ID"] for r in rows} == {fresh, stale}, "0 disables the age filter"


def test_read_job_page_returns_display_dicts_with_status():
    job = _add(days_old=5)
    tracker.log_application(UID, "Acme", "Director of Engineering", "https://x/1",
                            "resumes/generated_resumes/a.html",
                            "resumes/generated_resumes/a.pdf", job_id=job)
    row = tracker.read_job_page(UID, tab="pending", max_age_days=64)[0]
    assert row["Status"] == "Generated"
    assert row["Company"] == "Acme"
    assert row["Date Posted"] == date.today() - timedelta(days=5)   # a real date


def test_count_job_tabs_uses_the_same_window():
    _add(days_old=10)
    _add(days_old=200)
    assert tracker.count_job_tabs(UID, max_age_days=64)["pending"] == 1
    assert tracker.count_job_tabs(UID, max_age_days=0)["pending"] == 2


# ── The Under Qualified threshold, now stored per user instead of in localStorage ──────

def test_underqualified_threshold_defaults_to_five():
    from src import filter_config
    assert filter_config.load(UID)["underqualified_at_or_below"] == 5


def test_underqualified_threshold_round_trips_through_the_database():
    from src import filter_config
    cfg = filter_config.load(UID)
    cfg["underqualified_at_or_below"] = 3
    filter_config.save(UID, cfg)
    assert filter_config.load(UID)["underqualified_at_or_below"] == 3


def test_saving_other_filters_does_not_reset_the_threshold():
    from src import filter_config
    cfg = filter_config.load(UID)
    cfg["underqualified_at_or_below"] = 7
    filter_config.save(UID, cfg)
    later = filter_config.load(UID)
    later["salary_floor"] = 210000          # the Settings page saving an unrelated field
    filter_config.save(UID, later)
    assert filter_config.load(UID)["underqualified_at_or_below"] == 7
