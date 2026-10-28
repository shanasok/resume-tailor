# tests/test_generate_guard.py — the "refuse to generate without a base résumé" guard on
# the shared _generate_for_job_core (relocated from the retired test_upload_route.py, which
# was removed with the .docx upload flow), plus the Manual tab's add-only /add-job route.
import pytest

import app as app_module


@pytest.fixture
def client(monkeypatch):
    """Flask test client. The index/settings pages list tracked jobs from Postgres; these
    tests are about the generation guards, so stub the tracker read — no database needed."""
    monkeypatch.setattr(app_module, "read_applications", lambda uid: [])
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_add_job_files_the_job_without_generating(client, monkeypatch):
    """/add-job just records the job (no résumé, no base-résumé requirement) — the pipeline
    must never run, and add_manual_job is called with the posted fields."""
    calls = {}

    def _add(uid, company, role, source):
        calls.update(uid=uid, company=company, role=role, source=source)
        return {"ok": True, "added": True, "job_id": "manual:1:acme_dev_resume",
                "message": "Added Acme — Dev to your list."}
    monkeypatch.setattr(app_module, "add_manual_job", _add)

    def _pipeline_must_not_run(*args, **kwargs):
        raise AssertionError("pipeline ran — /add-job must not generate a résumé")
    monkeypatch.setattr(app_module, "generate", _pipeline_must_not_run)

    resp = client.post("/add-job", data={"ajax": "1", "company": "Acme", "role": "Dev",
                                         "job_text": "A job posting"})
    body = resp.get_json()
    assert body["ok"] is True
    assert calls["company"] == "Acme" and calls["role"] == "Dev"
    assert calls["source"] == "A job posting"


def test_add_job_requires_company_and_role(client, monkeypatch):
    """Missing company/role is rejected before anything is stored."""
    monkeypatch.setattr(app_module, "add_manual_job",
                        lambda *a, **k: pytest.fail("should not store without company/role"))
    resp = client.post("/add-job", data={"ajax": "1", "role": "Dev", "job_text": "x"})
    body = resp.get_json()
    assert body["ok"] is False
    assert "required" in body["message"].lower()


def test_generate_for_job_core_blocked_without_base_resume(client, monkeypatch):
    """The shared core used by the per-row button AND scheduled runs must refuse to generate
    without a base résumé. The guard runs before the job lookup, so a fake job_id never gets
    that far."""
    monkeypatch.setattr(app_module.base_resume, "has_base_resume", lambda uid: False)
    result = app_module._generate_for_job_core(1, "no-such-job", "claude")
    assert result["ok"] is False
    assert "base résumé" in result["message"].lower()
