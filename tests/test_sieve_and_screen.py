from src.job_search import _sieve_and_screen
from src.matching import Filters


class _RecAudit:
    """A stand-in ScanAudit that records calls instead of writing a file."""

    def __init__(self):
        self.sieved = []          # (title, how)
        self.stages = []          # (title, stage, outcome, tuple(reasons))

    def sieve_pass(self, job, how):
        self.sieved.append((job["title"], how))

    def stage(self, job, stage, outcome, reasons=None):
        self.stages.append((job["title"], stage, outcome, tuple(reasons or ())))


def _filters():
    return Filters({
        "domain_keywords": ["engineering"],
        "title_negative_keywords": ["junior"],
        "salary_floor": 0,
        "commutable_locations": [],
    })


def _job(title, location, url):
    return {"title": title, "company": "Acme", "location": location, "url": url,
            "source": "greenhouse", "board_token": "acme", "id": url}


def test_sieve_aware_split_logs_only_sieve_passers():
    f = _filters()
    jobs = [
        _job("Engineering Manager", "Remote - US", "u1"),         # passes sieve + list screen
        _job("Engineering Lead", "Seattle, WA (On-site)", "u2"),  # passes sieve, on-site drop
        _job("Widget Wrangler", "Remote - US", "u3"),              # sieve-only fail → borderline
        _job("Junior Analyst", "Remote - US", "u4"),               # sieve + negative → silent
        _job("Engineering Manager", "Remote - US", "seen"),        # already-seen → skipped
    ]
    audit = _RecAudit()
    candidates, borderline = _sieve_and_screen(jobs, f, {"seen"}, audit)

    assert [c["title"] for c in candidates] == ["Engineering Manager"]
    assert [b["title"] for b in borderline] == ["Widget Wrangler"]

    # Only the two sieve-passers were logged; the borderline, silent, and seen jobs were not.
    assert audit.sieved == [("Engineering Manager", "keyword"),
                            ("Engineering Lead", "keyword")]
    assert ("Engineering Manager", "list_screen", "advanced", ()) in audit.stages
    dropped = [s for s in audit.stages if s[0] == "Engineering Lead" and s[2] == "dropped"]
    assert dropped and "on-site role" in dropped[0][3]
    assert not any(s[0] in ("Widget Wrangler", "Junior Analyst") for s in audit.stages)
