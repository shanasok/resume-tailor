# tests/test_generation_message.py — _generation_message() turns the pipeline's meta into the
# banner the user sees. The key case: a résumé the critic never scored (unaudited) must be
# called out as UNVERIFIED, not reported as a clean-but-low "score 0/100".
import app as app_module

_gen_msg = app_module._generation_message


def _meta(**over):
    base = {"pages": 2, "attempts": 1, "external_review_used": False, "accepted": False,
            "unaudited": False, "fit": True, "too_short": False, "score": 88,
            "fidelity_issues": [], "style_issues": []}
    base.update(over)
    return base


def test_accepted_is_a_success_banner():
    msg, cat = _gen_msg("Acme", "Director", _meta(accepted=True, score=91))
    assert cat == "success" and "quality score 91/100" in msg


def test_unaudited_warns_that_the_resume_is_unverified():
    msg, cat = _gen_msg("Harvey", "Automation Engineer",
                        _meta(unaudited=True, score=0))
    assert cat == "error"
    assert "could NOT be truthfulness-checked" in msg and "UNVERIFIED" in msg
    # It must NOT masquerade as a scored result — no misleading "score 0/100".
    assert "score 0/100" not in msg


def test_normal_not_accepted_lists_flags():
    msg, cat = _gen_msg("Acme", "Director",
                        _meta(score=60, fidelity_issues=["inflated scope"], style_issues=["echoes posting"]))
    assert cat == "error"
    assert "1 truthfulness flag(s)" in msg and "1 AI-style flag(s)" in msg
