# tests/test_connector_salary_max.py — the structured-pay helpers now return the range
# HIGH as salary_max (the matcher screens the floor on the top of the range). No network.
from src.ats import greenhouse, lever, rippling


def test_greenhouse_returns_min_max_from_pay_input_ranges():
    data = {"pay_input_ranges": [
        {"currency_type": "USD", "min_value": 150000, "max_value": 250000}]}
    lo, hi, text = greenhouse._extract_structured_pay(data)
    assert (lo, hi) == (150000, 250000)
    assert text == "$150,000 - $250,000"


def test_greenhouse_no_structured_pay_is_none_none():
    assert greenhouse._extract_structured_pay({}) == (None, None, "")


def test_lever_returns_min_max_from_salary_range():
    posting = {"salaryRange": {"currency": "USD", "min": 150000, "max": 250000}}
    lo, hi, text = lever._salary(posting)
    assert (lo, hi) == (150000, 250000)
    assert text == "$150,000 - $250,000"


def test_lever_max_falls_back_to_min_when_absent():
    posting = {"salaryRange": {"currency": "USD", "min": 200000}}
    lo, hi, _ = lever._salary(posting)
    assert (lo, hi) == (200000, 200000)


def test_rippling_returns_min_max_from_pay_range_details():
    data = {"payRangeDetails": [{"currency": "USD", "min": 150000, "max": 250000}]}
    lo, hi, text = rippling._salary(data)
    assert (lo, hi) == (150000, 250000)
    assert text == "$150,000 - $250,000"


def test_rippling_no_structured_pay_is_none_none():
    assert rippling._salary({}) == (None, None, "")
