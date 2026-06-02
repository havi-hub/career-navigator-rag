"""
Structural sanity checks for run_live_job_search() output.

These tests validate the *shape* of results — no API calls, no pipeline re-runs.
A valid result dict must have:
  - score: int in [1, 10]
  - matching_skills: non-empty list
  - missing_skills: non-empty list
  - summary: non-blank string
  - domain gate: if is_same_domain=False, score must be 1
"""

import pytest
from live_jobs import validate_results


def _job(**overrides) -> dict:
    base = {
        "title": "Senior Data Scientist",
        "url": "https://boards.greenhouse.io/example/jobs/123",
        "score": 8,
        "is_same_domain": True,
        "score_rationale": "Strong ML background matches role requirements.",
        "matching_skills": ["PyTorch", "NLP", "MLOps"],
        "missing_skills": ["Kubernetes", "Spark"],
        "summary": "Strong fit — candidate's NLP and MLOps experience aligns well.",
    }
    base.update(overrides)
    return base


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_valid_single_result():
    assert validate_results([_job()]) == []


def test_valid_three_results():
    jobs = [_job(score=9), _job(score=7), _job(score=5)]
    assert validate_results(jobs) == []


def test_off_domain_score_1_is_valid():
    job = _job(is_same_domain=False, score=1, score_rationale="Sales role — off domain.")
    assert validate_results([job]) == []


# ── Empty results ──────────────────────────────────────────────────────────────

def test_empty_results_is_violation():
    violations = validate_results([])
    assert any("empty" in v for v in violations)


# ── Score range ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_score", [0, 11, -1, 100])
def test_score_out_of_range(bad_score):
    violations = validate_results([_job(score=bad_score)])
    assert any("score" in v for v in violations)


@pytest.mark.parametrize("bad_score", [7.5, "8", None])
def test_score_wrong_type(bad_score):
    violations = validate_results([_job(score=bad_score)])
    assert any("score" in v for v in violations)


@pytest.mark.parametrize("valid_score", [1, 5, 10])
def test_score_boundary_values_are_valid(valid_score):
    assert validate_results([_job(score=valid_score)]) == []


# ── Skills lists ───────────────────────────────────────────────────────────────

def test_empty_matching_skills_is_violation():
    violations = validate_results([_job(matching_skills=[])])
    assert any("matching_skills" in v for v in violations)


def test_empty_missing_skills_is_violation():
    violations = validate_results([_job(missing_skills=[])])
    assert any("missing_skills" in v for v in violations)


def test_missing_skills_key_is_violation():
    job = _job()
    del job["matching_skills"]
    violations = validate_results([job])
    assert any("matching_skills" in v for v in violations)


# ── Summary ────────────────────────────────────────────────────────────────────

def test_blank_summary_is_violation():
    violations = validate_results([_job(summary="")])
    assert any("summary" in v for v in violations)


def test_whitespace_only_summary_is_violation():
    violations = validate_results([_job(summary="   ")])
    assert any("summary" in v for v in violations)


# ── Domain gate invariant ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_score", [2, 5, 8, 10])
def test_off_domain_with_high_score_is_violation(bad_score):
    job = _job(is_same_domain=False, score=bad_score)
    violations = validate_results([job])
    assert any("is_same_domain" in v for v in violations)


def test_in_domain_job_is_not_constrained_to_score_1():
    job = _job(is_same_domain=True, score=9)
    assert validate_results([job]) == []


# ── Multiple results — each is checked independently ──────────────────────────

def test_violation_in_second_result_is_caught():
    jobs = [_job(score=9), _job(score=0)]
    violations = validate_results(jobs)
    assert len(violations) == 1
    assert "result[1]" in violations[0]


def test_multiple_violations_all_reported():
    job = _job(score=0, matching_skills=[], summary="")
    violations = validate_results([job])
    assert len(violations) == 3
