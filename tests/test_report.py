"""Report generation: template, structured Gemini narratives and verification.

Gemini is never called for real here: `report._call_gemini` is replaced with
canned raw outputs, so every path (valid, inconsistent, hallucinated, invalid
JSON, API failure) is deterministic and offline.
"""

import json
from dataclasses import replace

import pytest

from bloodcell import report
from bloodcell.config import get_settings
from bloodcell.qc import run_quality_control
from bloodcell.report_validation import LLMNarrative, collect_facts, verify_narrative

# sample_prediction (conftest): 640x480, RBC 2, WBC 1, Platelets 1, total 4,
# confidences 0.91/0.88/0.95/0.62 -> mean 0.84, QC PASS, 12.3 ms.

VALID = {
    "summary": "The model detected 4 cells (2 RBCs, 1 WBC and 1 platelet) and the run passed QC with status PASS.",
    "quality_assessment": "Mean confidence is 84% with no low-confidence detections and no overlapping boxes.",
    "reviewer_actions": ["Spot-check the single platelet box.", "Confirm the WBC boundary."],
}


@pytest.fixture
def qc(sample_prediction):
    return run_quality_control(sample_prediction.detections, 640, 480)


@pytest.fixture
def llm_settings():
    return replace(get_settings(), gemini_api_key="test-key", gemini_model="gemini-test")


def _narrative(**overrides) -> LLMNarrative:
    return LLMNarrative(**{**VALID, **overrides})


def _payload(sample_prediction, qc):
    return report._payload(sample_prediction, qc)


def fake_gemini(monkeypatch, raw: str | Exception):
    def _call(payload, settings):
        if isinstance(raw, Exception):
            raise raw
        return raw

    monkeypatch.setattr(report, "_call_gemini", _call)


# --------------------------------------------------------------------------- #
# Template
# --------------------------------------------------------------------------- #
def test_template_report_contains_sections_and_numbers(sample_prediction, qc):
    text = report.template_report(sample_prediction, qc)
    for section in ("### Summary", "### Detection Breakdown", "### Quality Assessment", "### Recommended Reviewer Actions"):
        assert section in text
    assert "**4 cells**" in text
    assert "| RBC | 2 | 50% |" in text
    assert "not a clinical diagnosis" in text


def test_no_key_uses_template(sample_prediction, qc):
    settings = replace(get_settings(), gemini_api_key=None)
    _, source, validation = report.generate_report(sample_prediction, qc, settings)
    assert source == "template"
    assert validation.status == "not_applicable"


def test_llm_disabled_uses_template(sample_prediction, qc, llm_settings, monkeypatch):
    fake_gemini(monkeypatch, AssertionError("must not be called"))
    _, source, _ = report.generate_report(sample_prediction, qc, llm_settings, use_llm=False)
    assert source == "template"


# --------------------------------------------------------------------------- #
# Grounding: the LLM sees aggregates only
# --------------------------------------------------------------------------- #
def test_payload_is_aggregates_only(sample_prediction, qc):
    payload = _payload(sample_prediction, qc)
    assert payload["counts"] == {"RBC": 2, "WBC": 1, "Platelets": 1}
    assert payload["total_detections"] == 4
    assert "bbox" not in json.dumps(payload)


def test_facts_include_derived_shares(sample_prediction, qc):
    facts = collect_facts(_payload(sample_prediction, qc))
    assert 50.0 in facts and 25.0 in facts  # RBC and WBC shares in %
    assert 4.0 in facts and 2.0 in facts


# --------------------------------------------------------------------------- #
# Verification (unit level)
# --------------------------------------------------------------------------- #
def test_valid_narrative_is_verified(sample_prediction, qc):
    v = verify_narrative(_narrative(), _payload(sample_prediction, qc))
    assert v.status == "verified", v.issues
    assert v.numbers_checked >= 5


def test_rounded_and_percent_forms_are_accepted(sample_prediction, qc):
    n = _narrative(
        quality_assessment="Mean confidence 0.84 (84.0%), minimum 62%; inference took 12 ms on a 640x480 image. RBCs are 50% of cells."
    )
    v = verify_narrative(n, _payload(sample_prediction, qc))
    assert v.status == "verified", v.issues


@pytest.mark.parametrize(
    "summary, expected",
    [
        ("The model detected 4 cells including 3 RBCs; QC status PASS.", "inconsistent RBC count"),
        ("The model detected 4 cells; WBC count: 2. QC status PASS.", "inconsistent WBC count"),
        ("The model detected 5 cells in total and QC status is PASS.", "inconsistent total"),
        ("A total of 4 cells, with 4 platelets, passed QC (PASS).", "inconsistent Platelets count"),
    ],
)
def test_inconsistent_counts_are_rejected(sample_prediction, qc, summary, expected):
    v = verify_narrative(_narrative(summary=summary), _payload(sample_prediction, qc))
    assert v.status == "rejected"
    assert any(expected in issue for issue in v.issues), v.issues


def test_inconsistent_count_that_matches_another_fact_is_still_rejected(sample_prediction, qc):
    # "2 WBCs" is traceable (RBC count is 2) but wrong for WBC: context check must catch it.
    v = verify_narrative(
        _narrative(summary="The model detected 4 cells, including 2 WBCs. QC status PASS."),
        _payload(sample_prediction, qc),
    )
    assert v.status == "rejected"
    assert any("inconsistent WBC count" in i for i in v.issues)


@pytest.mark.parametrize(
    "text",
    [
        "Mean RBC diameter is 7.5 µm, within normal limits.",
        "The estimated hematocrit is 42%.",
        "Confidence was 97% on average.",
        "Roughly 5000 cells per microlitre were estimated.",
    ],
)
def test_hallucinated_measurements_are_rejected(sample_prediction, qc, text):
    v = verify_narrative(_narrative(quality_assessment=text), _payload(sample_prediction, qc))
    assert v.status == "rejected"
    assert any("untraceable number" in i for i in v.issues), v.issues


def test_missing_total_and_status_are_rejected(sample_prediction, qc):
    v = verify_narrative(
        _narrative(summary="The run looks technically reliable with clear cell boundaries."),
        _payload(sample_prediction, qc),
    )
    assert v.status == "rejected"
    assert any("omits the total" in i for i in v.issues)
    assert any("omits the QC status" in i for i in v.issues)


def test_clinical_terms_are_rejected(sample_prediction, qc):
    v = verify_narrative(
        _narrative(quality_assessment="Low platelet count could indicate thrombocytopenia."),
        _payload(sample_prediction, qc),
    )
    assert v.status == "rejected"
    assert any("clinical term" in i for i in v.issues)


def test_filename_and_model_digits_are_ignored(sample_prediction, qc):
    n = _narrative(summary=f"In {sample_prediction.filename}, test-model detected 4 cells; QC status PASS.")
    assert verify_narrative(n, _payload(sample_prediction, qc)).status == "verified"


# --------------------------------------------------------------------------- #
# End to end through generate_report with a fake Gemini
# --------------------------------------------------------------------------- #
def test_gemini_valid_report(sample_prediction, qc, llm_settings, monkeypatch):
    fake_gemini(monkeypatch, json.dumps(VALID))
    text, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "gemini"
    assert validation.status == "verified"
    assert validation.llm_model == "gemini-test"
    assert VALID["summary"] in text
    assert "| RBC | 2 | 50% |" in text  # table rendered by code, not by the LLM
    assert "checked for traceability to detector outputs" in text
    assert "reported cell counts were cross-checked" in text
    assert "does not guarantee the correctness of all interpretations" in text
    assert "verified against" not in text  # footer must not overstate the check
    assert text.rstrip().endswith(report.DISCLAIMER)


def test_gemini_json_in_code_fence_is_accepted(sample_prediction, qc, llm_settings, monkeypatch):
    fake_gemini(monkeypatch, "```json\n" + json.dumps(VALID) + "\n```")
    _, source, _ = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "gemini"


def test_gemini_inconsistent_report_falls_back(sample_prediction, qc, llm_settings, monkeypatch):
    bad = {**VALID, "summary": "The model detected 4 cells including 3 RBCs; QC status PASS."}
    fake_gemini(monkeypatch, json.dumps(bad))
    text, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "template_fallback"
    assert validation.status == "rejected"
    assert "3 RBCs" not in text


def test_gemini_hallucination_falls_back(sample_prediction, qc, llm_settings, monkeypatch):
    bad = {**VALID, "quality_assessment": "Average RBC diameter was 7.8 µm."}
    fake_gemini(monkeypatch, json.dumps(bad))
    text, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "template_fallback"
    assert "7.8" not in text
    assert any("7.8" in i for i in validation.issues)


@pytest.mark.parametrize(
    "raw",
    [
        "Here is your report: everything looks fine.",  # not JSON
        json.dumps({"summary": "too short"}),  # schema violation / missing fields
        json.dumps({**VALID, "reviewer_actions": []}),  # empty list violates schema
        "",  # empty
    ],
)
def test_invalid_structured_output_falls_back(sample_prediction, qc, llm_settings, monkeypatch, raw):
    fake_gemini(monkeypatch, raw)
    _, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "template_fallback"
    assert validation.status == "rejected"
    assert any("invalid structured output" in i for i in validation.issues)


@pytest.mark.parametrize(
    "exc", [RuntimeError("quota exceeded"), TimeoutError("timed out"), ConnectionError("network down")]
)
def test_api_failure_falls_back(sample_prediction, qc, llm_settings, monkeypatch, exc):
    fake_gemini(monkeypatch, exc)
    text, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "template_fallback"
    assert any("LLM call failed" in i for i in validation.issues)
    assert "### Summary" in text


def test_secrets_never_reach_validation_issues(sample_prediction, qc, llm_settings, monkeypatch):
    fake_gemini(monkeypatch, RuntimeError("401 invalid key test-key"))
    _, _, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert "test-key" not in json.dumps(validation.model_dump())


def test_sentence_final_numbers_are_parsed(sample_prediction, qc):
    # A full stop after a number must not be mistaken for a decimal point.
    ok = _narrative(summary="QC status is PASS and the total detection count is 4.")
    assert verify_narrative(ok, _payload(sample_prediction, qc)).status == "verified"
    bad = _narrative(summary="QC status is PASS. The WBC count was 3.")
    assert verify_narrative(bad, _payload(sample_prediction, qc)).status == "rejected"


def test_gemini_limits_leave_room_for_thinking_and_bound_latency():
    # Regression: a live call spent ~980 "thinking" tokens, truncating the JSON at a 1,024 budget.
    assert report.GEMINI_MAX_OUTPUT_TOKENS >= 4096
    assert 0 < report.GEMINI_TIMEOUT_MS <= 60_000


def test_truncated_json_falls_back(sample_prediction, qc, llm_settings, monkeypatch):
    # Exactly what the live API returned when the token budget ran out mid-string.
    fake_gemini(monkeypatch, '{"summary":"Automated detection yielded 4 total detections with a quality control status of')
    text, source, validation = report.generate_report(sample_prediction, qc, llm_settings)
    assert source == "template_fallback"
    assert any("invalid structured output" in i for i in validation.issues)
    assert "### Summary" in text
