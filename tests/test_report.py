from dataclasses import replace

from bloodcell import report
from bloodcell.config import get_settings
from bloodcell.qc import run_quality_control


def test_template_report_contains_sections_and_numbers(sample_prediction):
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    text = report.template_report(sample_prediction, qc)
    for section in ("### Summary", "### Detection Breakdown", "### Quality Assessment", "### Recommended Reviewer Actions"):
        assert section in text
    assert "**4 cells**" in text
    assert "**RBC**: 2" in text
    assert "not a clinical diagnosis" in text


def test_generate_report_without_key_uses_template(sample_prediction):
    settings = replace(get_settings(), gemini_api_key=None)
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    _, source = report.generate_report(sample_prediction, qc, settings)
    assert source == "template"


def test_generate_report_falls_back_when_gemini_fails(sample_prediction, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(report, "gemini_report", boom)
    settings = replace(get_settings(), gemini_api_key="fake-key")
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    text, source = report.generate_report(sample_prediction, qc, settings)
    assert source == "template"
    assert "### Summary" in text


def test_gemini_success_path(sample_prediction, monkeypatch):
    monkeypatch.setattr(report, "gemini_report", lambda p, q, s: "### Summary\nLLM text")
    settings = replace(get_settings(), gemini_api_key="fake-key")
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    text, source = report.generate_report(sample_prediction, qc, settings)
    assert source == "gemini"
    assert text.startswith("### Summary")


def test_llm_payload_is_grounded(sample_prediction):
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    payload = report._payload(sample_prediction, qc)
    assert payload["counts"] == {"RBC": 2, "WBC": 1, "Platelets": 1}
    assert payload["total_detections"] == 4
    assert "bbox" not in str(payload)  # only aggregates are sent to the LLM
