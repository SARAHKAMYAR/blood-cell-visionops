"""Technical report generation: grounded Gemini narrative with verified numbers.

Design
------
* Every measurement in a report is rendered by code from the detector/QC
  output (counts table, confidence and QC figures). The LLM never produces a
  table of numbers.
* Gemini receives only aggregate measurements (no image, no patient data) and
  returns a *structured* JSON narrative (summary, quality assessment,
  reviewer actions) validated against a Pydantic schema.
* The narrative is then verified deterministically
  (:mod:`bloodcell.report_validation`): any untraceable, inconsistent or
  missing number, or any clinical term, rejects it.
* Any failure (no key, network/quota error, safety block, invalid JSON,
  schema violation, failed verification) falls back to the deterministic
  template. The API never fails because of the LLM.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from bloodcell.config import Settings, get_settings
from bloodcell.report_validation import LLMNarrative, verify_narrative
from bloodcell.schemas import Prediction, QualityControl, ReportValidation

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "_Research use only. Automated output, not a clinical diagnosis. "
    "Not validated for patient care._"
)

SYSTEM_INSTRUCTION = """\
You are a laboratory computer-vision QA assistant. You write the narrative \
parts of a technical report about one automated blood-smear detection run. \
Tables of measurements are added separately by software; do not reproduce them.

Rules:
- Every number you write must be copied exactly from the JSON measurements \
(you may round, and you may express a 0-1 confidence as a percentage). \
Never estimate, compute new statistics, or introduce measurements that are \
not in the JSON (no cell sizes, ratios to reference ranges, or clinical values).
- The summary must state the total number of detections and the QC status.
- Do NOT diagnose or name any disease or clinical condition. Comment only on \
the technical reliability of the run and what a human reviewer should check.
- Explain QC flags in plain language.
- Be concise: at most 140 words in total.
"""

FLAG_TEXT = {
    "NO_DETECTIONS": "No cells were detected: the image may be blank, out of focus or not a blood smear.",
    "LOW_MEAN_CONFIDENCE": "Mean confidence is below 0.50, so the model is uncertain on this image.",
    "HIGH_LOW_CONFIDENCE_RATIO": "More than 30% of detections are low-confidence.",
    "OVERLAPPING_BOXES": "Some boxes overlap strongly; clumped or overlapping cells may be double-counted.",
    "NO_WBC_DETECTED": "No white blood cell was found, although BCCD fields are centred on one.",
    "UNUSUAL_WBC_COUNT": "More than three WBCs in one field is unusual for this dataset.",
}


def _payload(prediction: Prediction, qc: QualityControl) -> dict:
    """The only information the LLM sees: aggregates, never boxes or pixels."""
    return {
        "filename": prediction.filename,
        "model": prediction.model_name,
        "image_size": [prediction.image_width, prediction.image_height],
        "inference_ms": prediction.inference_ms,
        "counts": prediction.counts,
        "total_detections": len(prediction.detections),
        "quality_control": qc.model_dump(),
        "flag_explanations": {f: FLAG_TEXT.get(f, f) for f in qc.flags},
    }


# --------------------------------------------------------------------------- #
# Deterministic sections (shared by template and LLM reports)
# --------------------------------------------------------------------------- #
def _breakdown_lines(prediction: Prediction) -> list[str]:
    total = len(prediction.detections)
    lines = ["### Detection Breakdown", "| Cell type | Count | Share |", "|---|---|---|"]
    for name, n in prediction.counts.items():
        share = f"{n / total:.0%}" if total else "0%"
        lines.append(f"| {name} | {n} | {share} |")
    lines.append(f"| **Total** | **{total}** | 100% |")
    return lines


def _qc_lines(qc: QualityControl) -> list[str]:
    avg = f"{qc.average_confidence:.1%}" if qc.average_confidence is not None else "n/a"
    return [
        f"- QC status: **{qc.status}**",
        f"- Mean confidence: {avg}; low-confidence detections "
        f"(< {qc.low_confidence_threshold}): {qc.low_confidence_count}",
        f"- Overlapping box pairs (IoU > {qc.overlap_iou_threshold}): {qc.overlapping_pair_count}",
        f"- Cells touching the image border: {qc.edge_truncated_count}",
    ]


def template_report(prediction: Prediction, qc: QualityControl) -> str:
    total = len(prediction.detections)
    lines = [
        "### Summary",
        f"**{prediction.model_name}** detected **{total} cells** in "
        f"`{prediction.filename}` ({prediction.image_width}x{prediction.image_height}) "
        f"in {prediction.inference_ms:.0f} ms. QC status: **{qc.status}**.",
        "",
        *_breakdown_lines(prediction),
        "",
        "### Quality Assessment",
        *_qc_lines(qc),
    ]
    for flag in qc.flags:
        lines.append(f"- ⚠️ {FLAG_TEXT.get(flag, flag)}")
    lines += ["", "### Recommended Reviewer Actions"]
    if qc.status == "PASS":
        lines.append("- No automated issues found. Spot-check a sample of boxes.")
    else:
        lines.append("- Visually verify flagged regions before using these counts.")
        if "OVERLAPPING_BOXES" in qc.flags:
            lines.append("- Check clumped RBC areas for duplicate boxes.")
        if qc.low_confidence_count:
            lines.append("- Raise the confidence threshold and compare the counts.")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def render_llm_report(
    prediction: Prediction, qc: QualityControl, narrative: LLMNarrative, validation: ReportValidation
) -> str:
    lines = [
        "### Summary",
        narrative.summary.strip(),
        "",
        *_breakdown_lines(prediction),
        "",
        "### Quality Assessment",
        *_qc_lines(qc),
        "",
        narrative.quality_assessment.strip(),
        "",
        "### Recommended Reviewer Actions",
        *[f"- {a.strip()}" for a in narrative.reviewer_actions],
        "",
        f"_Narrative written by {validation.llm_model} (structured output). Tables and measurements are "
        f"computed by the detector and QC. The {validation.numbers_checked} numeric value(s) in the "
        f"narrative were checked for traceability to detector outputs, and reported cell counts were "
        f"cross-checked. This does not guarantee the correctness of all interpretations or statements._",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def _call_gemini(payload: dict, settings: Settings) -> str:
    """Return Gemini's raw JSON text. Isolated so tests can replace it."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents="Measurements (JSON):\n" + json.dumps(payload, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=LLMNarrative,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response (possibly safety-blocked)")
    return text


def parse_narrative(raw: str) -> LLMNarrative:
    """Parse and schema-validate the LLM output (tolerates a ```json fence)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return LLMNarrative.model_validate_json(cleaned)


def gemini_report(
    prediction: Prediction, qc: QualityControl, settings: Settings
) -> tuple[str | None, ReportValidation]:
    """Return (report, validation); report is None when the narrative is rejected."""
    payload = _payload(prediction, qc)
    narrative = parse_narrative(_call_gemini(payload, settings))
    validation = verify_narrative(narrative, payload, llm_model=settings.gemini_model)
    if validation.status != "verified":
        return None, validation
    return render_llm_report(prediction, qc, narrative, validation), validation


def generate_report(
    prediction: Prediction,
    qc: QualityControl,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[str, str, ReportValidation]:
    """Return (markdown_report, source, validation).

    source is 'gemini' (verified narrative), 'template' (LLM not requested or
    not configured) or 'template_fallback' (LLM failed or was rejected).
    """
    settings = settings or get_settings()
    if not (use_llm and settings.gemini_api_key):
        return template_report(prediction, qc), "template", ReportValidation(status="not_applicable")

    try:
        report, validation = gemini_report(prediction, qc, settings)
    except (ValidationError, ValueError) as exc:
        validation = ReportValidation(
            status="rejected",
            issues=[f"invalid structured output: {type(exc).__name__}"],
            llm_model=settings.gemini_model,
        )
        report = None
    except Exception as exc:  # network, auth, quota, safety block, SDK errors
        logger.warning("Gemini unavailable, using template: %s", type(exc).__name__)
        validation = ReportValidation(
            status="rejected",
            issues=[f"LLM call failed: {type(exc).__name__}"],
            llm_model=settings.gemini_model,
        )
        report = None

    if report is None:
        logger.warning("Gemini narrative rejected: %s", "; ".join(validation.issues))
        return template_report(prediction, qc), "template_fallback", validation
    return report, "gemini", validation
