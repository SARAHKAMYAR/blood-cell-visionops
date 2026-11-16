"""Technical report generation: Gemini LLM with a deterministic fallback.

The LLM never sees the image. It only receives the structured, already
computed measurements (counts, confidences, QC flags), which keeps it
grounded: it explains numbers, it does not invent findings.
"""

from __future__ import annotations

import json
import logging

from bloodcell.config import Settings, get_settings
from bloodcell.schemas import Prediction, QualityControl

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "_Research use only. Automated output, not a clinical diagnosis. "
    "Not validated for patient care._"
)

SYSTEM_INSTRUCTION = """\
You are a laboratory computer-vision QA assistant writing a short technical \
report about an automated blood-smear object-detection run.

Rules:
- Use ONLY the JSON measurements provided. Never invent numbers.
- Do NOT give a medical diagnosis or suggest a disease. You may say whether \
the run looks technically reliable and what a human reviewer should check.
- Explain QC flags in plain language.
- Output GitHub-flavoured Markdown with exactly these sections:
  ### Summary
  ### Detection Breakdown
  ### Quality Assessment
  ### Recommended Reviewer Actions
- Maximum 220 words.
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
    return {
        "filename": prediction.filename,
        "model": prediction.model_name,
        "image_size": [prediction.image_width, prediction.image_height],
        "inference_ms": prediction.inference_ms,
        "counts": prediction.counts,
        "total_detections": len(prediction.detections),
        "quality_control": qc.model_dump(),
    }


def template_report(prediction: Prediction, qc: QualityControl) -> str:
    counts = prediction.counts
    total = len(prediction.detections)
    avg = (
        f"{qc.average_confidence:.1%}" if qc.average_confidence is not None else "n/a"
    )
    lines = [
        "### Summary",
        f"**{prediction.model_name}** detected **{total} cells** in "
        f"`{prediction.filename}` ({prediction.image_width}x{prediction.image_height}) "
        f"in {prediction.inference_ms:.0f} ms. QC status: **{qc.status}**.",
        "",
        "### Detection Breakdown",
    ]
    for name, n in counts.items():
        share = f"{n / total:.0%}" if total else "0%"
        lines.append(f"- **{name}**: {n} ({share})")
    lines += [
        "",
        "### Quality Assessment",
        f"- Mean confidence: {avg}; low-confidence detections "
        f"(< {qc.low_confidence_threshold}): {qc.low_confidence_count}",
        f"- Overlapping box pairs (IoU > {qc.overlap_iou_threshold}): "
        f"{qc.overlapping_pair_count}",
        f"- Cells touching the image border: {qc.edge_truncated_count}",
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


def gemini_report(
    prediction: Prediction, qc: QualityControl, settings: Settings
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=(
            "Measurements (JSON):\n"
            + json.dumps(_payload(prediction, qc), indent=2)
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=800,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return f"{text}\n\n{DISCLAIMER}"


def generate_report(
    prediction: Prediction,
    qc: QualityControl,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> tuple[str, str]:
    """Return (markdown_report, source) where source is 'gemini' or 'template'."""
    settings = settings or get_settings()
    if use_llm and settings.gemini_api_key:
        try:
            return gemini_report(prediction, qc, settings), "gemini"
        except Exception as exc:  # network, quota, safety block...
            logger.warning("Gemini report failed, using template: %s", exc)
    return template_report(prediction, qc), "template"
