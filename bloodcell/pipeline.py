"""End-to-end analysis: detect -> quality control -> report."""

from __future__ import annotations

from PIL import Image

from bloodcell.config import Settings, get_settings
from bloodcell.inference import Detector
from bloodcell.qc import run_quality_control
from bloodcell.report import generate_report
from bloodcell.schemas import Analysis


def analyze_image(
    detector: Detector,
    image: Image.Image,
    filename: str,
    settings: Settings | None = None,
    use_llm: bool = True,
) -> Analysis:
    settings = settings or get_settings()
    prediction = detector.predict(image, filename=filename)
    qc = run_quality_control(
        prediction.detections,
        prediction.image_width,
        prediction.image_height,
        low_conf_threshold=settings.low_conf_threshold,
        overlap_iou_threshold=settings.overlap_iou_threshold,
    )
    report, source, validation = generate_report(prediction, qc, settings, use_llm=use_llm)
    return Analysis(
        filename=prediction.filename,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        model_name=prediction.model_name,
        inference_ms=prediction.inference_ms,
        total_detections=len(prediction.detections),
        counts=prediction.counts,
        detections=prediction.detections,
        quality_control=qc,
        ai_report=report,
        report_source=source,
        report_validation=validation,
    )
