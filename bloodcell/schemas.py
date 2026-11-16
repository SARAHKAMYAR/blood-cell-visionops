"""Typed data contracts shared by the API, pipeline, database and dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from bloodcell.config import CLASS_NAMES


class Detection(BaseModel):
    cell_class: str = Field(alias="class", description="RBC, WBC or Platelets")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(
        min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels"
    )

    model_config = {"populate_by_name": True}

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class QualityControl(BaseModel):
    average_confidence: float | None
    min_confidence: float | None
    low_confidence_threshold: float
    low_confidence_count: int
    overlap_iou_threshold: float
    overlapping_pair_count: int
    edge_truncated_count: int
    flags: list[str] = Field(default_factory=list)
    status: str = Field(description="PASS, REVIEW or FAIL")


class Prediction(BaseModel):
    filename: str
    image_width: int
    image_height: int
    model_name: str
    inference_ms: float
    detections: list[Detection]

    @property
    def counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for det in self.detections:
            counts[det.cell_class] = counts.get(det.cell_class, 0) + 1
        return counts


class ReportValidation(BaseModel):
    """Outcome of deterministic verification of an LLM narrative."""

    status: Literal["verified", "rejected", "not_applicable"]
    numbers_checked: int = 0
    issues: list[str] = Field(default_factory=list)
    llm_model: str | None = None


class Analysis(BaseModel):
    """A full analysis: detections + QC + generated report."""

    id: int | None = None
    filename: str
    image_width: int
    image_height: int
    model_name: str
    inference_ms: float
    total_detections: int
    counts: dict[str, int]
    detections: list[Detection]
    quality_control: QualityControl
    ai_report: str
    report_source: str = Field(description="gemini, template or template_fallback")
    report_validation: ReportValidation | None = None
    created_at: datetime | None = None
