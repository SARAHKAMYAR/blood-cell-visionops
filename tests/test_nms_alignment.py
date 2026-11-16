"""Regression: the served NMS IoU must equal the one counting thresholds were calibrated with.

Counting thresholds (models/model_info.json) were selected on the val split with
NMS IoU 0.5; changing NMS changes which boxes survive and silently invalidates
them. These tests spy on the YOLO call made by the real Detector class (no
weights are executed), so they run in CI.
"""

import json
from dataclasses import replace

import pytest
from PIL import Image

from bloodcell.config import PROJECT_ROOT, get_settings

CALIBRATION_NMS_IOU = 0.5  # value used by scripts/evaluate.py when thresholds were selected
ARTEFACT = PROJECT_ROOT / "reports" / "predictions" / "test_yolo11n_e100.json"


class SpyYOLO:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.names = {0: "RBC", 1: "WBC", 2: "Platelets"}

    def predict(self, *args, **kwargs):
        SpyYOLO.calls.append(kwargs)

        class _Boxes:
            xyxy = cls = conf = type("T", (), {"tolist": staticmethod(list)})()

        return [type("R", (), {"boxes": _Boxes()})()]


@pytest.fixture
def spy(monkeypatch):
    import ultralytics

    SpyYOLO.calls = []
    monkeypatch.setattr(ultralytics, "YOLO", SpyYOLO)
    return SpyYOLO


def test_default_serving_iou_equals_calibration_iou():
    assert get_settings().iou_threshold == CALIBRATION_NMS_IOU


def test_detector_passes_configured_iou_to_model(spy):
    from bloodcell.inference import Detector

    settings = get_settings()
    if not settings.model_path.exists():
        pytest.skip("served weights not present")
    Detector(settings).predict(Image.new("RGB", (64, 64)))
    served = spy.calls[-1]  # last call is the real request (first is warm-up)
    assert served["iou"] == CALIBRATION_NMS_IOU


def test_detector_follows_iou_override(spy):
    from bloodcell.inference import Detector

    settings = replace(get_settings(), iou_threshold=0.65)
    if not settings.model_path.exists():
        pytest.skip("served weights not present")
    Detector(settings).predict(Image.new("RGB", (64, 64)))
    assert spy.calls[-1]["iou"] == 0.65


@pytest.mark.skipif(not ARTEFACT.exists(), reason="evaluation artefacts not exported")
def test_evaluation_artefacts_were_produced_with_calibration_iou():
    assert json.loads(ARTEFACT.read_text())["serving_settings"]["nms_iou"] == CALIBRATION_NMS_IOU
