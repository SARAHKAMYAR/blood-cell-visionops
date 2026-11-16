"""Integration test against the real trained model (skipped if weights absent)."""

from pathlib import Path

import pytest
from PIL import Image

from bloodcell.config import PROJECT_ROOT, get_settings

WEIGHTS = get_settings().model_path
SAMPLE = PROJECT_ROOT / "BCCD_Dataset" / "example.jpg"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not WEIGHTS.exists(), reason="trained weights not available"),
    pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not available"),
]


def test_real_model_detects_cells():
    from bloodcell.inference import Detector

    pred = Detector().predict(Image.open(SAMPLE), filename=Path(SAMPLE).name)
    counts = pred.counts
    assert counts["RBC"] >= 5
    assert counts["WBC"] >= 1
    assert all(0 <= d.confidence <= 1 for d in pred.detections)


def test_calibrated_thresholds_filter_per_class():
    from bloodcell.inference import Detector

    det = Detector()
    if not det.class_thresholds:
        pytest.skip("no calibrated thresholds in model_info.json")
    pred = det.predict(Image.open(SAMPLE))
    for d in pred.detections:
        assert d.confidence >= det.class_thresholds[d.cell_class]
    # An explicit conf overrides calibration.
    low = det.predict(Image.open(SAMPLE), conf=0.05)
    assert len(low.detections) >= len(pred.detections)
