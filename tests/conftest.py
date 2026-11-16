from __future__ import annotations

import pytest

from bloodcell.schemas import Detection, Prediction


def det(cls: str, conf: float, bbox: list[float]) -> Detection:
    return Detection(cell_class=cls, confidence=conf, bbox=bbox)


@pytest.fixture
def sample_prediction() -> Prediction:
    return Prediction(
        filename="BloodImage_00001.jpg",
        image_width=640,
        image_height=480,
        model_name="test-model",
        inference_ms=12.3,
        detections=[
            det("RBC", 0.91, [100, 100, 160, 160]),
            det("RBC", 0.88, [300, 200, 360, 260]),
            det("WBC", 0.95, [200, 150, 320, 270]),
            det("Platelets", 0.62, [50, 300, 70, 320]),
        ],
    )


class FakeDetector:
    """Stands in for the YOLO model so API tests run without weights."""

    model_name = "fake-yolo"
    threshold_mode = "global"
    class_thresholds: dict = {}

    def __init__(self, prediction: Prediction):
        self._prediction = prediction

    def predict(self, image, filename="image.jpg", conf=None):
        p = self._prediction.model_copy(deep=True)
        p.filename = filename
        p.image_width, p.image_height = image.size
        if conf is not None:
            p.detections = [d for d in p.detections if d.confidence >= conf]
        return p
