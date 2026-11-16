import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bloodcell import api
from tests.conftest import FakeDetector


@pytest.fixture
def client(sample_prediction, monkeypatch):
    monkeypatch.setattr(api, "Detector", lambda settings=None: FakeDetector(sample_prediction))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api.get_settings.cache_clear()
    with TestClient(api.app) as c:
        yield c
    api.get_settings.cache_clear()


def _jpeg(size=(640, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 140)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-yolo"
    assert body["classes"] == ["RBC", "WBC", "Platelets"]
    assert body["database"] is False


def test_predict(client):
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "a.jpg"
    assert len(body["detections"]) == 4
    assert body["detections"][0]["class"] == "RBC"
    assert "X-Process-Time-Ms" in r.headers


def test_predict_conf_filter(client):
    r = client.post("/predict?conf=0.9", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert len(r.json()["detections"]) == 2


def test_analyze_returns_qc_and_report(client):
    r = client.post("/analyze?llm=false", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"RBC": 2, "WBC": 1, "Platelets": 1}
    assert body["quality_control"]["status"] == "PASS"
    assert body["report_source"] == "template"
    assert body["id"] is None  # no DB configured


def test_rejects_wrong_content_type(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/predict", files={"file": ("a.jpg", b"not-a-jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_history_requires_db(client):
    assert client.get("/analyses").status_code == 503
