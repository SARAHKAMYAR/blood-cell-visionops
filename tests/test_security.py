"""Security regression tests (from docs/SECURITY_AUDIT.md findings).

All tests run offline with a fake detector; no network, database or LLM calls.
"""

import asyncio
import io
import json
import shutil
import struct
import time
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bloodcell import api
from bloodcell.config import PROJECT_ROOT
from bloodcell.inference import ModelIntegrityError, verify_model_integrity
from bloodcell.qc import run_quality_control
from bloodcell.report import _payload
from bloodcell.report_validation import LLMNarrative, verify_narrative
from tests.conftest import FakeDetector


def _png_with_header(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims a huge size (decompression-bomb shape)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    body = zlib.compress(b"\x00" * 64)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", body) + chunk(b"IEND", b"")


def _jpeg(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 140)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(sample_prediction, monkeypatch):
    monkeypatch.setattr(api, "Detector", lambda settings=None: FakeDetector(sample_prediction))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api.get_settings.cache_clear()
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c
    api.get_settings.cache_clear()


def _post(client, data, ctype="image/png", name="a.png", headers=None):
    return client.post("/predict", files={"file": (name, data, ctype)}, headers=headers or {})


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def test_decompression_bomb_is_rejected_cleanly(client):
    r = _post(client, _png_with_header(30_000, 30_000))  # 900 MP claimed by a ~100-byte file
    assert r.status_code == 413


def test_pixel_cap_rejects_huge_but_legal_images(client):
    r = _post(client, _png_with_header(9_000, 9_000))  # 81 MP, under Pillow's own bomb limit
    assert r.status_code == 413


def test_upload_size_limit(client):
    r = _post(client, b"\xff\xd8" + b"\x00" * (11 * 1024 * 1024), "image/jpeg", "a.jpg")
    assert r.status_code == 413


def test_error_messages_do_not_echo_client_input(client):
    r = client.post("/predict", files={"file": ("a.txt", b"x", "text/<script>alert(1)</script>")})
    assert r.status_code == 415
    assert "<script>" not in r.text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../../etc/passwd.jpg", "passwd.jpg"),
        ("..\\..\\windows\\system32\\a.jpg", "a.jpg"),
        ("<img src=x onerror=alert(1)>.jpg", "_img src_x onerror_alert_1__.jpg"),
        ("![x](javascript:alert(1)).png", "__x__javascript_alert_1__.png"),
        ("", "upload.jpg"),
        (None, "upload.jpg"),
        ("....", "upload.jpg"),
    ],
)
def test_safe_filename(raw, expected):
    assert api.safe_filename(raw) == expected


def test_filename_is_sanitised_before_use(client):
    r = _post(client, _jpeg(), "image/jpeg", "`<b>x</b>` [a](http://e).jpg")
    name = r.json()["filename"]
    assert "<" not in name and "`" not in name and "(" not in name and "/" not in name


# --------------------------------------------------------------------------- #
# Cross-site requests
# --------------------------------------------------------------------------- #
def test_cross_origin_browser_requests_are_refused(client):
    r = _post(client, _jpeg(), "image/jpeg", "a.jpg", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get("/health", headers={"Origin": "https://evil.example"}).status_code == 403


def test_no_permissive_cors_headers(client):
    r = client.options("/analyses", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") is None


def test_same_origin_and_server_side_requests_still_work(client):
    assert client.get("/health", headers={"Origin": "http://testserver"}).status_code == 200  # Swagger UI
    assert client.get("/health").status_code == 200  # dashboard (server-side, no Origin)


# --------------------------------------------------------------------------- #
# Blocking work stays off the event loop
# --------------------------------------------------------------------------- #
def test_slow_inference_does_not_block_health(sample_prediction, monkeypatch):
    class SlowDetector(FakeDetector):
        def predict(self, image, filename="image.jpg", conf=None):
            time.sleep(1.0)
            return super().predict(image, filename, conf)

    monkeypatch.setattr(api, "Detector", lambda settings=None: SlowDetector(sample_prediction))

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with api.lifespan(api.app), httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            slow = asyncio.create_task(c.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}))
            await asyncio.sleep(0.2)
            start = time.perf_counter()
            health = await c.get("/health")
            elapsed = time.perf_counter() - start
            await slow
            return health.status_code, elapsed

    status, elapsed = asyncio.run(scenario())
    assert status == 200
    assert elapsed < 0.5  # would be ~0.8 s if inference blocked the loop


# --------------------------------------------------------------------------- #
# LLM output cannot carry links, images or HTML
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "See [details](https://evil.example/x) for more.",
        "![x](https://evil.example/pixel.png) Looks fine.",
        "<img src=x onerror=alert(1)> Looks fine.",
        "[click](javascript:alert(1)) Looks fine.",
        "Visit https://evil.example for help.",
        "Visit www.evil.example for help.",
        "Data at data:text/html;base64,AAAA here.",
    ],
)
def test_markup_in_llm_narrative_is_rejected(sample_prediction, text):
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    narrative = LLMNarrative(
        summary="The model detected 4 cells; QC status PASS.", quality_assessment=text, reviewer_actions=["Check boxes."]
    )
    v = verify_narrative(narrative, _payload(sample_prediction, qc))
    assert v.status == "rejected"
    assert any("not allowed" in i for i in v.issues)


def test_plain_prose_with_less_than_is_still_accepted(sample_prediction):
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    narrative = LLMNarrative(
        summary="The model detected 4 cells; QC status PASS.",
        quality_assessment="No detections fell below the 0.5 threshold (confidence < 0.5).",
        reviewer_actions=["Spot-check boxes."],
    )
    assert verify_narrative(narrative, _payload(sample_prediction, qc)).status == "verified"


# --------------------------------------------------------------------------- #
# Model integrity (checkpoints are unpickled)
# --------------------------------------------------------------------------- #
def test_served_models_match_recorded_hashes():
    for name in ("best.pt", "best.onnx"):
        path = PROJECT_ROOT / "models" / name
        if not path.exists():
            pytest.skip("served weights not present")
        assert verify_model_integrity(path) is True


def test_tampered_checkpoint_is_refused_before_loading(tmp_path):
    src = PROJECT_ROOT / "models"
    if not (src / "best.pt").exists():
        pytest.skip("served weights not present")
    shutil.copy(src / "model_info.json", tmp_path / "model_info.json")
    (tmp_path / "best.pt").write_bytes((src / "best.pt").read_bytes() + b"tampered")
    with pytest.raises(ModelIntegrityError):
        verify_model_integrity(tmp_path / "best.pt")


def test_no_recorded_hash_means_no_claim(tmp_path):
    (tmp_path / "model_info.json").write_text(json.dumps({"source_run": "x"}))
    (tmp_path / "best.pt").write_bytes(b"x")
    assert verify_model_integrity(tmp_path / "best.pt") is False


def test_model_metadata_contains_no_local_paths():
    for name in ("best.pt", "best.onnx"):
        path = PROJECT_ROOT / "models" / name
        if not path.exists():
            pytest.skip("served weights not present")
        data = path.read_bytes()
        assert b"/Users/" not in data and b"/home/" not in data
