"""FastAPI service.

Run locally:  uvicorn bloodcell.api:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from bloodcell import __version__
from bloodcell.config import get_settings
from bloodcell.inference import Detector
from bloodcell.pipeline import analyze_image
from bloodcell.schemas import Analysis, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bloodcell.api")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading model %s on %s", settings.model_path, settings.device)
    state["detector"] = Detector(settings)
    if settings.database_url:
        try:
            from bloodcell.db import init_db

            init_db()
            state["db"] = True
        except Exception as exc:
            logger.warning("Database unavailable, persistence disabled: %s", exc)
            state["db"] = False
    else:
        state["db"] = False
    yield
    state.clear()


app = FastAPI(
    title="Blood Cell VisionOps API",
    description=(
        "Detect and count RBCs, WBCs and platelets in blood-smear microscopy "
        "images, with automated quality control and an LLM-written technical "
        "report. **Research use only.**"
    ),
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


async def _read_image(file: UploadFile) -> Image.Image:
    settings = get_settings()
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported type {file.content_type}; use JPEG/PNG/BMP/TIFF")
    contents = await file.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {settings.max_upload_mb} MB")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Invalid or corrupted image file") from exc
    return image.convert("RGB")


def _detector() -> Detector:
    detector = state.get("detector")
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    return detector


@app.get("/health", tags=["system"])
def health():
    settings = get_settings()
    detector = state.get("detector")
    return {
        "status": "ok" if detector else "loading",
        "version": __version__,
        "model": detector.model_name if detector else None,
        "device": settings.device,
        "thresholds": {
            "mode": detector.threshold_mode if detector else None,
            "per_class": detector.class_thresholds if detector else {},
            "default": settings.conf_threshold,
        },
        "classes": list(settings.class_names),
        "database": state.get("db", False),
        "llm_reports": bool(settings.gemini_api_key),
    }


@app.post("/predict", response_model=Prediction, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    conf: float | None = Query(None, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """Run detection only (fast path, nothing stored)."""
    image = await _read_image(file)
    return _detector().predict(image, filename=file.filename or "upload.jpg", conf=conf)


@app.post("/analyze", response_model=Analysis, tags=["inference"])
async def analyze(
    file: UploadFile = File(...),
    persist: bool = Query(True, description="Store the result in PostgreSQL"),
    llm: bool = Query(True, description="Use Gemini for the report if configured"),
):
    """Detection + quality control + technical report (+ optional storage)."""
    image = await _read_image(file)
    analysis = analyze_image(
        _detector(), image, filename=file.filename or "upload.jpg", use_llm=llm
    )
    if persist and state.get("db"):
        from bloodcell.db import encode_jpeg, save_analysis

        analysis.id = save_analysis(analysis, image_jpeg=encode_jpeg(image))
    return analysis


@app.get("/analyses", tags=["history"])
def analyses(limit: int = Query(50, ge=1, le=500)):
    if not state.get("db"):
        raise HTTPException(503, "Database not configured")
    from bloodcell.db import list_analyses

    return list_analyses(limit)


@app.get("/analyses/{analysis_id}", tags=["history"])
def analysis_detail(analysis_id: int):
    if not state.get("db"):
        raise HTTPException(503, "Database not configured")
    from bloodcell.db import get_analysis

    row = get_analysis(analysis_id)
    if row is None:
        raise HTTPException(404, "Analysis not found")
    return row
