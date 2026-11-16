"""Central configuration.

Every setting can be overridden with an environment variable (or a `.env`
file at the project root), so the same code runs on a laptop, in Docker and
in CI without editing source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES: tuple[str, ...] = ("RBC", "WBC", "Platelets")


def _load_dotenv() -> None:
    """Load `.env` if python-dotenv is installed (optional dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str | None = None) -> str:
    """Pick the best available accelerator: CUDA > Apple MPS > CPU."""
    requested = (requested or os.getenv("BCV_DEVICE") or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Settings:
    model_path: Path
    dataset_yaml: Path
    raw_dataset_dir: Path
    images_dir: Path
    device: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    low_conf_threshold: float
    overlap_iou_threshold: float
    database_url: str | None
    gemini_api_key: str | None
    gemini_model: str
    max_upload_mb: int
    max_image_pixels: int
    allowed_origins: tuple[str, ...]
    use_calibrated: bool
    class_names: tuple[str, ...] = field(default=CLASS_NAMES)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        model_path=_env_path(
            "BCV_MODEL_PATH", PROJECT_ROOT / "models" / "best.pt"
        ),
        dataset_yaml=_env_path(
            "BCV_DATASET_YAML", PROJECT_ROOT / "data" / "blood_cells" / "data.yaml"
        ),
        raw_dataset_dir=_env_path(
            "BCV_RAW_DATASET", PROJECT_ROOT / "BCCD_Dataset" / "BCCD"
        ),
        images_dir=_env_path(
            "BCV_IMAGES_DIR", PROJECT_ROOT / "BCCD_Dataset" / "BCCD" / "JPEGImages"
        ),
        device=resolve_device(),
        conf_threshold=float(os.getenv("BCV_CONF", "0.25")),
        # NMS IoU used for serving AND for counting evaluation/calibration
        # (scripts read this same setting so the two cannot drift apart).
        iou_threshold=float(os.getenv("BCV_IOU", "0.5")),
        imgsz=int(os.getenv("BCV_IMGSZ", "640")),
        low_conf_threshold=float(os.getenv("BCV_LOW_CONF", "0.5")),
        overlap_iou_threshold=float(os.getenv("BCV_OVERLAP_IOU", "0.5")),
        database_url=os.getenv("DATABASE_URL") or None,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        max_upload_mb=int(os.getenv("BCV_MAX_UPLOAD_MB", "10")),
        # Decoded-size cap: a 10 MB compressed file can still expand to gigabytes of pixels.
        max_image_pixels=int(os.getenv("BCV_MAX_PIXELS", str(40_000_000))),
        # Extra browser origins allowed to call the API (same-origin, e.g. /docs, is always allowed).
        allowed_origins=tuple(o.strip() for o in os.getenv("BCV_ALLOWED_ORIGINS", "").split(",") if o.strip()),
        use_calibrated=os.getenv("BCV_USE_CALIBRATED", "true").lower() in {"1", "true", "yes"},
    )
