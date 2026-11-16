"""Model loading and inference."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from PIL import Image

from bloodcell.config import Settings, get_settings
from bloodcell.schemas import Detection, Prediction


class ModelIntegrityError(RuntimeError):
    """The weights file does not match the hash recorded when it was promoted."""


HASH_KEYS = {".pt": "pt_sha256", ".onnx": "onnx_sha256"}


def verify_model_integrity(path: Path) -> bool:
    """Check `path` against the SHA-256 in model_info.json beside it.

    `.pt` checkpoints are unpickled by Ultralytics (``weights_only=False``), so a
    tampered file could execute code when loaded. If model_info.json records a
    hash for this file type, the file must match it before it is loaded.
    Returns True when a recorded hash was verified, False when none is recorded.
    """
    info_file = path.parent / "model_info.json"
    key = HASH_KEYS.get(path.suffix)
    if key is None or not info_file.exists():
        return False
    expected = json.loads(info_file.read_text()).get(key)
    if not expected:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ModelIntegrityError(
            f"{path.name} does not match the {key} recorded in model_info.json; refusing to load it."
        )
    return True


class Detector:
    """Thin wrapper around an Ultralytics YOLO model (.pt or .onnx)."""

    def __init__(self, settings: Settings | None = None):
        from ultralytics import YOLO

        self.settings = settings or get_settings()
        path = self.settings.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. Train a model "
                "(scripts/train.py) and run scripts/promote_model.py, or set "
                "BCV_MODEL_PATH."
            )
        verify_model_integrity(path)  # before any unpickling happens
        self.model = YOLO(str(path), task="detect")
        self.model_name = self._describe(path)
        self.names: dict[int, str] = self.model.names
        self.class_thresholds = self._load_thresholds(path) if self.settings.use_calibrated else {}
        # Warm-up so the first real request isn't penalised.
        self.model.predict(
            Image.new("RGB", (64, 64)),
            imgsz=self.settings.imgsz,
            device=self._device,
            verbose=False,
        )

    @staticmethod
    def _describe(path: Path) -> str:
        """Human-readable model id, e.g. 'yolo11n_e100 (onnx)'."""
        info = path.parent / "model_info.json"
        if info.exists():
            try:
                run = json.loads(info.read_text()).get("source_run")
                if run:
                    return f"{run} ({path.suffix.lstrip('.')})"
            except (OSError, ValueError):
                pass
        if path.stem == "best" and path.parent.name == "weights":
            return path.parent.parent.name
        return path.stem

    @staticmethod
    def _load_thresholds(path: Path) -> dict[str, float]:
        """Per-class confidence thresholds calibrated on the validation split
        (written by scripts/promote_model.py)."""
        info = path.parent / "model_info.json"
        try:
            return {k: float(v) for k, v in json.loads(info.read_text()).get("count_thresholds", {}).items()}
        except (OSError, ValueError):
            return {}

    @property
    def threshold_mode(self) -> str:
        return "calibrated-per-class" if self.class_thresholds else "global"

    @property
    def _device(self) -> str:
        # ONNX Runtime does not use torch device strings like "mps".
        if self.settings.model_path.suffix == ".onnx":
            return "cpu"
        return self.settings.device

    def predict(
        self,
        image: Image.Image,
        filename: str = "image.jpg",
        conf: float | None = None,
    ) -> Prediction:
        image = image.convert("RGB")
        # An explicit `conf` always wins; otherwise use calibrated per-class
        # thresholds when available, else the global default.
        per_class = self.class_thresholds if conf is None else {}
        base_conf = conf if conf is not None else self.settings.conf_threshold
        if per_class:
            base_conf = min(per_class.values())
        start = time.perf_counter()
        result = self.model.predict(
            source=image,
            conf=base_conf,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.imgsz,
            device=self._device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = [
            Detection(
                cell_class=self.names[int(cls)],
                confidence=round(float(score), 4),
                bbox=[round(float(v), 2) for v in xyxy],
            )
            for xyxy, cls, score in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            )
            if not per_class or score >= per_class.get(self.names[int(cls)], base_conf)
        ]
        return Prediction(
            filename=Path(filename).name,
            image_width=image.width,
            image_height=image.height,
            model_name=self.model_name,
            inference_ms=round(elapsed_ms, 2),
            detections=detections,
        )
