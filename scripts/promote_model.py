"""Promote a trained run to the serving model and export it to ONNX.

Copies runs/<run>/weights/best.pt -> models/best.pt, exports models/best.onnx
and writes models/model_info.json (provenance for the model card).

Usage:
    python scripts/promote_model.py runs/yolo11n_e100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402

from bloodcell.config import PROJECT_ROOT  # noqa: E402

MODELS = PROJECT_ROOT / "models"


def _relative(value):
    """Rewrite absolute paths inside the project to project-relative ones."""
    if isinstance(value, str) and str(PROJECT_ROOT) in value:
        return value.replace(str(PROJECT_ROOT) + "/", "").replace(str(PROJECT_ROOT), ".")
    return value


def scrub_checkpoint(path: Path) -> None:
    """Remove local filesystem paths and git info from an Ultralytics .pt file."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own freshly trained file
    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = {k: _relative(v) for k, v in ckpt["train_args"].items()}
    ckpt.pop("git", None)
    for key in ("model", "ema"):  # the pickled nn.Module carries its own copy of the args
        module = ckpt.get(key)
        if module is None:
            continue
        if isinstance(getattr(module, "args", None), dict):
            module.args = {k: _relative(v) for k, v in module.args.items()}
        if hasattr(module, "pt_path"):
            module.pt_path = _relative(module.pt_path)
    torch.save(ckpt, path)


def scrub_onnx(path: Path) -> None:
    """Remove local filesystem paths from ONNX metadata (weights untouched)."""
    import onnx

    model = onnx.load(str(path))
    for prop in model.metadata_props:
        prop.value = _relative(prop.value)
    onnx.save(model, str(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()

    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    src = run / "weights" / "best.pt"
    MODELS.mkdir(exist_ok=True)
    dst = MODELS / "best.pt"
    shutil.copy2(src, dst)
    scrub_checkpoint(dst)
    print(f"Promoted {src} -> {dst}")

    info = {
        "source_run": run.name,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pt_sha256": sha256(dst),
    }
    metrics_file = PROJECT_ROOT / "reports" / "metrics" / f"{run.name}.json"
    if metrics_file.exists():
        info["test_metrics"] = json.loads(metrics_file.read_text())
        info["count_thresholds"] = info["test_metrics"]["counting"]["calibrated"]["conf"]

    if not args.no_onnx:
        onnx_path = Path(YOLO(str(dst)).export(format="onnx", imgsz=640, simplify=True, dynamic=False))
        scrub_onnx(onnx_path)
        info["onnx_sha256"] = sha256(onnx_path)
        print(f"Exported ONNX -> {onnx_path}")

    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
