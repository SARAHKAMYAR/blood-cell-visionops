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
        info["onnx_sha256"] = sha256(onnx_path)
        print(f"Exported ONNX -> {onnx_path}")

    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
