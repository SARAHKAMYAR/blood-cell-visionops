"""Train a YOLO11 detector on the prepared BCCD dataset.

Examples
--------
    python scripts/train.py --model yolo11n.pt --epochs 100 --name yolo11n_e100
    python scripts/train.py --model yolo11s.pt --epochs 100 --name yolo11s_e100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402

from bloodcell.config import PROJECT_ROOT, get_settings, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default=None, help="Run name under runs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume runs/<name> from last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    name = args.name or f"{Path(args.model).stem}_e{args.epochs}"

    last = PROJECT_ROOT / "runs" / name / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True, device=resolve_device(args.device), workers=0)
        print(f"\nTraining finished -> {PROJECT_ROOT / 'runs' / name}")
        return

    model = YOLO(args.model)
    model.train(
        data=str(settings.dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=resolve_device(args.device),
        workers=0,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        # Microscopy has no canonical orientation: vertical flips and
        # rotations are label-preserving augmentations here.
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        project=str(PROJECT_ROOT / "runs"),
        name=name,
        exist_ok=True,
        plots=True,
    )
    print(f"\nTraining finished -> {PROJECT_ROOT / 'runs' / name}")


if __name__ == "__main__":
    main()
