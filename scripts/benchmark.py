"""Latency benchmark across models, formats and devices.

Usage:
    python scripts/benchmark.py models/best.pt models/best.onnx runs/yolo11s_e100/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bloodcell.config import PROJECT_ROOT, get_settings, resolve_device  # noqa: E402


def bench(weights: Path, device: str, images: list[Path], warmup: int = 5) -> dict:
    model = YOLO(str(weights), task="detect")
    frames = [Image.open(p).convert("RGB") for p in images]
    for frame in frames[:warmup]:
        model.predict(frame, imgsz=640, device=device, verbose=False)
    times = []
    for frame in frames:
        start = time.perf_counter()
        model.predict(frame, imgsz=640, device=device, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "weights": str(weights.relative_to(PROJECT_ROOT)) if weights.is_relative_to(PROJECT_ROOT) else str(weights),
        "format": weights.suffix.lstrip("."),
        "device": device,
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 2),
        "fps": round(1000 / statistics.mean(times), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="+", type=Path)
    parser.add_argument("-n", type=int, default=50, help="Number of test images")
    args = parser.parse_args()

    test_dir = get_settings().dataset_yaml.parent / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))[: args.n]
    accel = resolve_device()

    rows = []
    for w in args.weights:
        w = w if w.is_absolute() else PROJECT_ROOT / w
        devices = ["cpu"] if w.suffix == ".onnx" else sorted({"cpu", accel})
        for device in devices:
            row = bench(w, device, images)
            rows.append(row)
            print(row)

    out = PROJECT_ROOT / "reports" / "metrics" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("\n| Weights | Format | Device | Mean (ms) | p95 (ms) | FPS |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['weights']} | {r['format']} | {r['device']} | {r['mean_ms']} | {r['p95_ms']} | {r['fps']} |")


if __name__ == "__main__":
    main()
