"""Evaluate trained models on the held-out TEST split.

Reports two kinds of metrics:
  1. Detection: precision, recall, mAP50, mAP50-95 (overall + per class).
  2. Counting: per-image absolute count error vs. ground truth, which is what
     a cell-counting product actually delivers to users.

Usage:
    python scripts/evaluate.py runs/yolo11n_e100 runs/yolo11s_e100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bloodcell.config import CLASS_NAMES, PROJECT_ROOT, get_settings, resolve_device  # noqa: E402

OUT_DIR = PROJECT_ROOT / "reports" / "metrics"


def ground_truth_counts(label_file: Path) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=int)
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def _collect(model: YOLO, data_root: Path, split: str, device: str):
    """Ground-truth counts and low-threshold predictions for every image in a split."""
    images = sorted((data_root / "images" / split).glob("*.jpg"))
    gts, preds = [], []
    for image in images:
        gts.append(ground_truth_counts(data_root / "labels" / split / f"{image.stem}.txt"))
        r = model.predict(str(image), conf=0.01, device=device, verbose=False)[0]
        preds.append((r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()))
    return np.array(gts), preds


def _counts_at(preds, conf: float) -> np.ndarray:
    return np.array([np.bincount(c[s >= conf], minlength=len(CLASS_NAMES)) for c, s in preds])


def _summary(gt: np.ndarray, pred: np.ndarray, conf: float) -> dict:
    err = pred - gt
    return {
        "conf": round(conf, 2),
        "count_mae": {n: round(float(np.abs(err[:, i]).mean()), 3) for i, n in enumerate(CLASS_NAMES)},
        "count_bias": {n: round(float(err[:, i].mean()), 3) for i, n in enumerate(CLASS_NAMES)},
        "gt_mean": {n: round(float(gt[:, i].mean()), 2) for i, n in enumerate(CLASS_NAMES)},
        "wbc_exact_match": round(float((err[:, 1] == 0).mean()), 3),
    }


def counting_metrics(model: YOLO, data_root: Path, device: str, conf: float) -> dict:
    """Counting error on TEST at the default threshold and at per-class
    thresholds calibrated on VAL (to minimise count MAE, no test leakage)."""
    val_gt, val_preds = _collect(model, data_root, "val", device)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    calibrated = {}
    for i, name in enumerate(CLASS_NAMES):
        maes = [np.abs(_counts_at(val_preds, t)[:, i] - val_gt[:, i]).mean() for t in grid]
        calibrated[name] = float(grid[int(np.argmin(maes))])

    test_gt, test_preds = _collect(model, data_root, "test", device)
    default = _summary(test_gt, _counts_at(test_preds, conf), conf)
    cal_counts = np.stack(
        [_counts_at(test_preds, calibrated[n])[:, i] for i, n in enumerate(CLASS_NAMES)], axis=1
    )
    cal = _summary(test_gt, cal_counts, float("nan"))
    cal["conf"] = calibrated
    return {"images": len(test_gt), **default, "calibrated": cal}


def evaluate(run_dir: Path, device: str, conf: float) -> dict:
    settings = get_settings()
    weights = run_dir / "weights" / "best.pt"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(settings.dataset_yaml),
        split="test",
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project=str(PROJECT_ROOT / "runs" / "eval"),
        name=run_dir.name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    box = metrics.box
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        p, r, ap50, ap = box.class_result(i)
        per_class[name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        }
    data_root = Path(settings.dataset_yaml).parent
    return {
        "run": run_dir.name,
        "params_m": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
        "weights_mb": round(weights.stat().st_size / 1e6, 1),
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "counting": counting_metrics(model, data_root, device, conf),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    device = resolve_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        run = run if run.is_absolute() else PROJECT_ROOT / run
        res = evaluate(run, device, args.conf)
        (OUT_DIR / f"{run.name}.json").write_text(json.dumps(res, indent=2))
        results.append(res)

    print("\n| Model | Params (M) | Precision | Recall | mAP50 | mAP50-95 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        o = r["overall"]
        print(
            f"| {r['run']} | {r['params_m']} | {o['precision']:.3f} | {o['recall']:.3f} | "
            f"{o['mAP50']:.3f} | {o['mAP50_95']:.3f} |"
        )
    print("\n| Model | Threshold | RBC MAE (bias) | WBC MAE | Platelet MAE (bias) |")
    print("|---|---|---|---|---|")
    for r in results:
        for label, c in (("0.25", r["counting"]), ("val-calibrated", r["counting"]["calibrated"])):
            m, b = c["count_mae"], c["count_bias"]
            print(
                f"| {r['run']} | {label} | {m['RBC']:.2f} ({b['RBC']:+.2f}) | {m['WBC']:.2f} | "
                f"{m['Platelets']:.2f} ({b['Platelets']:+.2f}) |"
            )
    print(f"\nSaved JSON to {OUT_DIR}")


if __name__ == "__main__":
    main()
