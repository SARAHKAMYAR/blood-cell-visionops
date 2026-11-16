"""Save per-image test-set evaluation artefacts for each model.

Two kinds of per-image data are saved, because Ultralytics evaluates and
serves differently:

* ``val_*``: the validator's own per-image match statistics (TP matrix over
  IoU 0.50:0.95, confidences, classes), captured during ``model.val()``. These
  reproduce the reported mAP exactly (validation uses multi-label NMS).
* ``count_*``: per-image cell counts produced through the SERVING
  configuration (``model.predict`` with the API's NMS IoU and image size;
  default = global conf 0.25, calibrated = per-class thresholds with the
  API's confidence floor) plus ground-truth counts. Counts are computed
  exactly as served because NMS settings change which boxes survive.

These artefacts make scripts/bootstrap_eval.py reproducible without a GPU.
Inference only: weights are not modified.

Usage:
    python scripts/export_predictions.py runs/yolo11n_e100 runs/yolo11s_e100 runs/blood_cells_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bloodcell.config import CLASS_NAMES, PROJECT_ROOT, get_settings, resolve_device  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "predictions"


def load_ground_truth(data_root: Path, images: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_idx, boxes, cls = [], [], []
    for i, path in enumerate(images):
        w, h = Image.open(path).size
        for line in (data_root / "labels" / "test" / f"{path.stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            c, xc, yc, bw, bh = map(float, line.split())
            img_idx.append(i)
            cls.append(int(c))
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(img_idx, dtype=np.int32), np.array(boxes, dtype=np.float32), np.array(cls, dtype=np.int16)


def serve(model: YOLO, path: Path, conf: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Mirror bloodcell.inference.Detector.predict: same NMS IoU and image size as the API."""
    settings = get_settings()
    r = model.predict(str(path), conf=conf, iou=settings.iou_threshold, imgsz=settings.imgsz,
                      device=device, verbose=False)[0]
    return r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()


def capture_val_stats(model: YOLO, device: str) -> tuple[list[dict], dict]:
    """Run model.val() on the test split and record per-image match statistics."""
    captured: list[dict] = []

    def patch(validator) -> None:
        metrics = validator.metrics
        if getattr(metrics, "_bcv_patched", False):
            return
        original = metrics.update_stats

        def update_stats(stat: dict) -> None:
            captured.append({k: (v if k == "im_name" else np.asarray(v).copy()) for k, v in stat.items()})
            return original(stat)

        metrics.update_stats = update_stats
        metrics._bcv_patched = True

    model.add_callback("on_val_batch_start", patch)
    res = model.val(
        data=str(get_settings().dataset_yaml), split="test", imgsz=640, batch=8,
        device=device, workers=0, plots=False, verbose=False,
    )
    reported = {"mAP50": float(res.box.map50), "mAP50_95": float(res.box.map)}
    return captured, reported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_root = get_settings().dataset_yaml.parent
    images = sorted((data_root / "images" / "test").glob("*.jpg"))
    if not images:
        sys.exit(f"No test images in {data_root}/images/test: run scripts/prepare_dataset.py first")
    gt_idx, gt_boxes, gt_cls = load_ground_truth(data_root, images)
    device = resolve_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        run = run if run.is_absolute() else PROJECT_ROOT / run
        weights = run / "weights" / "best.pt"
        model = YOLO(str(weights))
        settings = get_settings()
        thresholds = json.loads((PROJECT_ROOT / "reports" / "metrics" / f"{run.name}.json").read_text())[
            "counting"]["calibrated"]["conf"]
        thr = np.array([thresholds[n] for n in CLASS_NAMES])

        count_default, count_calibrated = [], []
        for path in images:
            k, c = serve(model, path, settings.conf_threshold, device)
            count_default.append(np.bincount(k, minlength=len(CLASS_NAMES)))
            k, c = serve(model, path, float(thr.min()), device)
            count_calibrated.append(np.bincount(k[c >= thr[k]], minlength=len(CLASS_NAMES)))
        count_gt = np.stack([np.bincount(gt_cls[gt_idx == i], minlength=len(CLASS_NAMES)) for i in range(len(images))])
        val_stats, val_reported = capture_val_stats(YOLO(str(weights)), device)
        names = [p.name for p in images]
        if sorted(st["im_name"] for st in val_stats) != names:
            sys.exit(f"{run.name}: validator saw a different image set than the test folder")
        v_img = np.concatenate([np.full(len(st["conf"]), names.index(st["im_name"]), np.int32) for st in val_stats])
        v_tgt_img = np.concatenate(
            [np.full(len(st["target_cls"]), names.index(st["im_name"]), np.int32) for st in val_stats]
        )
        out = OUT / f"test_{run.name}.npz"
        np.savez_compressed(
            out,
            image_names=np.array([p.name for p in images]),
            count_default=np.stack(count_default).astype(np.int32),
            count_calibrated=np.stack(count_calibrated).astype(np.int32),
            count_gt=count_gt.astype(np.int32),
            val_img=v_img,
            val_tp=np.concatenate([st["tp"] for st in val_stats]).astype(bool),
            val_conf=np.concatenate([st["conf"] for st in val_stats]).astype(np.float32),
            val_pred_cls=np.concatenate([st["pred_cls"] for st in val_stats]).astype(np.int16),
            val_target_img=v_tgt_img,
            val_target_cls=np.concatenate([st["target_cls"] for st in val_stats]).astype(np.int16),
        )
        meta = {
            "run": run.name,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "split": "test",
            "images": len(images),
            "labels": int(len(gt_cls)),
            "class_names": list(CLASS_NAMES),
            "serving_settings": {"nms_iou": settings.iou_threshold, "imgsz": settings.imgsz,
                                 "default_conf": settings.conf_threshold, "calibrated_thresholds": thresholds},
            "val_reported": val_reported,
            "val_note": "val_* arrays are Ultralytics validator stats (multi-label NMS, rect batches)",
            "device": device,
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"{run.name}: {len(images)} images, val mAP50-95 {val_reported['mAP50_95']:.4f} -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
