"""Image-level bootstrap confidence intervals and paired model comparisons.

Inputs are the saved per-image test predictions (scripts/export_predictions.py)
and the val-calibrated counting thresholds in reports/metrics/<run>.json.

Method
------
* Resampling unit: the IMAGE (cells within an image are correlated, so
  resampling boxes would understate uncertainty).
* B resamples of the 56 test images with replacement, fixed seed.
* For each resample, the same image indices are used for every model
  (paired design), so model differences are estimated on identical data.
* mAP is recomputed per resample with Ultralytics' own matching
  (per-image TP stats captured from model.val()) and ap_per_class, so point estimates are
  comparable with `model.val()`.
* 95% percentile intervals. Interpretation: an interval for a difference that
  excludes 0 is "resolved"; one that includes 0 is "not resolved by this test
  set". It is NOT evidence of equivalence.

Usage:
    python scripts/bootstrap_eval.py --runs yolo11n_e100 yolo11s_e100 blood_cells_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from ultralytics.utils.metrics import ap_per_class  # noqa: E402

from bloodcell.config import CLASS_NAMES, PROJECT_ROOT  # noqa: E402

PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
DOC = PROJECT_ROOT / "docs" / "EVALUATION_UNCERTAINTY.md"


class ModelData:
    """Per-image matched predictions for one model."""

    def __init__(self, run: str):
        path = PRED_DIR / f"test_{run}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/export_predictions.py first")
        z = np.load(path)
        self.run = run
        self.image_names = [str(n) for n in z["image_names"]]
        n_img = len(self.image_names)
        # Detection: the validator's own per-image statistics (exactly what val() scored).
        self.tp, self.conf, self.pcls, self.tcls = [], [], [], []
        for i in range(n_img):
            pm, tm = z["val_img"] == i, z["val_target_img"] == i
            self.tp.append(z["val_tp"][pm])
            self.conf.append(z["val_conf"][pm])
            self.pcls.append(z["val_pred_cls"][pm].astype(np.int64))
            self.tcls.append(z["val_target_cls"][tm].astype(np.int64))

        # Counting: per-image counts produced through the serving configuration.
        metrics = json.loads((METRICS_DIR / f"{run}.json").read_text())
        self.thresholds = metrics["counting"]["calibrated"]["conf"]
        meta = json.loads((PRED_DIR / f"test_{run}.json").read_text())
        self.ultralytics_overall = meta["val_reported"]
        self.serving = meta["serving_settings"]
        gt = z["count_gt"]
        self.abs_count_err = np.abs(z["count_calibrated"] - gt).astype(float)  # (n_img, n_cls)
        self.abs_count_err_default = np.abs(z["count_default"] - gt).astype(float)

    def detection_metrics(self, idx: np.ndarray) -> dict[str, float]:
        tp = np.concatenate([self.tp[i] for i in idx])
        conf = np.concatenate([self.conf[i] for i in idx])
        pcls = np.concatenate([self.pcls[i] for i in idx])
        tcls = np.concatenate([self.tcls[i] for i in idx])
        out = ap_per_class(tp, conf, pcls, tcls)
        ap, classes = out[5], out[6]
        res = {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean())}
        for row, c in zip(ap, classes):
            res[f"{CLASS_NAMES[c]}_mAP50_95"] = float(row.mean())
        return res

    def counting_metrics(self, idx: np.ndarray) -> dict[str, float]:
        err = self.abs_count_err[idx]
        return {f"{n}_count_MAE": float(err[:, j].mean()) for j, n in enumerate(CLASS_NAMES)}

    def all_metrics(self, idx: np.ndarray) -> dict[str, float]:
        return {**self.detection_metrics(idx), **self.counting_metrics(idx)}


def verdict(lo: float, hi: float) -> str:
    if lo == 0 and hi == 0:
        return "no: identical on every resample"
    if lo > 0 or hi < 0:
        return "yes (borderline)" if min(abs(lo), abs(hi)) < 0.0015 else "yes"
    return "no: interval includes 0"


def ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="first run is the reference (served) model")
    parser.add_argument("-B", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = [ModelData(r) for r in args.runs]
    names0 = models[0].image_names
    for m in models[1:]:
        if m.image_names != names0:
            sys.exit(f"Cannot pair: {m.run} was evaluated on different images than {models[0].run}")
    n = len(names0)
    n_labels = int(sum(len(t) for t in models[0].tcls))

    full = np.arange(n)
    point = {m.run: m.all_metrics(full) for m in models}
    for m in models:  # refuse to report intervals around numbers we cannot reproduce
        gap = abs(point[m.run]["mAP50_95"] - m.ultralytics_overall["mAP50_95"])
        if gap > 0.002:
            sys.exit(
                f"BLOCKER: {m.run} recomputed mAP50-95 {point[m.run]['mAP50_95']:.4f} differs from "
                f"model.val() {m.ultralytics_overall['mAP50_95']:.4f}; re-run export_predictions.py"
            )
    keys = list(point[models[0].run])

    rng = np.random.default_rng(args.seed)
    samples = {m.run: {k: np.empty(args.B) for k in keys} for m in models}
    calib = {m.run: np.empty((args.B, len(CLASS_NAMES))) for m in models}
    t0 = time.time()
    for b in range(args.B):
        idx = rng.integers(0, n, n)  # same indices for every model -> paired
        for m in models:
            calib[m.run][b] = m.abs_count_err[idx].mean(0) - m.abs_count_err_default[idx].mean(0)
        for m in models:
            res = m.all_metrics(idx)
            for k in keys:
                samples[m.run][k][b] = res.get(k, np.nan)  # a class absent from a resample -> NaN
    elapsed = time.time() - t0

    report = {
        "method": {
            "resampling_unit": "image",
            "design": "paired (identical resampled image sets for all models)",
            "B": args.B,
            "seed": args.seed,
            "interval": "95% percentile bootstrap",
            "split": "test",
            "n_images": n,
            "n_labels": n_labels,
            "detection_matching": "per-image TP statistics captured from Ultralytics model.val() "
            "(multi-label NMS), re-aggregated with ultralytics ap_per_class (IoU 0.50:0.95)",
            "counting_source": "counts produced through the serving configuration "
            "(model.predict, single-label NMS, API NMS IoU and confidence floor)",
            "counting_thresholds": "per-class, calibrated on the val split (see reports/metrics/<run>.json)",
        },
        "models": {},
        "comparisons": {},
    }
    for m in models:
        entry = {"point": point[m.run], "ci95": {}, "ultralytics_val_reported": m.ultralytics_overall,
                 "count_thresholds": m.thresholds}
        for k in keys:
            vals = samples[m.run][k][~np.isnan(samples[m.run][k])]
            entry["ci95"][k] = ci(vals)
        report["models"][m.run] = entry

    ref = models[0]
    for m in models[1:]:
        comp = {}
        for k in keys:
            d = samples[m.run][k] - samples[ref.run][k]
            d = d[~np.isnan(d)]
            lo, hi = ci(d)
            comp[k] = {
                "difference": point[m.run][k] - point[ref.run][k],
                "ci95": [lo, hi],
                "resolved": bool(lo > 0 or hi < 0),
                "verdict": verdict(lo, hi),
                "share_of_resamples_positive": float((d > 0).mean()),
            }
        report["comparisons"][f"{m.run} minus {ref.run}"] = comp

    report["threshold_calibration"] = {}
    for m in models:
        entry = {}
        for j, name in enumerate(CLASS_NAMES):
            d_point = float(m.abs_count_err[:, j].mean() - m.abs_count_err_default[:, j].mean())
            lo, hi = ci(calib[m.run][:, j])
            entry[name] = {
                "count_MAE_conf025": float(m.abs_count_err_default[:, j].mean()),
                "count_MAE_calibrated": float(m.abs_count_err[:, j].mean()),
                "difference": d_point, "ci95": [lo, hi], "verdict": verdict(lo, hi),
            }
        report["threshold_calibration"][m.run] = entry

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "bootstrap.json").write_text(json.dumps(report, indent=2))
    DOC.write_text(render_markdown(report, ref.run, elapsed))
    print(DOC.read_text())


def _fmt(x: float, k: str) -> str:
    return f"{x:.2f}" if "MAE" in k else f"{x:.3f}"


def render_markdown(r: dict, ref: str, elapsed: float) -> str:
    m = r["method"]
    lines = [
        "# Evaluation uncertainty",
        "",
        "_Generated by `scripts/bootstrap_eval.py`; do not edit by hand._",
        "",
        "## Method",
        f"- **Data:** held-out test split, **{m['n_images']} images / {m['n_labels']} labelled cells**; "
        "per-image predictions in `reports/predictions/` (exported with Ultralytics validation settings).",
        f"- **Resampling:** image-level bootstrap, **B = {m['B']}**, seed {m['seed']}, {m['interval']} intervals.",
        f"- **Design:** {m['design']}; model differences are computed within each resample.",
        f"- **Detection metrics:** {m['detection_matching']}.",
        f"- **Counting metrics:** mean absolute per-image count error at {m['counting_thresholds']}; "
        f"source: {m['counting_source']}.",
        "",
        "## Per-model estimates (95% CI)",
        "",
        "| Metric | " + " | ".join(r["models"]) + " |",
        "|---|" + "---|" * len(r["models"]),
    ]
    keys = list(next(iter(r["models"].values()))["point"])
    for k in keys:
        cells = []
        for e in r["models"].values():
            lo, hi = e["ci95"][k]
            cells.append(f"{_fmt(e['point'][k], k)} [{_fmt(lo, k)}, {_fmt(hi, k)}]")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Sanity check against Ultralytics `model.val()` on the same split (mAP50-95): "
        + ", ".join(
            f"{run} {e['point']['mAP50_95']:.3f} vs {e['ultralytics_val_reported']['mAP50_95']:.3f}"
            for run, e in r["models"].items()
        )
        + ". Detection statistics are the validator's own, so these should match to rounding.",
        "",
        f"## Paired differences vs. `{ref}`",
        "",
        "Positive = the other model has a higher value (for MAE, higher is worse).",
        "",
        "| Comparison | Metric | Difference | 95% CI | Resolved? |",
        "|---|---|---|---|---|",
    ]
    for name, comp in r["comparisons"].items():
        for k, c in comp.items():
            lo, hi = c["ci95"]
            lines.append(f"| {name} | {k} | {c['difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict(lo, hi)} |")

    lines += [
        "",
        "## Effect of val-calibrated thresholds on counting (paired, same model and images)",
        "",
        "Negative = calibration reduced the per-image count error relative to the default conf 0.25.",
        "",
        "| Model | Class | MAE @0.25 | MAE calibrated | Difference | 95% CI | Resolved? |",
        "|---|---|---|---|---|---|---|",
    ]
    for run, entry in r["threshold_calibration"].items():
        for cls, c in entry.items():
            lo, hi = c["ci95"]
            lines.append(
                f"| {run} | {cls} | {c['count_MAE_conf025']:.2f} | {c['count_MAE_calibrated']:.2f} | "
                f"{c['difference']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {c['verdict']} |"
            )
    lines += [
        "",
        "## How to read this",
        "- **Resolved** means the 95% interval of the paired difference excludes 0 on this test set.",
        "- **Not resolved** means this test set cannot tell the models apart on that metric. It does "
        "**not** show the models are equivalent; an equivalence claim would need a pre-registered margin and "
        "an interval that falls entirely inside it.",
        "",
        "## Limitations",
        f"- Small test set ({m['n_images']} images): percentile intervals from small samples tend to be "
        "too narrow; treat them as approximate.",
        "- Single source (one lab, stain and microscope); intervals describe sampling variability within "
        "BCCD, not generalisation to other sites.",
        "- Image independence is assumed. BCCD does not document patient or slide IDs, so correlated images "
        "could make intervals optimistic.",
        "- Ground truth has missing RBC annotations, which biases RBC precision/count metrics; the bootstrap "
        "quantifies sampling noise, not label noise.",
        "- Several metrics and comparisons are reported without multiplicity correction.",
        "- Counting thresholds were tuned on the val split (not test), but only one train/val split was used.",
        "",
        f"_Bootstrap runtime: {elapsed:.0f} s._",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
