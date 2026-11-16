"""Statistical analysis helpers and reproducibility of committed artefacts."""

import importlib.util
import json

import numpy as np
import pytest

from bloodcell.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("bootstrap_eval", PROJECT_ROOT / "scripts" / "bootstrap_eval.py")
bootstrap_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap_eval)

PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
RUNS = ["yolo11n_e100", "yolo11s_e100", "blood_cells_baseline"]


def test_percentile_ci():
    lo, hi = bootstrap_eval.ci(np.arange(1001, dtype=float))
    assert lo == pytest.approx(25.0)
    assert hi == pytest.approx(975.0)


@pytest.mark.parametrize(
    "lo, hi, expected",
    [
        (0.01, 0.05, "yes"),
        (-0.05, -0.01, "yes"),
        (0.0005, 0.05, "yes (borderline)"),
        (-0.01, 0.02, "no: interval includes 0"),
        (0.0, 0.0, "no: identical on every resample"),
    ],
)
def test_verdict_never_claims_equivalence(lo, hi, expected):
    v = bootstrap_eval.verdict(lo, hi)
    assert v == expected
    assert "equivalent" not in v


artefacts = pytest.mark.skipif(
    not all((PRED_DIR / f"test_{r}.npz").exists() for r in RUNS), reason="prediction artefacts not exported"
)


@artefacts
@pytest.mark.parametrize("run", RUNS)
def test_recomputed_map_matches_ultralytics_val(run):
    m = bootstrap_eval.ModelData(run)
    point = m.detection_metrics(np.arange(len(m.image_names)))
    assert point["mAP50_95"] == pytest.approx(m.ultralytics_overall["mAP50_95"], abs=1e-3)
    assert point["mAP50"] == pytest.approx(m.ultralytics_overall["mAP50"], abs=1e-3)


@artefacts
@pytest.mark.parametrize("run", RUNS)
def test_served_counts_reproduce_reported_counting_metrics(run):
    m = bootstrap_eval.ModelData(run)
    reported = json.loads((PROJECT_ROOT / "reports" / "metrics" / f"{run}.json").read_text())["counting"]
    assert m.abs_count_err[:, 0].mean() == pytest.approx(reported["calibrated"]["count_mae"]["RBC"], abs=0.01)
    assert m.abs_count_err_default[:, 0].mean() == pytest.approx(reported["count_mae"]["RBC"], abs=0.01)


@artefacts
def test_models_are_paired_on_identical_images():
    names = [bootstrap_eval.ModelData(r).image_names for r in RUNS]
    assert all(n == names[0] for n in names)
    assert len(names[0]) == 56


def test_serving_nms_iou_matches_counting_evaluation():
    # Counting thresholds were calibrated with NMS IoU 0.5; serving must use the same.
    from bloodcell.config import get_settings

    info = json.loads((PRED_DIR / "test_yolo11n_e100.json").read_text()) if (PRED_DIR / "test_yolo11n_e100.json").exists() else None
    if info is None:
        pytest.skip("artefacts not exported")
    assert get_settings().iou_threshold == info["serving_settings"]["nms_iou"]
