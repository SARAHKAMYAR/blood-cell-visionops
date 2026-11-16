import numpy as np
import pytest

from bloodcell.qc import count_edge_truncated, count_overlapping_pairs, pairwise_iou, run_quality_control
from tests.conftest import det


def test_pairwise_iou_identity_and_disjoint():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = pairwise_iou(boxes)
    assert iou[0, 1] == pytest.approx(1.0)
    assert iou[0, 2] == 0.0
    assert np.allclose(np.diag(iou), 1.0)


def test_pairwise_iou_half_overlap():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    # intersection 50, union 150
    assert pairwise_iou(boxes)[0, 1] == pytest.approx(1 / 3)


def test_pairwise_iou_empty():
    assert pairwise_iou(np.zeros((0, 4))).shape == (0, 0)


def test_overlapping_pairs_counts_unique_pairs():
    dets = [det("RBC", 0.9, [0, 0, 10, 10]), det("RBC", 0.9, [1, 1, 10, 10]), det("RBC", 0.9, [50, 50, 60, 60])]
    assert count_overlapping_pairs(dets, 0.5) == 1
    assert count_overlapping_pairs(dets[:1], 0.5) == 0


def test_edge_truncated():
    dets = [det("RBC", 0.9, [0, 10, 20, 30]), det("RBC", 0.9, [100, 100, 120, 120])]
    assert count_edge_truncated(dets, 640, 480) == 1


def test_qc_pass(sample_prediction):
    qc = run_quality_control(sample_prediction.detections, 640, 480)
    assert qc.status == "PASS"
    assert qc.flags == []
    assert qc.average_confidence == pytest.approx(0.84, abs=1e-3)
    assert qc.min_confidence == 0.62


def test_qc_no_detections_fails():
    qc = run_quality_control([], 640, 480)
    assert qc.status == "FAIL"
    assert "NO_DETECTIONS" in qc.flags
    assert qc.average_confidence is None


def test_qc_flags_low_confidence_and_missing_wbc():
    dets = [det("RBC", 0.3, [10, 10, 50, 50]), det("RBC", 0.35, [100, 100, 150, 150])]
    qc = run_quality_control(dets, 640, 480)
    assert qc.status == "REVIEW"
    assert {"LOW_MEAN_CONFIDENCE", "HIGH_LOW_CONFIDENCE_RATIO", "NO_WBC_DETECTED"} <= set(qc.flags)
    assert qc.low_confidence_count == 2
