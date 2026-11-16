"""Automated quality control for a single image's detections.

The QC layer turns raw model output into signals a reviewer can act on:
confidence statistics, duplicate/overlapping boxes that survived NMS, and
cells cut off by the image border (which bias counts).
"""

from __future__ import annotations

import numpy as np

from bloodcell.schemas import Detection, QualityControl


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """IoU matrix for an (N, 4) array of xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros((0, 0))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def count_overlapping_pairs(
    detections: list[Detection], iou_threshold: float
) -> int:
    """Number of unique box pairs whose IoU exceeds the threshold."""
    if len(detections) < 2:
        return 0
    boxes = np.array([d.bbox for d in detections], dtype=float)
    iou = pairwise_iou(boxes)
    upper = np.triu(iou, k=1)
    return int((upper > iou_threshold).sum())


def count_edge_truncated(
    detections: list[Detection], width: int, height: int, margin: float = 2.0
) -> int:
    """Boxes touching the image border (likely partial cells)."""
    count = 0
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
            count += 1
    return count


def run_quality_control(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    low_conf_threshold: float = 0.5,
    overlap_iou_threshold: float = 0.5,
) -> QualityControl:
    confidences = [d.confidence for d in detections]
    avg = float(np.mean(confidences)) if confidences else None
    low = sum(c < low_conf_threshold for c in confidences)
    overlaps = count_overlapping_pairs(detections, overlap_iou_threshold)
    edges = count_edge_truncated(detections, image_width, image_height)
    wbc = sum(d.cell_class == "WBC" for d in detections)

    flags: list[str] = []
    if not detections:
        flags.append("NO_DETECTIONS")
    if avg is not None and avg < 0.5:
        flags.append("LOW_MEAN_CONFIDENCE")
    if confidences and low / len(confidences) > 0.3:
        flags.append("HIGH_LOW_CONFIDENCE_RATIO")
    if overlaps > 0:
        flags.append("OVERLAPPING_BOXES")
    if detections and wbc == 0:
        # Every BCCD smear is centred on at least one leukocyte.
        flags.append("NO_WBC_DETECTED")
    if wbc > 3:
        flags.append("UNUSUAL_WBC_COUNT")

    if "NO_DETECTIONS" in flags:
        status = "FAIL"
    elif flags:
        status = "REVIEW"
    else:
        status = "PASS"

    return QualityControl(
        average_confidence=round(avg, 4) if avg is not None else None,
        min_confidence=round(min(confidences), 4) if confidences else None,
        low_confidence_threshold=low_conf_threshold,
        low_confidence_count=int(low),
        overlap_iou_threshold=overlap_iou_threshold,
        overlapping_pair_count=overlaps,
        edge_truncated_count=edges,
        flags=flags,
        status=status,
    )
