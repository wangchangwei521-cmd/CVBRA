from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from buse_uav.schemas import Box


@dataclass(frozen=True)
class BoxMatch:
    reference_index: int
    prediction_index: int
    iou: float


def hungarian_match(
    reference: Sequence[Box],
    predictions: Sequence[Box],
    *,
    iou_threshold: float,
) -> tuple[BoxMatch, ...]:
    """Perform class-constrained Hungarian IoU matching."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not reference or not predictions:
        return ()
    cost = np.full((len(reference), len(predictions)), 1e6, dtype=np.float64)
    overlaps = np.zeros_like(cost)
    for reference_index, reference_box in enumerate(reference):
        for prediction_index, prediction_box in enumerate(predictions):
            if reference_box.class_id != prediction_box.class_id:
                continue
            overlap = box_iou(reference_box.xyxy, prediction_box.xyxy)
            overlaps[reference_index, prediction_index] = overlap
            if overlap >= iou_threshold:
                cost[reference_index, prediction_index] = 1.0 - overlap
    reference_indices, prediction_indices = linear_sum_assignment(cost)
    matches = [
        BoxMatch(
            reference_index=int(reference_index),
            prediction_index=int(prediction_index),
            iou=float(overlaps[reference_index, prediction_index]),
        )
        for reference_index, prediction_index in zip(  # noqa: B905 - Phase 9 uses Python 3.9
            reference_indices,
            prediction_indices,
        )
        if cost[reference_index, prediction_index] < 1e6
    ]
    return tuple(sorted(matches, key=lambda match: (match.reference_index, match.prediction_index)))


def fuse_reference(
    base_boxes: Sequence[Box],
    identity_boxes: Sequence[Box],
    *,
    iou_threshold: float,
) -> tuple[Box, ...]:
    """Deduplicate the base/identity reference union with same-class WBF."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    by_class: dict[int, list[Box]] = defaultdict(list)
    for box in (*base_boxes, *identity_boxes):
        by_class[box.class_id].append(box)
    fused: list[Box] = []
    for class_id in sorted(by_class):
        clusters: list[list[Box]] = []
        for box in sorted(by_class[class_id], key=lambda item: (-item.score, item.xyxy)):
            best_index = -1
            best_iou = iou_threshold
            for index, cluster in enumerate(clusters):
                overlap = box_iou(box.xyxy, _weighted_coordinates(cluster))
                if overlap >= best_iou:
                    best_index = index
                    best_iou = overlap
            if best_index < 0:
                clusters.append([box])
            else:
                clusters[best_index].append(box)
        for cluster in clusters:
            fused.append(
                Box(
                    xyxy=_weighted_coordinates(cluster),
                    score=float(np.mean([box.score for box in cluster])),
                    class_id=class_id,
                    source="utility_reference",
                )
            )
    return tuple(sorted(fused, key=lambda box: (box.class_id, -box.score, box.xyxy)))


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _weighted_coordinates(boxes: Sequence[Box]) -> tuple[float, float, float, float]:
    weights = np.asarray([max(box.score, 1e-12) for box in boxes], dtype=np.float64)
    coordinates = np.asarray([box.xyxy for box in boxes], dtype=np.float64)
    result = np.average(coordinates, axis=0, weights=weights)
    return (
        float(result[0]),
        float(result[1]),
        float(result[2]),
        float(result[3]),
    )
