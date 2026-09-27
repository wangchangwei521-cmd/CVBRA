from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from buse_uav.schemas import Box, DetectionBatch, Region
from buse_uav.utility.matching import hungarian_match

UNCERTAINTY_COMPONENTS = (
    "confidence_entropy",
    "threshold_proximity",
    "class_conflict",
    "low_conf_density",
)
EPSILON = 1e-12


@dataclass(frozen=True)
class UncertaintyScore:
    region_id: int
    uncertainty: float
    components: dict[str, float]
    raw_components: dict[str, float]
    candidate_count: int
    edge_density: float


def score_uncertainty(
    image: np.ndarray,
    regions: Sequence[Region],
    detections: DetectionBatch,
    *,
    publish_conf: float,
    threshold_sigma: float,
    density_kappa: float,
    rank_mix: float,
    weights: Mapping[str, float],
    flip_detections: DetectionBatch | None = None,
    flip_match_iou: float = 0.30,
    flip_weight: float = 0.20,
) -> tuple[UncertaintyScore, ...]:
    """Compute the four section 6.3 uncertainty proxies for every core region."""
    _validate_inputs(
        image,
        regions,
        publish_conf=publish_conf,
        threshold_sigma=threshold_sigma,
        density_kappa=density_kappa,
        rank_mix=rank_mix,
        weights=weights,
    )
    raw_rows, boxes_by_region, flip_boxes_by_region = _raw_uncertainty_rows(
        regions,
        detections,
        publish_conf=publish_conf,
        threshold_sigma=threshold_sigma,
        density_kappa=density_kappa,
        flip_detections=flip_detections,
    )
    edge_by_region = region_edge_densities(image, regions)
    if flip_boxes_by_region is not None:
        for index, region in enumerate(regions):
            raw_rows[index]["flip_consistency"] = _flip_inconsistency(
                boxes_by_region[region.id],
                flip_boxes_by_region[region.id],
                iou_threshold=flip_match_iou,
            )
    normalized = _normalized_components(raw_rows, rank_mix=rank_mix)
    output: list[UncertaintyScore] = []
    for index, region in enumerate(regions):
        components, uncertainty = _weighted_uncertainty(
            raw_rows[index],
            normalized,
            index=index,
            weights=weights,
            flip_weight=flip_weight if flip_boxes_by_region is not None else None,
        )
        output.append(
            UncertaintyScore(
                region_id=region.id,
                uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
                components=components,
                raw_components=raw_rows[index],
                candidate_count=len(boxes_by_region[region.id]),
                edge_density=edge_by_region[region.id],
            )
        )
    return tuple(output)


def max_region_uncertainty(
    regions: Sequence[Region],
    detections: DetectionBatch,
    *,
    publish_conf: float,
    threshold_sigma: float,
    density_kappa: float,
    rank_mix: float,
    weights: Mapping[str, float],
) -> float:
    """Return the image gate score using only the already-computed B0 boxes."""
    _validate_score_inputs(
        regions,
        publish_conf=publish_conf,
        threshold_sigma=threshold_sigma,
        density_kappa=density_kappa,
        rank_mix=rank_mix,
        weights=weights,
    )
    raw_rows, _, _ = _raw_uncertainty_rows(
        regions,
        detections,
        publish_conf=publish_conf,
        threshold_sigma=threshold_sigma,
        density_kappa=density_kappa,
        flip_detections=None,
    )
    normalized = _normalized_components(raw_rows, rank_mix=rank_mix)
    return max(
        _weighted_uncertainty(
            row,
            normalized,
            index=index,
            weights=weights,
            flip_weight=None,
        )[1]
        for index, row in enumerate(raw_rows)
    )


def _raw_uncertainty_rows(
    regions: Sequence[Region],
    detections: DetectionBatch,
    *,
    publish_conf: float,
    threshold_sigma: float,
    density_kappa: float,
    flip_detections: DetectionBatch | None,
) -> tuple[
    list[dict[str, float]],
    dict[int, tuple[Box, ...]],
    dict[int, tuple[Box, ...]] | None,
]:
    boxes_by_region = assign_boxes_to_regions(detections.boxes, regions)
    flip_boxes_by_region = (
        assign_boxes_to_regions(flip_detections.boxes, regions)
        if flip_detections is not None
        else None
    )
    raw_rows: list[dict[str, float]] = []
    for region in regions:
        boxes = boxes_by_region[region.id]
        scores = np.asarray([box.score for box in boxes], dtype=np.float64)
        if scores.size:
            clipped = np.clip(scores, EPSILON, 1.0 - EPSILON)
            confidence_entropy = float(
                np.mean(
                    (-clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped))
                    / math.log(2.0)
                )
            )
            threshold_proximity = float(
                np.mean(np.exp(-np.abs(scores - publish_conf) / threshold_sigma))
            )
        else:
            confidence_entropy = 0.0
            threshold_proximity = 0.0
        x1, y1, x2, y2 = region.core_xyxy
        area = float((x2 - x1) * (y2 - y1))
        residual_sum = sum(1.0 - box.score for box in boxes)
        raw_rows.append(
            {
                "confidence_entropy": confidence_entropy,
                "threshold_proximity": threshold_proximity,
                "class_conflict": _class_conflict(boxes),
                "low_conf_density": 1.0 - math.exp(-residual_sum / (density_kappa * area)),
            }
        )
    return raw_rows, boxes_by_region, flip_boxes_by_region


def _normalized_components(
    raw_rows: Sequence[Mapping[str, float]],
    *,
    rank_mix: float,
) -> dict[str, np.ndarray]:
    return {
        name: _mixed_unit_normalization(
            np.asarray([row[name] for row in raw_rows], dtype=np.float64),
            rank_mix=rank_mix,
        )
        for name in UNCERTAINTY_COMPONENTS
    }


def _weighted_uncertainty(
    raw_row: Mapping[str, float],
    normalized: Mapping[str, np.ndarray],
    *,
    index: int,
    weights: Mapping[str, float],
    flip_weight: float | None,
) -> tuple[dict[str, float], float]:
    components = {name: float(normalized[name][index]) for name in UNCERTAINTY_COMPONENTS}
    uncertainty = sum(float(weights[name]) * components[name] for name in components)
    if flip_weight is not None:
        flip_value = float(raw_row["flip_consistency"])
        components["flip_consistency"] = flip_value
        uncertainty = (1.0 - flip_weight) * uncertainty + flip_weight * flip_value
    return components, float(np.clip(uncertainty, 0.0, 1.0))


def _flip_inconsistency(
    original: Sequence[Box],
    flipped: Sequence[Box],
    *,
    iou_threshold: float,
) -> float:
    matches = hungarian_match(original, flipped, iou_threshold=iou_threshold)
    return float(
        np.clip(
            1.0 - (2.0 * len(matches)) / (len(original) + len(flipped) + EPSILON),
            0.0,
            1.0,
        )
    )


def assign_boxes_to_regions(
    boxes: Sequence[Box],
    regions: Sequence[Region],
) -> dict[int, tuple[Box, ...]]:
    assigned: dict[int, list[Box]] = {region.id: [] for region in regions}
    for box in boxes:
        center_x = (box.xyxy[0] + box.xyxy[2]) / 2.0
        center_y = (box.xyxy[1] + box.xyxy[3]) / 2.0
        for region in regions:
            x1, y1, x2, y2 = region.core_xyxy
            if x1 <= center_x < x2 and y1 <= center_y < y2:
                assigned[region.id].append(box)
                break
    return {region_id: tuple(values) for region_id, values in assigned.items()}


def region_edge_densities(
    image: np.ndarray,
    regions: Sequence[Region],
) -> dict[int, float]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("edge density expects uint8 RGB HWC")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    densities: dict[int, float] = {}
    for region in regions:
        x1, y1, x2, y2 = region.core_xyxy
        densities[region.id] = float(np.count_nonzero(edges[y1:y2, x1:x2])) / float(
            (x2 - x1) * (y2 - y1)
        )
    return densities


def _class_conflict(boxes: Sequence[Box]) -> float:
    numerator = 0.0
    different_class_pairs = 0
    for first_index, first in enumerate(boxes):
        for second in boxes[first_index + 1 :]:
            if first.class_id == second.class_id:
                continue
            different_class_pairs += 1
            numerator += _iou(first.xyxy, second.xyxy) * math.sqrt(first.score * second.score)
    return numerator / (different_class_pairs + EPSILON)


def _iou(
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
    return intersection / union if union > 0 else 0.0


def _mixed_unit_normalization(values: np.ndarray, *, rank_mix: float) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    if values.size <= 1 or np.allclose(values, values[0]):
        return cast(np.ndarray, clipped)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    ranks /= values.size - 1.0
    return cast(np.ndarray, rank_mix * ranks + (1.0 - rank_mix) * clipped)


def _validate_inputs(
    image: np.ndarray,
    regions: Sequence[Region],
    *,
    publish_conf: float,
    threshold_sigma: float,
    density_kappa: float,
    rank_mix: float,
    weights: Mapping[str, float],
) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("uncertainty scoring expects uint8 RGB HWC")
    _validate_score_inputs(
        regions,
        publish_conf=publish_conf,
        threshold_sigma=threshold_sigma,
        density_kappa=density_kappa,
        rank_mix=rank_mix,
        weights=weights,
    )


def _validate_score_inputs(
    regions: Sequence[Region],
    *,
    publish_conf: float,
    threshold_sigma: float,
    density_kappa: float,
    rank_mix: float,
    weights: Mapping[str, float],
) -> None:
    if not regions:
        raise ValueError("uncertainty scoring requires at least one region")
    if not 0.0 < publish_conf <= 1.0:
        raise ValueError("publish_conf must be in (0, 1]")
    if threshold_sigma <= 0.0 or density_kappa <= 0.0:
        raise ValueError("uncertainty scale constants must be positive")
    if not 0.0 <= rank_mix <= 1.0:
        raise ValueError("rank_mix must be in [0, 1]")
    if set(weights) != set(UNCERTAINTY_COMPONENTS):
        raise ValueError(f"uncertainty weights must be exactly {UNCERTAINTY_COMPONENTS}")
    if any(value < 0 for value in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0, abs_tol=1e-6
    ):
        raise ValueError("uncertainty weights must be nonnegative and sum to one")
