from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import cv2
import numpy as np

from buse_uav.schemas import Box, DetectionBatch, ImageRecord


class MultiFidelityFlipError(ValueError):
    """Raised when an MFFG input violates the registered method contract."""


def horizontal_flip_records(
    records: Sequence[ImageRecord],
    *,
    uri_prefix: str = "memory://mffg-flip",
) -> tuple[ImageRecord, ...]:
    """Create in-memory horizontal flips while retaining original dimensions."""
    output: list[ImageRecord] = []
    for record in records:
        image = record.image_bgr
        if image is None:
            try:
                encoded = np.fromfile(record.path, dtype=np.uint8)
            except OSError as exc:
                raise MultiFidelityFlipError(f"cannot read MFFG image: {record.path}") from exc
            image = (
                cv2.imdecode(
                    encoded,
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
                )
                if encoded.size
                else None
            )
        if image is None:
            raise MultiFidelityFlipError(f"cannot read MFFG image: {record.path}")
        if image.shape[:2] != (record.height, record.width):
            raise MultiFidelityFlipError(
                f"image {record.image_id} shape {image.shape[:2]} does not match "
                f"record {(record.height, record.width)}"
            )
        output.append(
            replace(
                record,
                path=f"{uri_prefix}/{record.image_id}",
                image_bgr=np.ascontiguousarray(cv2.flip(image, 1)),
            )
        )
    return tuple(output)


def mffg_features(
    base: DetectionBatch,
    scout: DetectionBatch,
    record: ImageRecord,
) -> dict[str, float]:
    """Return detector-shared B0/scout features allowed by the MFFG protocol."""
    if base.image_id != record.image_id or scout.image_id != record.image_id:
        raise MultiFidelityFlipError("MFFG feature inputs must share one image ID")
    if record.width <= 0 or record.height <= 0:
        raise MultiFidelityFlipError("MFFG image dimensions must be positive")

    features: dict[str, float] = {
        "image_width_kpx": record.width / 1000.0,
        "image_height_kpx": record.height / 1000.0,
        "image_aspect": record.width / record.height,
        "image_megapixels": record.width * record.height / 1_000_000.0,
    }
    features.update(_view_features(base.boxes, record, prefix="b0"))
    features.update(_view_features(scout.boxes, record, prefix="scout"))
    features.update(_cross_view_features(base.boxes, scout.boxes, record))
    if not all(math.isfinite(value) for value in features.values()):
        raise MultiFidelityFlipError("MFFG features contain non-finite values")
    return features


def merge_mffg_bypass(
    base: Sequence[DetectionBatch],
    scout: Sequence[DetectionBatch],
    *,
    iou_threshold: float = 0.70,
    publish_conf: float = 0.25,
    max_det: int = 500,
) -> tuple[DetectionBatch, ...]:
    """Fuse B0 and the restored low-resolution scout for bypassed images."""
    _validate_thresholds(iou_threshold, publish_conf, max_det)
    scout_by_id = {batch.image_id: batch for batch in scout}
    if set(scout_by_id) != {batch.image_id for batch in base}:
        raise MultiFidelityFlipError("MFFG B0 and scout coverage differ")
    merged = tuple(
        _merge_pair_classwise_nms(
            batch,
            scout_by_id[batch.image_id],
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
        for batch in base
    )
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= publish_conf),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "mffg_escalated": False},
        )
        for batch in merged
    )


def select_mffg_outputs(
    bypass: Sequence[DetectionBatch],
    exact_full_flip: Sequence[DetectionBatch],
    active_ids: set[int | str],
) -> tuple[DetectionBatch, ...]:
    """Choose scout fusion or the exact frozen full-Flip output per image."""
    flip_by_id = {batch.image_id: batch for batch in exact_full_flip}
    bypass_ids = {batch.image_id for batch in bypass}
    if set(flip_by_id) != bypass_ids:
        raise MultiFidelityFlipError("MFFG bypass and full-Flip coverage differ")
    if not active_ids.issubset(bypass_ids):
        raise MultiFidelityFlipError("MFFG active IDs fall outside the prediction bank")
    output: list[DetectionBatch] = []
    for batch in bypass:
        if batch.image_id not in active_ids:
            output.append(batch)
            continue
        exact = flip_by_id[batch.image_id]
        output.append(
            DetectionBatch(
                image_id=exact.image_id,
                boxes=exact.boxes,
                latency_ms=exact.latency_ms,
                meta={**exact.meta, "method": "mffg_full_escalated", "mffg_escalated": True},
            )
        )
    return tuple(output)


def rank_mffg_scores(
    scores: Mapping[int | str, float],
    *,
    activation_rate: float,
) -> tuple[int | str, ...]:
    """Apply the registered cohort ranking and deterministic tie break."""
    if not scores:
        raise MultiFidelityFlipError("MFFG scores must not be empty")
    if not 0.0 < activation_rate <= 1.0:
        raise MultiFidelityFlipError("MFFG activation rate must be in (0, 1]")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise MultiFidelityFlipError("MFFG scores contain non-finite values")
    active_count = min(len(scores), max(1, round(len(scores) * activation_rate)))
    ranked = sorted(scores, key=lambda image_id: (-float(scores[image_id]), str(image_id)))
    return tuple(ranked[:active_count])


def _validate_thresholds(iou_threshold: float, publish_conf: float, max_det: int) -> None:
    if not 0.0 < iou_threshold <= 1.0:
        raise MultiFidelityFlipError("MFFG fusion IoU must be in (0, 1]")
    if not 0.0 <= publish_conf <= 1.0:
        raise MultiFidelityFlipError("MFFG publish confidence must be in [0, 1]")
    if max_det <= 0:
        raise MultiFidelityFlipError("MFFG max_det must be positive")


def _view_features(
    boxes: Sequence[Box],
    record: ImageRecord,
    *,
    prefix: str,
) -> dict[str, float]:
    ordered = sorted(boxes, key=lambda box: (-box.score, box.class_id, box.xyxy))
    scores = np.asarray([box.score for box in ordered], dtype=np.float64)
    features = {
        f"{prefix}_count": float(len(ordered)),
        f"{prefix}_log_count": math.log1p(len(ordered)),
        f"{prefix}_count_per_megapixel": len(ordered)
        / (record.width * record.height / 1_000_000.0),
    }
    features.update(_stats(scores, f"{prefix}_score"))
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70):
        count = int(np.count_nonzero(scores >= threshold))
        features[f"{prefix}_count_ge_{threshold:.2f}"] = float(count)
        features[f"{prefix}_fraction_ge_{threshold:.2f}"] = count / max(1, len(scores))
    for class_id in range(3):
        count = sum(box.class_id == class_id for box in ordered)
        features[f"{prefix}_class_{class_id}_count"] = float(count)
        features[f"{prefix}_class_{class_id}_fraction"] = count / max(1, len(ordered))

    if not ordered:
        for name in ("area", "width", "height", "center_x", "center_y"):
            features.update(_stats(np.empty(0, dtype=np.float64), f"{prefix}_{name}"))
        features[f"{prefix}_border_fraction"] = 0.0
        features[f"{prefix}_left_right_imbalance"] = 0.0
        features[f"{prefix}_top_bottom_imbalance"] = 0.0
        return features

    coordinates = np.asarray([box.xyxy for box in ordered], dtype=np.float64)
    width = (coordinates[:, 2] - coordinates[:, 0]) / record.width
    height = (coordinates[:, 3] - coordinates[:, 1]) / record.height
    center_x = (coordinates[:, 0] + coordinates[:, 2]) / (2.0 * record.width)
    center_y = (coordinates[:, 1] + coordinates[:, 3]) / (2.0 * record.height)
    area = width * height
    for name, values in (
        ("area", area),
        ("width", width),
        ("height", height),
        ("center_x", center_x),
        ("center_y", center_y),
    ):
        features.update(_stats(values, f"{prefix}_{name}"))
    features[f"{prefix}_border_fraction"] = float(
        np.mean((center_x < 0.10) | (center_x > 0.90) | (center_y < 0.10) | (center_y > 0.90))
    )
    features[f"{prefix}_left_right_imbalance"] = abs(
        np.count_nonzero(center_x < 0.5) - np.count_nonzero(center_x >= 0.5)
    ) / len(center_x)
    features[f"{prefix}_top_bottom_imbalance"] = abs(
        np.count_nonzero(center_y < 0.5) - np.count_nonzero(center_y >= 0.5)
    ) / len(center_y)
    return features


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not values.size:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_max": 0.0,
        }
    quantiles = np.quantile(values, (0.25, 0.50, 0.75, 0.90))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q25": float(quantiles[0]),
        f"{prefix}_q50": float(quantiles[1]),
        f"{prefix}_q75": float(quantiles[2]),
        f"{prefix}_q90": float(quantiles[3]),
        f"{prefix}_max": float(np.max(values)),
    }


def _cross_view_features(
    base: Sequence[Box],
    scout: Sequence[Box],
    record: ImageRecord,
) -> dict[str, float]:
    base_ordered = sorted(base, key=lambda box: (-box.score, box.class_id, box.xyxy))
    scout_ordered = sorted(scout, key=lambda box: (-box.score, box.class_id, box.xyxy))
    features: dict[str, float] = {
        "cross_count_delta": float(len(scout_ordered) - len(base_ordered)),
        "cross_count_ratio": (len(scout_ordered) + 1.0) / (len(base_ordered) + 1.0),
        "cross_score_sum_delta": sum(box.score for box in scout_ordered)
        - sum(box.score for box in base_ordered),
        "cross_score_sum_ratio": (sum(box.score for box in scout_ordered) + 1e-6)
        / (sum(box.score for box in base_ordered) + 1e-6),
    }
    for class_id in range(3):
        base_count = sum(box.class_id == class_id for box in base_ordered)
        scout_count = sum(box.class_id == class_id for box in scout_ordered)
        features[f"cross_class_{class_id}_count_delta"] = float(scout_count - base_count)
        features[f"cross_class_{class_id}_count_ratio"] = (scout_count + 1.0) / (
            base_count + 1.0
        )
    candidates = _match_candidates(base_ordered, scout_ordered, minimum_iou=0.30)
    for confidence in (0.25, 0.35, 0.50):
        allowed_base = {index for index, box in enumerate(base_ordered) if box.score >= confidence}
        allowed_scout = {
            index for index, box in enumerate(scout_ordered) if box.score >= confidence
        }
        matches = _greedy_from_candidates(
            candidates,
            iou_threshold=0.50,
            allowed_base=allowed_base,
            allowed_scout=allowed_scout,
        )
        features[f"cross_unmatched_b0_ge_{confidence:.2f}"] = float(
            len(allowed_base) - len(matches)
        )
        features[f"cross_unmatched_scout_ge_{confidence:.2f}"] = float(
            len(allowed_scout) - len(matches)
        )
    for threshold in (0.30, 0.50, 0.70):
        matches = _greedy_from_candidates(candidates, iou_threshold=threshold)
        prefix = f"cross_iou_{threshold:.2f}"
        features[f"{prefix}_matches"] = float(len(matches))
        features[f"{prefix}_b0_fraction"] = len(matches) / max(1, len(base_ordered))
        features[f"{prefix}_scout_fraction"] = len(matches) / max(1, len(scout_ordered))
        features[f"{prefix}_unmatched_b0"] = float(len(base_ordered) - len(matches))
        features[f"{prefix}_unmatched_scout"] = float(len(scout_ordered) - len(matches))
        features.update(_match_statistics(matches, base_ordered, scout_ordered, record, prefix))
    return features


def _greedy_matches(
    base: Sequence[Box],
    scout: Sequence[Box],
    *,
    iou_threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    candidates = _match_candidates(base, scout, minimum_iou=iou_threshold)
    return _greedy_from_candidates(candidates, iou_threshold=iou_threshold)


def _match_candidates(
    base: Sequence[Box],
    scout: Sequence[Box],
    *,
    minimum_iou: float,
) -> tuple[tuple[int, int, float], ...]:
    if not base or not scout:
        return ()
    base_coordinates = np.asarray([box.xyxy for box in base], dtype=np.float64)
    scout_coordinates = np.asarray([box.xyxy for box in scout], dtype=np.float64)
    base_classes = np.asarray([box.class_id for box in base], dtype=np.int64)
    scout_classes = np.asarray([box.class_id for box in scout], dtype=np.int64)
    base_area = np.maximum(0.0, base_coordinates[:, 2] - base_coordinates[:, 0]) * np.maximum(
        0.0, base_coordinates[:, 3] - base_coordinates[:, 1]
    )
    scout_area = np.maximum(
        0.0, scout_coordinates[:, 2] - scout_coordinates[:, 0]
    ) * np.maximum(0.0, scout_coordinates[:, 3] - scout_coordinates[:, 1])
    collected_base: list[np.ndarray] = []
    collected_scout: list[np.ndarray] = []
    collected_overlap: list[np.ndarray] = []
    for class_id in np.intersect1d(base_classes, scout_classes):
        base_indices = np.flatnonzero(base_classes == class_id)
        scout_indices = np.flatnonzero(scout_classes == class_id)
        first = base_coordinates[base_indices]
        second = scout_coordinates[scout_indices]
        x1 = np.maximum(first[:, None, 0], second[None, :, 0])
        y1 = np.maximum(first[:, None, 1], second[None, :, 1])
        x2 = np.minimum(first[:, None, 2], second[None, :, 2])
        y2 = np.minimum(first[:, None, 3], second[None, :, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = (
            base_area[base_indices, None]
            + scout_area[None, scout_indices]
            - intersection
        )
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        local_base, local_scout = np.nonzero(overlap >= minimum_iou)
        if local_base.size:
            collected_base.append(base_indices[local_base])
            collected_scout.append(scout_indices[local_scout])
            collected_overlap.append(overlap[local_base, local_scout])
    if not collected_base:
        return ()
    base_indices = np.concatenate(collected_base)
    scout_indices = np.concatenate(collected_scout)
    overlaps = np.concatenate(collected_overlap)
    order = np.lexsort((scout_indices, base_indices, -overlaps))
    return tuple(
        (int(base_indices[index]), int(scout_indices[index]), float(overlaps[index]))
        for index in order
    )


def _greedy_from_candidates(
    candidates: Sequence[tuple[int, int, float]],
    *,
    iou_threshold: float,
    allowed_base: set[int] | None = None,
    allowed_scout: set[int] | None = None,
) -> tuple[tuple[int, int, float], ...]:
    used_base: set[int] = set()
    used_scout: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for base_index, scout_index, overlap in candidates:
        if overlap < iou_threshold:
            continue
        if allowed_base is not None and base_index not in allowed_base:
            continue
        if allowed_scout is not None and scout_index not in allowed_scout:
            continue
        if base_index in used_base or scout_index in used_scout:
            continue
        used_base.add(base_index)
        used_scout.add(scout_index)
        matches.append((base_index, scout_index, overlap))
    return tuple(matches)


def _merge_pair_classwise_nms(
    first: DetectionBatch,
    second: DetectionBatch,
    *,
    iou_threshold: float,
    max_det: int,
) -> DetectionBatch:
    candidates = sorted(
        (*first.boxes, *second.boxes),
        key=lambda box: (-box.score, box.class_id, box.xyxy),
    )
    kept: list[Box] = []
    kept_by_class: dict[int, list[Box]] = {}
    for candidate in candidates:
        same_class = kept_by_class.get(candidate.class_id, [])
        if same_class and np.any(
            _iou_vector(candidate.xyxy, same_class) > iou_threshold
        ):
            continue
        kept.append(candidate)
        kept_by_class.setdefault(candidate.class_id, []).append(candidate)
        if len(kept) >= max_det:
            break
    return DetectionBatch(
        image_id=first.image_id,
        boxes=tuple(kept),
        latency_ms=first.latency_ms + second.latency_ms,
        meta={"method": "mffg_scout_bypass", "sources": 2},
    )


def _iou_vector(
    candidate: tuple[float, float, float, float],
    boxes: Sequence[Box],
) -> np.ndarray:
    coordinates = np.asarray([box.xyxy for box in boxes], dtype=np.float64)
    x1 = np.maximum(candidate[0], coordinates[:, 0])
    y1 = np.maximum(candidate[1], coordinates[:, 1])
    x2 = np.minimum(candidate[2], coordinates[:, 2])
    y2 = np.minimum(candidate[3], coordinates[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(
        0.0, candidate[3] - candidate[1]
    )
    other_area = np.maximum(0.0, coordinates[:, 2] - coordinates[:, 0]) * np.maximum(
        0.0, coordinates[:, 3] - coordinates[:, 1]
    )
    union = candidate_area + other_area - intersection
    return np.asarray(
        np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        ),
        dtype=np.float64,
    )


def _match_statistics(
    matches: Sequence[tuple[int, int, float]],
    base: Sequence[Box],
    scout: Sequence[Box],
    record: ImageRecord,
    prefix: str,
) -> dict[str, float]:
    if not matches:
        return {
            f"{prefix}_overlap_mean": 0.0,
            f"{prefix}_overlap_min": 0.0,
            f"{prefix}_score_delta_mean": 0.0,
            f"{prefix}_score_abs_delta_mean": 0.0,
            f"{prefix}_center_shift_mean": 0.0,
            f"{prefix}_center_shift_max": 0.0,
            f"{prefix}_log_area_ratio_mean": 0.0,
            f"{prefix}_log_area_ratio_abs_mean": 0.0,
        }
    overlaps: list[float] = []
    score_deltas: list[float] = []
    shifts: list[float] = []
    log_area_ratios: list[float] = []
    for base_index, scout_index, overlap in matches:
        first = base[base_index]
        second = scout[scout_index]
        overlaps.append(overlap)
        score_deltas.append(second.score - first.score)
        first_center = (
            (first.xyxy[0] + first.xyxy[2]) / (2.0 * record.width),
            (first.xyxy[1] + first.xyxy[3]) / (2.0 * record.height),
        )
        second_center = (
            (second.xyxy[0] + second.xyxy[2]) / (2.0 * record.width),
            (second.xyxy[1] + second.xyxy[3]) / (2.0 * record.height),
        )
        shifts.append(
            math.hypot(
                first_center[0] - second_center[0],
                first_center[1] - second_center[1],
            )
        )
        first_area = max(1e-12, (first.xyxy[2] - first.xyxy[0]) * (first.xyxy[3] - first.xyxy[1]))
        second_area = max(
            1e-12,
            (second.xyxy[2] - second.xyxy[0]) * (second.xyxy[3] - second.xyxy[1]),
        )
        log_area_ratios.append(math.log(second_area / first_area))
    overlap_values = np.asarray(overlaps, dtype=np.float64)
    score_values = np.asarray(score_deltas, dtype=np.float64)
    shift_values = np.asarray(shifts, dtype=np.float64)
    area_values = np.asarray(log_area_ratios, dtype=np.float64)
    return {
        f"{prefix}_overlap_mean": float(np.mean(overlap_values)),
        f"{prefix}_overlap_min": float(np.min(overlap_values)),
        f"{prefix}_score_delta_mean": float(np.mean(score_values)),
        f"{prefix}_score_abs_delta_mean": float(np.mean(np.abs(score_values))),
        f"{prefix}_center_shift_mean": float(np.mean(shift_values)),
        f"{prefix}_center_shift_max": float(np.max(shift_values)),
        f"{prefix}_log_area_ratio_mean": float(np.mean(area_values)),
        f"{prefix}_log_area_ratio_abs_mean": float(np.mean(np.abs(area_values))),
    }


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
    return intersection / union if union > 0.0 else 0.0
