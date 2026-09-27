from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from buse_uav.schemas import Box
from buse_uav.utility.matching import box_iou


@dataclass(frozen=True)
class WeightedDetection:
    box: Box
    source_weight: float


def weighted_box_fusion(
    base_boxes: Sequence[Box],
    local_boxes: Sequence[Box],
    *,
    local_q_by_region: Mapping[int, float],
    iou_threshold: float,
    probe_conf: float,
    publish_conf: float,
    base_weight: float,
    local_weight_min: float,
    local_weight_max: float,
    max_det: int,
) -> tuple[Box, ...]:
    """Fuse same-class base/local boxes with Q-dependent source weights."""
    _validate_parameters(
        iou_threshold=iou_threshold,
        probe_conf=probe_conf,
        publish_conf=publish_conf,
        base_weight=base_weight,
        local_weight_min=local_weight_min,
        local_weight_max=local_weight_max,
        max_det=max_det,
    )
    detections = [
        WeightedDetection(box=box, source_weight=base_weight)
        for box in base_boxes
        if box.score >= probe_conf
    ]
    detections.extend(
        WeightedDetection(
            box=box,
            source_weight=_local_weight(
                box,
                local_q_by_region=local_q_by_region,
                minimum=local_weight_min,
                maximum=local_weight_max,
            ),
        )
        for box in local_boxes
        if box.score >= probe_conf
    )
    clusters: list[list[WeightedDetection]] = []
    for detection in sorted(
        detections,
        key=lambda item: (
            item.box.class_id,
            -item.box.score * item.source_weight,
            item.box.xyxy,
            item.box.source,
        ),
    ):
        best_index = -1
        best_overlap = iou_threshold
        for index, cluster in enumerate(clusters):
            fused = _fuse_cluster(cluster)
            if fused.class_id != detection.box.class_id:
                continue
            overlap = box_iou(fused.xyxy, detection.box.xyxy)
            if overlap >= best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index < 0:
            clusters.append([detection])
        else:
            clusters[best_index].append(detection)

    _merge_overlapping_clusters(clusters, iou_threshold=iou_threshold)
    output = [
        _fuse_cluster(cluster)
        for cluster in clusters
        if _fuse_cluster(cluster).score >= publish_conf
    ]
    return tuple(sorted(output, key=lambda box: (-box.score, box.class_id, box.xyxy))[:max_det])


def _merge_overlapping_clusters(
    clusters: list[list[WeightedDetection]],
    *,
    iou_threshold: float,
) -> None:
    changed = True
    while changed:
        changed = False
        for first_index in range(len(clusters)):
            first = _fuse_cluster(clusters[first_index])
            for second_index in range(first_index + 1, len(clusters)):
                second = _fuse_cluster(clusters[second_index])
                if (
                    first.class_id == second.class_id
                    and box_iou(first.xyxy, second.xyxy) >= iou_threshold
                ):
                    clusters[first_index].extend(clusters.pop(second_index))
                    changed = True
                    break
            if changed:
                break


def _fuse_cluster(cluster: Sequence[WeightedDetection]) -> Box:
    coordinate_weights = [detection.box.score * detection.source_weight for detection in cluster]
    denominator = sum(coordinate_weights)
    if denominator <= 0.0:
        denominator = float(len(cluster))
        coordinate_weights = [1.0] * len(cluster)
    coordinates = tuple(
        sum(
            detection.box.xyxy[coordinate] * weight
            for detection, weight in zip(  # noqa: B905 - Phase 9 uses Python 3.9
                cluster, coordinate_weights
            )
        )
        / denominator
        for coordinate in range(4)
    )
    source_denominator = sum(detection.source_weight for detection in cluster)
    score = (
        sum(detection.box.score * detection.source_weight for detection in cluster)
        / source_denominator
    )
    return Box(
        xyxy=(coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
        score=max(0.0, min(1.0, score)),
        class_id=cluster[0].box.class_id,
        source="wbf",
    )


def _local_weight(
    box: Box,
    *,
    local_q_by_region: Mapping[int, float],
    minimum: float,
    maximum: float,
) -> float:
    q = local_q_by_region.get(box.region_id, 0.0) if box.region_id is not None else 0.0
    return max(minimum, min(maximum, 1.0 + q))


def _validate_parameters(
    *,
    iou_threshold: float,
    probe_conf: float,
    publish_conf: float,
    base_weight: float,
    local_weight_min: float,
    local_weight_max: float,
    max_det: int,
) -> None:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("WBF IoU threshold must be in (0, 1]")
    if not 0.0 <= probe_conf < publish_conf <= 1.0:
        raise ValueError("WBF requires 0 <= probe_conf < publish_conf <= 1")
    if base_weight <= 0.0 or local_weight_min <= 0.0:
        raise ValueError("WBF source weights must be positive")
    if local_weight_min > local_weight_max:
        raise ValueError("WBF local weight bounds are reversed")
    if max_det <= 0:
        raise ValueError("WBF max_det must be positive")
