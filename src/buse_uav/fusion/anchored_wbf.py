from __future__ import annotations

from collections.abc import Mapping, Sequence

from buse_uav.fusion.wbf import WeightedDetection, _fuse_cluster
from buse_uav.schemas import Box
from buse_uav.utility.matching import box_iou


def anchored_weighted_box_fusion(
    base_boxes: Sequence[Box],
    local_boxes: Sequence[Box],
    *,
    local_q_by_region: Mapping[int, float],
    iou_threshold: float,
    anchor_iou: float,
    q_min: float,
    probe_conf: float,
    publish_conf: float,
    rescue_conf: float,
    base_weight: float,
    local_weight_min: float,
    local_weight_max: float,
    max_det: int,
) -> tuple[Box, ...]:
    """Fuse only reliable local evidence while preserving published B0 anchors.

    Base detections are never removed by local evidence. A reliable local box may
    refine a same-class anchor only at ``anchor_iou`` or above, and the refined
    score cannot fall below the anchor score. Unmatched local boxes are admitted
    only as high-confidence rescues and cannot suppress an existing anchor.
    """
    _validate_parameters(
        iou_threshold=iou_threshold,
        anchor_iou=anchor_iou,
        probe_conf=probe_conf,
        publish_conf=publish_conf,
        rescue_conf=rescue_conf,
        base_weight=base_weight,
        local_weight_min=local_weight_min,
        local_weight_max=local_weight_max,
        max_det=max_det,
    )
    anchors = sorted(
        (box for box in base_boxes if box.score >= publish_conf),
        key=lambda box: (-box.score, box.class_id, box.xyxy),
    )[:max_det]
    reliable = [
        box
        for box in local_boxes
        if box.score >= probe_conf
        and box.region_id is not None
        and local_q_by_region.get(box.region_id, float("-inf")) >= q_min
    ]
    assigned: set[int] = set()
    fused_anchors: list[Box] = []
    for anchor in anchors:
        matches = [
            (index, local)
            for index, local in enumerate(reliable)
            if index not in assigned
            and local.class_id == anchor.class_id
            and local.score >= publish_conf
            and box_iou(anchor.xyxy, local.xyxy) >= anchor_iou
        ]
        if not matches:
            fused_anchors.append(_with_source(anchor, "anchored_wbf_base"))
            continue
        cluster = [WeightedDetection(box=anchor, source_weight=base_weight)]
        for index, local in matches:
            assigned.add(index)
            if local.region_id is None:
                raise AssertionError("reliable CAWBF local box lost its region ID")
            q = local_q_by_region.get(local.region_id, q_min)
            cluster.append(
                WeightedDetection(
                    box=local,
                    source_weight=max(
                        local_weight_min,
                        min(local_weight_max, 1.0 + q),
                    ),
                )
            )
        fused = _fuse_cluster(cluster)
        fused_anchors.append(
            Box(
                xyxy=fused.xyxy,
                score=max(anchor.score, *(local.score for _, local in matches)),
                class_id=anchor.class_id,
                source="anchored_wbf_match",
            )
        )

    rescues: list[Box] = []
    remaining_capacity = max(0, max_det - len(fused_anchors))
    if remaining_capacity:
        for index, local in sorted(
            enumerate(reliable),
            key=lambda item: (-item[1].score, item[1].class_id, item[1].xyxy),
        ):
            if index in assigned or local.score < rescue_conf:
                continue
            if any(
                existing.class_id == local.class_id
                and box_iou(existing.xyxy, local.xyxy) >= iou_threshold
                for existing in (*fused_anchors, *rescues)
            ):
                continue
            rescues.append(_with_source(local, "anchored_wbf_rescue"))
            if len(rescues) >= remaining_capacity:
                break

    return tuple(
        sorted(
            (*fused_anchors, *rescues),
            key=lambda box: (-box.score, box.class_id, box.xyxy),
        )
    )


def _with_source(box: Box, source: str) -> Box:
    return Box(
        xyxy=box.xyxy,
        score=box.score,
        class_id=box.class_id,
        source=source,
        region_id=box.region_id,
        operation=box.operation,
    )


def _validate_parameters(
    *,
    iou_threshold: float,
    anchor_iou: float,
    probe_conf: float,
    publish_conf: float,
    rescue_conf: float,
    base_weight: float,
    local_weight_min: float,
    local_weight_max: float,
    max_det: int,
) -> None:
    if not 0.0 < iou_threshold <= 1.0 or not 0.0 < anchor_iou <= 1.0:
        raise ValueError("CAWBF IoU thresholds must be in (0, 1]")
    if anchor_iou < iou_threshold:
        raise ValueError("CAWBF anchor_iou must not be below fusion iou_threshold")
    if not 0.0 <= probe_conf < publish_conf <= rescue_conf <= 1.0:
        raise ValueError("CAWBF requires 0 <= probe_conf < publish_conf <= rescue_conf <= 1")
    if base_weight <= 0.0 or local_weight_min <= 0.0:
        raise ValueError("CAWBF source weights must be positive")
    if local_weight_min > local_weight_max:
        raise ValueError("CAWBF local weight bounds are reversed")
    if max_det <= 0:
        raise ValueError("CAWBF max_det must be positive")
