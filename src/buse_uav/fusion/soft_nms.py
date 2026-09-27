from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from buse_uav.schemas import Box
from buse_uav.utility.matching import box_iou


def soft_nms(
    base_boxes: Sequence[Box],
    local_boxes: Sequence[Box],
    *,
    local_q_by_region: Mapping[int, float],
    sigma: float,
    probe_conf: float,
    publish_conf: float,
    local_weight_min: float,
    local_weight_max: float,
    max_det: int,
) -> tuple[Box, ...]:
    """Apply deterministic class-wise Gaussian Soft-NMS."""
    if sigma <= 0.0:
        raise ValueError("Soft-NMS sigma must be positive")
    candidates = [replace(box, source="soft_nms") for box in base_boxes if box.score >= probe_conf]
    candidates.extend(
        replace(
            box,
            score=min(
                1.0,
                box.score
                * max(
                    local_weight_min,
                    min(
                        local_weight_max,
                        1.0
                        + (
                            local_q_by_region.get(box.region_id, 0.0)
                            if box.region_id is not None
                            else 0.0
                        ),
                    ),
                ),
            ),
            source="soft_nms",
        )
        for box in local_boxes
        if box.score >= probe_conf
    )
    kept: list[Box] = []
    while candidates and len(kept) < max_det:
        candidates.sort(key=lambda box: (-box.score, box.class_id, box.xyxy))
        best = candidates.pop(0)
        if best.score < publish_conf:
            break
        kept.append(best)
        decayed: list[Box] = []
        for candidate in candidates:
            score = candidate.score
            if candidate.class_id == best.class_id:
                overlap = box_iou(best.xyxy, candidate.xyxy)
                score *= math.exp(-(overlap * overlap) / sigma)
            if score >= probe_conf:
                decayed.append(replace(candidate, score=score))
        candidates = decayed
    return tuple(kept)
