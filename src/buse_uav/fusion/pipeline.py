from __future__ import annotations

import math
from collections.abc import Sequence

from buse_uav.evaluation.timing import SynchronizedTimer
from buse_uav.fusion.anchored_wbf import anchored_weighted_box_fusion
from buse_uav.fusion.soft_nms import soft_nms
from buse_uav.fusion.wbf import weighted_box_fusion
from buse_uav.schemas import AppConfig, Box, DetectionBatch, ImageRecord


def fuse_detection_batches(
    config: AppConfig,
    base_batches: Sequence[DetectionBatch],
    local_batches: Sequence[DetectionBatch],
    records: Sequence[ImageRecord],
) -> tuple[DetectionBatch, ...]:
    """Fuse base/local batches and validate all published geometry."""
    local_by_id = {batch.image_id: batch for batch in local_batches}
    record_by_id = {record.image_id: record for record in records}
    if set(local_by_id) != set(record_by_id):
        raise ValueError("local fusion batches must cover every input image")
    effective_method = config.fusion.method if config.method.fusion_enabled else "hard_nms"
    output: list[DetectionBatch] = []
    for base in base_batches:
        record = record_by_id[base.image_id]
        local = local_by_id[base.image_id]
        with SynchronizedTimer(config.experiment.device) as fusion_timer:
            if not local.boxes:
                boxes = tuple(
                    box for box in base.boxes if box.score >= config.detector.publish_conf
                )
            else:
                raw_q = local.meta.get("local_q_by_region", {})
                if not isinstance(raw_q, dict):
                    raise ValueError("local_q_by_region must be a mapping")
                local_q_by_region = {int(key): float(value) for key, value in raw_q.items()}
                if not config.method.fusion_enabled:
                    boxes = _hard_nms(
                        (*base.boxes, *local.boxes),
                        iou_threshold=config.fusion.iou,
                        publish_conf=config.detector.publish_conf,
                        max_det=config.detector.max_det,
                    )
                elif config.fusion.method == "wbf":
                    boxes = weighted_box_fusion(
                        base.boxes,
                        local.boxes,
                        local_q_by_region=local_q_by_region,
                        iou_threshold=config.fusion.iou,
                        probe_conf=config.detector.probe_conf,
                        publish_conf=config.detector.publish_conf,
                        base_weight=config.fusion.base_weight,
                        local_weight_min=config.fusion.local_weight_min,
                        local_weight_max=config.fusion.local_weight_max,
                        max_det=config.detector.max_det,
                    )
                elif config.fusion.method == "anchored_wbf":
                    boxes = anchored_weighted_box_fusion(
                        base.boxes,
                        local.boxes,
                        local_q_by_region=local_q_by_region,
                        iou_threshold=config.fusion.iou,
                        anchor_iou=config.fusion.anchor_iou,
                        q_min=config.fusion.reliability_q_min,
                        probe_conf=config.detector.probe_conf,
                        publish_conf=config.detector.publish_conf,
                        rescue_conf=config.fusion.rescue_conf,
                        base_weight=config.fusion.base_weight,
                        local_weight_min=config.fusion.local_weight_min,
                        local_weight_max=config.fusion.local_weight_max,
                        max_det=config.detector.max_det,
                    )
                elif config.fusion.method == "soft_nms":
                    boxes = soft_nms(
                        base.boxes,
                        local.boxes,
                        local_q_by_region=local_q_by_region,
                        sigma=config.fusion.soft_nms_sigma,
                        probe_conf=config.detector.probe_conf,
                        publish_conf=config.detector.publish_conf,
                        local_weight_min=config.fusion.local_weight_min,
                        local_weight_max=config.fusion.local_weight_max,
                        max_det=config.detector.max_det,
                    )
                else:
                    boxes = _hard_nms(
                        (*base.boxes, *local.boxes),
                        iou_threshold=config.fusion.iou,
                        publish_conf=config.detector.publish_conf,
                        max_det=config.detector.max_det,
                    )
            boxes, max_boundary_correction = _clip_fusion_boxes(boxes, record)
            _validate_boxes(boxes, record)
        output.append(
            DetectionBatch(
                image_id=base.image_id,
                boxes=boxes,
                latency_ms=base.latency_ms + local.latency_ms,
                meta={
                    "method": effective_method,
                    "base_boxes": len(base.boxes),
                    "local_boxes": len(local.boxes),
                    "max_boundary_correction_px": max_boundary_correction,
                    "fusion_ms": fusion_timer.elapsed_ms,
                    "cuda_synchronized": fusion_timer.synchronized,
                },
            )
        )
    return tuple(output)


def concatenate_pre_fusion(
    base_batches: Sequence[DetectionBatch],
    local_batches: Sequence[DetectionBatch],
) -> tuple[DetectionBatch, ...]:
    local_by_id = {batch.image_id: batch for batch in local_batches}
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=(*batch.boxes, *local_by_id[batch.image_id].boxes),
            latency_ms=batch.latency_ms + local_by_id[batch.image_id].latency_ms,
            meta={"method": "pre_fusion"},
        )
        for batch in base_batches
    )


def _hard_nms(
    boxes: Sequence[Box],
    *,
    iou_threshold: float,
    publish_conf: float,
    max_det: int,
) -> tuple[Box, ...]:
    from buse_uav.utility.matching import box_iou

    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: (-item.score, item.class_id, item.xyxy)):
        if box.score < publish_conf:
            continue
        if any(
            existing.class_id == box.class_id and box_iou(existing.xyxy, box.xyxy) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(replace_source(box, "hard_nms"))
        if len(kept) >= max_det:
            break
    return tuple(kept)


def replace_source(box: Box, source: str) -> Box:
    return Box(
        xyxy=box.xyxy,
        score=box.score,
        class_id=box.class_id,
        source=source,
        region_id=box.region_id,
        operation=box.operation,
    )


def _validate_boxes(boxes: Sequence[Box], record: ImageRecord) -> None:
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        if not all(math.isfinite(value) for value in (*box.xyxy, box.score)):
            raise ValueError(f"fusion produced non-finite box for image {record.image_id}")
        if not (0.0 <= x1 < x2 <= record.width and 0.0 <= y1 < y2 <= record.height):
            raise ValueError(f"fusion produced out-of-bounds box for image {record.image_id}")
        if not 0.0 <= box.score <= 1.0:
            raise ValueError(f"fusion produced invalid score for image {record.image_id}")


def _clip_fusion_boxes(
    boxes: Sequence[Box],
    record: ImageRecord,
) -> tuple[tuple[Box, ...], float]:
    clipped: list[Box] = []
    max_correction = 0.0
    for box in boxes:
        if not all(math.isfinite(value) for value in (*box.xyxy, box.score)):
            raise ValueError(f"fusion produced non-finite box for image {record.image_id}")
        x1, y1, x2, y2 = box.xyxy
        coordinates = (
            max(0.0, min(float(record.width), x1)),
            max(0.0, min(float(record.height), y1)),
            max(0.0, min(float(record.width), x2)),
            max(0.0, min(float(record.height), y2)),
        )
        correction = max(
            abs(original - bounded)
            for original, bounded in zip(box.xyxy, coordinates)  # noqa: B905
        )
        max_correction = max(max_correction, correction)
        if correction > 1.0 + 1e-9:
            raise ValueError(
                f"fusion box for image {record.image_id} requires "
                f"{correction:.6f}px boundary correction: {box.xyxy}"
            )
        if coordinates[2] - coordinates[0] < 1.0 or coordinates[3] - coordinates[1] < 1.0:
            continue
        clipped.append(
            Box(
                xyxy=coordinates,
                score=box.score,
                class_id=box.class_id,
                source=box.source,
                region_id=box.region_id,
                operation=box.operation,
            )
        )
    return tuple(clipped), max_correction
