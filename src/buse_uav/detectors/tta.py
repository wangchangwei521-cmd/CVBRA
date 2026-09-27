from __future__ import annotations

from collections.abc import Sequence

from buse_uav.schemas import Box, DetectionBatch, ImageRecord


def restore_horizontal_flip(
    batches: Sequence[DetectionBatch],
    records: Sequence[ImageRecord],
) -> tuple[DetectionBatch, ...]:
    records_by_id = {record.image_id: record for record in records}
    restored: list[DetectionBatch] = []
    for batch in batches:
        record = records_by_id[batch.image_id]
        boxes = tuple(
            Box(
                xyxy=(
                    record.width - box.xyxy[2],
                    box.xyxy[1],
                    record.width - box.xyxy[0],
                    box.xyxy[3],
                ),
                score=box.score,
                class_id=box.class_id,
                source="flip_tta",
            )
            for box in batch.boxes
        )
        restored.append(
            DetectionBatch(
                image_id=batch.image_id,
                boxes=boxes,
                latency_ms=batch.latency_ms,
                meta={**batch.meta, "restored_horizontal_flip": True},
            )
        )
    return tuple(restored)


def merge_classwise_nms(
    first: Sequence[DetectionBatch],
    second: Sequence[DetectionBatch],
    *,
    iou_threshold: float,
    max_det: int,
    method: str = "flip_tta",
) -> tuple[DetectionBatch, ...]:
    second_by_id = {batch.image_id: batch for batch in second}
    merged: list[DetectionBatch] = []
    for batch in first:
        other = second_by_id[batch.image_id]
        candidates = sorted(
            (*batch.boxes, *other.boxes),
            key=lambda box: (-box.score, box.class_id, box.xyxy),
        )
        kept: list[Box] = []
        for candidate in candidates:
            if any(
                existing.class_id == candidate.class_id
                and _iou(existing.xyxy, candidate.xyxy) > iou_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)
            if len(kept) >= max_det:
                break
        merged.append(
            DetectionBatch(
                image_id=batch.image_id,
                boxes=tuple(kept),
                latency_ms=batch.latency_ms + other.latency_ms,
                meta={"method": method, "sources": 2},
            )
        )
    return tuple(merged)


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
