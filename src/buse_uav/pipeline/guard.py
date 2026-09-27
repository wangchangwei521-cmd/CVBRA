from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from buse_uav.detectors.base import DetectorAdapter
from buse_uav.detectors.tta import restore_horizontal_flip
from buse_uav.evaluation.timing import cpu_memory_mb, peak_vram_mb
from buse_uav.pipeline.selective import SelectiveOutput, apply_identity_crop_selection
from buse_uav.pipeline.trace import RunDirectory
from buse_uav.regions.grid import make_grid
from buse_uav.schemas import AppConfig, Box, DetectionBatch, ImageRecord
from buse_uav.scoring.uncertainty import max_region_uncertainty
from buse_uav.utils.io import atomic_write_json, atomic_write_text


def apply_duq_guard(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
    base_probe: Sequence[DetectionBatch],
) -> SelectiveOutput:
    """Execute the registered U gate and the bounded active-image action."""
    if config.method.name not in {"duq_guard", "u_flip_guard"}:
        raise ValueError("Guard requires method.name=duq_guard or u_flip_guard")
    if config.guard.score != "max_region_uncertainty":
        raise ValueError(f"unsupported Guard score: {config.guard.score}")

    base_by_id = {batch.image_id: batch for batch in base_probe}
    expected_ids = {record.image_id for record in records}
    if set(base_by_id) != expected_ids:
        raise ValueError("Guard base detections must cover every input image")

    scores: dict[int | str, float] = {}
    gate_ms_by_id: dict[int | str, float] = {}
    for record in records:
        started = perf_counter()
        regions = make_grid(
            (record.height, record.width, 3),
            rows=config.regions.rows,
            cols=config.regions.cols,
            context_padding=config.regions.context_padding,
        )
        scores[record.image_id] = max_region_uncertainty(
            regions,
            base_by_id[record.image_id],
            publish_conf=config.detector.publish_conf,
            threshold_sigma=config.scoring.threshold_sigma,
            density_kappa=config.scoring.density_kappa,
            rank_mix=config.scoring.rank_mix,
            weights=config.scoring.uncertainty_weights,
        )
        gate_ms_by_id[record.image_id] = (perf_counter() - started) * 1000.0

    ranked_ids = tuple(sorted(scores, key=lambda image_id: (-scores[image_id], str(image_id))))
    active_count = max(1, round(len(records) * config.guard.activation_rate))
    active_count = min(active_count, len(records))
    active_ids = set(ranked_ids[:active_count])
    active_records = tuple(record for record in records if record.image_id in active_ids)
    active_base = tuple(base_by_id[record.image_id] for record in active_records)

    flipped_records = _flipped_records(active_records)
    flipped_batches = detector.predict(
        flipped_records,
        imgsz=config.detector.full_imgsz,
        conf=config.detector.probe_conf,
        iou=config.detector.nms_iou,
        max_det=config.detector.max_det,
        fp16=config.detector.fp16,
    )
    active_flip = restore_horizontal_flip(flipped_batches, active_records)
    flip_by_id = {batch.image_id: batch for batch in active_flip}

    active_local_output = (
        apply_identity_crop_selection(config, run, detector, active_records, active_base)
        if config.method.name == "duq_guard"
        else _empty_local_output(config, run, active_records, active_base)
    )
    local_by_id = {batch.image_id: batch for batch in active_local_output.local_pre_fusion}
    local_timing_by_id = {row["image_id"]: row for row in active_local_output.timings}

    active_final: dict[int | str, DetectionBatch] = {}
    guard_fusion_ms: dict[int | str, float] = {}
    for record in active_records:
        image_id = record.image_id
        started = perf_counter()
        if config.method.name == "u_flip_guard" or config.guard.fusion_mode == "hard_nms_union":
            boxes = _hard_nms_union(
                base_by_id[image_id].boxes,
                flip_by_id[image_id].boxes,
                local_by_id[image_id].boxes,
                iou_threshold=config.detector.nms_iou,
                max_det=config.detector.max_det,
            )
            rescues = 0
        else:
            boxes, rescues = _consensus_anchor(
                base_by_id[image_id],
                flip_by_id[image_id],
                local_by_id[image_id],
                publish_conf=config.detector.publish_conf,
                rescue_conf=config.guard.rescue_conf,
                consensus_iou=config.guard.consensus_iou,
                max_det=config.detector.max_det,
            )
        guard_fusion_ms[image_id] = (perf_counter() - started) * 1000.0
        active_final[image_id] = DetectionBatch(
            image_id=image_id,
            boxes=boxes,
            latency_ms=(
                base_by_id[image_id].latency_ms
                + flip_by_id[image_id].latency_ms
                + local_by_id[image_id].latency_ms
            ),
            meta={
                "method": config.guard.fusion_mode,
                "guard_activated": True,
                "guard_rescues": rescues,
                "fusion_ms": guard_fusion_ms[image_id],
            },
        )

    vram = peak_vram_mb(config.experiment.device)
    memory = cpu_memory_mb()
    final: list[DetectionBatch] = []
    local_pre_fusion: list[DetectionBatch] = []
    pre_fusion: list[DetectionBatch] = []
    guard_flip: list[DetectionBatch] = []
    timings: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for record in records:
        image_id = record.image_id
        gate_ms = gate_ms_by_id[image_id]
        if image_id in active_ids:
            local = local_by_id[image_id]
            flip = flip_by_id[image_id]
            result = active_final[image_id]
            final.append(result)
            local_pre_fusion.append(local)
            guard_flip.append(flip)
            pre_fusion.append(
                DetectionBatch(
                    image_id=image_id,
                    boxes=(
                        *base_by_id[image_id].boxes,
                        *local.boxes,
                        *flip.boxes,
                    ),
                    latency_ms=result.latency_ms,
                    meta={"method": "guard_raw_union"},
                )
            )
            timing = dict(local_timing_by_id[image_id])
            previous_fusion_ms = float(timing["fusion_ms"])
            timing["flip_ms"] = flip.latency_ms
            timing["scoring_ms"] = float(timing["scoring_ms"]) + gate_ms
            timing["fusion_ms"] = guard_fusion_ms[image_id]
            timing["total_ms"] = (
                float(timing["total_ms"])
                - previous_fusion_ms
                + gate_ms
                + flip.latency_ms
                + guard_fusion_ms[image_id]
            )
            timing["num_calls"] = int(timing["num_calls"]) + 1
            timing["num_full_calls"] = 2
            timing["eic"] = float(timing["eic"]) + 1.0
            timings.append(timing)
            candidate_rows.append(
                {
                    "image_id": image_id,
                    "local_q_by_region": local.meta.get("local_q_by_region", {}),
                    "local_boxes": len(local.boxes),
                    "flip_boxes": len(flip.boxes),
                }
            )
            continue

        base = base_by_id[image_id]
        bypass_boxes = tuple(box for box in base.boxes if box.score >= config.detector.publish_conf)
        final.append(
            DetectionBatch(
                image_id=image_id,
                boxes=bypass_boxes,
                latency_ms=base.latency_ms,
                meta={"method": "guard_b0_bypass", "guard_activated": False},
            )
        )
        empty = DetectionBatch(
            image_id=image_id,
            boxes=(),
            latency_ms=0.0,
            meta={"method": "guard_b0_bypass", "local_q_by_region": {}},
        )
        local_pre_fusion.append(empty)
        guard_flip.append(empty)
        pre_fusion.append(
            DetectionBatch(
                image_id=image_id,
                boxes=base.boxes,
                latency_ms=base.latency_ms,
                meta={"method": "guard_b0_bypass_pre_fusion"},
            )
        )
        timings.append(
            {
                "run_id": run.run_id,
                "image_id": image_id,
                "load_ms": 0.0,
                "base_ms": base.latency_ms,
                "flip_ms": 0.0,
                "scoring_ms": gate_ms,
                "crop_ms": 0.0,
                "enhance_ms": 0.0,
                "candidate_ms": 0.0,
                "utility_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": base.latency_ms + gate_ms,
                "num_regions": 0,
                "num_candidates": 0,
                "num_calls": 1,
                "num_full_calls": 1,
                "eic": 1.0,
                "peak_vram_mb": vram,
                "cpu_memory_mb": memory,
            }
        )

    decisions = [
        {
            "image_id": image_id,
            "score": scores[image_id],
            "rank": rank,
            "activated": image_id in active_ids,
            "gate_ms": gate_ms_by_id[image_id],
        }
        for rank, image_id in enumerate(ranked_ids, start=1)
    ]
    atomic_write_json(
        run.path / "traces" / "guard_summary.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "method": config.method.name,
            "score": config.guard.score,
            "activation_rate": config.guard.activation_rate,
            "activated_images": active_count,
            "fusion_mode": config.guard.fusion_mode,
            "consensus_iou": config.guard.consensus_iou,
            "rescue_conf": config.guard.rescue_conf,
            "decisions": decisions,
            "active_candidates": candidate_rows,
        },
    )
    return SelectiveOutput(
        final=tuple(final),
        local_pre_fusion=tuple(local_pre_fusion),
        pre_fusion=tuple(pre_fusion),
        timings=tuple(timings),
        extra_predictions={"guard_flip": tuple(guard_flip)},
    )


def _flipped_records(records: Sequence[ImageRecord]) -> tuple[ImageRecord, ...]:
    output: list[ImageRecord] = []
    for record in records:
        image = record.image_bgr
        if image is None:
            image = cv2.imread(record.path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read Guard image: {record.path}")
        flipped = np.ascontiguousarray(cv2.flip(image, 1))
        output.append(
            replace(
                record,
                path=f"memory://guard-flip/{record.image_id}",
                image_bgr=flipped,
            )
        )
    return tuple(output)


def _empty_local_output(
    config: AppConfig,
    run: RunDirectory,
    records: Sequence[ImageRecord],
    base_probe: Sequence[DetectionBatch],
) -> SelectiveOutput:
    atomic_write_text(run.path / "traces" / "regions.jsonl", "")
    atomic_write_text(run.path / "traces" / "candidates.jsonl", "")
    atomic_write_json(
        run.path / "traces" / "selection_summary.json",
        {"schema_version": 1, "run_id": run.run_id, "method": config.method.name, "images": []},
    )
    empty = tuple(
        DetectionBatch(
            image_id=record.image_id,
            boxes=(),
            latency_ms=0.0,
            meta={"method": "u_flip_guard", "local_q_by_region": {}},
        )
        for record in records
    )
    timings = tuple(
        {
            "run_id": run.run_id,
            "image_id": record.image_id,
            "load_ms": 0.0,
            "base_ms": base.latency_ms,
            "flip_ms": 0.0,
            "scoring_ms": 0.0,
            "crop_ms": 0.0,
            "enhance_ms": 0.0,
            "candidate_ms": 0.0,
            "utility_ms": 0.0,
            "fusion_ms": 0.0,
            "total_ms": base.latency_ms,
            "num_regions": 0,
            "num_candidates": 0,
            "num_calls": 1,
            "num_full_calls": 1,
            "eic": 1.0,
            "peak_vram_mb": 0.0,
            "cpu_memory_mb": 0.0,
        }
        for record, base in zip(records, base_probe, strict=True)
    )
    return SelectiveOutput(
        final=tuple(base_probe),
        local_pre_fusion=empty,
        pre_fusion=tuple(base_probe),
        timings=timings,
    )


def _hard_nms_union(
    base: Sequence[Box],
    flip: Sequence[Box],
    local: Sequence[Box],
    *,
    iou_threshold: float,
    max_det: int,
) -> tuple[Box, ...]:
    candidates = sorted(
        (*base, *flip, *local),
        key=lambda box: (-box.score, box.class_id, box.xyxy, box.source),
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
    return tuple(kept)


def _consensus_anchor(
    base: DetectionBatch,
    flip: DetectionBatch,
    local: DetectionBatch,
    *,
    publish_conf: float,
    rescue_conf: float,
    consensus_iou: float,
    max_det: int,
) -> tuple[tuple[Box, ...], int]:
    anchors = tuple(box for box in base.boxes if box.score >= publish_conf)
    q_by_region = {
        int(region_id): float(value)
        for region_id, value in dict(local.meta.get("local_q_by_region", {})).items()
    }
    rescues: list[Box] = []
    for candidate in sorted(local.boxes, key=lambda box: (-box.score, box.class_id, box.xyxy)):
        if candidate.region_id is None or q_by_region.get(candidate.region_id, 0.0) <= 0.0:
            continue
        if candidate.score < rescue_conf:
            continue
        if any(
            anchor.class_id == candidate.class_id
            and _iou(anchor.xyxy, candidate.xyxy) >= consensus_iou
            for anchor in anchors
        ):
            continue
        supports = [
            support
            for support in flip.boxes
            if support.class_id == candidate.class_id
            and support.score >= rescue_conf
            and _iou(support.xyxy, candidate.xyxy) >= consensus_iou
        ]
        if not supports:
            continue
        support = max(supports, key=lambda box: (_iou(box.xyxy, candidate.xyxy), box.score))
        rescue = Box(
            xyxy=candidate.xyxy,
            score=min(candidate.score, support.score),
            class_id=candidate.class_id,
            source="guard_consensus_rescue",
            region_id=candidate.region_id,
            operation=candidate.operation,
        )
        if any(
            existing.class_id == rescue.class_id and _iou(existing.xyxy, rescue.xyxy) > 0.70
            for existing in rescues
        ):
            continue
        rescues.append(rescue)
    capacity = max(0, max_det - len(anchors))
    rescues = sorted(rescues, key=lambda box: (-box.score, box.class_id, box.xyxy))[:capacity]
    return (*anchors, *rescues), len(rescues)


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
