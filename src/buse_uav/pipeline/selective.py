from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from buse_uav.data.common import DataError
from buse_uav.detectors.base import DetectorAdapter
from buse_uav.enhancements.candidates import make_enhancer, route_top_operations
from buse_uav.evaluation.timing import (
    SynchronizedTimer,
    cpu_memory_mb,
    equivalent_inference_cost,
    peak_vram_mb,
)
from buse_uav.fusion.pipeline import concatenate_pre_fusion, fuse_detection_batches
from buse_uav.pipeline.intermediates import intermediate_shard_path, score_map_path
from buse_uav.pipeline.trace import RunDirectory
from buse_uav.regions.grid import make_grid
from buse_uav.regions.selection import SelectionResult, combine_scores, select_regions
from buse_uav.schemas import AppConfig, Box, DetectionBatch, ImageRecord, Region, RegionScore
from buse_uav.scoring.degradation import QualityCalibration, score_degradation
from buse_uav.scoring.uncertainty import score_uncertainty
from buse_uav.utility.matching import fuse_reference
from buse_uav.utility.task_utility import (
    CandidateEvaluation,
    choose_best_candidate,
    evaluate_candidate,
    identity_evaluation,
    should_early_stop,
    with_operation,
)
from buse_uav.utils.io import atomic_write_json, atomic_write_text


@dataclass(frozen=True)
class SelectiveOutput:
    final: tuple[DetectionBatch, ...]
    local_pre_fusion: tuple[DetectionBatch, ...]
    pre_fusion: tuple[DetectionBatch, ...]
    timings: tuple[dict[str, Any], ...]
    extra_predictions: dict[str, tuple[DetectionBatch, ...]] = field(default_factory=dict)


def apply_identity_crop_selection(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
    base_probe: Sequence[DetectionBatch],
    flip_probe: Sequence[DetectionBatch] | None = None,
) -> SelectiveOutput:
    """Run Phase 4/5 selection, local candidates, and traceable redetection."""
    calibration = QualityCalibration.from_file(config.scoring.calibration)
    base_by_id = {batch.image_id: batch for batch in base_probe}
    flip_by_id = {batch.image_id: batch for batch in flip_probe} if flip_probe is not None else {}
    crop_records: list[ImageRecord] = []
    crop_context: dict[int | str, tuple[ImageRecord, Region]] = {}
    assessment_by_crop: dict[int | str, RegionScore] = {}
    trace_rows: list[dict[str, Any]] = []
    selection_summaries: list[dict[str, Any]] = []
    timings = {
        record.image_id: {
            "load_ms": 0.0,
            "base_ms": base_by_id[record.image_id].latency_ms,
            "flip_ms": flip_by_id[record.image_id].latency_ms if flip_by_id else 0.0,
            "scoring_ms": 0.0,
            "crop_ms": 0.0,
            "enhance_ms": 0.0,
            "candidate_ms": 0.0,
            "utility_ms": 0.0,
            "fusion_ms": 0.0,
            "num_regions": 0,
            "num_candidates": 0,
        }
        for record in records
    }

    for record in records:
        with SynchronizedTimer(config.experiment.device) as load_timer:
            image_bgr = _record_bgr(record)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        timings[record.image_id]["load_ms"] += load_timer.elapsed_ms
        with SynchronizedTimer(config.experiment.device) as scoring_timer:
            regions = make_grid(
                image_rgb.shape,
                rows=config.regions.rows,
                cols=config.regions.cols,
                context_padding=config.regions.context_padding,
            )
            degradation = score_degradation(
                image_rgb,
                regions,
                calibration=calibration,
                rank_mix=config.scoring.rank_mix,
                weights=config.scoring.degradation_weights,
                workers=config.runtime.degradation_workers,
            )
            uncertainty = score_uncertainty(
                image_rgb,
                regions,
                base_by_id[record.image_id],
                publish_conf=config.detector.publish_conf,
                threshold_sigma=config.scoring.threshold_sigma,
                density_kappa=config.scoring.density_kappa,
                rank_mix=config.scoring.rank_mix,
                weights=config.scoring.uncertainty_weights,
                flip_detections=flip_by_id.get(record.image_id),
                flip_match_iou=config.utility.match_iou,
                flip_weight=config.scoring.flip_consistency.weight,
            )
            assessments = combine_scores(
                regions,
                degradation,
                uncertainty,
                alpha=config.difficulty.alpha,
                interaction_lambda=config.difficulty.interaction_lambda,
                blank_suppression=config.regions.blank_suppression,
            )
            selection = select_regions(
                regions,
                assessments,
                selection=config.method.selection,
                area_budget=config.regions.area_budget,
                max_regions=config.regions.max_regions,
                seed=config.project.seed,
                image_id=record.image_id,
            )
            selected_ids = {region.id for region in selection.regions}
            assessment_by_id = {score.region_id: score for score in assessments}
            selection_rank = {
                region_id: rank for rank, region_id in enumerate(selection.ranking, start=1)
            }
            trace_rows.extend(
                _trace_rows(
                    run,
                    record,
                    regions,
                    assessments,
                    selection,
                    selection_rank=selection_rank,
                )
            )
        timings[record.image_id]["scoring_ms"] += scoring_timer.elapsed_ms
        timings[record.image_id]["num_regions"] = len(selection.regions)
        timings[record.image_id]["num_candidates"] = len(selection.regions)
        selection_summaries.append(
            {
                "image_id": record.image_id,
                "selection": config.method.selection,
                "selected_region_ids": [region.id for region in selection.regions],
                "actual_area_ratio": selection.actual_area_ratio,
                "area_budget": config.regions.area_budget,
                "max_regions": config.regions.max_regions,
            }
        )
        diagnostic_path = score_map_path(config, run, record.image_id)
        if diagnostic_path is not None:
            _save_score_map(
                diagnostic_path,
                image_bgr,
                regions,
                assessments,
                selected_ids=selected_ids,
            )
        with SynchronizedTimer(config.experiment.device) as crop_timer:
            for region in selection.regions:
                crop_id = f"{record.image_id}::region::{region.id}"
                x1, y1, x2, y2 = region.crop_xyxy
                crop = image_bgr[y1:y2, x1:x2]
                crop_image: np.ndarray | None = None
                if config.runtime.in_memory_candidates:
                    crop_image = np.ascontiguousarray(crop.copy())
                    crop_path = f"memory://identity/{record.image_id}/{region.id}"
                else:
                    destination = intermediate_shard_path(
                        config,
                        run,
                        "identity",
                        f"{record.image_id}_{region.id}.png",
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), crop):
                        raise DataError(f"OpenCV cannot write identity crop {destination}")
                    crop_path = str(destination)
                crop_record = ImageRecord(
                    image_id=crop_id,
                    path=crop_path,
                    width=x2 - x1,
                    height=y2 - y1,
                    image_bgr=crop_image,
                )
                crop_records.append(crop_record)
                crop_context[crop_id] = (record, region)
                assessment_by_crop[crop_id] = assessment_by_id[region.id]
        timings[record.image_id]["crop_ms"] += crop_timer.elapsed_ms

    _write_region_trace(run.path / "traces" / "regions.jsonl", trace_rows)
    atomic_write_json(
        run.path / "traces" / "selection_summary.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "method": config.method.name,
            "selection": config.method.selection,
            "images": selection_summaries,
        },
    )
    if not crop_records:
        empty_local = tuple(
            DetectionBatch(
                image_id=record.image_id,
                boxes=(),
                latency_ms=0.0,
                meta={"method": "zero_budget", "local_q_by_region": {}},
            )
            for record in records
        )
        final = fuse_detection_batches(config, base_probe, empty_local, records)
        return SelectiveOutput(
            final=final,
            local_pre_fusion=empty_local,
            pre_fusion=concatenate_pre_fusion(base_probe, empty_local),
            timings=_finalize_timing_rows(config, run, records, final, timings),
        )

    crop_batches = detector.predict(
        crop_records,
        imgsz=config.detector.crop_imgsz,
        conf=config.detector.probe_conf,
        iou=config.detector.nms_iou,
        max_det=config.detector.max_det,
        fp16=config.detector.fp16,
    )
    local_latency: dict[int | str, float] = defaultdict(float)
    mapped_identity_by_crop: dict[int | str, tuple[Box, ...]] = {}
    for crop_batch in crop_batches:
        original_record, region = crop_context[crop_batch.image_id]
        local_latency[original_record.image_id] += crop_batch.latency_ms
        timings[original_record.image_id]["candidate_ms"] += crop_batch.latency_ms
        mapped_boxes: list[Box] = []
        for box in crop_batch.boxes:
            mapped = _map_and_filter_local_box(
                box,
                region=region,
                image_width=original_record.width,
                image_height=original_record.height,
                operation="identity",
            )
            if mapped is not None:
                mapped_boxes.append(mapped)
        mapped_identity_by_crop[crop_batch.image_id] = tuple(mapped_boxes)

    if config.method.enhancement_enabled:
        local_batches = _apply_enhancement_candidates(
            config,
            run,
            detector,
            records,
            base_by_id=base_by_id,
            crop_records=crop_records,
            crop_context=crop_context,
            assessment_by_crop=assessment_by_crop,
            mapped_identity_by_crop=mapped_identity_by_crop,
            local_latency=local_latency,
            timings=timings,
        )
    else:
        local_by_image: dict[int | str, list[Box]] = defaultdict(list)
        local_q_by_image: dict[int | str, dict[int, float]] = defaultdict(dict)
        for identity_crop_id, boxes in mapped_identity_by_crop.items():
            original_record, region = crop_context[identity_crop_id]
            local_by_image[original_record.image_id].extend(boxes)
            local_q_by_image[original_record.image_id][region.id] = 0.0
        selected_count = {
            item["image_id"]: len(item["selected_region_ids"]) for item in selection_summaries
        }
        local_batches = tuple(
            DetectionBatch(
                image_id=record.image_id,
                boxes=tuple(local_by_image[record.image_id]),
                latency_ms=local_latency[record.image_id],
                meta={
                    "method": "selected_identity_crop",
                    "selected_regions": selected_count[record.image_id],
                    "local_q_by_region": local_q_by_image[record.image_id],
                },
            )
            for record in records
        )
    pre_fusion = concatenate_pre_fusion(base_probe, local_batches)
    final = fuse_detection_batches(
        config,
        base_probe,
        local_batches,
        records,
    )
    return SelectiveOutput(
        final=final,
        local_pre_fusion=local_batches,
        pre_fusion=pre_fusion,
        timings=_finalize_timing_rows(config, run, records, final, timings),
    )


def _apply_enhancement_candidates(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
    *,
    base_by_id: dict[int | str, DetectionBatch],
    crop_records: Sequence[ImageRecord],
    crop_context: dict[int | str, tuple[ImageRecord, Region]],
    assessment_by_crop: dict[int | str, RegionScore],
    mapped_identity_by_crop: dict[int | str, tuple[Box, ...]],
    local_latency: dict[int | str, float],
    timings: dict[int | str, dict[str, float | int]],
) -> tuple[DetectionBatch, ...]:
    crop_record_by_id = {record.image_id: record for record in crop_records}
    references: dict[int | str, tuple[Box, ...]] = {}
    evaluations: dict[int | str, list[CandidateEvaluation]] = {}
    operation_plans: dict[int | str, tuple[str, ...]] = {}
    for crop_id, (original_record, region) in crop_context.items():
        with SynchronizedTimer(config.experiment.device) as utility_timer:
            reference = fuse_reference(
                _boxes_centered_in_region(base_by_id[original_record.image_id].boxes, region),
                mapped_identity_by_crop[crop_id],
                iou_threshold=config.fusion.iou,
            )
            references[crop_id] = reference
            evaluations[crop_id] = [
                identity_evaluation(
                    mapped_identity_by_crop[crop_id],
                    num_reference=len(reference),
                )
            ]
            assessment = assessment_by_crop[crop_id]
            degradation_components = {
                name: assessment.components[f"d_{name}"]
                for name in ("luminance", "contrast", "blur", "haze", "entropy")
            }
            operation_plans[crop_id] = route_top_operations(
                degradation_components,
                available=config.enhancement.candidates,
                max_ops=config.enhancement.max_ops_per_region,
            )
        timings[original_record.image_id]["utility_ms"] += utility_timer.elapsed_ms

    stopped: set[int | str] = set()
    for round_index in range(config.enhancement.max_ops_per_region):
        candidate_records: list[ImageRecord] = []
        candidate_context: dict[int | str, tuple[int | str, str]] = {}
        for crop_id, operations in operation_plans.items():
            if crop_id in stopped or round_index >= len(operations):
                continue
            operation = operations[round_index]
            original_record, _ = crop_context[crop_id]
            with SynchronizedTimer(config.experiment.device) as enhancement_timer:
                candidate_record = _make_candidate_record(
                    config,
                    run,
                    crop_record_by_id[crop_id],
                    operation=operation,
                    round_index=round_index,
                )
            timings[original_record.image_id]["enhance_ms"] += enhancement_timer.elapsed_ms
            timings[original_record.image_id]["num_candidates"] += 1
            candidate_records.append(candidate_record)
            candidate_context[candidate_record.image_id] = (crop_id, operation)
        if not candidate_records:
            continue
        candidate_batches = _predict_candidate_records(
            config,
            detector,
            candidate_records,
        )
        for candidate_batch in candidate_batches:
            crop_id, operation = candidate_context[candidate_batch.image_id]
            original_record, region = crop_context[crop_id]
            local_latency[original_record.image_id] += candidate_batch.latency_ms
            timings[original_record.image_id]["candidate_ms"] += candidate_batch.latency_ms
            with SynchronizedTimer(config.experiment.device) as utility_timer:
                mapped = tuple(
                    mapped_box
                    for box in candidate_batch.boxes
                    if (
                        mapped_box := _map_and_filter_local_box(
                            box,
                            region=region,
                            image_width=original_record.width,
                            image_height=original_record.height,
                            operation=operation,
                        )
                    )
                    is not None
                )
                evaluation = with_operation(
                    evaluate_candidate(
                        references[crop_id],
                        mapped,
                        publish_conf=config.detector.publish_conf,
                        match_iou=config.utility.match_iou,
                        max_count_ratio=config.utility.max_count_ratio,
                        crop_imgsz=config.detector.crop_imgsz,
                        full_imgsz=config.detector.full_imgsz,
                        weights=config.utility.weights,
                    ),
                    operation,
                )
                early_stopped = (
                    round_index == 0
                    and config.method.early_stop_enabled
                    and should_early_stop(
                        evaluation.utility,
                        q_threshold=config.enhancement.early_stop_q,
                        min_stability=config.enhancement.early_stop_min_stability,
                        max_unsupported_fp=(config.enhancement.early_stop_max_unsupported_fp),
                    )
                )
                if early_stopped:
                    evaluation = replace(evaluation, early_stopped=True)
                    stopped.add(crop_id)
                evaluations[crop_id].append(evaluation)
            timings[original_record.image_id]["utility_ms"] += utility_timer.elapsed_ms

    local_by_image: dict[int | str, list[Box]] = defaultdict(list)
    local_q_by_image: dict[int | str, dict[int, float]] = defaultdict(dict)
    candidate_trace: list[dict[str, Any]] = []
    choice_summaries: list[dict[str, Any]] = []
    for crop_id, crop_evaluations in evaluations.items():
        original_record, region = crop_context[crop_id]
        with SynchronizedTimer(config.experiment.device) as utility_timer:
            identity = crop_evaluations[0]
            choice = choose_best_candidate(identity, crop_evaluations[1:])
        timings[original_record.image_id]["utility_ms"] += utility_timer.elapsed_ms
        conservative_gated = config.utility.conservative_gate and choice.q <= 0.0
        if not conservative_gated:
            local_by_image[original_record.image_id].extend(choice.predictions)
        local_q_by_image[original_record.image_id][region.id] = choice.q
        choice_summaries.append(
            {
                "image_id": original_record.image_id,
                "region_id": region.id,
                "operation": choice.operation,
                "q": choice.q,
                "conservative_gated": conservative_gated,
                "attempted": [evaluation.operation for evaluation in crop_evaluations],
            }
        )
        for evaluation in crop_evaluations:
            utility = evaluation.utility
            candidate_trace.append(
                {
                    "run_id": run.run_id,
                    "image_id": original_record.image_id,
                    "region_id": region.id,
                    "operation": evaluation.operation,
                    "num_reference": evaluation.num_reference,
                    "num_predictions": evaluation.num_predictions,
                    "num_matches": evaluation.num_matches,
                    "confidence_gain": utility.confidence_gain,
                    "stability": utility.stability,
                    "rescue": utility.rescue,
                    "unsupported_fp": utility.unsupported_fp,
                    "count_explosion": utility.count_explosion,
                    "compute": utility.compute,
                    "Q": utility.q,
                    "selected": evaluation.operation == choice.operation,
                    "early_stopped": evaluation.early_stopped,
                }
            )
    if config.runtime.save_trace:
        _write_region_trace(run.path / "traces" / "candidates.jsonl", candidate_trace)
        atomic_write_json(
            run.path / "traces" / "candidate_summary.json",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "images": choice_summaries,
            },
        )
    selected_counts: dict[int | str, int] = defaultdict(int)
    for original_record, _ in crop_context.values():
        selected_counts[original_record.image_id] += 1
    return tuple(
        DetectionBatch(
            image_id=record.image_id,
            boxes=tuple(local_by_image[record.image_id]),
            latency_ms=local_latency[record.image_id],
            meta={
                "method": "buse_candidates",
                "selected_regions": selected_counts[record.image_id],
                "local_q_by_region": local_q_by_image[record.image_id],
            },
        )
        for record in records
    )


def _make_candidate_record(
    config: AppConfig,
    run: RunDirectory,
    crop_record: ImageRecord,
    *,
    operation: str,
    round_index: int,
) -> ImageRecord:
    crop_bgr = _record_bgr(crop_record)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    enhancer = make_enhancer(
        operation,
        gamma=config.enhancement.gamma,
        clahe_clip_limit=config.enhancement.clahe_clip_limit,
        clahe_tile_grid=config.enhancement.clahe_tile_grid,
        unsharp_amount=config.enhancement.unsharp_amount,
        unsharp_sigma=config.enhancement.unsharp_sigma,
    )
    enhanced_rgb = enhancer.apply(crop_rgb)
    original_record, region = _parse_crop_id(crop_record.image_id)
    enhanced_bgr = np.ascontiguousarray(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
    candidate_image: np.ndarray | None = None
    if config.runtime.in_memory_candidates:
        candidate_image = enhanced_bgr
        candidate_path = f"memory://candidate/{original_record}/{region}/{round_index}/{operation}"
    else:
        destination = intermediate_shard_path(
            config,
            run,
            operation,
            f"{original_record}_{region}_{round_index}.png",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), enhanced_bgr):
            raise DataError(f"OpenCV cannot write enhanced candidate {destination}")
        candidate_path = str(destination)
    return ImageRecord(
        image_id=f"{crop_record.image_id}::candidate::{round_index}::{operation}",
        path=candidate_path,
        width=crop_record.width,
        height=crop_record.height,
        image_bgr=candidate_image,
    )


def _predict_candidate_records(
    config: AppConfig,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
) -> tuple[DetectionBatch, ...]:
    if config.runtime.batch_candidates:
        if config.runtime.cross_shape_candidate_batching:
            return detector.predict(
                records,
                imgsz=config.detector.crop_imgsz,
                conf=config.detector.probe_conf,
                iou=config.detector.nms_iou,
                max_det=config.detector.max_det,
                fp16=config.detector.fp16,
            )
        groups: dict[tuple[int, int], list[ImageRecord]] = defaultdict(list)
        for record in records:
            groups[(record.width, record.height)].append(record)
        by_id: dict[int | str, DetectionBatch] = {}
        for group in groups.values():
            for batch in detector.predict(
                group,
                imgsz=config.detector.crop_imgsz,
                conf=config.detector.probe_conf,
                iou=config.detector.nms_iou,
                max_det=config.detector.max_det,
                fp16=config.detector.fp16,
            ):
                by_id[batch.image_id] = batch
        return tuple(by_id[record.image_id] for record in records)
    batches: list[DetectionBatch] = []
    for record in records:
        batches.extend(
            detector.predict(
                [record],
                imgsz=config.detector.crop_imgsz,
                conf=config.detector.probe_conf,
                iou=config.detector.nms_iou,
                max_det=config.detector.max_det,
                fp16=config.detector.fp16,
            )
        )
    return tuple(batches)


def _map_and_filter_local_box(
    box: Box,
    *,
    region: Region,
    image_width: int,
    image_height: int,
    operation: str,
) -> Box | None:
    if not all(math.isfinite(value) for value in (*box.xyxy, box.score)):
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = region.crop_xyxy
    x1 = max(0.0, min(float(image_width), box.xyxy[0] + crop_x1))
    y1 = max(0.0, min(float(image_height), box.xyxy[1] + crop_y1))
    x2 = max(0.0, min(float(image_width), box.xyxy[2] + crop_x1))
    y2 = max(0.0, min(float(image_height), box.xyxy[3] + crop_y1))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    core_x1, core_y1, core_x2, core_y2 = region.core_xyxy
    if not (core_x1 <= center_x < core_x2 and core_y1 <= center_y < core_y2):
        return None
    intersection_x1 = max(x1, float(crop_x1))
    intersection_y1 = max(y1, float(crop_y1))
    intersection_x2 = min(x2, float(crop_x2))
    intersection_y2 = min(y2, float(crop_y2))
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    if intersection / ((x2 - x1) * (y2 - y1)) < 0.8:
        return None
    return Box(
        xyxy=(x1, y1, x2, y2),
        score=box.score,
        class_id=box.class_id,
        source=f"selected_{operation}_crop",
        region_id=region.id,
        operation=operation,
    )


def _boxes_centered_in_region(
    boxes: Sequence[Box],
    region: Region,
) -> tuple[Box, ...]:
    x1, y1, x2, y2 = region.core_xyxy
    return tuple(
        box
        for box in boxes
        if x1 <= (box.xyxy[0] + box.xyxy[2]) / 2.0 < x2
        and y1 <= (box.xyxy[1] + box.xyxy[3]) / 2.0 < y2
    )


def _parse_crop_id(crop_id: int | str) -> tuple[str, str]:
    parts = str(crop_id).split("::")
    if len(parts) != 3 or parts[1] != "region":
        raise ValueError(f"invalid crop image ID: {crop_id}")
    return parts[0], parts[2]


def _trace_rows(
    run: RunDirectory,
    record: ImageRecord,
    regions: Sequence[Region],
    assessments: Sequence[RegionScore],
    selection: SelectionResult,
    *,
    selection_rank: dict[int, int],
) -> list[dict[str, Any]]:
    region_by_id = {region.id: region for region in regions}
    selected_ids = {region.id for region in selection.regions}
    rows: list[dict[str, Any]] = []
    for score in assessments:
        region = region_by_id[score.region_id]
        components = score.components
        rows.append(
            {
                "run_id": run.run_id,
                "image_id": record.image_id,
                "region_id": region.id,
                "core_xyxy": list(region.core_xyxy),
                "crop_xyxy": list(region.crop_xyxy),
                "area_ratio": region.area_ratio,
                "d_lum": components["d_luminance"],
                "d_con": components["d_contrast"],
                "d_blur": components["d_blur"],
                "d_haze": components["d_haze"],
                "d_ent": components["d_entropy"],
                "D": score.degradation,
                "u_entropy": components["u_confidence_entropy"],
                "u_threshold": components["u_threshold_proximity"],
                "u_conflict": components["u_class_conflict"],
                "u_density": components["u_low_conf_density"],
                "u_flip": components.get("u_flip_consistency", 0.0),
                "U": score.uncertainty,
                "S": score.difficulty,
                "edge_density": components["edge_density"],
                "candidate_count": round(components["candidate_count"]),
                "blank_suppressed": bool(components["blank_suppressed"]),
                "selected": region.id in selected_ids,
                "selection_rank": selection_rank[region.id],
                "selection_score": selection.scores_by_region[region.id],
            }
        )
    return rows


def _write_region_trace(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _save_score_map(
    path: Path,
    image: np.ndarray,
    regions: Sequence[Region],
    scores: Sequence[RegionScore],
    *,
    selected_ids: set[int],
) -> None:
    score_by_id = {score.region_id: score for score in scores}
    panels: list[np.ndarray] = []
    target_width = min(720, image.shape[1])
    target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]
    for title, attribute in (
        ("D degradation", "degradation"),
        ("U uncertainty", "uncertainty"),
        ("S difficulty", "difficulty"),
    ):
        panel = cv2.resize(image, (target_width, target_height))
        for region in regions:
            value = float(getattr(score_by_id[region.id], attribute))
            color = cv2.applyColorMap(
                np.asarray([[[round(value * 255.0)]]], dtype=np.uint8),
                cv2.COLORMAP_TURBO,
            )[0, 0]
            x1, y1, x2, y2 = region.core_xyxy
            x1 = round(x1 * scale_x)
            y1 = round(y1 * scale_y)
            x2 = round(x2 * scale_x)
            y2 = round(y2 * scale_y)
            patch = panel[y1:y2, x1:x2]
            tint = np.empty_like(patch)
            tint[:] = color
            panel[y1:y2, x1:x2] = cv2.addWeighted(patch, 0.55, tint, 0.45, 0.0)
            thickness = 3 if region.id in selected_ids else 1
            cv2.rectangle(panel, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), thickness)
            cv2.putText(
                panel,
                f"{value:.2f}",
                (x1 + 4, min(y2 - 4, y1 + 17)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 230), 30), (0, 0, 0), -1)
        cv2.putText(
            panel,
            title,
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise DataError(f"OpenCV cannot write D/U/S score map {path}")


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DataError(f"OpenCV cannot read image {path}")
    return image


def _record_bgr(record: ImageRecord) -> np.ndarray:
    if record.image_bgr is not None:
        image = record.image_bgr
        if image.ndim != 3 or image.shape[2] != 3:
            raise DataError(f"in-memory image {record.image_id} must be BGR HWC")
        if image.shape[:2] != (record.height, record.width):
            raise DataError(
                f"in-memory image {record.image_id} shape {image.shape[:2]} does not "
                f"match record {(record.height, record.width)}"
            )
        return image
    return _read_bgr(Path(record.path))


def _finalize_timing_rows(
    config: AppConfig,
    run: RunDirectory,
    records: Sequence[ImageRecord],
    final: Sequence[DetectionBatch],
    timings: dict[int | str, dict[str, float | int]],
) -> tuple[dict[str, Any], ...]:
    final_by_id = {batch.image_id: batch for batch in final}
    vram = peak_vram_mb(config.experiment.device)
    memory = cpu_memory_mb()
    output: list[dict[str, Any]] = []
    for record in records:
        values = timings[record.image_id]
        values["fusion_ms"] = float(final_by_id[record.image_id].meta.get("fusion_ms", 0.0))
        num_candidates = int(values["num_candidates"])
        num_full_calls = 2 if config.scoring.flip_consistency.enabled else 1
        num_calls = num_full_calls + num_candidates
        eic = equivalent_inference_cost(
            [config.detector.crop_imgsz] * num_candidates,
            full_size=config.detector.full_imgsz,
            full_calls=num_full_calls,
        )
        stage_total = sum(
            float(values[name])
            for name in (
                "load_ms",
                "base_ms",
                "flip_ms",
                "scoring_ms",
                "crop_ms",
                "enhance_ms",
                "candidate_ms",
                "utility_ms",
                "fusion_ms",
            )
        )
        output.append(
            {
                "run_id": run.run_id,
                "image_id": record.image_id,
                **values,
                "total_ms": stage_total,
                "num_calls": num_calls,
                "num_full_calls": num_full_calls,
                "eic": eic,
                "peak_vram_mb": vram,
                "cpu_memory_mb": memory,
            }
        )
    return tuple(output)
