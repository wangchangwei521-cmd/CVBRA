from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from buse_uav.data.common import DataError
from buse_uav.detectors.base import DetectorAdapter
from buse_uav.evaluation.timing import cpu_memory_mb, equivalent_inference_cost, peak_vram_mb
from buse_uav.fusion.pipeline import concatenate_pre_fusion, fuse_detection_batches
from buse_uav.pipeline.selective import SelectiveOutput
from buse_uav.pipeline.trace import RunDirectory
from buse_uav.regions.grid import make_grid
from buse_uav.regions.selection import combine_scores
from buse_uav.schemas import AppConfig, Box, DetectionBatch, ImageRecord, Region
from buse_uav.scoring.degradation import QualityCalibration, score_degradation
from buse_uav.scoring.uncertainty import max_region_uncertainty, score_uncertainty
from buse_uav.utility.task_utility import (
    CandidateEvaluation,
    choose_best_candidate,
    evaluate_candidate,
    identity_evaluation,
    with_operation,
)
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVELOPMENT_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "20260801T052108Z_hazydet_flip_tta_yolo11n_8f96292f"
    / "data_manifest.json"
)
DEVELOPMENT_MANIFEST_SHA256 = "d2ddfa6cd968f70a25f55d8ea1a08ad6b7aab90799740d4a09a7717e47478eed"


@dataclass(frozen=True)
class PackedTileTransform:
    region: Region
    slot: int
    cell_xyxy: tuple[int, int, int, int]
    content_xyxy: tuple[int, int, int, int]
    crop_width: int
    crop_height: int
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class PackedView:
    image_bgr: np.ndarray
    transforms: tuple[PackedTileTransform, ...]


def validate_duq_pvf_development_bank(
    config: AppConfig,
    records: Sequence[ImageRecord],
    *,
    manifest_path: Path = DEVELOPMENT_MANIFEST,
    expected_manifest_sha256: str = DEVELOPMENT_MANIFEST_SHA256,
) -> None:
    """Fail before inference unless HazyDet records match the registered bank prefix."""
    if config.method.name != "duq_pvf" or config.dataset.name != "hazydet":
        return
    if not manifest_path.is_file():
        raise ValueError(f"registered duq_pvf development manifest is missing: {manifest_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "registered duq_pvf development manifest hash drift: "
            f"expected {expected_manifest_sha256}, received {actual_manifest_sha256}"
        )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_rows = document["images"]
        expected_annotation = Path(document["annotation"])
        expected_annotation_sha256 = str(document["annotation_sha256"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid registered duq_pvf development manifest: {exc}") from exc
    if document.get("dataset") != "hazydet" or document.get("split") != "val":
        raise ValueError("registered duq_pvf development manifest is not HazyDet validation")
    if not isinstance(expected_rows, list) or len(expected_rows) != 300:
        raise ValueError("registered duq_pvf development manifest must contain exactly 300 images")
    if len(records) != config.experiment.max_images:
        raise ValueError("duq_pvf development records do not match experiment.max_images")
    if len(records) > len(expected_rows):
        raise ValueError("duq_pvf development records exceed the registered bank")
    if (
        not expected_annotation.is_file()
        or sha256_file(expected_annotation) != expected_annotation_sha256
    ):
        raise ValueError("duq_pvf development annotation hash drift")
    for index, (record, expected) in enumerate(
        zip(records, expected_rows[: len(records)], strict=True)
    ):
        if not isinstance(expected, dict):
            raise ValueError(f"invalid duq_pvf development manifest row {index}")
        expected_path = Path(str(expected.get("path", ""))).resolve()
        if (
            str(record.image_id) != str(expected.get("image_id"))
            or Path(record.path).resolve() != expected_path
            or record.width != int(expected.get("width", -1))
            or record.height != int(expected.get("height", -1))
        ):
            raise ValueError(f"duq_pvf development bank mismatch at row {index}")
        if sha256_file(Path(record.path)) != str(expected.get("sha256", "")):
            raise ValueError(f"duq_pvf development image hash drift at row {index}")


def pack_flipped_regions(
    image_bgr: np.ndarray,
    regions: Sequence[Region],
    *,
    canvas_imgsz: int,
    fill_value: int = 114,
) -> PackedView:
    """Flip four context crops and letterbox them into a deterministic 2x2 canvas."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("packed-view input must be uint8 BGR HWC")
    if len(regions) != 4:
        raise ValueError("packed view requires exactly four regions")
    if canvas_imgsz <= 0 or canvas_imgsz % 2:
        raise ValueError("packed-view canvas must have a positive even side length")
    if not 0 <= fill_value <= 255:
        raise ValueError("packed-view fill value must be in [0, 255]")
    if len({region.id for region in regions}) != len(regions):
        raise ValueError("packed-view regions must have unique IDs")

    height, width = image_bgr.shape[:2]
    cell_side = canvas_imgsz // 2
    canvas = np.full(
        (canvas_imgsz, canvas_imgsz, 3),
        fill_value,
        dtype=np.uint8,
    )
    transforms: list[PackedTileTransform] = []
    for slot, region in enumerate(regions):
        crop_x1, crop_y1, crop_x2, crop_y2 = region.crop_xyxy
        if not (0 <= crop_x1 < crop_x2 <= width and 0 <= crop_y1 < crop_y2 <= height):
            raise ValueError(f"region {region.id} crop lies outside the packed-view source")
        crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        flipped = np.ascontiguousarray(cv2.flip(crop, 1))
        crop_height, crop_width = flipped.shape[:2]
        scale = min(cell_side / crop_width, cell_side / crop_height)
        resized_width = max(1, min(cell_side, round(crop_width * scale)))
        resized_height = max(1, min(cell_side, round(crop_height * scale)))
        resized = cv2.resize(
            flipped,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        cell_x1 = (slot % 2) * cell_side
        cell_y1 = (slot // 2) * cell_side
        cell_x2 = cell_x1 + cell_side
        cell_y2 = cell_y1 + cell_side
        pad_x = (cell_side - resized_width) // 2
        pad_y = (cell_side - resized_height) // 2
        content_x1 = cell_x1 + pad_x
        content_y1 = cell_y1 + pad_y
        content_x2 = content_x1 + resized_width
        content_y2 = content_y1 + resized_height
        canvas[content_y1:content_y2, content_x1:content_x2] = resized
        transforms.append(
            PackedTileTransform(
                region=region,
                slot=slot,
                cell_xyxy=(cell_x1, cell_y1, cell_x2, cell_y2),
                content_xyxy=(content_x1, content_y1, content_x2, content_y2),
                crop_width=crop_width,
                crop_height=crop_height,
                scale_x=resized_width / crop_width,
                scale_y=resized_height / crop_height,
            )
        )
    return PackedView(image_bgr=np.ascontiguousarray(canvas), transforms=tuple(transforms))


def restore_packed_box(
    box: Box,
    transforms: Sequence[PackedTileTransform],
    *,
    image_width: int,
    image_height: int,
    content_overlap_min: float = 0.80,
) -> Box | None:
    """Restore one packed detection and apply the frozen local-crop filters."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("restored image dimensions must be positive")
    if not 0.0 < content_overlap_min <= 1.0:
        raise ValueError("content_overlap_min must be in (0, 1]")
    if not all(math.isfinite(value) for value in (*box.xyxy, box.score)):
        return None
    box_x1, box_y1, box_x2, box_y2 = box.xyxy
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return None
    center_x = (box_x1 + box_x2) / 2.0
    center_y = (box_y1 + box_y2) / 2.0
    matching = [
        transform
        for transform in transforms
        if transform.content_xyxy[0] <= center_x < transform.content_xyxy[2]
        and transform.content_xyxy[1] <= center_y < transform.content_xyxy[3]
    ]
    if len(matching) != 1:
        return None
    transform = matching[0]
    content_x1, content_y1, content_x2, content_y2 = transform.content_xyxy
    clipped_x1 = max(box_x1, float(content_x1))
    clipped_y1 = max(box_y1, float(content_y1))
    clipped_x2 = min(box_x2, float(content_x2))
    clipped_y2 = min(box_y2, float(content_y2))
    intersection = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
    if intersection / box_area < content_overlap_min:
        return None

    flipped_x1 = (clipped_x1 - content_x1) / transform.scale_x
    flipped_y1 = (clipped_y1 - content_y1) / transform.scale_y
    flipped_x2 = (clipped_x2 - content_x1) / transform.scale_x
    flipped_y2 = (clipped_y2 - content_y1) / transform.scale_y
    crop_local_x1 = transform.crop_width - flipped_x2
    crop_local_x2 = transform.crop_width - flipped_x1
    crop_x1, crop_y1, crop_x2, crop_y2 = transform.region.crop_xyxy
    mapped_x1 = crop_x1 + crop_local_x1
    mapped_y1 = crop_y1 + flipped_y1
    mapped_x2 = crop_x1 + crop_local_x2
    mapped_y2 = crop_y1 + flipped_y2

    crop_intersection_x1 = max(mapped_x1, float(crop_x1))
    crop_intersection_y1 = max(mapped_y1, float(crop_y1))
    crop_intersection_x2 = min(mapped_x2, float(crop_x2))
    crop_intersection_y2 = min(mapped_y2, float(crop_y2))
    crop_intersection = max(0.0, crop_intersection_x2 - crop_intersection_x1) * max(
        0.0, crop_intersection_y2 - crop_intersection_y1
    )
    mapped_area = max(0.0, mapped_x2 - mapped_x1) * max(0.0, mapped_y2 - mapped_y1)
    if mapped_area <= 0.0 or crop_intersection / mapped_area < content_overlap_min:
        return None

    mapped_x1 = max(0.0, min(float(image_width), mapped_x1))
    mapped_y1 = max(0.0, min(float(image_height), mapped_y1))
    mapped_x2 = max(0.0, min(float(image_width), mapped_x2))
    mapped_y2 = max(0.0, min(float(image_height), mapped_y2))
    if mapped_x2 - mapped_x1 < 1.0 or mapped_y2 - mapped_y1 < 1.0:
        return None
    mapped_center_x = (mapped_x1 + mapped_x2) / 2.0
    mapped_center_y = (mapped_y1 + mapped_y2) / 2.0
    core_x1, core_y1, core_x2, core_y2 = transform.region.core_xyxy
    if not (core_x1 <= mapped_center_x < core_x2 and core_y1 <= mapped_center_y < core_y2):
        return None
    return Box(
        xyxy=(mapped_x1, mapped_y1, mapped_x2, mapped_y2),
        score=box.score,
        class_id=box.class_id,
        source="packed_view_flip",
        region_id=transform.region.id,
        operation="packed_flip",
    )


def admit_packed_region(
    config: AppConfig,
    reference: Sequence[Box],
    predictions: Sequence[Box],
) -> tuple[tuple[Box, ...], CandidateEvaluation]:
    """Apply the frozen Q formula against the B0 regional reference."""
    evaluation = with_operation(
        evaluate_candidate(
            reference,
            predictions,
            publish_conf=config.detector.publish_conf,
            match_iou=config.utility.match_iou,
            max_count_ratio=config.utility.max_count_ratio,
            crop_imgsz=config.packed_view.canvas_imgsz,
            full_imgsz=config.detector.full_imgsz,
            weights=config.utility.weights,
        ),
        "packed_flip",
    )
    choice = choose_best_candidate(
        identity_evaluation((), num_reference=len(reference)),
        (evaluation,),
    )
    return choice.predictions, evaluation


def apply_duq_pvf(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
    base_probe: Sequence[DetectionBatch],
) -> SelectiveOutput:
    """Execute the registered one-call D-U-Q packed-view policy."""
    if config.method.name != "duq_pvf":
        raise ValueError("packed-view pipeline requires method.name=duq_pvf")
    if config.packed_view.score != "max_region_uncertainty":
        raise ValueError(f"unsupported packed-view score: {config.packed_view.score}")
    base_by_id = {batch.image_id: batch for batch in base_probe}
    expected_ids = {record.image_id for record in records}
    if len(base_by_id) != len(base_probe) or set(base_by_id) != expected_ids:
        raise ValueError("packed-view base detections must cover every input image exactly once")
    if not records:
        raise ValueError("packed-view inference requires at least one image")

    gate_scores: dict[int | str, float] = {}
    timings = {
        record.image_id: {
            "load_ms": 0.0,
            "base_ms": base_by_id[record.image_id].latency_ms,
            "flip_ms": 0.0,
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
    regions_by_id: dict[int | str, tuple[Region, ...]] = {}
    for record in records:
        started = perf_counter()
        regions = make_grid(
            (record.height, record.width, 3),
            rows=config.regions.rows,
            cols=config.regions.cols,
            context_padding=config.regions.context_padding,
        )
        regions_by_id[record.image_id] = regions
        gate_scores[record.image_id] = max_region_uncertainty(
            regions,
            base_by_id[record.image_id],
            publish_conf=config.detector.publish_conf,
            threshold_sigma=config.scoring.threshold_sigma,
            density_kappa=config.scoring.density_kappa,
            rank_mix=config.scoring.rank_mix,
            weights=config.scoring.uncertainty_weights,
        )
        timings[record.image_id]["scoring_ms"] += (perf_counter() - started) * 1000.0

    ranked_ids = tuple(
        sorted(gate_scores, key=lambda image_id: (-gate_scores[image_id], str(image_id)))
    )
    active_count = max(1, round(len(records) * config.packed_view.activation_rate))
    active_count = min(active_count, len(records))
    active_ids = set(ranked_ids[:active_count])
    calibration = QualityCalibration.from_file(config.scoring.calibration)
    packed_records: list[ImageRecord] = []
    packed_context: dict[
        int | str,
        tuple[ImageRecord, tuple[Region, ...], tuple[PackedTileTransform, ...]],
    ] = {}
    selection_rows: list[dict[str, Any]] = []

    for record in records:
        if record.image_id not in active_ids:
            continue
        started = perf_counter()
        image_bgr = _record_bgr(record)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        timings[record.image_id]["load_ms"] += (perf_counter() - started) * 1000.0

        started = perf_counter()
        regions = regions_by_id[record.image_id]
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
        )
        assessments = combine_scores(
            regions,
            degradation,
            uncertainty,
            alpha=config.difficulty.alpha,
            interaction_lambda=config.difficulty.interaction_lambda,
            blank_suppression=config.regions.blank_suppression,
        )
        score_by_region = {score.region_id: score.difficulty for score in assessments}
        region_by_id = {region.id: region for region in regions}
        ranking = tuple(
            sorted(region_by_id, key=lambda region_id: (-score_by_region[region_id], region_id))
        )
        selected = tuple(
            region_by_id[region_id] for region_id in ranking[: config.packed_view.selected_regions]
        )
        if len(selected) != config.packed_view.selected_regions:
            raise ValueError("packed-view selection did not produce exactly four regions")
        timings[record.image_id]["scoring_ms"] += (perf_counter() - started) * 1000.0
        timings[record.image_id]["num_regions"] = len(selected)

        started = perf_counter()
        packed = pack_flipped_regions(
            image_bgr,
            selected,
            canvas_imgsz=config.packed_view.canvas_imgsz,
            fill_value=config.packed_view.fill_value,
        )
        timings[record.image_id]["crop_ms"] += (perf_counter() - started) * 1000.0
        timings[record.image_id]["num_candidates"] = 1
        packed_id = f"{record.image_id}::duq_pvf"
        packed_records.append(
            ImageRecord(
                image_id=packed_id,
                path=f"memory://duq-pvf/{record.image_id}",
                width=config.packed_view.canvas_imgsz,
                height=config.packed_view.canvas_imgsz,
                image_bgr=packed.image_bgr,
            )
        )
        packed_context[packed_id] = (record, selected, packed.transforms)
        selection_rows.append(
            {
                "image_id": record.image_id,
                "gate_score": gate_scores[record.image_id],
                "selected_region_ids": [region.id for region in selected],
                "ranking": list(ranking),
            }
        )

    auxiliary = detector.predict(
        packed_records,
        imgsz=config.packed_view.canvas_imgsz,
        conf=config.detector.probe_conf,
        iou=config.detector.nms_iou,
        max_det=config.detector.max_det,
        fp16=config.detector.fp16,
    )
    if len(auxiliary) != len(packed_records):
        raise ValueError("packed-view detector did not return one batch per active image")

    local_by_image: dict[int | str, DetectionBatch] = {}
    auxiliary_by_image: dict[int | str, DetectionBatch] = {}
    candidate_rows: list[dict[str, Any]] = []
    for auxiliary_batch in auxiliary:
        if auxiliary_batch.image_id not in packed_context:
            raise ValueError(f"unknown packed-view prediction ID: {auxiliary_batch.image_id}")
        record, selected, transforms = packed_context[auxiliary_batch.image_id]
        timings[record.image_id]["candidate_ms"] += auxiliary_batch.latency_ms
        mapped_by_region: dict[int, list[Box]] = defaultdict(list)
        for box in auxiliary_batch.boxes:
            mapped = restore_packed_box(
                box,
                transforms,
                image_width=record.width,
                image_height=record.height,
                content_overlap_min=config.packed_view.content_overlap_min,
            )
            if mapped is not None and mapped.region_id is not None:
                mapped_by_region[mapped.region_id].append(mapped)

        started = perf_counter()
        accepted: list[Box] = []
        q_by_region: dict[int, float] = {}
        for region in selected:
            reference = _boxes_centered_in_region(base_by_id[record.image_id].boxes, region)
            predictions = tuple(mapped_by_region[region.id])
            admitted, evaluation = admit_packed_region(config, reference, predictions)
            q_by_region[region.id] = evaluation.utility.q
            accepted.extend(admitted)
            utility = evaluation.utility
            candidate_rows.append(
                {
                    "run_id": run.run_id,
                    "image_id": record.image_id,
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
                    "selected": bool(admitted),
                }
            )
        timings[record.image_id]["utility_ms"] += (perf_counter() - started) * 1000.0
        mapped_all = tuple(box for region in selected for box in mapped_by_region[region.id])
        auxiliary_by_image[record.image_id] = DetectionBatch(
            image_id=record.image_id,
            boxes=mapped_all,
            latency_ms=auxiliary_batch.latency_ms,
            meta={"method": "duq_pvf_auxiliary_mapped"},
        )
        local_by_image[record.image_id] = DetectionBatch(
            image_id=record.image_id,
            boxes=tuple(accepted),
            latency_ms=auxiliary_batch.latency_ms,
            meta={
                "method": "duq_pvf_q_admitted",
                "selected_regions": len(selected),
                "local_q_by_region": q_by_region,
            },
        )

    for record in records:
        if record.image_id in active_ids:
            continue
        empty = DetectionBatch(
            image_id=record.image_id,
            boxes=(),
            latency_ms=0.0,
            meta={"method": "duq_pvf_b0_bypass", "local_q_by_region": {}},
        )
        local_by_image[record.image_id] = empty
        auxiliary_by_image[record.image_id] = empty

    local_batches = tuple(local_by_image[record.image_id] for record in records)
    pre_fusion = concatenate_pre_fusion(base_probe, local_batches)
    fused = fuse_detection_batches(config, base_probe, local_batches, records)
    final: list[DetectionBatch] = []
    for batch in fused:
        activated = batch.image_id in active_ids
        final.append(
            replace(
                batch,
                meta={
                    **batch.meta,
                    "packed_view_activated": activated,
                    "packed_canvas_imgsz": (config.packed_view.canvas_imgsz if activated else None),
                },
            )
        )

    decisions = [
        {
            "image_id": image_id,
            "score": gate_scores[image_id],
            "rank": rank,
            "activated": image_id in active_ids,
        }
        for rank, image_id in enumerate(ranked_ids, start=1)
    ]
    atomic_write_json(
        run.path / "traces" / "packed_view_summary.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "method": config.method.name,
            "score": config.packed_view.score,
            "activation_rate": config.packed_view.activation_rate,
            "activated_images": active_count,
            "canvas_imgsz": config.packed_view.canvas_imgsz,
            "grid": list(config.packed_view.grid),
            "selected_regions": config.packed_view.selected_regions,
            "layout": list(config.packed_view.layout),
            "auxiliary_calls_per_activated_image": 1,
            "decisions": decisions,
            "selections": selection_rows,
        },
    )
    if config.runtime.save_trace:
        _write_jsonl(run.path / "traces" / "candidates.jsonl", candidate_rows)
        atomic_write_json(
            run.path / "traces" / "selection_summary.json",
            {
                "schema_version": 1,
                "run_id": run.run_id,
                "method": config.method.name,
                "images": selection_rows,
            },
        )

    final_by_id = {batch.image_id: batch for batch in final}
    vram = peak_vram_mb(config.experiment.device)
    memory = cpu_memory_mb()
    timing_rows: list[dict[str, Any]] = []
    for record in records:
        values = timings[record.image_id]
        activated = record.image_id in active_ids
        fusion_ms = (
            float(final_by_id[record.image_id].meta.get("fusion_ms", 0.0)) if activated else 0.0
        )
        values["fusion_ms"] = fusion_ms
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
        timing_rows.append(
            {
                "run_id": run.run_id,
                "image_id": record.image_id,
                **values,
                "total_ms": stage_total,
                "num_calls": 2 if activated else 1,
                "num_full_calls": 1,
                "eic": (
                    equivalent_inference_cost(
                        [config.packed_view.canvas_imgsz],
                        full_size=config.detector.full_imgsz,
                        full_calls=1,
                    )
                    if activated
                    else 1.0
                ),
                "peak_vram_mb": vram,
                "cpu_memory_mb": memory,
            }
        )
    return SelectiveOutput(
        final=tuple(final),
        local_pre_fusion=local_batches,
        pre_fusion=pre_fusion,
        timings=tuple(timing_rows),
        extra_predictions={
            "packed_auxiliary": tuple(auxiliary_by_image[record.image_id] for record in records)
        },
    )


def _boxes_centered_in_region(boxes: Sequence[Box], region: Region) -> tuple[Box, ...]:
    x1, y1, x2, y2 = region.core_xyxy
    return tuple(
        box
        for box in boxes
        if x1 <= (box.xyxy[0] + box.xyxy[2]) / 2.0 < x2
        and y1 <= (box.xyxy[1] + box.xyxy[3]) / 2.0 < y2
    )


def _record_bgr(record: ImageRecord) -> np.ndarray:
    if record.image_bgr is not None:
        image = record.image_bgr
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DataError(f"in-memory image {record.image_id} must be uint8 BGR HWC")
        if image.shape[:2] != (record.height, record.width):
            raise DataError(
                f"in-memory image {record.image_id} shape {image.shape[:2]} does not "
                f"match record {(record.height, record.width)}"
            )
        return image
    disk_image = cv2.imread(record.path, cv2.IMREAD_COLOR)
    if disk_image is None:
        raise DataError(f"OpenCV cannot read packed-view image {record.path}")
    if disk_image.shape[:2] != (record.height, record.width):
        raise DataError(
            f"packed-view image {record.path} shape {disk_image.shape[:2]} does not "
            f"match record {(record.height, record.width)}"
        )
    return disk_image


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)
