from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from buse_uav.data.auair import AUAIR_CLASS_NAMES, AuAirAdapter
from buse_uav.data.common import DataError
from buse_uav.data.dronevehicle import DRONEVEHICLE_CLASS_NAMES, DroneVehicleAdapter
from buse_uav.data.hazydet import HAZYDET_CLASS_NAMES, HazyDetAdapter
from buse_uav.data.uavdt import UAVDT_CLASS_NAMES, UavdtAdapter
from buse_uav.data.visdrone import VISDRONE_CLASS_NAMES, VisDroneAdapter
from buse_uav.detectors.base import DetectorAdapter, DetectorError
from buse_uav.detectors.factory import build_detector
from buse_uav.detectors.tta import merge_classwise_nms, restore_horizontal_flip
from buse_uav.enhancements.global_ops import enhance_file
from buse_uav.evaluation.coco import EvaluationError, evaluate_coco, write_coco_predictions
from buse_uav.evaluation.timing import (
    cpu_memory_mb,
    peak_vram_mb,
    reset_peak_vram,
    write_timing_traces,
)
from buse_uav.fusion.pipeline import fuse_detection_batches
from buse_uav.pipeline.fast_gate import apply_fast_cawbf_gate
from buse_uav.pipeline.guard import apply_duq_guard
from buse_uav.pipeline.intermediates import (
    cleanup_transient_intermediates,
    intermediate_shard_path,
)
from buse_uav.pipeline.packed_view import apply_duq_pvf, validate_duq_pvf_development_bank
from buse_uav.pipeline.selective import SelectiveOutput, apply_identity_crop_selection
from buse_uav.pipeline.trace import RunDirectory
from buse_uav.schemas import AppConfig, Box, DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text


def infer_detector(
    config: AppConfig,
    raw_config: dict[str, Any],
    *,
    command: list[str] | None = None,
    resume_run: Path | None = None,
    run_id: str | None = None,
    warmup_images: int = 0,
) -> RunDirectory:
    if config.dataset.name not in {
        "hazydet",
        "visdrone",
        "uavdt",
        "auair",
        "dronevehicle",
    }:
        raise DetectorError(
            "inference requires a supported HazyDet, VisDrone, UAVDT, AU-AIR, or "
            "DroneVehicle dataset"
        )
    actual_command = command or sys.argv
    run = (
        RunDirectory.resume(
            resume_run,
            config=raw_config,
            command=actual_command,
        )
        if resume_run is not None
        else RunDirectory.create(
            output_root=config.project.output_root,
            config=raw_config,
            command=actual_command,
            run_id=run_id,
        )
    )
    if run.successful:
        cleanup_transient_intermediates(config, run)
        run.logger.info("resume_skipped_successful_run")
        return run
    cleanup_transient_intermediates(config, run)
    run.logger.info("inference_started", method=config.method.name)
    try:
        adapter = _inference_adapter(config)
        records = tuple(adapter.records())
        if config.experiment.max_images is not None:
            records = records[: config.experiment.max_images]
        validate_duq_pvf_development_bank(config, records)
        detector = build_detector(
            config.detector,
            device=config.experiment.device,
            expected_class_names=config.dataset.class_names,
            runtime=config.runtime,
        )
        if warmup_images < 0:
            raise ValueError("warmup_images must be nonnegative")
        if warmup_images:
            warmup_records = records[: min(warmup_images, len(records))]
            detector.predict(
                warmup_records,
                imgsz=config.detector.full_imgsz,
                conf=config.detector.probe_conf,
                iou=config.detector.nms_iou,
                max_det=config.detector.max_det,
                fp16=config.detector.fp16,
            )
            run.logger.info(
                "inference_warmup_completed",
                images=len(warmup_records),
                requested=warmup_images,
            )
        reset_peak_vram(config.experiment.device)
        base_probe = detector.predict(
            records,
            imgsz=config.detector.full_imgsz,
            conf=config.detector.probe_conf,
            iou=config.detector.nms_iou,
            max_det=config.detector.max_det,
            fp16=config.detector.fp16,
        )
        method_output = _apply_method(config, run, detector, records, base_probe)
        if isinstance(method_output, SelectiveOutput):
            final = method_output.final
            local_pre_fusion = method_output.local_pre_fusion
            pre_fusion = method_output.pre_fusion
            timing_rows = method_output.timings
            for name, batches in method_output.extra_predictions.items():
                if not name.replace("_", "").isalnum():
                    raise DetectorError(f"invalid extra prediction name: {name!r}")
                write_coco_predictions(
                    run.path / "predictions" / f"{name}.coco.json",
                    batches,
                    category_id_by_class=_category_mapping(adapter),
                )
            atomic_write_json(
                run.path / "traces" / "fusion_summary.json",
                {
                    "schema_version": 1,
                    "method": (
                        config.guard.fusion_mode
                        if config.method.name in {"duq_guard", "u_flip_guard"}
                        else (config.fusion.method if config.method.fusion_enabled else "hard_nms")
                    ),
                    "images": [
                        {
                            "image_id": batch.image_id,
                            **batch.meta,
                            "final_boxes": len(batch.boxes),
                        }
                        for batch in final
                    ],
                },
            )
        else:
            final = method_output
            local_pre_fusion = None
            pre_fusion = None
            timing_rows = _simple_timing_rows(
                config,
                run,
                records,
                base_probe,
                final,
            )
        final = _filter_batches(final, threshold=config.detector.publish_conf)
        category_mapping = _category_mapping(adapter)
        write_coco_predictions(
            run.path / "predictions" / "base_probe.coco.json",
            base_probe,
            category_id_by_class=category_mapping,
        )
        if local_pre_fusion is not None and pre_fusion is not None:
            write_coco_predictions(
                run.path / "predictions" / "local_pre_fusion.coco.json",
                local_pre_fusion,
                category_id_by_class=category_mapping,
            )
            write_coco_predictions(
                run.path / "predictions" / "pre_fusion.coco.json",
                pre_fusion,
                category_id_by_class=category_mapping,
            )
        rows = write_coco_predictions(
            run.path / "predictions" / "final.coco.json",
            final,
            category_id_by_class=category_mapping,
        )
        if not rows:
            raise DetectorError(
                "no predictions survived publish_conf; keep this run invalid and "
                "complete the 20-epoch pilot before baseline evaluation"
            )
        fingerprint = detector.fingerprint()
        atomic_write_json(run.path / "model_fingerprint.json", fingerprint)
        atomic_write_json(run.path / "data_manifest.json", _data_manifest(config, adapter, records))
        write_timing_traces(run.path / "traces", timing_rows)
        atomic_write_json(
            run.path / "timing_protocol.json",
            {
                "schema_version": 1,
                "device": config.experiment.device,
                "batch_size": 1,
                "candidate_batching": config.runtime.batch_candidates,
                "cross_shape_candidate_batching": (config.runtime.cross_shape_candidate_batching),
                "degradation_workers": config.runtime.degradation_workers,
                "warmup_images": warmup_images,
                "cuda_synchronized": all(
                    bool(batch.meta.get("cuda_synchronized", False)) for batch in base_probe
                ),
                "stream_chunk_records": int(base_probe[0].meta.get("stream_chunk_records", 1)),
                "in_memory_candidates": config.runtime.in_memory_candidates,
                "release_cuda_cache_between_chunks": (
                    config.runtime.release_cuda_cache_between_chunks
                ),
                "fast_gate": (
                    {
                        "score": config.fast_gate.score,
                        "threshold": config.fast_gate.threshold,
                    }
                    if config.method.name == "buse_cawbf_fast"
                    else None
                ),
                "guard": (
                    config.guard.model_dump(mode="json")
                    if config.method.name in {"duq_guard", "u_flip_guard"}
                    else None
                ),
                "packed_view": (
                    config.packed_view.model_dump(mode="json")
                    if config.method.name == "duq_pvf"
                    else None
                ),
                "model_load_excluded": True,
                "data_download_excluded": True,
                "disk_read_reported_separately": True,
            },
        )
        atomic_write_text(
            run.path / "summary.md",
            f"# Inference summary\n\n"
            f"- Run: `{run.run_id}`\n"
            f"- Method: `{config.method.name}`\n"
            f"- Images: {len(records)}\n"
            f"- Published predictions: {len(rows)}\n"
            f"- Weight SHA256: `{fingerprint['sha256']}`\n",
        )
        run.logger.info(
            "inference_completed",
            images=len(records),
            predictions=len(rows),
        )
        cleanup_transient_intermediates(config, run)
        run.logger.info(
            "intermediate_cleanup_completed",
            save_intermediates=config.runtime.save_intermediates,
        )
        run.mark_success()
        return run
    except Exception as exc:
        run.logger.error("inference_failed", error=str(exc))
        raise
    finally:
        if not config.runtime.save_intermediates:
            try:
                cleanup_transient_intermediates(config, run)
            except OSError as cleanup_error:
                run.logger.error(
                    "intermediate_cleanup_failed",
                    error=str(cleanup_error),
                )


def evaluate_run(
    run_path: Path,
    *,
    prediction: str = "final",
) -> dict[str, Any]:
    resolved = run_path.resolve()
    config_path = resolved / "config_resolved.yaml"
    allowed_predictions = {
        "base_probe",
        "local_pre_fusion",
        "pre_fusion",
        "final",
    }
    if prediction not in allowed_predictions:
        raise EvaluationError(
            f"prediction must be one of {sorted(allowed_predictions)}; received {prediction!r}"
        )
    prediction_path = resolved / "predictions" / f"{prediction}.coco.json"
    if not config_path.is_file():
        raise EvaluationError(f"run configuration is missing: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EvaluationError(f"invalid run configuration: {config_path}")
    dataset = raw.get("dataset", {})
    detector = raw.get("detector", {})
    if not isinstance(dataset, dict) or not isinstance(detector, dict):
        raise EvaluationError("run configuration lacks dataset/detector sections")
    root = Path(str(dataset["root"]))
    split = str(dataset["split"])
    data_manifest_path = resolved / "data_manifest.json"
    if not data_manifest_path.is_file():
        raise EvaluationError(f"run data manifest is missing: {data_manifest_path}")
    data_manifest = yaml.safe_load(data_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data_manifest, dict) or not isinstance(data_manifest.get("images"), list):
        raise EvaluationError(f"invalid run data manifest: {data_manifest_path}")
    image_ids = [
        image["image_id"]
        for image in data_manifest["images"]
        if isinstance(image, dict) and "image_id" in image
    ]
    dataset_name = str(dataset.get("name", ""))
    if dataset_name == "hazydet":
        annotation = HazyDetAdapter(root, split=split).annotation_file()
    elif dataset_name == "uavdt":
        annotation = _uavdt_adapter_from_mapping(dataset).annotation_file()
    elif dataset_name == "auair":
        annotation = _auair_adapter_from_mapping(dataset).annotation_file()
    elif dataset_name == "dronevehicle":
        annotation = _dronevehicle_adapter_from_mapping(dataset).annotation_file()
    elif dataset_name == "visdrone":
        adapter = _visdrone_adapter_from_mapping(dataset)
        annotation = resolved / "ground_truth.coco.json"
        atomic_write_json(annotation, adapter.coco_document(image_ids=image_ids))
    else:
        raise EvaluationError(f"unsupported evaluation dataset: {dataset_name}")
    metrics = evaluate_coco(
        annotation,
        prediction_path,
        max_det=int(detector.get("max_det", 500)),
        image_ids=image_ids,
    )
    suffix = "" if prediction == "final" else f"_{prediction}"
    atomic_write_json(resolved / f"metrics{suffix}.json", metrics)
    _write_metrics_csv(resolved / f"metrics{suffix}.csv", metrics)
    metrics_path = resolved / f"metrics{suffix}.json"
    prediction_file = resolved / "predictions" / f"{prediction}.coco.json"
    provenance_suffix = "" if prediction == "final" else f"_{prediction}"
    atomic_write_json(
        resolved / f"metrics_provenance{provenance_suffix}.json",
        {
            "schema_version": 1,
            "evaluator": "pycocotools.COCOeval",
            "prediction": prediction,
            "metrics_sha256": sha256_file(metrics_path),
            "prediction_sha256": sha256_file(prediction_file),
            "config_sha256": sha256_file(config_path),
            "annotation_sha256": sha256_file(annotation),
        },
    )
    summary_path = resolved / "summary.md"
    previous = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    atomic_write_text(
        summary_path,
        previous
        + f"\n## COCO evaluation: {prediction}\n\n"
        + "\n".join(
            f"- `{name}`: {value:.6f}"
            for name, value in metrics.items()
            if isinstance(value, float)
        )
        + "\n",
    )
    return metrics


def _apply_method(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: tuple[ImageRecord, ...],
    base_probe: tuple[DetectionBatch, ...],
) -> tuple[DetectionBatch, ...] | SelectiveOutput:
    method = config.method.name
    if method == "baseline":
        return base_probe
    if method == "flip_tta":
        flipped = _derived_records(config, run, records, operation="identity", flip=True)
        flipped_batches = detector.predict(
            flipped,
            imgsz=config.detector.full_imgsz,
            conf=config.detector.probe_conf,
            iou=config.detector.nms_iou,
            max_det=config.detector.max_det,
            fp16=config.detector.fp16,
        )
        restored = restore_horizontal_flip(flipped_batches, records)
        return merge_classwise_nms(
            base_probe,
            restored,
            iou_threshold=config.detector.nms_iou,
            max_det=config.detector.max_det,
        )
    if method == "buse_cawbf_fast":
        return apply_fast_cawbf_gate(
            config,
            run,
            detector,
            records,
            base_probe,
        )
    if method in {"duq_guard", "u_flip_guard"}:
        return apply_duq_guard(
            config,
            run,
            detector,
            records,
            base_probe,
        )
    if method == "duq_pvf":
        return apply_duq_pvf(
            config,
            run,
            detector,
            records,
            base_probe,
        )
    if method in {
        "degradation_only",
        "uncertainty_only",
        "identity_crop",
        "random_identity_crop",
        "all_identity_grid",
        "buse",
        "buse_cawbf",
    }:
        flip_probe: tuple[DetectionBatch, ...] | None = None
        if config.scoring.flip_consistency.enabled:
            flipped = _derived_records(config, run, records, operation="identity", flip=True)
            flipped_batches = detector.predict(
                flipped,
                imgsz=config.detector.full_imgsz,
                conf=config.detector.probe_conf,
                iou=config.detector.nms_iou,
                max_det=config.detector.max_det,
                fp16=config.detector.fp16,
            )
            flip_probe = restore_horizontal_flip(flipped_batches, records)
        return apply_identity_crop_selection(
            config,
            run,
            detector,
            records,
            base_probe,
            flip_probe,
        )
    if method == "full_candidate_bank":
        candidate_sets: list[tuple[DetectionBatch, ...]] = []
        for operation in ("gamma", "clahe", "unsharp"):
            enhanced = _derived_records(
                config,
                run,
                records,
                operation=operation,
                flip=False,
            )
            candidate_sets.append(
                detector.predict(
                    enhanced,
                    imgsz=config.detector.full_imgsz,
                    conf=config.detector.probe_conf,
                    iou=config.detector.nms_iou,
                    max_det=config.detector.max_det,
                    fp16=config.detector.fp16,
                )
            )
        candidates_by_image = [
            {batch.image_id: batch for batch in candidate_set} for candidate_set in candidate_sets
        ]
        local = tuple(
            DetectionBatch(
                image_id=record.image_id,
                boxes=tuple(
                    box
                    for candidate_by_image in candidates_by_image
                    for box in candidate_by_image[record.image_id].boxes
                ),
                latency_ms=sum(
                    candidate_by_image[record.image_id].latency_ms
                    for candidate_by_image in candidates_by_image
                ),
                meta={"local_q_by_region": {}, "full_candidates": 3},
            )
            for record in records
        )
        return fuse_detection_batches(config, base_probe, local, records)
    operations = {
        "global_gamma": "gamma",
        "global_clahe": "clahe",
        "global_unsharp": "unsharp",
    }
    if method not in operations:
        raise DetectorError(
            f"inference supports baseline, flip_tta, global enhancements, "
            f"Guard, and identity-crop selection; "
            f"received {method}"
        )
    enhanced = _derived_records(
        config,
        run,
        records,
        operation=operations[method],
        flip=False,
    )
    return detector.predict(
        enhanced,
        imgsz=config.detector.full_imgsz,
        conf=config.detector.probe_conf,
        iou=config.detector.nms_iou,
        max_det=config.detector.max_det,
        fp16=config.detector.fp16,
    )


def _derived_records(
    config: AppConfig,
    run: RunDirectory,
    records: tuple[ImageRecord, ...],
    *,
    operation: str,
    flip: bool,
) -> tuple[ImageRecord, ...]:
    output: list[ImageRecord] = []
    suffix = "flip" if flip else operation
    for record in records:
        source = Path(record.path)
        destination = intermediate_shard_path(
            config,
            run,
            suffix,
            f"{record.image_id}{source.suffix}",
        )
        enhance_file(
            source,
            destination,
            operation=operation,
            gamma=config.enhancement.gamma,
            clahe_clip_limit=config.enhancement.clahe_clip_limit,
            clahe_tile_grid=config.enhancement.clahe_tile_grid,
            unsharp_amount=config.enhancement.unsharp_amount,
            unsharp_sigma=config.enhancement.unsharp_sigma,
            horizontal_flip=flip,
        )
        output.append(replace(record, path=str(destination)))
    return tuple(output)


def _filter_batches(
    batches: tuple[DetectionBatch, ...],
    *,
    threshold: float,
) -> tuple[DetectionBatch, ...]:
    return tuple(
        replace(
            batch,
            boxes=tuple(
                Box(
                    xyxy=box.xyxy,
                    score=box.score,
                    class_id=box.class_id,
                    source=box.source,
                    region_id=box.region_id,
                    operation=box.operation,
                )
                for box in batch.boxes
                if box.score >= threshold
            ),
        )
        for batch in batches
    )


def _simple_timing_rows(
    config: AppConfig,
    run: RunDirectory,
    records: tuple[ImageRecord, ...],
    base_probe: tuple[DetectionBatch, ...],
    final: tuple[DetectionBatch, ...],
) -> tuple[dict[str, Any], ...]:
    base_by_id = {batch.image_id: batch for batch in base_probe}
    final_by_id = {batch.image_id: batch for batch in final}
    vram = peak_vram_mb(config.experiment.device)
    memory = cpu_memory_mb()
    output: list[dict[str, Any]] = []
    for record in records:
        base_ms = base_by_id[record.image_id].latency_ms
        flip_ms = 0.0
        candidate_ms = 0.0
        num_candidates = 0
        num_full_calls = 1
        if config.method.name != "baseline":
            num_candidates = 3 if config.method.name == "full_candidate_bank" else 1
            num_full_calls = 1 + num_candidates
            if config.method.name == "flip_tta":
                flip_ms = max(0.0, final_by_id[record.image_id].latency_ms - base_ms)
            elif config.method.name == "full_candidate_bank":
                candidate_ms = max(0.0, final_by_id[record.image_id].latency_ms - base_ms)
            else:
                candidate_ms = final_by_id[record.image_id].latency_ms
        fusion_ms = float(final_by_id[record.image_id].meta.get("fusion_ms", 0.0))
        output.append(
            {
                "run_id": run.run_id,
                "image_id": record.image_id,
                "load_ms": 0.0,
                "base_ms": base_ms,
                "flip_ms": flip_ms,
                "scoring_ms": 0.0,
                "crop_ms": 0.0,
                "enhance_ms": 0.0,
                "candidate_ms": candidate_ms,
                "utility_ms": 0.0,
                "fusion_ms": fusion_ms,
                "total_ms": base_ms + flip_ms + candidate_ms + fusion_ms,
                "num_regions": 0,
                "num_candidates": num_candidates,
                "num_calls": 1 + num_candidates,
                "num_full_calls": num_full_calls,
                "eic": float(num_full_calls),
                "peak_vram_mb": vram,
                "cpu_memory_mb": memory,
            }
        )
    return tuple(output)


def _hazydet_category_mapping(adapter: HazyDetAdapter) -> dict[int, int]:
    categories = adapter.document().category_names_by_id
    id_by_name = {name: category_id for category_id, name in categories.items()}
    if set(id_by_name) != set(HAZYDET_CLASS_NAMES):
        raise DataError(f"unexpected HazyDet categories: {sorted(id_by_name)}")
    return {class_id: int(id_by_name[name]) for class_id, name in enumerate(HAZYDET_CLASS_NAMES)}


def _category_mapping(
    adapter: HazyDetAdapter | VisDroneAdapter | UavdtAdapter | AuAirAdapter | DroneVehicleAdapter,
) -> dict[int, int]:
    if isinstance(adapter, HazyDetAdapter):
        return _hazydet_category_mapping(adapter)
    if isinstance(adapter, UavdtAdapter):
        return {class_id: class_id + 1 for class_id in range(len(UAVDT_CLASS_NAMES))}
    if isinstance(adapter, AuAirAdapter):
        return {class_id: class_id + 1 for class_id in range(len(AUAIR_CLASS_NAMES))}
    if isinstance(adapter, DroneVehicleAdapter):
        return {class_id: class_id + 1 for class_id in range(len(DRONEVEHICLE_CLASS_NAMES))}
    return {class_id: class_id + 1 for class_id in range(len(VISDRONE_CLASS_NAMES))}


def _inference_adapter(
    config: AppConfig,
) -> HazyDetAdapter | VisDroneAdapter | UavdtAdapter | AuAirAdapter | DroneVehicleAdapter:
    if config.dataset.name == "hazydet":
        return HazyDetAdapter(
            config.dataset.root,
            split=config.dataset.split,
            image_variant=config.dataset.image_variant or "hazy",
        )
    if config.dataset.name == "uavdt":
        return UavdtAdapter(
            config.dataset.root,
            split=config.dataset.split,
            image_directory=_configured_path(
                config.dataset.images_root or config.dataset.root,
                config.dataset.images,
            ),
            annotation_file=_configured_path(
                config.dataset.root,
                config.dataset.annotations,
            ),
        )
    if config.dataset.name == "auair":
        return AuAirAdapter(
            config.dataset.root,
            split=config.dataset.split,
            image_directory=_configured_path(
                config.dataset.images_root or config.dataset.root,
                config.dataset.images,
            ),
            annotation_file=_configured_path(
                config.dataset.root,
                config.dataset.annotations,
            ),
        )
    if config.dataset.name == "dronevehicle":
        return DroneVehicleAdapter(
            config.dataset.root,
            split=config.dataset.split,
            image_directory=_configured_path(
                config.dataset.images_root or config.dataset.root,
                config.dataset.images,
            ),
            annotation_file=_configured_path(
                config.dataset.root,
                config.dataset.annotations,
            ),
        )
    if config.dataset.name == "visdrone":
        return VisDroneAdapter(
            config.dataset.root,
            split=config.dataset.split,
            image_directory=_configured_path(
                config.dataset.images_root or config.dataset.root,
                config.dataset.images,
            ),
            annotation_directory=_configured_path(
                config.dataset.root,
                config.dataset.annotations,
            ),
        )
    raise DetectorError(f"unsupported inference dataset: {config.dataset.name}")


def _visdrone_adapter_from_mapping(dataset: dict[str, Any]) -> VisDroneAdapter:
    root = Path(str(dataset["root"]))
    images_root_value = dataset.get("images_root")
    images_root = Path(str(images_root_value)) if images_root_value else root
    return VisDroneAdapter(
        root,
        split=str(dataset["split"]),
        image_directory=_configured_path(images_root, Path(str(dataset["images"]))),
        annotation_directory=_configured_path(root, Path(str(dataset["annotations"]))),
    )


def _uavdt_adapter_from_mapping(dataset: dict[str, Any]) -> UavdtAdapter:
    root = Path(str(dataset["root"]))
    images_root_value = dataset.get("images_root")
    images_root = Path(str(images_root_value)) if images_root_value else root
    return UavdtAdapter(
        root,
        split=str(dataset["split"]),
        image_directory=_configured_path(images_root, Path(str(dataset["images"]))),
        annotation_file=_configured_path(root, Path(str(dataset["annotations"]))),
    )


def _auair_adapter_from_mapping(dataset: dict[str, Any]) -> AuAirAdapter:
    root = Path(str(dataset["root"]))
    images_root_value = dataset.get("images_root")
    images_root = Path(str(images_root_value)) if images_root_value else root
    return AuAirAdapter(
        root,
        split=str(dataset["split"]),
        image_directory=_configured_path(images_root, Path(str(dataset["images"]))),
        annotation_file=_configured_path(root, Path(str(dataset["annotations"]))),
    )


def _dronevehicle_adapter_from_mapping(dataset: dict[str, Any]) -> DroneVehicleAdapter:
    root = Path(str(dataset["root"]))
    images_root_value = dataset.get("images_root")
    images_root = Path(str(images_root_value)) if images_root_value else root
    return DroneVehicleAdapter(
        root,
        split=str(dataset["split"]),
        image_directory=_configured_path(images_root, Path(str(dataset["images"]))),
        annotation_file=_configured_path(root, Path(str(dataset["annotations"]))),
    )


def _configured_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _data_manifest(
    config: AppConfig,
    adapter: HazyDetAdapter | VisDroneAdapter | UavdtAdapter | AuAirAdapter | DroneVehicleAdapter,
    records: tuple[ImageRecord, ...],
) -> dict[str, Any]:
    sequence_by_image = adapter.sequence_by_image() if isinstance(adapter, AuAirAdapter) else {}
    manifest: dict[str, Any] = {
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "corruption_name": config.dataset.corruption_name,
        "corruption_severity": config.dataset.corruption_severity,
        "images": [
            {
                "image_id": record.image_id,
                "path": str(Path(record.path).resolve()),
                "sha256": sha256_file(Path(record.path)),
                "width": record.width,
                "height": record.height,
                **(
                    {"sequence": sequence_by_image[int(record.image_id)]}
                    if isinstance(adapter, AuAirAdapter)
                    else {}
                ),
            }
            for record in records
        ],
    }
    if isinstance(adapter, (HazyDetAdapter, UavdtAdapter, AuAirAdapter, DroneVehicleAdapter)):
        annotation = adapter.annotation_file()
        manifest.update(
            annotation=str(annotation.resolve()),
            annotation_sha256=sha256_file(annotation),
        )
    else:
        annotation_rows = [
            {
                "image_id": record.image_id,
                "path": str(adapter.annotation_file(Path(record.path)).resolve()),
                "sha256": sha256_file(adapter.annotation_file(Path(record.path))),
            }
            for record in records
        ]
        manifest["annotations"] = annotation_rows
    return manifest


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    numeric = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numeric))
        writer.writeheader()
        writer.writerow(numeric)
    temporary.replace(path)
