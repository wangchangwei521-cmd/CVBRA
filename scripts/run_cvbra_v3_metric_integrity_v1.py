from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from scripts import analyze_cvbra_v1_uav_obb_official_validation as target_analysis
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_hazydet_source_retention_v3 as hazydet_reference
from scripts import run_cvbra_v1_training_seed_robustness as seed_evidence

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import (
    evaluate_coco,
    evaluate_coco_per_class,
    write_coco_predictions,
)
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/development/cvbra_v3_metric_integrity_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
POINT_REPORT = OUTPUT / "low_floor_point_report.json"
ROBUSTNESS_REPORT = OUTPUT / "score_floor_robustness.json"
STATISTICS_REPORT = OUTPUT / "core_paired_statistics.json"
COMPLETE = OUTPUT / "COMPLETE.json"

TARGET_ANNOTATION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations"
    / "official_validation_exact_car_truck_bus_hbb.coco.json"
)
HAZY_ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"

LOW_FLOOR = 0.001
SCORE_FLOORS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.25)
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
CLASS_NAMES = ("car", "truck", "bus")
TARGET_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}
IMGSZ = 1280
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
TARGET_RESAMPLES = 10_000
HAZY_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260824
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")

YOLO_CHECKPOINTS: dict[str, Path] = {
    "source": ROOT / "weights/hazydet/yolo11n_best.pt",
    "STF": ROOT / "runs/cvbra_v1_matched_baselines/STF/STF.pt",
    "CVBRA_noCV": (ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noCV/CVBRA_noCV.pt"),
    "CVBRA_noReplay": (ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noReplay/CVBRA_noReplay.pt"),
    "CVBRA_noFreeze": (ROOT / "runs/cvbra_v1_matched_baselines/CVBRA_noFreeze/CVBRA_noFreeze.pt"),
    "PlainMix": ROOT / "runs/cvbra_v1_acceptance_upgrade/PlainMix/PlainMix.pt",
    "CVBRA_EWC": (ROOT / "runs/cvbra_v1_quality_upgrade_v2/CVBRA_EWC/CVBRA_EWC.pt"),
    "qS_0p125_L10": (ROOT / "runs/cvbra_v1_allocation_sensitivity_v1/qS_0p125/qS_0p125.pt"),
    "qS_0p25_L15": (
        ROOT
        / "runs/cvbra_v1_allocation_sensitivity_v1/first_trainable_15"
        / "first_trainable_15.pt"
    ),
    "CVBRA_L5_s42": (
        ROOT / "runs/cvbra_v1_allocation_sensitivity_v1/first_trainable_5" / "first_trainable_5.pt"
    ),
    "CVBRA_L10_s42": ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt",
    "CVBRA_L5_s27182": (
        ROOT / "runs/cvbra_v2_reviewer_closure_v1/layer_5/seed_27182/cvbra_l5_s27182.pt"
    ),
    "CVBRA_L10_s27182": (
        ROOT / "runs/cvbra_v2_reviewer_closure_v1/layer_10/seed_27182/cvbra_l10_s27182.pt"
    ),
    "CVBRA_L5_s31415": (
        ROOT / "runs/cvbra_v2_reviewer_closure_v1/layer_5/seed_31415/cvbra_l5_s31415.pt"
    ),
    "CVBRA_L10_s31415": (
        ROOT / "runs/cvbra_v2_reviewer_closure_v1/layer_10/seed_31415/cvbra_l10_s31415.pt"
    ),
    "ALDIpp_AF_native": (
        ROOT / "runs/cvbra_v1_aldi_direct_baseline_v1/native_uda" / "ALDIpp_AF_Y11_native_UDA.pt"
    ),
    "ALDIpp_AF_equal": (
        ROOT
        / "runs/cvbra_v1_aldi_direct_baseline_v1/equal_supervision"
        / "ALDIpp_AF_Y11_equal_supervision.pt"
    ),
    "ALDIpp_AF_HN": (ROOT / "runs/cvbra_v2_aldi_horizon_normalized_v1/ALDIpp_AF_HN_equal.pt"),
}

RTDETR_CHECKPOINTS: dict[str, Path] = {
    "RTDETR_source": ROOT / "weights/hazydet/rtdetr_l_best.pt",
    "RTDETR_CVBRA_L10": ROOT / "runs/cvbra_v1_rtdetr_l/cvbra_v1_rtdetr_l.pt",
}

CORE_MODELS = (
    "source",
    "CVBRA_L10_s42",
    "CVBRA_L5_s42",
    "ALDIpp_AF_HN",
)
TRAJECTORY_MODELS = tuple(
    f"CVBRA_L{boundary}_s{seed}" for boundary in (5, 10) for seed in (42, 27182, 31415)
)


class MetricIntegrityError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction(model: str, domain: str, view: str) -> Path:
    return OUTPUT / "predictions" / model / domain / view / "predictions.coco.json"


def _success(model: str, domain: str, view: str) -> Path:
    return _prediction(model, domain, view).parent / "SUCCESS.json"


def _all_checkpoints() -> dict[str, Path]:
    return {**YOLO_CHECKPOINTS, **RTDETR_CHECKPOINTS}


def register() -> dict[str, Any]:
    inputs = {
        "runner": Path(__file__),
        "target_annotation": TARGET_ANNOTATION,
        "hazydet_annotation": HAZY_ANNOTATION,
        **{f"checkpoint:{name}": path for name, path in _all_checkpoints().items()},
    }
    missing = [label for label, path in inputs.items() if not path.is_file()]
    if missing:
        raise MetricIntegrityError(f"metric-integrity inputs are missing: {missing}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_METRIC_INTEGRITY_REGISTERED_BEFORE_INFERENCE",
        "registered_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "target_annotation_sha256": sha256_file(TARGET_ANNOTATION),
        "hazydet_annotation_sha256": sha256_file(HAZY_ANNOTATION),
        "checkpoint_sha256": {name: sha256_file(path) for name, path in _all_checkpoints().items()},
        "candidate_score_floor": LOW_FLOOR,
        "score_floor_sensitivity": list(SCORE_FLOORS),
        "nms_iou": NMS_IOU,
        "max_detections_per_image": MAX_DET,
        "target_resamples": TARGET_RESAMPLES,
        "hazydet_resamples": HAZY_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fixed_checkpoints_only": True,
        "method_or_hyperparameter_selection": False,
        "experimental_conclusion_reselection": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = tuple(
            key for key in payload if key not in {"schema_version", "status", "registered_at_utc"}
        )
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise MetricIntegrityError("metric-integrity registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _write_prediction(
    *,
    model: str,
    domain: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    path = _prediction(model, domain, view)
    marker = _success(model, domain, view)
    if marker.exists():
        value = _load(marker)
        if value.get("prediction_sha256") != sha256_file(path):
            raise MetricIntegrityError(f"low-floor prediction changed: {path}")
        return value
    if path.parent.exists():
        raise MetricIntegrityError(f"partial low-floor prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    rows = write_coco_predictions(path, batches, category_id_by_class=category_map)
    if not rows:
        raise MetricIntegrityError(f"low-floor prediction is empty: {model}/{domain}/{view}")
    minimum = min(float(row["score"]) for row in rows)
    maximum = max(float(row["score"]) for row in rows)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_LOW_FLOOR_PREDICTION_COMPLETE",
        "model": model,
        "domain": domain,
        "view": view,
        "images": len(records),
        "detections": len(rows),
        "minimum_score": minimum,
        "maximum_score": maximum,
        "candidate_score_floor": LOW_FLOOR,
        "checkpoint_sha256": sha256_file(_all_checkpoints()[model]),
        "prediction_sha256": sha256_file(path),
    }
    atomic_write_json(marker, payload)
    return payload


def infer() -> dict[str, Any]:
    register()
    if PREDICTION_LOCK.exists():
        return _load(PREDICTION_LOCK)
    target_records, _ = sensitivity._target_records()
    hazy_records = seed_evidence._hazy_records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    configure_ultralytics_environment(ROOT)
    for model, checkpoint in _all_checkpoints().items():
        detector = UltralyticsDetector(
            checkpoint,
            model_name="rtdetr_l" if model.startswith("RTDETR_") else "yolo11n",
            device="cuda:0",
            expected_class_names=CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        detector.predict(
            target_records["original"][:WARMUP_IMAGES],
            imgsz=IMGSZ,
            conf=LOW_FLOOR,
            iou=NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        for view in TARGET_VIEWS:
            artifacts.append(
                _write_prediction(
                    model=model,
                    domain="UAV_OBB_validation",
                    view=view,
                    detector=detector,
                    records=target_records[view],
                    category_map=TARGET_CATEGORY_ID_BY_CLASS,
                )
            )
        artifacts.append(
            _write_prediction(
                model=model,
                domain="HazyDet_validation",
                view="hazy",
                detector=detector,
                records=hazy_records,
                category_map=HAZY_CATEGORY_ID_BY_CLASS,
            )
        )
        del detector
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"low_floor_inference_complete": model}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_LOW_FLOOR_PREDICTIONS_LOCKED",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "artifacts": artifacts,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def _metrics(
    annotation: Path,
    prediction: Path,
    *,
    image_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    result = evaluate_coco(
        annotation,
        prediction,
        max_det=MAX_DET,
        image_ids=image_ids,
    )
    return {key: float(result[key]) for key in METRIC_KEYS} | {
        "images_evaluated": int(result["images_evaluated"])
    }


def points() -> dict[str, Any]:
    infer()
    if POINT_REPORT.exists():
        return _load(POINT_REPORT)
    _, primary_ids = sensitivity._target_records()
    target_rows: list[dict[str, Any]] = []
    hazy_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model in _all_checkpoints():
        for view in TARGET_VIEWS:
            prediction = _prediction(model, "UAV_OBB_validation", view)
            target_rows.append(
                {
                    "model": model,
                    "view": view,
                    **_metrics(TARGET_ANNOTATION, prediction, image_ids=primary_ids),
                    "prediction_sha256": sha256_file(prediction),
                }
            )
            if model in CORE_MODELS:
                for row in evaluate_coco_per_class(
                    TARGET_ANNOTATION,
                    prediction,
                    max_det=MAX_DET,
                    image_ids=primary_ids,
                ):
                    per_class_rows.append({"model": model, "view": view, **row})
        prediction = _prediction(model, "HazyDet_validation", "hazy")
        hazy_rows.append(
            {
                "model": model,
                **_metrics(HAZY_ANNOTATION, prediction),
                "prediction_sha256": sha256_file(prediction),
            }
        )
    point_lookup = {(str(row["model"]), str(row["view"])): float(row["AP"]) for row in target_rows}
    point_lookup.update({(str(row["model"]), "HazyDet"): float(row["AP"]) for row in hazy_rows})
    trajectory_summary: list[dict[str, Any]] = []
    for boundary in (5, 10):
        for coordinate in (*TARGET_VIEWS, "HazyDet"):
            values = [
                point_lookup[(f"CVBRA_L{boundary}_s{seed}", coordinate)]
                for seed in (42, 27182, 31415)
            ]
            trajectory_summary.append(
                {
                    "boundary": boundary,
                    "coordinate": coordinate,
                    "values": values,
                    "mean": fmean(values),
                    "sample_standard_deviation": stdev(values),
                }
            )
    allocation_models = {
        "qS_0_L10": "CVBRA_noReplay",
        "qS_0p125_L10": "qS_0p125_L10",
        "qS_0p25_L0": "CVBRA_noFreeze",
        "qS_0p25_L5": "CVBRA_L5_s42",
        "qS_0p25_L10": "CVBRA_L10_s42",
        "qS_0p25_L15": "qS_0p25_L15",
    }
    profiles = {
        allocation: {
            coordinate: point_lookup[(model, coordinate)]
            for coordinate in (*TARGET_VIEWS, "HazyDet")
        }
        for allocation, model in allocation_models.items()
    }
    nondominated: list[str] = []
    for allocation, profile in profiles.items():
        dominated = False
        for other, other_profile in profiles.items():
            if allocation == other:
                continue
            no_smaller = all(other_profile[key] >= profile[key] for key in profile)
            strictly_larger = any(other_profile[key] > profile[key] for key in profile)
            if no_smaller and strictly_larger:
                dominated = True
                break
        if not dominated:
            nondominated.append(allocation)
    report = {
        "schema_version": 1,
        "status": "CVBRA_V3_LOW_FLOOR_POINT_ESTIMATES_COMPLETE",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "candidate_score_floor": LOW_FLOOR,
        "target_rows": target_rows,
        "hazydet_rows": hazy_rows,
        "primary_target_per_class_rows": per_class_rows,
        "trajectory_summary": trajectory_summary,
        "allocation_profiles": profiles,
        "nondominated_allocations": nondominated,
        "point_estimates_conditional_on_fixed_checkpoints": True,
    }
    atomic_write_json(POINT_REPORT, report)
    return report


def _evaluate_rows(
    annotation: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    image_ids: Sequence[int],
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise MetricIntegrityError("pycocotools is not installed") from exc
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation))
        ground_truth.dataset.setdefault("info", {})
        detections = ground_truth.loadRes(list(rows))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.params.maxDets = [1, min(100, MAX_DET), MAX_DET]
        evaluator.evaluate()
        evaluator.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.summarize()
    return {key: float(evaluator.stats[index]) for index, key in enumerate(METRIC_KEYS)}


def robustness() -> dict[str, Any]:
    points()
    if ROBUSTNESS_REPORT.exists():
        return _load(ROBUSTNESS_REPORT)
    _, primary_ids = sensitivity._target_records()
    hazy_ids = [int(record.image_id) for record in seed_evidence._hazy_records(verify_hashes=False)]
    rows: list[dict[str, Any]] = []
    for model in CORE_MODELS:
        for domain, views, annotation, image_ids in (
            ("UAV_OBB_validation", TARGET_VIEWS, TARGET_ANNOTATION, primary_ids),
            ("HazyDet_validation", ("hazy",), HAZY_ANNOTATION, hazy_ids),
        ):
            for view in views:
                source_rows = _load(_prediction(model, domain, view))
                if not isinstance(source_rows, list):
                    raise MetricIntegrityError("prediction document is not a list")
                for floor in SCORE_FLOORS:
                    filtered = [row for row in source_rows if float(row["score"]) >= floor]
                    result = _evaluate_rows(
                        annotation,
                        filtered,
                        image_ids=image_ids,
                    )
                    rows.append(
                        {
                            "model": model,
                            "domain": domain,
                            "view": view,
                            "score_floor": floor,
                            "detections": len(filtered),
                            **result,
                        }
                    )
    by_cell = {
        (str(row["model"]), str(row["domain"]), str(row["view"]), float(row["score_floor"])): row
        for row in rows
    }
    convergence: list[dict[str, Any]] = []
    for model in CORE_MODELS:
        for domain, views in (
            ("UAV_OBB_validation", TARGET_VIEWS),
            ("HazyDet_validation", ("hazy",)),
        ):
            for view in views:
                reference = float(by_cell[(model, domain, view, LOW_FLOOR)]["AP"])
                for floor in SCORE_FLOORS[1:]:
                    value = float(by_cell[(model, domain, view, floor)]["AP"])
                    convergence.append(
                        {
                            "model": model,
                            "domain": domain,
                            "view": view,
                            "score_floor": floor,
                            "AP_change_from_0p001": value - reference,
                        }
                    )
    report = {
        "schema_version": 1,
        "status": "CVBRA_V3_SCORE_FLOOR_ROBUSTNESS_COMPLETE",
        "candidate_score_floor": LOW_FLOOR,
        "rows": rows,
        "convergence": convergence,
        "maximum_absolute_AP_change_0p001_to_0p005": max(
            abs(float(row["AP_change_from_0p001"]))
            for row in convergence
            if math.isclose(float(row["score_floor"]), 0.005)
        ),
        "threshold_or_model_selection": False,
    }
    atomic_write_json(ROBUSTNESS_REPORT, report)
    return report


def _checkpoint_deltas(path: Path, expected: int) -> list[float]:
    document = _load(path)
    raw = document.get("deltas")
    if not isinstance(raw, list) or len(raw) != expected:
        raise MetricIntegrityError(f"bootstrap checkpoint is incomplete: {path}")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise MetricIntegrityError(f"bootstrap checkpoint is nonfinite: {path}")
    return values


def _paired(
    *,
    family: str,
    name: str,
    annotation: Path,
    baseline: Path,
    method: Path,
    clusters: Mapping[str, Sequence[int]],
    resamples: int,
) -> tuple[dict[str, Any], list[float]]:
    checkpoint = OUTPUT / "bootstrap" / family / f"{name}.json"
    result = paired_coco_ap_cluster_bootstrap_scopes(
        annotation,
        baseline,
        method,
        {
            "analysis": ClusterBootstrapScope(
                clusters=clusters,
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "study": "cvbra_v3_metric_integrity_v1",
                    "family": family,
                    "comparison": name,
                    "registration_sha256": sha256_file(REGISTRATION),
                    "candidate_score_floor": LOW_FLOOR,
                },
            )
        },
        resamples=resamples,
        seed=BOOTSTRAP_SEED,
        max_det=MAX_DET,
        workers=4,
        chunk_resamples=100,
        accelerate_ap_only=True,
    )["analysis"]
    deltas = _checkpoint_deltas(checkpoint, resamples)
    spread = stdev(deltas)
    row = {
        "comparison": name,
        **result,
        "bootstrap_mean_delta": fmean(deltas),
        "bootstrap_standard_deviation": spread,
        "standardized_effect": float(result["delta"]) / spread if spread > 0 else 0.0,
        "p_two_sided": bootstrap_sign_pvalue(deltas),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return row, deltas


def statistics() -> dict[str, Any]:
    points()
    if STATISTICS_REPORT.exists():
        return _load(STATISTICS_REPORT)
    target_clusters = target_analysis._clusters(TARGET_ANNOTATION)
    hazy_ids = [int(record.image_id) for record in seed_evidence._hazy_records(verify_hashes=False)]
    hazy_clusters = hazydet_reference.singleton_image_clusters(hazy_ids)
    target_rows: list[dict[str, Any]] = []
    target_specs = (
        ("L10_minus_source", "source", "CVBRA_L10_s42"),
        ("L10_minus_ALDI_HN", "ALDIpp_AF_HN", "CVBRA_L10_s42"),
        ("L5_minus_ALDI_HN", "ALDIpp_AF_HN", "CVBRA_L5_s42"),
    )
    for stem, baseline, method in target_specs:
        for view in TARGET_VIEWS:
            row, _ = _paired(
                family="target_core",
                name=f"{stem}_{view}",
                annotation=TARGET_ANNOTATION,
                baseline=_prediction(baseline, "UAV_OBB_validation", view),
                method=_prediction(method, "UAV_OBB_validation", view),
                clusters=target_clusters,
                resamples=TARGET_RESAMPLES,
            )
            row.update({"view": view, "baseline": baseline, "method": method})
            target_rows.append(row)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in target_rows])
    for row, value in zip(target_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05

    hazy_rows: list[dict[str, Any]] = []
    for stem, baseline, method in target_specs:
        row, _ = _paired(
            family="hazydet_core",
            name=f"{stem}_hazy",
            annotation=HAZY_ANNOTATION,
            baseline=_prediction(baseline, "HazyDet_validation", "hazy"),
            method=_prediction(method, "HazyDet_validation", "hazy"),
            clusters=hazy_clusters,
            resamples=HAZY_RESAMPLES,
        )
        row.update({"view": "HazyDet", "baseline": baseline, "method": method})
        hazy_rows.append(row)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in hazy_rows])
    for row, value in zip(hazy_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    report = {
        "schema_version": 1,
        "status": "CVBRA_V3_LOW_FLOOR_CORE_PAIRED_STATISTICS_COMPLETE",
        "candidate_score_floor": LOW_FLOOR,
        "target_rows": target_rows,
        "hazydet_rows": hazy_rows,
        "sampling_variability_conditional_on_fixed_checkpoints": True,
    }
    atomic_write_json(STATISTICS_REPORT, report)
    return report


def analyze() -> dict[str, Any]:
    register()
    point_report = points()
    robustness_report = robustness()
    statistics_report = statistics()
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V3_METRIC_INTEGRITY_V1",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "point_report_sha256": sha256_file(POINT_REPORT),
        "robustness_report_sha256": sha256_file(ROBUSTNESS_REPORT),
        "statistics_report_sha256": sha256_file(STATISTICS_REPORT),
        "nondominated_allocations": point_report["nondominated_allocations"],
        "maximum_absolute_AP_change_0p001_to_0p005": robustness_report[
            "maximum_absolute_AP_change_0p001_to_0p005"
        ],
        "core_target_intervals_all_positive": all(
            float(row["ci_low"]) > 0 for row in statistics_report["target_rows"]
        ),
        "method_or_hyperparameter_selection": False,
    }
    atomic_write_json(COMPLETE, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "infer", "points", "robustness", "statistics", "analyze"),
    )
    args = parser.parse_args()
    result = {
        "register": register,
        "infer": infer,
        "points": points,
        "robustness": robustness,
        "statistics": statistics,
        "analyze": analyze,
    }[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
