from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from buse_uav.evaluation.aggregate import AggregateError, ValidatedRun, validate_run
from buse_uav.evaluation.bootstrap import paired_coco_ap_bootstrap
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.final_bootstrap import load_final_bootstrap_pairs
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text


class SupplementaryAnalysisError(RuntimeError):
    """Raised when frozen supplementary evidence is incomplete or inconsistent."""


BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_RECONSTRUCTION_TOLERANCE = 5e-5
VISDRONE_CORRUPTIONS = (
    "fog",
    "low_light",
    "gaussian_blur",
    "motion_blur",
    "rain",
    "jpeg",
)
HAZYDET_VAL_BASELINE = "20260801T050311Z_hazydet_baseline_yolo11n_5691371a_bench20260801T050311Z_r1"
HAZYDET_VAL_FULL = "20260801T072527Z_hazydet_buse_yolo11n_438532a6"


def build_supplementary_analysis(
    root: Path,
    *,
    output: Path,
    state_dir: Path,
    report_path: Path,
    workers: int = 8,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    smoke: bool = False,
) -> dict[str, Any]:
    """Build reporting-only statistics from frozen predictions and validation traces."""
    root = root.resolve()
    output = _inside_root(root, output)
    state_dir = _inside_root(root, state_dir)
    report_path = _inside_root(root, report_path)
    if workers <= 0:
        raise ValueError("supplementary bootstrap workers must be positive")
    if smoke:
        if resamples <= 0 or resamples > 10:
            raise ValueError("supplementary smoke requires 1-10 resamples")
    elif resamples != BOOTSTRAP_RESAMPLES or seed != BOOTSTRAP_SEED:
        raise ValueError("final supplementary analysis requires 1,000 resamples and seed 42")

    output.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    rtdetr_rows, rtdetr_details = _rtdetr_bootstrap_rows(
        root,
        state_dir=state_dir,
        workers=workers,
        resamples=resamples,
        seed=seed,
        smoke=smoke,
    )
    visdrone_rows, cluster_rows, visdrone_details = _visdrone_bootstrap_rows(
        root,
        state_dir=state_dir,
        workers=workers,
        resamples=resamples,
        seed=seed,
        smoke=smoke,
    )
    image_rows, mechanism_rows, fusion_rows, mechanism_details = _mechanism_rows(root)

    artifacts: dict[str, dict[str, Any]] = {}
    if not smoke:
        table_rows = {
            "bootstrap_rtdetr.csv": rtdetr_rows,
            "visdrone_cluster_bootstrap.csv": cluster_rows,
            "visdrone_multiplicity.csv": visdrone_rows,
            "mechanism_val_images.csv": image_rows,
            "mechanism_val_summary.csv": mechanism_rows,
            "fusion_val_analysis.csv": fusion_rows,
        }
        for name, rows in table_rows.items():
            path = output / name
            _write_csv(path, rows)
            artifacts[name] = _artifact(path, root=root)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "protocol": "frozen_predictions_reporting_only_supplementary_analysis",
        "mode": "smoke" if smoke else "final",
        "resamples": resamples,
        "seed": seed,
        "workers": workers,
        "invariants": {
            "detector_inference_rerun": False,
            "training_rerun": False,
            "test_tuning": False,
            "frozen_predictions_only": True,
            "mechanism_split": "val",
            "mechanism_used_for_tuning": False,
            "visdrone_cluster_unit": "original_image",
        },
        "rtdetr": rtdetr_details,
        "visdrone": visdrone_details,
        "mechanism": mechanism_details,
        "artifacts": artifacts,
    }
    atomic_write_json(report_path, report)
    return {
        "status": "PASS",
        "mode": report["mode"],
        "rtdetr_comparisons": len(rtdetr_rows),
        "visdrone_cell_comparisons": len(visdrone_rows),
        "visdrone_cluster_scopes": len(cluster_rows),
        "mechanism_images": len(image_rows),
        "report": str(report_path),
    }


def bootstrap_sign_pvalue(deltas: Sequence[float]) -> float:
    """Two-sided sign-tail bootstrap p-value with finite-sample correction."""
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise ValueError("bootstrap deltas must be a nonempty finite sequence")
    size = len(deltas)
    nonpositive = sum(value <= 0.0 for value in deltas)
    nonnegative = sum(value >= 0.0 for value in deltas)
    tail = min((nonpositive + 1) / (size + 1), (nonnegative + 1) / (size + 1))
    return min(1.0, 2.0 * tail)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm family-wise error correction in original row order."""
    _validate_pvalues(pvalues)
    total = len(pvalues)
    ordered = sorted(range(total), key=lambda index: pvalues[index])
    adjusted = [0.0] * total
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery correction in original row order."""
    _validate_pvalues(pvalues)
    total = len(pvalues)
    ordered = sorted(range(total), key=lambda index: pvalues[index])
    adjusted = [0.0] * total
    running = 1.0
    for rank in range(total, 0, -1):
        index = ordered[rank - 1]
        running = min(running, pvalues[index] * total / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def cluster_bootstrap_summary(
    distributions: Mapping[str, Sequence[float]],
    observed: Mapping[str, float],
    keys: Sequence[str],
) -> dict[str, float | int]:
    """Average aligned cell AP deltas while retaining original-image clusters."""
    if not keys:
        raise ValueError("cluster bootstrap requires at least one cell")
    lengths = {len(distributions[key]) for key in keys}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("cluster bootstrap distributions must have equal positive lengths")
    matrix = np.asarray([distributions[key] for key in keys], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("cluster bootstrap distributions must be finite")
    deltas = np.mean(matrix, axis=0)
    return {
        "delta": fmean(observed[key] for key in keys),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_p_two_sided": bootstrap_sign_pvalue(deltas.tolist()),
        "cells": len(keys),
        "resamples": int(deltas.size),
    }


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-aware Spearman correlation without an additional runtime dependency."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have the same length of at least two")
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = fmean(left_ranks)
    right_mean = fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else math.nan


def _rtdetr_bootstrap_rows(
    root: Path,
    *,
    state_dir: Path,
    workers: int,
    resamples: int,
    seed: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "data/processed/second_detector_rtdetr_l_final_runs.json"
    manifest = _load_mapping(manifest_path)
    raw_runs = manifest.get("runs")
    if (
        manifest.get("protocol") != "second_detector_rtdetr_l_frozen_test_rddts"
        or manifest.get("tune") is not False
        or not isinstance(raw_runs, list)
        or len(raw_runs) != 4
    ):
        raise SupplementaryAnalysisError("RT-DETR final manifest is not the frozen four-cell route")
    by_key = {
        (str(row["split"]), str(row["configuration"])): row
        for row in raw_runs
        if isinstance(row, dict)
    }
    splits = ("test",) if smoke else ("test", "RDDTS")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for split in splits:
        baseline_row = by_key[(split, "B0")]
        method_row = by_key[(split, "Full")]
        baseline = validate_run(Path(str(baseline_row["run_path"])))
        method = validate_run(Path(str(method_row["run_path"])))
        image_ids = sorted(_validate_pair(baseline, method, split=split), key=str)
        checkpoint = state_dir / f"rtdetr_{split.casefold()}_sorted_bootstrap.json"
        result = paired_coco_ap_bootstrap(
            _annotation_path(method),
            baseline.path / "predictions/final.coco.json",
            method.path / "predictions/final.coco.json",
            resamples=resamples,
            seed=seed,
            max_det=int(method.config["detector"]["max_det"]),
            image_ids=image_ids,
            workers=workers,
            checkpoint_path=checkpoint,
            checkpoint_identity={
                "protocol": "rtdetr_reporting_only_bootstrap",
                "split": split,
                "manifest_sha256": sha256_file(manifest_path),
            },
            chunk_resamples=min(5, resamples),
        )
        expected_delta = float(method.metrics["AP"]) - float(baseline.metrics["AP"])
        reference_baseline = evaluate_coco(
            _annotation_path(baseline),
            baseline.path / "predictions/final.coco.json",
            max_det=int(baseline.config["detector"]["max_det"]),
            image_ids=image_ids,
        )
        reference_method = evaluate_coco(
            _annotation_path(method),
            method.path / "predictions/final.coco.json",
            max_det=int(method.config["detector"]["max_det"]),
            image_ids=image_ids,
        )
        _require_close(
            float(reference_baseline["AP"]),
            float(baseline.metrics["AP"]),
            f"RT-DETR {split} production baseline AP",
        )
        _require_close(
            float(reference_method["AP"]),
            float(method.metrics["AP"]),
            f"RT-DETR {split} production method AP",
        )
        reconstruction_drift = float(result["delta"]) - expected_delta
        if abs(reconstruction_drift) > BOOTSTRAP_RECONSTRUCTION_TOLERANCE:
            raise SupplementaryAnalysisError(
                f"RT-DETR {split} bootstrap reconstruction drift exceeds "
                f"{BOOTSTRAP_RECONSTRUCTION_TOLERANCE}: {reconstruction_drift}"
            )
        deltas = _checkpoint_deltas(checkpoint, expected=resamples)
        checkpoint_hashes[split] = sha256_file(checkpoint)
        rows.append(
            {
                "dataset": "hazydet",
                "split": split,
                "detector": "rtdetr_l",
                "baseline_run_id": baseline.run_id,
                "method_run_id": method.run_id,
                "model_sha256": method.model_fingerprint["sha256"],
                "delta": expected_delta,
                "bootstrap_observed_delta": result["delta"],
                "bootstrap_reconstruction_drift": reconstruction_drift,
                "bootstrap_reconstruction_tolerance": BOOTSTRAP_RECONSTRUCTION_TOLERANCE,
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "bootstrap_p_two_sided": bootstrap_sign_pvalue(deltas),
                "images": result["images"],
                "resamples": result["resamples"],
                "seed": result["seed"],
                "max_det": method.config["detector"]["max_det"],
                "posthoc_reporting_only": True,
                "used_for_tuning": False,
            }
        )
    return rows, {
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_file(manifest_path),
        "comparisons": len(rows),
        "production_metric_recheck": True,
        "bootstrap_reconstruction_tolerance": BOOTSTRAP_RECONSTRUCTION_TOLERANCE,
        "max_abs_bootstrap_reconstruction_drift": max(
            abs(float(row["bootstrap_reconstruction_drift"])) for row in rows
        ),
        "checkpoint_sha256": checkpoint_hashes,
    }


def _visdrone_bootstrap_rows(
    root: Path,
    *,
    state_dir: Path,
    workers: int,
    resamples: int,
    seed: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = root / "data/processed/visdrone_testdev_final_runs.json"
    frozen_table = root / "reports/tables/bootstrap_visdrone.csv"
    frozen_rows = _read_csv(frozen_table)
    frozen_by_group = {str(row["group"]): row for row in frozen_rows}
    pairs = load_final_bootstrap_pairs(manifest_path)
    if smoke:
        pairs = pairs[:1]
    distributions: dict[str, list[float]] = {}
    observed: dict[str, float] = {}
    image_ids_hash: str | None = None
    raw_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for pair in pairs:
        baseline = validate_run(pair.baseline_run_path)
        method = validate_run(pair.method_run_path)
        image_ids = _validate_pair(baseline, method, split="test-dev")
        current_ids_hash = stable_hash(image_ids, length=64)
        if image_ids_hash is None:
            image_ids_hash = current_ids_hash
        elif current_ids_hash != image_ids_hash:
            raise SupplementaryAnalysisError(
                "VisDrone cells do not retain identical original-image ordering"
            )
        safe_key = pair.key.replace(":", "_")
        checkpoint = state_dir / f"visdrone_{safe_key}_bootstrap.json"
        result = paired_coco_ap_bootstrap(
            _annotation_path(method),
            baseline.path / "predictions/final.coco.json",
            method.path / "predictions/final.coco.json",
            resamples=resamples,
            seed=seed,
            max_det=int(method.config["detector"]["max_det"]),
            image_ids=image_ids,
            workers=workers,
            checkpoint_path=checkpoint,
            checkpoint_identity={
                "protocol": "visdrone_original_image_cluster_bootstrap",
                "comparison_key": pair.key,
                "manifest_sha256": sha256_file(manifest_path),
            },
            chunk_resamples=min(5, resamples),
        )
        expected_delta = float(method.metrics["AP"]) - float(baseline.metrics["AP"])
        _require_close(float(result["delta"]), expected_delta, f"VisDrone {pair.key} delta")
        frozen = frozen_by_group.get(pair.group)
        if frozen is None:
            raise SupplementaryAnalysisError(f"frozen bootstrap row missing for {pair.key}")
        _require_close(float(result["delta"]), float(frozen["delta"]), f"{pair.key} frozen delta")
        if not smoke:
            _require_close(float(result["ci_low"]), float(frozen["ci_low"]), f"{pair.key} CI low")
            _require_close(
                float(result["ci_high"]), float(frozen["ci_high"]), f"{pair.key} CI high"
            )
        deltas = _checkpoint_deltas(checkpoint, expected=resamples)
        distributions[pair.key] = deltas
        observed[pair.key] = expected_delta
        checkpoint_hashes[pair.key] = sha256_file(checkpoint)
        raw_rows.append(
            {
                "key": pair.key,
                "corruption": pair.corruption,
                "severity": pair.severity,
                "baseline_run_id": baseline.run_id,
                "method_run_id": method.run_id,
                "delta": result["delta"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "bootstrap_p_two_sided": bootstrap_sign_pvalue(deltas),
                "images": result["images"],
                "resamples": result["resamples"],
                "seed": result["seed"],
                "cluster_unit": "original_image",
                "posthoc_reporting_only": True,
                "used_for_tuning": False,
            }
        )
    if smoke:
        return (
            raw_rows,
            [],
            {
                "manifest_sha256": sha256_file(manifest_path),
                "cells": len(raw_rows),
                "image_ids_sha256": image_ids_hash,
                "checkpoint_sha256": checkpoint_hashes,
                "frozen_cell_reproduction": True,
            },
        )

    pvalues = [float(row["bootstrap_p_two_sided"]) for row in raw_rows]
    holm = holm_adjust(pvalues)
    fdr = benjamini_hochberg_adjust(pvalues)
    for row, holm_value, fdr_value in zip(raw_rows, holm, fdr, strict=True):
        row["holm_adjusted_p"] = holm_value
        row["fdr_bh_adjusted_p"] = fdr_value
        row["significant_unadjusted_0_05"] = float(row["bootstrap_p_two_sided"]) < 0.05
        row["significant_holm_0_05"] = holm_value < 0.05
        row["significant_fdr_bh_0_05"] = fdr_value < 0.05

    cluster_rows: list[dict[str, Any]] = []
    all_corruption_keys = [
        f"{corruption}:{severity}" for corruption in VISDRONE_CORRUPTIONS for severity in (1, 2, 3)
    ]
    scopes = [("all_corruptions", "all", all_corruption_keys)] + [
        (
            "corruption_family",
            corruption,
            [f"{corruption}:{severity}" for severity in (1, 2, 3)],
        )
        for corruption in VISDRONE_CORRUPTIONS
    ]
    for scope, corruption, keys in scopes:
        summary = cluster_bootstrap_summary(distributions, observed, keys)
        cluster_rows.append(
            {
                "scope": scope,
                "corruption": corruption,
                **summary,
                "images": 1610,
                "variants_per_original": len(keys),
                "cluster_unit": "original_image",
                "source_sha256": sha256_file(manifest_path),
                "posthoc_reporting_only": True,
                "used_for_tuning": False,
            }
        )
    return (
        raw_rows,
        cluster_rows,
        {
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256_file(manifest_path),
            "frozen_table_sha256": sha256_file(frozen_table),
            "cells": len(raw_rows),
            "cluster_scopes": len(cluster_rows),
            "image_ids_sha256": image_ids_hash,
            "checkpoint_sha256": checkpoint_hashes,
            "frozen_cell_reproduction": True,
        },
    )


def _mechanism_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline = validate_run(root / "runs" / HAZYDET_VAL_BASELINE)
    method = validate_run(root / "runs" / HAZYDET_VAL_FULL)
    image_ids = _validate_pair(baseline, method, split="val")
    annotation_path = _annotation_path(method)
    ground_truth = _load_mapping(annotation_path)
    baseline_predictions = _load_rows(baseline.path / "predictions/final.coco.json")
    method_predictions = _load_rows(method.path / "predictions/final.coco.json")
    gt_by_image = _group_rows(ground_truth.get("annotations"), source=annotation_path)
    baseline_by_image = _group_rows(
        baseline_predictions, source=baseline.path / "predictions/final.coco.json"
    )
    method_by_image = _group_rows(
        method_predictions, source=method.path / "predictions/final.coco.json"
    )
    regions = _read_jsonl(method.path / "traces/regions.jsonl")
    candidates = _read_jsonl(method.path / "traces/candidates.jsonl")
    selected_regions: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    selected_candidates: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in regions:
        if bool(row.get("selected", False)):
            selected_regions[row.get("image_id")].append(row)
    for row in candidates:
        if bool(row.get("selected", False)):
            selected_candidates[row.get("image_id")].append(row)

    image_rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        region_rows = selected_regions.get(image_id, [])
        candidate_rows = selected_candidates.get(image_id, [])
        if not region_rows or len(candidate_rows) != 1:
            raise SupplementaryAnalysisError(
                f"val mechanism trace is incomplete for image {image_id}"
            )
        baseline_score = _detection_f1(gt_by_image[image_id], baseline_by_image[image_id])
        method_score = _detection_f1(gt_by_image[image_id], method_by_image[image_id])
        candidate = candidate_rows[0]
        image_rows.append(
            {
                "image_id": image_id,
                "dataset": "hazydet",
                "split": "val",
                "baseline_run_id": baseline.run_id,
                "method_run_id": method.run_id,
                "selected_regions": len(region_rows),
                "D": fmean(float(row["D"]) for row in region_rows),
                "U": fmean(float(row["U"]) for row in region_rows),
                "D_times_U": fmean(float(row["D"]) * float(row["U"]) for row in region_rows),
                "S": fmean(float(row["S"]) for row in region_rows),
                "Q": float(candidate["Q"]),
                "operation": str(candidate["operation"]),
                "baseline_f1_iou50": baseline_score,
                "full_f1_iou50": method_score,
                "delta_f1_iou50": method_score - baseline_score,
                "posthoc_reporting_only": True,
                "used_for_tuning": False,
            }
        )

    deltas = [float(row["delta_f1_iou50"]) for row in image_rows]
    summary_rows: list[dict[str, Any]] = []
    for feature in ("D", "U", "D_times_U", "S", "Q"):
        values = [float(row[feature]) for row in image_rows]
        summary_rows.append(
            _mechanism_summary_row(
                baseline,
                method,
                analysis="spearman_vs_delta_f1_iou50",
                group=feature,
                count=len(values),
                value=spearman_correlation(values, deltas),
                deltas=deltas,
            )
        )

    ordered = sorted(image_rows, key=lambda row: (float(row["Q"]), str(row["image_id"])))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for index, row in enumerate(ordered):
        bins[min(4, index * 5 // len(ordered))].append(row)
    for index, rows in enumerate(bins, start=1):
        bin_deltas = [float(row["delta_f1_iou50"]) for row in rows]
        summary_rows.append(
            _mechanism_summary_row(
                baseline,
                method,
                analysis="Q_quintile_calibration",
                group=f"Q{index}",
                count=len(rows),
                value=fmean(float(row["Q"]) for row in rows),
                deltas=bin_deltas,
                q_min=min(float(row["Q"]) for row in rows),
                q_max=max(float(row["Q"]) for row in rows),
            )
        )
    operations = sorted({str(row["operation"]) for row in image_rows})
    for operation in operations:
        operation_deltas = [
            float(row["delta_f1_iou50"]) for row in image_rows if row["operation"] == operation
        ]
        summary_rows.append(
            _mechanism_summary_row(
                baseline,
                method,
                analysis="selected_operation_outcome",
                group=operation,
                count=len(operation_deltas),
                value=len(operation_deltas) / len(image_rows),
                deltas=operation_deltas,
            )
        )
    summary_rows.append(
        _mechanism_summary_row(
            baseline,
            method,
            analysis="overall_outcome",
            group="all",
            count=len(image_rows),
            value=sum(row["operation"] == "identity" for row in image_rows) / len(image_rows),
            deltas=deltas,
        )
    )

    fusion_rows = _fusion_rows(root)
    trace_sources = {
        "annotation_sha256": sha256_file(annotation_path),
        "baseline_prediction_sha256": sha256_file(baseline.path / "predictions/final.coco.json"),
        "method_prediction_sha256": sha256_file(method.path / "predictions/final.coco.json"),
        "regions_sha256": sha256_file(method.path / "traces/regions.jsonl"),
        "candidates_sha256": sha256_file(method.path / "traces/candidates.jsonl"),
    }
    return (
        image_rows,
        summary_rows,
        fusion_rows,
        {
            "dataset": "hazydet",
            "split": "val",
            "images": len(image_rows),
            "baseline_run_id": baseline.run_id,
            "method_run_id": method.run_id,
            "outcome": "per_image_greedy_class_aware_f1_iou50",
            "trace_sources": trace_sources,
        },
    )


def _fusion_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "reports/tables/ablation.csv"
    rows = [row for row in _read_csv(path) if row.get("ablation_axis") == "fusion"]
    if {str(row.get("ablation_value")) for row in rows} != {"hard_nms", "soft_nms", "wbf"}:
        raise SupplementaryAnalysisError("frozen val fusion ablation is incomplete")
    hard_nms = next(row for row in rows if row["ablation_value"] == "hard_nms")
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["ablation_value"])):
        output.append(
            {
                "dataset": "hazydet",
                "split": "val",
                "fusion": row["ablation_value"],
                "run_id": row["run_id"],
                "AP": row["AP"],
                "AP50": row["AP50"],
                "AP75": row["AP75"],
                "latency_mean_ms": row["latency_mean_ms"],
                "eic_mean": row["eic_mean"],
                "delta_AP_vs_hard_nms": float(row["AP"]) - float(hard_nms["AP"]),
                "reference_run_id": hard_nms["run_id"],
                "source_sha256": sha256_file(path),
                "posthoc_reporting_only": True,
                "used_for_tuning": False,
            }
        )
    return output


def _mechanism_summary_row(
    baseline: ValidatedRun,
    method: ValidatedRun,
    *,
    analysis: str,
    group: str,
    count: int,
    value: float,
    deltas: Sequence[float],
    q_min: float = math.nan,
    q_max: float = math.nan,
) -> dict[str, Any]:
    tolerance = 1e-12
    return {
        "dataset": "hazydet",
        "split": "val",
        "analysis": analysis,
        "group": group,
        "n": count,
        "value": value,
        "mean_delta_f1_iou50": fmean(deltas),
        "positive_rate": sum(value > tolerance for value in deltas) / len(deltas),
        "harm_rate": sum(value < -tolerance for value in deltas) / len(deltas),
        "neutral_rate": sum(abs(value) <= tolerance for value in deltas) / len(deltas),
        "q_min": q_min,
        "q_max": q_max,
        "baseline_run_id": baseline.run_id,
        "method_run_id": method.run_id,
        "posthoc_reporting_only": True,
        "used_for_tuning": False,
    }


def _detection_f1(
    ground_truth: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> float:
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(
        predictions, key=lambda row: float(row.get("score", 0.0)), reverse=True
    ):
        category = prediction.get("category_id")
        prediction_box = _bbox_xyxy(prediction.get("bbox"))
        best_index: int | None = None
        best_iou = 0.5
        for index, annotation in enumerate(ground_truth):
            if index in matched or annotation.get("category_id") != category:
                continue
            iou = _iou(prediction_box, _bbox_xyxy(annotation.get("bbox")))
            if iou >= best_iou:
                best_index = index
                best_iou = iou
        if best_index is not None:
            matched.add(best_index)
            true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 1.0


def _validate_pair(baseline: ValidatedRun, method: ValidatedRun, *, split: str) -> list[Any]:
    if (
        str(baseline.config["dataset"]["split"]).casefold() != split.casefold()
        or str(method.config["dataset"]["split"]).casefold() != split.casefold()
    ):
        raise SupplementaryAnalysisError(f"run pair does not match split {split}")
    if baseline.config["runtime"].get("tune") or method.config["runtime"].get("tune"):
        raise SupplementaryAnalysisError("supplementary analysis refuses tuned final runs")
    if baseline.model_fingerprint["sha256"] != method.model_fingerprint["sha256"]:
        raise SupplementaryAnalysisError("supplementary pair mixes detector weights")
    baseline_ids = [row["image_id"] for row in baseline.data_manifest["images"]]
    method_ids = [row["image_id"] for row in method.data_manifest["images"]]
    if baseline_ids != method_ids or len(method_ids) != int(method.metrics["images_evaluated"]):
        raise SupplementaryAnalysisError("supplementary pair image coverage differs")
    if int(baseline.config["detector"]["max_det"]) != int(method.config["detector"]["max_det"]):
        raise SupplementaryAnalysisError("supplementary pair max_det differs")
    return method_ids


def _annotation_path(run: ValidatedRun) -> Path:
    raw = run.data_manifest.get("annotation")
    path = Path(str(raw)) if raw else run.path / "ground_truth.coco.json"
    if not path.is_file():
        raise AggregateError(f"bootstrap annotation is unavailable: {path}")
    return path


def _checkpoint_deltas(path: Path, *, expected: int) -> list[float]:
    document = _load_mapping(path)
    raw = document.get("deltas")
    if not isinstance(raw, list) or len(raw) != expected:
        raise SupplementaryAnalysisError(f"bootstrap checkpoint is incomplete: {path}")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise SupplementaryAnalysisError(f"bootstrap checkpoint has nonfinite deltas: {path}")
    return values


def _group_rows(value: Any, *, source: Path) -> defaultdict[Any, list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise SupplementaryAnalysisError(f"COCO rows are invalid: {source}")
    grouped: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in value:
        if not isinstance(row, dict) or "image_id" not in row:
            raise SupplementaryAnalysisError(f"COCO row is invalid: {source}")
        grouped[row["image_id"]].append(row)
    return grouped


def _bbox_xyxy(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise SupplementaryAnalysisError("invalid COCO bounding box")
    x, y, width, height = (float(number) for number in value)
    if width <= 0.0 or height <= 0.0:
        raise SupplementaryAnalysisError("nonpositive COCO bounding box")
    return x, y, x + width, y + height


def _iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("rank inputs must be nonempty and finite")
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _validate_pvalues(pvalues: Sequence[float]) -> None:
    if not pvalues or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pvalues):
        raise ValueError("p-values must be a nonempty sequence within [0, 1]")


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SupplementaryAnalysisError(f"{label} mismatch: {actual} != {expected}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (json.JSONDecodeError, OSError) as exc:
        raise SupplementaryAnalysisError(f"cannot read trace {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise SupplementaryAnalysisError(f"trace contains a non-object row: {path}")
    return rows


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SupplementaryAnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SupplementaryAnalysisError(f"JSON document is not an object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SupplementaryAnalysisError(f"cannot read JSON rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SupplementaryAnalysisError(f"JSON document is not an object list: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (csv.Error, OSError) as exc:
        raise SupplementaryAnalysisError(f"cannot read CSV {path}: {exc}") from exc


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SupplementaryAnalysisError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _inside_root(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"supplementary path escapes project root: {path}") from exc
    return resolved
