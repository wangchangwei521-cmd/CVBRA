from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np

from buse_uav.evaluation import bootstrap as bootstrap_engine
from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve()
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_quality_upgrade_v2.yaml"
PROTOCOL_SHA256 = "14ce7f1edec4248080e545c5a4832a7f20959157c7ef8f84ae692f758aa980d7"
REGISTRATION = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/REGISTRATION_LOCK.json"
REGISTRATION_SHA256 = "1b218d9a585d82d01222be7990f749a6aa615887a8080e7b17f7994a26063dd7"
REGISTRATION_MARKER = REGISTRATION.parent / "REGISTERED"
REGISTRATION_MARKER_SHA256 = (
    "6ef505f3a5431ba9a84886f5cb951e78eb9e3be3080a391aad27fe4c508b8343"
)

UAV_CONVERSION_LOCK = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations/conversion_lock.json"
)
UAV_CONVERSION_LOCK_SHA256 = (
    "aee05f6635fd4d9f56cce016a8cfd79f9a91da966a6284278865ffcbcf59e179"
)
UAV_ANNOTATION = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/annotations/"
    "official_validation_exact_car_truck_bus_hbb.coco.json"
)
UAV_ANNOTATION_SHA256 = "21d2431e0c6d70b21dc9bf9a3da03ae19a2d6e1cba96530c8e4be44e8bfe945e"
HAZY_ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"
HAZY_ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"

UAV_PRIMARY_LOCK = (
    ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation/"
    "joint_prediction_lock.json"
)
UAV_MATCHED_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/uav_obb_official_validation/"
    "evaluation/joint_prediction_lock.json"
)
UAV_PLAINMIX_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/"
    "uav_obb_official_validation/evaluation/joint_prediction_lock.json"
)
UAV_EWC_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_quality_upgrade_v2/CVBRA_EWC/"
    "uav_obb_official_validation/evaluation/joint_prediction_lock.json"
)
UAV_RAW_PRIMARY_LOCK = UAV_PRIMARY_LOCK.with_name("raw_observation_lock.json")
UAV_RAW_MATCHED_LOCK = UAV_MATCHED_LOCK.with_name("raw_observation_lock.json")
UAV_RAW_PLAINMIX_LOCK = UAV_PLAINMIX_LOCK.with_name("raw_observation_lock.json")

HAZY_REFERENCE_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/"
    "hazydet_source_retention_v2/prediction_lock.json"
)
HAZY_REFERENCE_LOCK_SHA256 = (
    "8733301dc19faa2b65acd668200fd2842a6223a785e6879ea50a9d209e5637e6"
)
HAZY_PLAINMIX_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/"
    "hazydet_validation/prediction_lock.json"
)
HAZY_EWC_LOCK = REGISTRATION.parent / "CVBRA_EWC/hazydet_validation_v2/prediction_lock.json"
HAZY_EWC_AMENDMENT = REGISTRATION.parent / "HAZYDET_INFERENCE_AMENDMENT_V2.json"
HAZY_EWC_AMENDMENT_SHA256 = (
    "12cccb11dd37ef6fbe4733c1ee472208d14c4a13862d83522bdec80278d04bad"
)

ORDER_REPORT = (
    ROOT
    / "reports/development/cvbra_v1_training_order_robustness_v4/"
    "effective_order_robustness_v4_report.json"
)
ORDER_REPORT_SHA256 = "11348a02dba2b9f6a3724768b520b4f85330e65ea8e6ff55c8d51502b7ad2138"

OUTPUT = REGISTRATION.parent / "analysis"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
POINTS = OUTPUT / "point_estimates.json"
POINTS_MARKER = OUTPUT / "POINTS_LOCKED"
FACTORIAL = OUTPUT / "factorial_statistics.json"
FACTORIAL_MARKER = OUTPUT / "FACTORIAL_LOCKED"
RETENTION = OUTPUT / "ewc_retention_statistics.json"
RETENTION_MARKER = OUTPUT / "RETENTION_LOCKED"
THRESHOLDS = OUTPUT / "threshold_robustness.json"
THRESHOLDS_MARKER = OUTPUT / "THRESHOLDS_LOCKED"
INFLUENCE = OUTPUT / "group_influence.json"
INFLUENCE_MARKER = OUTPUT / "INFLUENCE_LOCKED"
REPORT = OUTPUT / "quality_upgrade_report.json"
COMPLETE = OUTPUT / "QUALITY_UPGRADE_COMPLETE"

MODELS = ("source", "CVBRA_v1", "CVBRA_noCV", "CVBRA_noFreeze", "PlainMix", "CVBRA_EWC")
FACTORIAL_MODELS = ("CVBRA_v1", "CVBRA_noCV", "CVBRA_noFreeze", "PlainMix")
VIEWS = ("original", "fog_0p6", "fog_1p0")
FACTORIAL_VIEWS = ("original", "fog_1p0")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
THRESHOLD_VALUES = (0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
UAV_IMAGES = 167
UAV_GROUPS = 141
HAZY_IMAGES = 1000
UAV_RESAMPLES = 10000
HAZY_RESAMPLES = 2000
SEED = 20260821
MAX_DET = 500


class QualityUpgradeAnalysisError(RuntimeError):
    """Raised when registered post-freeze analysis cannot fail closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityUpgradeAnalysisError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityUpgradeAnalysisError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityUpgradeAnalysisError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise QualityUpgradeAnalysisError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected).casefold():
        raise QualityUpgradeAnalysisError(f"locked {label} changed: {path}")


def _validate_prediction_lock(path: Path, *, expected_models: set[str]) -> dict[str, Any]:
    if not path.is_file():
        raise QualityUpgradeAnalysisError(f"prediction lock is missing: {path}")
    lock = _load_mapping(path)
    marker_candidates = (path.parent / "PREDICTIONS_LOCKED", path.parent / "PREDICTION_LOCKED")
    marker = next((candidate for candidate in marker_candidates if candidate.is_file()), None)
    if marker is None:
        raise QualityUpgradeAnalysisError(f"prediction marker is missing: {path}")
    marker_data = _load_mapping(marker)
    marker_hashes = {
        str(value)
        for key, value in marker_data.items()
        if key.endswith("prediction_lock_sha256") or key == "joint_prediction_lock_sha256"
    }
    if sha256_file(path) not in marker_hashes:
        raise QualityUpgradeAnalysisError(f"prediction marker changed: {path}")
    observed: set[str] = set()
    artifacts = lock.get("artifacts")
    if isinstance(artifacts, list):
        for row in artifacts:
            if not isinstance(row, dict):
                raise QualityUpgradeAnalysisError(f"invalid prediction row: {path}")
            model = str(row.get("model"))
            if model not in expected_models:
                continue
            prediction_key = (
                "corrected_prediction" if "corrected_prediction" in row else "prediction"
            )
            digest_key = f"{prediction_key}_sha256"
            _assert_hash(_rooted(row[prediction_key]), row[digest_key], label=f"{model} prediction")
            observed.add(model)
    else:
        model = str(lock.get("model"))
        if model in expected_models:
            _assert_hash(_rooted(lock["prediction"]), lock["prediction_sha256"], label=model)
            observed.add(model)
    if observed != expected_models:
        raise QualityUpgradeAnalysisError(
            "prediction coverage changed for "
            f"{path}: expected {expected_models}, observed {observed}"
        )
    return lock


def _validate_evidence() -> dict[str, dict[str, Any]]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "quality-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "quality-upgrade registration"),
        (REGISTRATION_MARKER, REGISTRATION_MARKER_SHA256, "registration marker"),
        (UAV_CONVERSION_LOCK, UAV_CONVERSION_LOCK_SHA256, "UAV annotation conversion lock"),
        (UAV_ANNOTATION, UAV_ANNOTATION_SHA256, "UAV validation annotation"),
        (HAZY_ANNOTATION, HAZY_ANNOTATION_SHA256, "HazyDet validation annotation"),
        (HAZY_REFERENCE_LOCK, HAZY_REFERENCE_LOCK_SHA256, "HazyDet reference lock"),
        (HAZY_EWC_AMENDMENT, HAZY_EWC_AMENDMENT_SHA256, "HazyDet inference amendment"),
        (ORDER_REPORT, ORDER_REPORT_SHA256, "effective training-order report"),
    ):
        _assert_hash(path, digest, label=label)
    registration = _load_mapping(REGISTRATION)
    order = _load_mapping(ORDER_REPORT)
    if (
        registration.get("retention_baseline") != "CVBRA_EWC"
        or order.get("status")
        != "COMPLETE_CVBRA_V1_EFFECTIVE_ORDER_ROBUSTNESS_V4_AUDIT"
        or order.get("evidence_boundary", {}).get("independent_confirmation_claim") is not False
    ):
        raise QualityUpgradeAnalysisError("registered analysis scope changed")
    return {
        "uav_primary": _validate_prediction_lock(
            UAV_PRIMARY_LOCK, expected_models={"source", "CVBRA_v1"}
        ),
        "uav_matched": _validate_prediction_lock(
            UAV_MATCHED_LOCK,
            expected_models={"CVBRA_noCV", "CVBRA_noFreeze"},
        ),
        "uav_plainmix": _validate_prediction_lock(
            UAV_PLAINMIX_LOCK, expected_models={"PlainMix"}
        ),
        "uav_ewc": _validate_prediction_lock(UAV_EWC_LOCK, expected_models={"CVBRA_EWC"}),
        "hazy_reference": _validate_prediction_lock(
            HAZY_REFERENCE_LOCK,
            expected_models={"source", "CVBRA_v1", "CVBRA_noCV", "CVBRA_noFreeze"},
        ),
        "hazy_plainmix": _validate_prediction_lock(
            HAZY_PLAINMIX_LOCK, expected_models={"PlainMix"}
        ),
        "hazy_ewc": _validate_prediction_lock(
            HAZY_EWC_LOCK, expected_models={"CVBRA_EWC"}
        ),
    }


def _implementation_lock() -> dict[str, Any]:
    evidence = _validate_evidence()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise QualityUpgradeAnalysisError("analysis implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(RUNNER)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise QualityUpgradeAnalysisError("analysis implementation changed")
        return lock
    if any(
        path.exists()
        for path in (POINTS, FACTORIAL, RETENTION, THRESHOLDS, INFLUENCE, REPORT, COMPLETE)
    ):
        raise QualityUpgradeAnalysisError("derived analysis appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_QUALITY_UPGRADE_ANALYSIS_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(RUNNER),
        "runner_sha256": sha256_file(RUNNER),
        "prediction_locks": {
            name: {"path": _relative(path), "sha256": sha256_file(path)}
            for name, path in (
                ("uav_primary", UAV_PRIMARY_LOCK),
                ("uav_matched", UAV_MATCHED_LOCK),
                ("uav_plainmix", UAV_PLAINMIX_LOCK),
                ("uav_ewc", UAV_EWC_LOCK),
                ("hazy_reference", HAZY_REFERENCE_LOCK),
                ("hazy_plainmix", HAZY_PLAINMIX_LOCK),
                ("hazy_ewc", HAZY_EWC_LOCK),
            )
        },
        "prediction_lock_statuses": {name: value["status"] for name, value in evidence.items()},
        "UAV_annotation_sha256": UAV_ANNOTATION_SHA256,
        "HazyDet_annotation_sha256": HAZY_ANNOTATION_SHA256,
        "effective_order_report_sha256": ORDER_REPORT_SHA256,
        "factorial_resamples": {"UAV": UAV_RESAMPLES, "HazyDet": HAZY_RESAMPLES},
        "seed": SEED,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "validation_labels_previously_accessed": True,
        "method_threshold_checkpoint_or_hyperparameter_reselection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(IMPLEMENTATION_LOCK, payload)
    atomic_write_json(
        IMPLEMENTATION_MARKER,
        {
            "status": payload["status"],
            "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        },
    )
    return payload


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_V1_QUALITY_UPGRADE_ANALYSIS_PREFLIGHT",
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "models": list(MODELS),
        "official_test_access": "prohibited",
    }


def _artifact_prediction(
    lock: Mapping[str, Any],
    *,
    model: str,
    view: str | None = None,
    method: str | None = None,
    raw: bool = False,
) -> Path:
    rows = lock.get("cells" if raw else "artifacts")
    if not isinstance(rows, list):
        if str(lock.get("model")) == model:
            return _rooted(lock["prediction"])
        raise QualityUpgradeAnalysisError(f"prediction rows are missing for {model}")
    matches: list[Path] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("model")) != model:
            continue
        if view is not None and str(row.get("view")) != view:
            continue
        if raw:
            if str(row.get("orientation")) != "identity" or int(row.get("repetition", 0)) != 1:
                continue
        elif method is not None and str(row.get("method")) != method:
            continue
        key = "corrected_prediction" if "corrected_prediction" in row else "prediction"
        matches.append(_rooted(row[key]))
    if len(matches) != 1:
        raise QualityUpgradeAnalysisError(
            f"expected one prediction for {model}/{view}/{method}, observed {len(matches)}"
        )
    return matches[0]


def _uav_prediction_lookup(
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], Path]:
    owners = {
        "source": "uav_primary",
        "CVBRA_v1": "uav_primary",
        "CVBRA_noCV": "uav_matched",
        "CVBRA_noFreeze": "uav_matched",
        "PlainMix": "uav_plainmix",
        "CVBRA_EWC": "uav_ewc",
    }
    return {
        (model, view): _artifact_prediction(
            evidence[owners[model]], model=model, view=view, method="Identity"
        )
        for model in MODELS
        for view in VIEWS
    }


def _hazy_prediction_lookup(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    reference = evidence["hazy_reference"]
    return {
        "source": _artifact_prediction(reference, model="source"),
        "CVBRA_v1": _artifact_prediction(reference, model="CVBRA_v1"),
        "CVBRA_noCV": _artifact_prediction(reference, model="CVBRA_noCV"),
        "CVBRA_noFreeze": _artifact_prediction(reference, model="CVBRA_noFreeze"),
        "PlainMix": _artifact_prediction(evidence["hazy_plainmix"], model="PlainMix"),
        "CVBRA_EWC": _artifact_prediction(evidence["hazy_ewc"], model="CVBRA_EWC"),
    }


def _uav_clusters() -> dict[str, list[int]]:
    document = _load_mapping(UAV_ANNOTATION)
    images = document.get("images")
    if not isinstance(images, list):
        raise QualityUpgradeAnalysisError("UAV annotation image registry is invalid")
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in sorted(
        (row for row in images if isinstance(row, dict)),
        key=lambda item: int(item["id"]),
    ):
        if bool(row.get("primary_eligible")):
            clusters[str(row["source_key"])].append(int(row["id"]))
    flattened = [image_id for group in clusters.values() for image_id in group]
    if len(flattened) != UAV_IMAGES or len(clusters) != UAV_GROUPS:
        raise QualityUpgradeAnalysisError("UAV source-key groups changed")
    return dict(clusters)


def _hazy_clusters() -> dict[str, list[int]]:
    document = _load_mapping(HAZY_ANNOTATION)
    images = document.get("images")
    if not isinstance(images, list):
        raise QualityUpgradeAnalysisError("HazyDet image registry is invalid")
    image_ids = sorted(int(row["id"]) for row in images if isinstance(row, dict))
    if len(image_ids) != HAZY_IMAGES or len(set(image_ids)) != HAZY_IMAGES:
        raise QualityUpgradeAnalysisError("HazyDet image coverage changed")
    return {str(image_id): [image_id] for image_id in image_ids}


def points() -> dict[str, Any]:
    evidence = _validate_evidence()
    _implementation_lock()
    if POINTS.exists():
        report = _load_mapping(POINTS)
        if not POINTS_MARKER.is_file() or _load_mapping(POINTS_MARKER).get(
            "point_estimates_sha256"
        ) != sha256_file(POINTS):
            raise QualityUpgradeAnalysisError("point estimates are not locked")
        return report
    if any(path.exists() for path in (POINTS_MARKER, FACTORIAL, RETENTION, THRESHOLDS, INFLUENCE)):
        raise QualityUpgradeAnalysisError("partial point analysis requires audit")
    uav = _uav_prediction_lookup(evidence)
    hazy = _hazy_prediction_lookup(evidence)
    primary_ids = [image_id for group in _uav_clusters().values() for image_id in group]
    uav_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for view in VIEWS:
            result = evaluate_coco(
                UAV_ANNOTATION,
                uav[(model, view)],
                max_det=MAX_DET,
                image_ids=primary_ids,
            )
            uav_rows.append(
                {
                    "model": model,
                    "view": view,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images": int(result["images_evaluated"]),
                    "prediction_sha256": sha256_file(uav[(model, view)]),
                }
            )
    hazy_rows: list[dict[str, Any]] = []
    for model in MODELS:
        result = evaluate_coco(HAZY_ANNOTATION, hazy[model], max_det=MAX_DET)
        hazy_rows.append(
            {
                "model": model,
                **{key: float(result[key]) for key in METRIC_KEYS},
                "images": int(result["images_evaluated"]),
                "prediction_sha256": sha256_file(hazy[model]),
            }
        )
    if any(int(row["images"]) != UAV_IMAGES for row in uav_rows) or any(
        int(row["images"]) != HAZY_IMAGES for row in hazy_rows
    ):
        raise QualityUpgradeAnalysisError("point-estimate image coverage changed")
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_QUALITY_UPGRADE_POINT_ESTIMATES_LOCKED",
        "completed_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "UAV_OBB_train_disjoint": uav_rows,
        "HazyDet_validation": hazy_rows,
        "method_or_threshold_reselection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(POINTS, report)
    atomic_write_json(
        POINTS_MARKER,
        {"status": report["status"], "point_estimates_sha256": sha256_file(POINTS)},
    )
    return report


def _point_lookup(point_report: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    uav_rows = point_report.get("UAV_OBB_train_disjoint")
    hazy_rows = point_report.get("HazyDet_validation")
    if not isinstance(uav_rows, list) or not isinstance(hazy_rows, list):
        raise QualityUpgradeAnalysisError("point-estimate rows are invalid")
    lookup: dict[tuple[str, str], float] = {}
    for row in uav_rows:
        if isinstance(row, dict):
            lookup[(str(row["model"]), str(row["view"]))] = float(row["AP"])
    for row in hazy_rows:
        if isinstance(row, dict):
            lookup[(str(row["model"]), "HazyDet")] = float(row["AP"])
    expected = len(MODELS) * (len(VIEWS) + 1)
    if len(lookup) != expected:
        raise QualityUpgradeAnalysisError("point-estimate lookup coverage changed")
    return lookup


def _checkpoint_deltas(path: Path, *, resamples: int) -> list[float]:
    checkpoint = _load_mapping(path)
    raw = checkpoint.get("deltas")
    if (
        not isinstance(raw, list)
        or len(raw) != resamples
        or checkpoint.get("completed_resamples") != resamples
    ):
        raise QualityUpgradeAnalysisError(f"bootstrap checkpoint is incomplete: {path}")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise QualityUpgradeAnalysisError(f"bootstrap checkpoint is non-finite: {path}")
    return values


def _paired_result(
    *,
    annotation: Path,
    baseline_prediction: Path,
    method_prediction: Path,
    clusters: Mapping[str, Sequence[int]],
    resamples: int,
    family: str,
    domain: str,
    comparison: str,
) -> tuple[dict[str, float | int], list[float], Path]:
    checkpoint = OUTPUT / "bootstrap" / family / domain / f"{comparison}.json"
    result = paired_coco_ap_cluster_bootstrap_scopes(
        annotation,
        baseline_prediction,
        method_prediction,
        {
            "registered_scope": ClusterBootstrapScope(
                clusters=clusters,
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "study": "cvbra_v1_quality_upgrade_v2",
                    "family": family,
                    "domain": domain,
                    "comparison": comparison,
                    "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                    "method_or_hyperparameter_selection": False,
                },
            )
        },
        resamples=resamples,
        seed=SEED,
        max_det=MAX_DET,
        workers=4,
        chunk_resamples=100,
        accelerate_ap_only=True,
    )["registered_scope"]
    return result, _checkpoint_deltas(checkpoint, resamples=resamples), checkpoint


def factorial() -> dict[str, Any]:
    point_report = points()
    evidence = _validate_evidence()
    if FACTORIAL.exists():
        report = _load_mapping(FACTORIAL)
        if not FACTORIAL_MARKER.is_file() or _load_mapping(FACTORIAL_MARKER).get(
            "factorial_statistics_sha256"
        ) != sha256_file(FACTORIAL):
            raise QualityUpgradeAnalysisError("factorial statistics are not locked")
        return report
    if FACTORIAL_MARKER.exists():
        raise QualityUpgradeAnalysisError("partial factorial statistics require audit")
    points_by_cell = _point_lookup(point_report)
    uav = _uav_prediction_lookup(evidence)
    hazy = _hazy_prediction_lookup(evidence)
    domains: list[
        tuple[str, Path, Mapping[str, Sequence[int]], int, dict[str, Path]]
    ] = []
    uav_clusters = _uav_clusters()
    for view in FACTORIAL_VIEWS:
        domains.append(
            (
                view,
                UAV_ANNOTATION,
                uav_clusters,
                UAV_RESAMPLES,
                {model: uav[(model, view)] for model in FACTORIAL_MODELS},
            )
        )
    domains.append(
        (
            "HazyDet",
            HAZY_ANNOTATION,
            _hazy_clusters(),
            HAZY_RESAMPLES,
            {model: hazy[model] for model in FACTORIAL_MODELS},
        )
    )
    rows: list[dict[str, Any]] = []
    for domain, annotation, clusters, resamples, predictions in domains:
        pair_specs = {
            "visibility_when_frozen": ("CVBRA_noCV", "CVBRA_v1"),
            "visibility_when_unfrozen": ("PlainMix", "CVBRA_noFreeze"),
            "freezing_with_visibility": ("CVBRA_noFreeze", "CVBRA_v1"),
            "freezing_without_visibility": ("PlainMix", "CVBRA_noCV"),
        }
        pair_points: dict[str, float] = {}
        pair_deltas: dict[str, list[float]] = {}
        pair_checkpoints: dict[str, dict[str, str]] = {}
        for name, (baseline, method) in pair_specs.items():
            result, deltas, checkpoint = _paired_result(
                annotation=annotation,
                baseline_prediction=predictions[baseline],
                method_prediction=predictions[method],
                clusters=clusters,
                resamples=resamples,
                family="factorial",
                domain=domain,
                comparison=name,
            )
            direct = points_by_cell[(method, domain)] - points_by_cell[(baseline, domain)]
            if not math.isclose(float(result["delta"]), direct, rel_tol=0.0, abs_tol=1e-10):
                raise QualityUpgradeAnalysisError(
                    f"factorial point estimate drifted: {domain}/{name}"
                )
            pair_points[name] = direct
            pair_deltas[name] = deltas
            pair_checkpoints[name] = {
                "path": _relative(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
        effects = {
            "visibility_main_effect": (
                0.5
                * (
                    pair_points["visibility_when_frozen"]
                    + pair_points["visibility_when_unfrozen"]
                ),
                [
                    0.5 * (left + right)
                    for left, right in zip(
                        pair_deltas["visibility_when_frozen"],
                        pair_deltas["visibility_when_unfrozen"],
                        strict=True,
                    )
                ],
            ),
            "freezing_main_effect": (
                0.5
                * (
                    pair_points["freezing_with_visibility"]
                    + pair_points["freezing_without_visibility"]
                ),
                [
                    0.5 * (left + right)
                    for left, right in zip(
                        pair_deltas["freezing_with_visibility"],
                        pair_deltas["freezing_without_visibility"],
                        strict=True,
                    )
                ],
            ),
            "visibility_by_freezing_interaction": (
                pair_points["freezing_with_visibility"]
                - pair_points["freezing_without_visibility"],
                [
                    left - right
                    for left, right in zip(
                        pair_deltas["freezing_with_visibility"],
                        pair_deltas["freezing_without_visibility"],
                        strict=True,
                    )
                ],
            ),
        }
        for effect, (estimate, distribution) in effects.items():
            standard_deviation = stdev(distribution)
            if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
                raise QualityUpgradeAnalysisError(
                    f"factorial variance is invalid: {domain}/{effect}"
                )
            rows.append(
                {
                    "domain": domain,
                    "effect": effect,
                    "estimate": estimate,
                    "ci_low": float(np.percentile(distribution, 2.5)),
                    "ci_high": float(np.percentile(distribution, 97.5)),
                    "bootstrap_mean": fmean(distribution),
                    "bootstrap_standard_deviation": standard_deviation,
                    "standardized_effect": estimate / standard_deviation,
                    "p_two_sided": bootstrap_sign_pvalue(distribution),
                    "resamples": resamples,
                    "seed": SEED,
                    "unit": "canonical_source_key_group" if domain != "HazyDet" else "image",
                    "component_checkpoints": pair_checkpoints,
                }
            )
        print(json.dumps({"factorial_complete": domain}), flush=True)
    if len(rows) != 9:
        raise QualityUpgradeAnalysisError("factorial effect coverage changed")
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_REGISTERED_TWO_BY_TWO_FACTORIAL_COMPLETE",
        "completed_at_utc": _now(),
        "point_estimates_sha256": sha256_file(POINTS),
        "factors": ["visibility_view_coverage", "early_feature_freezing"],
        "cells": {
            "present_present": "CVBRA_v1",
            "absent_present": "CVBRA_noCV",
            "present_absent": "CVBRA_noFreeze",
            "absent_absent": "PlainMix",
        },
        "multiple_testing": "Holm across all nine registered factorial estimands",
        "rows": rows,
        "method_or_hyperparameter_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(FACTORIAL, report)
    atomic_write_json(
        FACTORIAL_MARKER,
        {"status": report["status"], "factorial_statistics_sha256": sha256_file(FACTORIAL)},
    )
    return report


def retention() -> dict[str, Any]:
    point_report = points()
    evidence = _validate_evidence()
    if RETENTION.exists():
        report = _load_mapping(RETENTION)
        if not RETENTION_MARKER.is_file() or _load_mapping(RETENTION_MARKER).get(
            "retention_statistics_sha256"
        ) != sha256_file(RETENTION):
            raise QualityUpgradeAnalysisError("retention statistics are not locked")
        return report
    if RETENTION_MARKER.exists():
        raise QualityUpgradeAnalysisError("partial retention statistics require audit")
    points_by_cell = _point_lookup(point_report)
    uav = _uav_prediction_lookup(evidence)
    hazy = _hazy_prediction_lookup(evidence)
    uav_clusters = _uav_clusters()
    rows: list[dict[str, Any]] = []
    comparisons = {
        "CVBRA_EWC_minus_CVBRA_noFreeze": "CVBRA_noFreeze",
        "CVBRA_EWC_minus_CVBRA_v1": "CVBRA_v1",
    }
    for domain in (*FACTORIAL_VIEWS, "HazyDet"):
        annotation = HAZY_ANNOTATION if domain == "HazyDet" else UAV_ANNOTATION
        clusters: Mapping[str, Sequence[int]] = (
            _hazy_clusters() if domain == "HazyDet" else uav_clusters
        )
        resamples = HAZY_RESAMPLES if domain == "HazyDet" else UAV_RESAMPLES
        predictions = hazy if domain == "HazyDet" else {
            model: uav[(model, domain)] for model in MODELS
        }
        for name, baseline in comparisons.items():
            result, deltas, checkpoint = _paired_result(
                annotation=annotation,
                baseline_prediction=predictions[baseline],
                method_prediction=predictions["CVBRA_EWC"],
                clusters=clusters,
                resamples=resamples,
                family="retention",
                domain=domain,
                comparison=name,
            )
            direct = points_by_cell[("CVBRA_EWC", domain)] - points_by_cell[(baseline, domain)]
            if not math.isclose(float(result["delta"]), direct, rel_tol=0.0, abs_tol=1e-10):
                raise QualityUpgradeAnalysisError(
                    f"retention point estimate drifted: {domain}/{name}"
                )
            standard_deviation = stdev(deltas)
            rows.append(
                {
                    "domain": domain,
                    "comparison": name,
                    "baseline": baseline,
                    "method": "CVBRA_EWC",
                    "estimate": direct,
                    "ci_low": float(result["ci_low"]),
                    "ci_high": float(result["ci_high"]),
                    "bootstrap_mean": fmean(deltas),
                    "bootstrap_standard_deviation": standard_deviation,
                    "standardized_effect": direct / standard_deviation,
                    "resamples": resamples,
                    "seed": SEED,
                    "checkpoint": _relative(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
        print(json.dumps({"retention_complete": domain}), flush=True)
    report = {
        "schema_version": 1,
        "status": "CVBRA_EWC_DESCRIPTIVE_RETENTION_COMPARISONS_COMPLETE",
        "completed_at_utc": _now(),
        "point_estimates_sha256": sha256_file(POINTS),
        "inference_role": "descriptive_matched_standard_retention_control",
        "multiplicity_adjusted_significance_claim": False,
        "rows": rows,
        "method_or_hyperparameter_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(RETENTION, report)
    atomic_write_json(
        RETENTION_MARKER,
        {"status": report["status"], "retention_statistics_sha256": sha256_file(RETENTION)},
    )
    return report


def _validate_raw_lock(path: Path, *, expected_models: set[str]) -> dict[str, Any]:
    lock = _load_mapping(path)
    marker_path = path.parent / "RAW_OBSERVATIONS_LOCKED"
    marker = _load_mapping(marker_path)
    if marker.get("raw_observation_lock_sha256") != sha256_file(path):
        raise QualityUpgradeAnalysisError(f"raw-observation marker changed: {path}")
    cells = lock.get("cells")
    if not isinstance(cells, list):
        raise QualityUpgradeAnalysisError(f"raw-observation cells are invalid: {path}")
    observed: set[str] = set()
    for row in cells:
        if not isinstance(row, dict):
            raise QualityUpgradeAnalysisError(f"raw-observation row is invalid: {path}")
        model = str(row.get("model"))
        if model not in expected_models:
            continue
        _assert_hash(_rooted(row["prediction"]), row["prediction_sha256"], label=f"{model} raw")
        observed.add(model)
    if observed != expected_models:
        raise QualityUpgradeAnalysisError(f"raw-observation model coverage changed: {path}")
    return lock


def _raw_uav_lookup() -> dict[tuple[str, str], Path]:
    locks = {
        "primary": _validate_raw_lock(
            UAV_RAW_PRIMARY_LOCK, expected_models={"source", "CVBRA_v1"}
        ),
        "matched": _validate_raw_lock(
            UAV_RAW_MATCHED_LOCK, expected_models={"CVBRA_noCV", "CVBRA_noFreeze"}
        ),
        "plainmix": _validate_raw_lock(
            UAV_RAW_PLAINMIX_LOCK, expected_models={"PlainMix"}
        ),
    }
    owners = {
        "source": "primary",
        "CVBRA_v1": "primary",
        "CVBRA_noCV": "matched",
        "CVBRA_noFreeze": "matched",
        "PlainMix": "plainmix",
    }
    return {
        (model, view): _artifact_prediction(
            locks[owners[model]],
            model=model,
            view=view,
            raw=True,
        )
        for model in owners
        for view in FACTORIAL_VIEWS
    }


def thresholds() -> dict[str, Any]:
    point_report = points()
    if THRESHOLDS.exists():
        report = _load_mapping(THRESHOLDS)
        if not THRESHOLDS_MARKER.is_file() or _load_mapping(THRESHOLDS_MARKER).get(
            "threshold_robustness_sha256"
        ) != sha256_file(THRESHOLDS):
            raise QualityUpgradeAnalysisError("threshold robustness is not locked")
        return report
    if THRESHOLDS_MARKER.exists():
        raise QualityUpgradeAnalysisError("partial threshold robustness requires audit")
    raw = _raw_uav_lookup()
    primary_ids = [image_id for group in _uav_clusters().values() for image_id in group]
    rows: list[dict[str, Any]] = []
    threshold_models = MODELS[:-1]
    for model in threshold_models:
        for view in FACTORIAL_VIEWS:
            raw_rows = _load_rows(raw[(model, view)])
            for threshold in THRESHOLD_VALUES:
                filtered = [row for row in raw_rows if float(row["score"]) >= threshold]
                prediction = (
                    OUTPUT
                    / "threshold_predictions"
                    / model
                    / view
                    / f"conf_{threshold:.2f}.coco.json"
                )
                if prediction.exists():
                    existing = _load_rows(prediction)
                    if existing != filtered:
                        raise QualityUpgradeAnalysisError(
                            f"threshold prediction changed: {prediction}"
                        )
                else:
                    atomic_write_json(prediction, filtered)
                result = evaluate_coco(
                    UAV_ANNOTATION,
                    prediction,
                    max_det=MAX_DET,
                    image_ids=primary_ids,
                )
                rows.append(
                    {
                        "model": model,
                        "view": view,
                        "confidence_threshold": threshold,
                        "AP": float(result["AP"]),
                        "AP50": float(result["AP50"]),
                        "AP75": float(result["AP75"]),
                        "detections": len(filtered),
                        "prediction": _relative(prediction),
                        "prediction_sha256": sha256_file(prediction),
                    }
                )
    if len(rows) != len(threshold_models) * len(FACTORIAL_VIEWS) * len(THRESHOLD_VALUES):
        raise QualityUpgradeAnalysisError("threshold-sweep coverage changed")
    lookup = {
        (str(row["model"]), str(row["view"]), float(row["confidence_threshold"])): row
        for row in rows
    }
    point_lookup = _point_lookup(point_report)
    for model in threshold_models:
        for view in FACTORIAL_VIEWS:
            observed = float(lookup[(model, view, 0.25)]["AP"])
            expected = point_lookup[(model, view)]
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                raise QualityUpgradeAnalysisError(f"threshold 0.25 anchor drifted: {model}/{view}")
    deltas: list[dict[str, Any]] = [
        {
            "view": view,
            "confidence_threshold": threshold,
            "CVBRA_v1_minus_source_AP": float(lookup[("CVBRA_v1", view, threshold)]["AP"])
            - float(lookup[("source", view, threshold)]["AP"]),
        }
        for view in FACTORIAL_VIEWS
        for threshold in THRESHOLD_VALUES
    ]
    all_positive = all(float(row["CVBRA_v1_minus_source_AP"]) > 0.0 for row in deltas)
    if not all_positive:
        raise QualityUpgradeAnalysisError("registered threshold-robustness check failed")
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_REGISTERED_THRESHOLD_ROBUSTNESS_COMPLETE",
        "completed_at_utc": _now(),
        "point_estimates_sha256": sha256_file(POINTS),
        "models": list(threshold_models),
        "views": list(FACTORIAL_VIEWS),
        "thresholds": list(THRESHOLD_VALUES),
        "rows": rows,
        "CVBRA_v1_minus_source": deltas,
        "positive_at_every_registered_threshold_and_view": all_positive,
        "minimum_CVBRA_v1_minus_source_AP": min(
            float(row["CVBRA_v1_minus_source_AP"]) for row in deltas
        ),
        "threshold_or_model_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(THRESHOLDS, report)
    atomic_write_json(
        THRESHOLDS_MARKER,
        {"status": report["status"], "threshold_robustness_sha256": sha256_file(THRESHOLDS)},
    )
    return report


def _accelerated_ap_cache(
    annotation: Path,
    prediction: Path,
    image_ids: Sequence[int],
) -> Any:
    ground_truth = bootstrap_engine._prepare_ground_truth(_load_mapping(annotation))
    prepared_prediction = bootstrap_engine._prepare_predictions(_load_rows(prediction))
    evaluation_cache = bootstrap_engine._build_evaluation_cache(
        ground_truth,
        prepared_prediction,
        image_ids,
        max_det=MAX_DET,
    )
    return bootstrap_engine._prepare_accelerated_coco_ap_only_cache(evaluation_cache)


def influence() -> dict[str, Any]:
    point_report = points()
    evidence = _validate_evidence()
    if INFLUENCE.exists():
        report = _load_mapping(INFLUENCE)
        if not INFLUENCE_MARKER.is_file() or _load_mapping(INFLUENCE_MARKER).get(
            "group_influence_sha256"
        ) != sha256_file(INFLUENCE):
            raise QualityUpgradeAnalysisError("group influence is not locked")
        return report
    if INFLUENCE_MARKER.exists():
        raise QualityUpgradeAnalysisError("partial group influence requires audit")
    predictions = _uav_prediction_lookup(evidence)
    clusters = _uav_clusters()
    image_ids = [image_id for group in clusters.values() for image_id in group]
    points_by_cell = _point_lookup(point_report)
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for view in FACTORIAL_VIEWS:
        source_cache = _accelerated_ap_cache(
            UAV_ANNOTATION,
            predictions[("source", view)],
            image_ids,
        )
        method_cache = _accelerated_ap_cache(
            UAV_ANNOTATION,
            predictions[("CVBRA_v1", view)],
            image_ids,
        )
        full_source = bootstrap_engine._coco_ap_from_accelerated_cache(source_cache, image_ids)
        full_method = bootstrap_engine._coco_ap_from_accelerated_cache(method_cache, image_ids)
        full_delta = full_method - full_source
        expected = points_by_cell[("CVBRA_v1", view)] - points_by_cell[("source", view)]
        if not math.isclose(full_delta, expected, rel_tol=0.0, abs_tol=1e-12):
            raise QualityUpgradeAnalysisError(f"influence point anchor drifted: {view}")
        rows: list[dict[str, Any]] = []
        for group, omitted_ids in clusters.items():
            omitted = set(omitted_ids)
            retained = [image_id for image_id in image_ids if image_id not in omitted]
            source_ap = bootstrap_engine._coco_ap_from_accelerated_cache(source_cache, retained)
            method_ap = bootstrap_engine._coco_ap_from_accelerated_cache(method_cache, retained)
            delta = method_ap - source_ap
            rows.append(
                {
                    "view": view,
                    "omitted_group": group,
                    "omitted_image_ids": omitted_ids,
                    "retained_images": len(retained),
                    "source_AP": source_ap,
                    "CVBRA_v1_AP": method_ap,
                    "delta": delta,
                    "shift_from_full_delta": delta - full_delta,
                }
            )
        most_influential = max(rows, key=lambda row: abs(float(row["shift_from_full_delta"])))
        summaries.append(
            {
                "view": view,
                "full_delta": full_delta,
                "minimum_leave_one_group_out_delta": min(float(row["delta"]) for row in rows),
                "maximum_leave_one_group_out_delta": max(float(row["delta"]) for row in rows),
                "maximum_absolute_shift_from_full_delta": abs(
                    float(most_influential["shift_from_full_delta"])
                ),
                "proportion_positive": fmean(float(row["delta"]) > 0.0 for row in rows),
                "most_influential_group": most_influential["omitted_group"],
                "most_influential_group_images": most_influential["omitted_image_ids"],
                "most_influential_group_delta": most_influential["delta"],
            }
        )
        all_rows.extend(rows)
    if any(float(row["proportion_positive"]) != 1.0 for row in summaries):
        raise QualityUpgradeAnalysisError("leave-one-group-out positivity check failed")
    report = {
        "schema_version": 1,
        "status": "CVBRA_V1_REGISTERED_GROUP_INFLUENCE_COMPLETE",
        "completed_at_utc": _now(),
        "point_estimates_sha256": sha256_file(POINTS),
        "comparison": "CVBRA_v1_minus_source",
        "groups": UAV_GROUPS,
        "operation": "leave_one_canonical_source_key_group_out",
        "summaries": summaries,
        "rows": all_rows,
        "method_or_hyperparameter_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(INFLUENCE, report)
    atomic_write_json(
        INFLUENCE_MARKER,
        {"status": report["status"], "group_influence_sha256": sha256_file(INFLUENCE)},
    )
    return report


def finalize() -> dict[str, Any]:
    point_report = points()
    factorial_report = factorial()
    retention_report = retention()
    threshold_report = thresholds()
    influence_report = influence()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        if not COMPLETE.is_file() or _load_mapping(COMPLETE).get(
            "quality_upgrade_report_sha256"
        ) != sha256_file(REPORT):
            raise QualityUpgradeAnalysisError("quality-upgrade report is not locked")
        return report
    order_report = _load_mapping(ORDER_REPORT)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_JEI_QUALITY_UPGRADE_EVIDENCE",
        "completed_at_utc": _now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "artifacts": {
            "point_estimates": {"path": _relative(POINTS), "sha256": sha256_file(POINTS)},
            "factorial_statistics": {
                "path": _relative(FACTORIAL),
                "sha256": sha256_file(FACTORIAL),
            },
            "EWC_retention_statistics": {
                "path": _relative(RETENTION),
                "sha256": sha256_file(RETENTION),
            },
            "threshold_robustness": {
                "path": _relative(THRESHOLDS),
                "sha256": sha256_file(THRESHOLDS),
            },
            "group_influence": {
                "path": _relative(INFLUENCE),
                "sha256": sha256_file(INFLUENCE),
            },
            "effective_training_order": {
                "path": _relative(ORDER_REPORT),
                "sha256": ORDER_REPORT_SHA256,
            },
        },
        "point_estimates": point_report,
        "factorial_rows": factorial_report["rows"],
        "EWC_retention_rows": retention_report["rows"],
        "threshold_summary": {
            "positive_at_every_registered_threshold_and_view": threshold_report[
                "positive_at_every_registered_threshold_and_view"
            ],
            "minimum_CVBRA_v1_minus_source_AP": threshold_report[
                "minimum_CVBRA_v1_minus_source_AP"
            ],
        },
        "group_influence_summaries": influence_report["summaries"],
        "effective_order_variability": order_report["metric_summary"]["variability"],
        "state_diversity": order_report["state_diversity"],
        "evidence_boundary": {
            "post_freeze_descriptive_quality_and_mechanism_evidence": True,
            "validation_labels_previously_accessed": True,
            "method_checkpoint_threshold_or_hyperparameter_reselection": False,
            "independent_seed_replication_claim": False,
            "effective_training_sequence_endpoints": 3,
            "HazyDet_test_access": "prohibited",
            "UAV_OBB_test_access": "prohibited",
        },
        "paper_result_integration_authorized": True,
    }
    atomic_write_json(REPORT, report)
    atomic_write_json(
        COMPLETE,
        {
            "status": report["status"],
            "quality_upgrade_report_sha256": sha256_file(REPORT),
            "factorial_statistics_sha256": sha256_file(FACTORIAL),
            "retention_statistics_sha256": sha256_file(RETENTION),
            "threshold_robustness_sha256": sha256_file(THRESHOLDS),
            "group_influence_sha256": sha256_file(INFLUENCE),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze registered JEI quality-upgrade evidence")
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "points",
            "factorial",
            "retention",
            "thresholds",
            "influence",
            "final",
        ),
        default="final",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        artifact = IMPLEMENTATION_LOCK
    elif args.stage == "points":
        result = points()
        artifact = POINTS
    elif args.stage == "factorial":
        result = factorial()
        artifact = FACTORIAL
    elif args.stage == "retention":
        result = retention()
        artifact = RETENTION
    elif args.stage == "thresholds":
        result = thresholds()
        artifact = THRESHOLDS
    elif args.stage == "influence":
        result = influence()
        artifact = INFLUENCE
    else:
        result = finalize()
        artifact = REPORT
    print(
        json.dumps(
            {"status": result["status"], "artifact_sha256": sha256_file(artifact)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
