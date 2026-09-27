from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, cast

import yaml
from scripts.run_cvbra_v1_hazydet_source_retention_v4 import recenter_bootstrap_deltas

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.coco import evaluate_coco, evaluate_coco_per_class
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.pipeline.detection import infer_detector
from buse_uav.schemas import AppConfig
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_dronevehicle_validation_protocol.yaml"
OUTPUT = ROOT / "reports" / "external" / "cvbra_v1_dronevehicle_validation"
PARENT_RUN = ROOT / "runs" / "dronevehicle_baseline_107eaabc808d"
PARENT_CONFIG = PARENT_RUN / "config_resolved.yaml"
PARENT_MANIFEST = PARENT_RUN / "data_manifest.json"
SOURCE_PREDICTION = PARENT_RUN / "predictions" / "final.coco.json"
SOURCE_TIMING = PARENT_RUN / "traces" / "timings.jsonl"
ANNOTATION = (
    ROOT
    / "data"
    / "processed"
    / "dronevehicle_rgb_val"
    / "annotations"
    / "dronevehicle_rgb_val.coco.json"
)
TEST_GUARD_COMPLETE = (
    ROOT
    / "reports"
    / "external"
    / "dronevehicle_test_guard"
    / "analysis"
    / "ANALYSIS_COMPLETE.json"
)
TEST_GUARD_REPORT = TEST_GUARD_COMPLETE.with_name("final_report.json")

PARENT_CONFIG_SHA256 = "07db30a028fd6d20d75220ad14cdbe0d770428d90c8df4da377349f46c6b3b63"
PARENT_MANIFEST_SHA256 = "c8eeab04c5a56f0ea02dc84a6a85a543d1880f6a483715659e863e1ee1a96ccd"
SOURCE_PREDICTION_SHA256 = "5235d3300b449e7d946f7fcd3c9f4dd5199ae409476eda0dcf09e3c2c514d7d1"
SOURCE_TIMING_SHA256 = "546e88606e735fbdb4eab01d6399136da0358519837090c6d01482edeb2b4be9"
ANNOTATION_SHA256 = "cb1d6dd992915beb76c87e2a0b4b6a85f5c69a7c3584d4a1fb35a0d8b59759b9"
TEST_GUARD_COMPLETE_SHA256 = (
    "d77a558c6a63b61f71981f3f293b7ec2a4b9dff7cf7c2d1ad0aaf8c66e5d9700"
)
TEST_GUARD_REPORT_SHA256 = "d761b4f3f4d79116183e096a28571dd3f3a2ee659f78f25182af407ab46f8908"
HISTORICAL_SOURCE_AP = 0.25292349701906747

MODELS = ("source", "CVBRA_v1", "STF")
NEW_MODELS = MODELS[1:]
WEIGHTS: dict[str, tuple[Path, str]] = {
    "source": (
        ROOT / "weights" / "hazydet" / "yolo11n_best.pt",
        "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430",
    ),
    "CVBRA_v1": (
        ROOT / "runs" / "cvbra_v1" / "yolo11n" / "cvbra_v1.pt",
        "d44f0926696e93b5f2e0ec5c9201f1e6360c43d4d40fccc68646e1ced633bd42",
    ),
    "STF": (
        ROOT / "runs" / "cvbra_v1_matched_baselines" / "STF" / "STF.pt",
        "90556b62756d0b1a944c42a7cb0e5e86935e2684058c88621ed92d7bfd78b6d2",
    ),
}
RUN_PATHS = {
    model: ROOT / "runs" / f"cvbra_v1_dronevehicle_val_{model.lower()}"
    for model in NEW_MODELS
}

REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTRATION_MARKER = OUTPUT / "REGISTERED"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
ANALYSIS_LOCK = OUTPUT / "analysis_authorization.json"
ANALYSIS_MARKER = OUTPUT / "ANALYSIS_AUTHORIZED"
METRICS = OUTPUT / "metrics.csv"
PER_CLASS = OUTPUT / "per_class.csv"
STATISTICS = OUTPUT / "paired_statistics.json"
REPORT = OUTPUT / "validation_report.json"
COMPLETE = OUTPUT / "DRONEVEHICLE_VALIDATION_COMPLETE"

IMAGES = 1469
MAX_DET = 500
RESAMPLES = 2000
SEED = 20260814
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class CvbraDroneVehicleError(RuntimeError):
    """Raised when validation-only DroneVehicle execution cannot fail closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraDroneVehicleError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CvbraDroneVehicleError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraDroneVehicleError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CvbraDroneVehicleError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise CvbraDroneVehicleError(f"locked {label} changed: {path}")


def ordered_image_ids() -> tuple[int, ...]:
    annotation = _load_mapping(ANNOTATION)
    raw_images = annotation.get("images")
    raw_categories = annotation.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise CvbraDroneVehicleError("DroneVehicle validation annotation changed")
    image_ids = [
        int(row["id"])
        for row in raw_images
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    category_ids = [
        int(row["id"])
        for row in raw_categories
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES or category_ids != [1, 2, 3]:
        raise CvbraDroneVehicleError("DroneVehicle validation image/category scope changed")
    return tuple(sorted(image_ids, key=str))


def singleton_image_clusters(image_ids: Sequence[int]) -> dict[str, tuple[int]]:
    if len(image_ids) != IMAGES or len(set(image_ids)) != IMAGES:
        raise CvbraDroneVehicleError("DroneVehicle image bootstrap scope changed")
    return {f"image_{index:04d}": (image_id,) for index, image_id in enumerate(image_ids)}


def locked_raw_config(model: str) -> dict[str, Any]:
    if model not in NEW_MODELS:
        raise CvbraDroneVehicleError(f"unsupported new model: {model}")
    try:
        loaded = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CvbraDroneVehicleError(f"cannot load parent config: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CvbraDroneVehicleError("parent DroneVehicle config is not a mapping")
    raw = cast(dict[str, Any], loaded)
    detector = raw.get("detector")
    experiment = raw.get("experiment")
    dataset = raw.get("dataset")
    if (
        not isinstance(detector, dict)
        or not isinstance(experiment, dict)
        or not isinstance(dataset, dict)
    ):
        raise CvbraDroneVehicleError("parent DroneVehicle config sections changed")
    if dataset.get("split") != "external_rgb_val" or "test" in str(dataset).lower():
        raise CvbraDroneVehicleError("DroneVehicle runner is not validation-only")
    detector["model"] = _relative(WEIGHTS[model][0])
    experiment["name"] = f"cvbra_v1_dronevehicle_val_{model.lower()}"
    return raw


def _render_config(model: str) -> str:
    return yaml.safe_dump(locked_raw_config(model), sort_keys=False, allow_unicode=True)


def _validate_parent() -> None:
    for path, digest, label in (
        (PARENT_CONFIG, PARENT_CONFIG_SHA256, "parent validation config"),
        (PARENT_MANIFEST, PARENT_MANIFEST_SHA256, "parent validation manifest"),
        (SOURCE_PREDICTION, SOURCE_PREDICTION_SHA256, "source validation prediction"),
        (SOURCE_TIMING, SOURCE_TIMING_SHA256, "source validation timing"),
        (ANNOTATION, ANNOTATION_SHA256, "validation annotation"),
        (TEST_GUARD_COMPLETE, TEST_GUARD_COMPLETE_SHA256, "historical test completion"),
        (TEST_GUARD_REPORT, TEST_GUARD_REPORT_SHA256, "historical test report"),
    ):
        _assert_hash(path, digest, label=label)
    for model, (path, digest) in WEIGHTS.items():
        _assert_hash(path, digest, label=f"{model} checkpoint")
    test_complete = _load_mapping(TEST_GUARD_COMPLETE)
    if test_complete.get("status") != "ANALYSIS_COMPLETE":
        raise CvbraDroneVehicleError("historical DroneVehicle test consumption state changed")
    manifest = _load_mapping(PARENT_MANIFEST)
    manifest_images = manifest.get("images")
    if (
        manifest.get("dataset") != "dronevehicle"
        or manifest.get("split") != "external_rgb_val"
        or "test" in json.dumps(manifest, ensure_ascii=False).lower()
        or not isinstance(manifest_images, list)
        or len(manifest_images) != IMAGES
    ):
        raise CvbraDroneVehicleError("DroneVehicle parent manifest is not validation-only")
    manifest_ids = tuple(
        sorted(
            (
                int(row["image_id"])
                for row in manifest_images
                if isinstance(row, dict) and isinstance(row.get("image_id"), int)
            ),
            key=str,
        )
    )
    if manifest_ids != ordered_image_ids():
        raise CvbraDroneVehicleError("DroneVehicle validation image IDs changed")
    source_rows = _load_rows(SOURCE_PREDICTION)
    valid_ids = set(manifest_ids)
    if not source_rows or any(
        int(row.get("category_id", -1)) not in {1, 2, 3}
        or int(row.get("image_id", -1)) not in valid_ids
        for row in source_rows
    ):
        raise CvbraDroneVehicleError("source validation prediction ontology changed")


def _registration() -> dict[str, Any]:
    _validate_parent()
    protocol_sha256 = sha256_file(PROTOCOL)
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
            raise CvbraDroneVehicleError("DroneVehicle registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTRATION_MARKER)
        if (
            lock.get("protocol_sha256") != protocol_sha256
            or lock.get("new_prediction_or_metric_before_registration") is not False
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise CvbraDroneVehicleError("DroneVehicle registration changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise CvbraDroneVehicleError("DroneVehicle output appeared before registration")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_DRONEVEHICLE_VALIDATION_REGISTERED_BEFORE_NEW_PREDICTIONS",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "models": list(MODELS),
        "new_prediction_or_metric_before_registration": False,
        "AU_AIR_results_observed_before_registration": True,
        "adaptive_supplementary_external_validation": True,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "DroneVehicle_test_consumed_historically": True,
        "new_test_access": "prohibited",
    }
    atomic_write_json(REGISTRATION, lock)
    atomic_write_json(
        REGISTRATION_MARKER,
        {"status": lock["status"], "registration_sha256": sha256_file(REGISTRATION)},
    )
    return lock


def _locked_config_path(model: str) -> Path:
    return OUTPUT / "locked_configs" / f"{model}.yaml"


def _implementation_lock() -> dict[str, Any]:
    registration = _registration()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise CvbraDroneVehicleError("DroneVehicle implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != sha256_file(PROTOCOL)
            or lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CvbraDroneVehicleError("DroneVehicle implementation lock changed")
        observed_hashes = lock.get("locked_config_sha256")
        if not isinstance(observed_hashes, dict):
            raise CvbraDroneVehicleError("DroneVehicle config lock changed")
        for model in NEW_MODELS:
            path = _locked_config_path(model)
            _assert_hash(path, str(observed_hashes[model]), label=f"{model} config")
            if path.read_text(encoding="utf-8") != _render_config(model):
                raise CvbraDroneVehicleError(f"{model} rendered config changed")
        return lock
    if any(RUN_PATHS[model].exists() for model in NEW_MODELS):
        raise CvbraDroneVehicleError("DroneVehicle run appeared before implementation lock")
    config_hashes: dict[str, str] = {}
    for model in NEW_MODELS:
        path = _locked_config_path(model)
        atomic_write_text(path, _render_config(model))
        config_hashes[model] = sha256_file(path)
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_DRONEVEHICLE_VALIDATION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "source_prediction_sha256": SOURCE_PREDICTION_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "test_guard_complete_sha256": TEST_GUARD_COMPLETE_SHA256,
        "locked_config_sha256": config_hashes,
        "models": list(MODELS),
        "images": IMAGES,
        "split": "external_rgb_val",
        "python": platform.python_version(),
        "new_test_access": "prohibited",
        "method_or_hyperparameter_selection": False,
        "registration_status": registration["status"],
    }
    atomic_write_json(IMPLEMENTATION_LOCK, lock)
    atomic_write_json(
        IMPLEMENTATION_MARKER,
        {
            "status": lock["status"],
            "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        },
    )
    return lock


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    return {
        "status": "PASS_CVBRA_V1_DRONEVEHICLE_VALIDATION_PREFLIGHT",
        "models": list(MODELS),
        "new_inference_cells": len(NEW_MODELS),
        "images": IMAGES,
        "split": "external_rgb_val",
        "new_test_access": "prohibited",
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
    }


def _validate_run(model: str) -> dict[str, Any]:
    run_path = RUN_PATHS[model]
    required = (
        run_path / "SUCCESS",
        run_path / "config_resolved.yaml",
        run_path / "data_manifest.json",
        run_path / "model_fingerprint.json",
        run_path / "predictions" / "final.coco.json",
        run_path / "traces" / "timings.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing or (run_path / "ERROR").exists():
        raise CvbraDroneVehicleError(f"incomplete validation run for {model}: {missing}")
    if sha256_file(run_path / "config_resolved.yaml") != sha256_file(_locked_config_path(model)):
        raise CvbraDroneVehicleError(f"validation config drifted for {model}")
    if sha256_file(run_path / "data_manifest.json") != PARENT_MANIFEST_SHA256:
        raise CvbraDroneVehicleError(f"validation manifest drifted for {model}")
    fingerprint = _load_mapping(run_path / "model_fingerprint.json")
    if fingerprint.get("sha256") != WEIGHTS[model][1]:
        raise CvbraDroneVehicleError(f"validation checkpoint drifted for {model}")
    prediction = run_path / "predictions" / "final.coco.json"
    rows = _load_rows(prediction)
    valid_ids = set(ordered_image_ids())
    if not rows or any(
        int(row.get("category_id", -1)) not in {1, 2, 3}
        or int(row.get("image_id", -1)) not in valid_ids
        for row in rows
    ):
        raise CvbraDroneVehicleError(f"validation prediction ontology drifted for {model}")
    forbidden = list(run_path.glob("metrics*.json")) + list(run_path.glob("metrics*.csv"))
    if forbidden:
        raise CvbraDroneVehicleError(f"metrics appeared before joint lock: {forbidden}")
    return {
        "model": model,
        "run": _relative(run_path),
        "checkpoint_sha256": WEIGHTS[model][1],
        "config_sha256": sha256_file(run_path / "config_resolved.yaml"),
        "data_manifest_sha256": sha256_file(run_path / "data_manifest.json"),
        "prediction": _relative(prediction),
        "prediction_sha256": sha256_file(prediction),
        "timing": _relative(run_path / "traces" / "timings.jsonl"),
        "timing_sha256": sha256_file(run_path / "traces" / "timings.jsonl"),
        "status": "SUCCESS",
    }


def _validate_prediction_lock() -> dict[str, Any]:
    _implementation_lock()
    if not PREDICTION_LOCK.is_file() or not PREDICTION_MARKER.is_file():
        raise CvbraDroneVehicleError("validation prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "CVBRA_V1_DRONEVEHICLE_VALIDATION_PREDICTIONS_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or lock.get("models") != list(MODELS)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
    ):
        raise CvbraDroneVehicleError("validation prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CvbraDroneVehicleError("validation prediction artifact changed")
        _assert_hash(
            ROOT / str(row["prediction"]),
            str(row["prediction_sha256"]),
            label=f"{row['model']} prediction",
        )
        _assert_hash(
            ROOT / str(row["timing"]),
            str(row["timing_sha256"]),
            label=f"{row['model']} timing",
        )
    return lock


def infer() -> dict[str, Any]:
    _implementation_lock()
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        return _validate_prediction_lock()
    artifacts: list[dict[str, Any]] = [
        {
            "model": "source",
            "run": _relative(PARENT_RUN),
            "checkpoint_sha256": WEIGHTS["source"][1],
            "config_sha256": PARENT_CONFIG_SHA256,
            "data_manifest_sha256": PARENT_MANIFEST_SHA256,
            "prediction": _relative(SOURCE_PREDICTION),
            "prediction_sha256": SOURCE_PREDICTION_SHA256,
            "timing": _relative(SOURCE_TIMING),
            "timing_sha256": SOURCE_TIMING_SHA256,
            "status": "REUSED_LOCKED_SOURCE_ANCHOR",
        }
    ]
    for model in NEW_MODELS:
        run_path = RUN_PATHS[model]
        if not (run_path / "SUCCESS").is_file():
            raw = locked_raw_config(model)
            config = AppConfig.model_validate(raw)
            infer_detector(
                config,
                raw,
                command=[sys.executable, *sys.argv, f"model={model}"],
                resume_run=run_path if run_path.exists() else None,
                run_id=None if run_path.exists() else run_path.name,
                warmup_images=16,
            )
        artifacts.append(_validate_run(model))
        print(json.dumps({"inference_complete": model}), flush=True)
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_DRONEVEHICLE_VALIDATION_PREDICTIONS_LOCKED",
        "locked_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "models": list(MODELS),
        "images": IMAGES,
        "split": "external_rgb_val",
        "artifacts": artifacts,
        "source_anchor_reused": True,
        "new_metric_accessed": False,
        "DroneVehicle_test_consumed_historically": True,
        "new_test_access": "prohibited",
        "method_or_hyperparameter_selection": False,
    }
    atomic_write_json(PREDICTION_LOCK, lock)
    atomic_write_json(
        PREDICTION_MARKER,
        {"status": lock["status"], "prediction_lock_sha256": sha256_file(PREDICTION_LOCK)},
    )
    return lock


def _analysis_authorization() -> dict[str, Any]:
    infer()
    if ANALYSIS_LOCK.exists() or ANALYSIS_MARKER.exists():
        if not ANALYSIS_LOCK.is_file() or not ANALYSIS_MARKER.is_file():
            raise CvbraDroneVehicleError("validation analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise CvbraDroneVehicleError("validation analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, PER_CLASS, STATISTICS, REPORT)):
        raise CvbraDroneVehicleError("validation metrics appeared before authorization")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_DRONEVEHICLE_VALIDATION_ANALYSIS_AUTHORIZED",
        "authorized_at_utc": _now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "split": "external_rgb_val",
        "new_metric_accessed_before_authorization": False,
        "new_test_access": "prohibited",
        "method_or_hyperparameter_selection": False,
    }
    atomic_write_json(ANALYSIS_LOCK, lock)
    atomic_write_json(
        ANALYSIS_MARKER,
        {"status": lock["status"], "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK)},
    )
    return lock


def _prediction_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise CvbraDroneVehicleError("validation prediction artifacts changed")
    paths = {
        str(row["model"]): ROOT / str(row["prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }
    if tuple(paths) != MODELS:
        raise CvbraDroneVehicleError("validation model coverage changed")
    return paths


def _mean_timing_ms(path: Path) -> float:
    values: list[float] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("total_ms"), (int, float)):
                raise CvbraDroneVehicleError(f"invalid timing row: {path}")
            values.append(float(row["total_ms"]))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraDroneVehicleError(f"cannot parse timing {path}: {exc}") from exc
    if len(values) != IMAGES or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise CvbraDroneVehicleError(f"validation timing coverage changed: {path}")
    return fmean(values)


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
    return buffer.getvalue()


def _checkpoint_deltas(path: Path) -> list[float]:
    values = _load_mapping(path).get("deltas")
    if not isinstance(values, list) or len(values) != RESAMPLES:
        raise CvbraDroneVehicleError(f"bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def score() -> dict[str, Any]:
    prediction_lock = infer()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise CvbraDroneVehicleError("existing validation report is not locked")
        return report
    paths = _prediction_paths(prediction_lock)
    artifacts = prediction_lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise CvbraDroneVehicleError("validation metadata changed")
    timing_paths = {
        str(row["model"]): ROOT / str(row["timing"])
        for row in artifacts
        if isinstance(row, dict)
    }
    image_ids = ordered_image_ids()
    clusters = singleton_image_clusters(image_ids)
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = evaluate_coco(ANNOTATION, paths[model], max_det=MAX_DET, image_ids=image_ids)
        if int(metrics["images_evaluated"]) != IMAGES:
            raise CvbraDroneVehicleError(f"{model} metrics omitted validation images")
        metric_rows.append(
            {
                "model": model,
                **{key: float(metrics[key]) for key in METRIC_KEYS},
                "images_evaluated": int(metrics["images_evaluated"]),
                "mean_ms": _mean_timing_ms(timing_paths[model]),
                "checkpoint_size_bytes": WEIGHTS[model][0].stat().st_size,
            }
        )
        for row in evaluate_coco_per_class(
            ANNOTATION,
            paths[model],
            max_det=MAX_DET,
            image_ids=image_ids,
        ):
            per_class_rows.append({"model": model, **row})
    metric_fields = (
        "model",
        *METRIC_KEYS,
        "images_evaluated",
        "mean_ms",
        "checkpoint_size_bytes",
    )
    per_class_fields = (
        "model",
        "category_id",
        "category_name",
        *METRIC_KEYS,
        "images_evaluated",
        "max_det",
    )
    expected_metrics = _csv_text(metric_rows, metric_fields)
    expected_per_class = _csv_text(per_class_rows, per_class_fields)
    if METRICS.exists() and METRICS.read_text(encoding="utf-8") != expected_metrics:
        raise CvbraDroneVehicleError("resumable validation metrics changed")
    if PER_CLASS.exists() and PER_CLASS.read_text(encoding="utf-8") != expected_per_class:
        raise CvbraDroneVehicleError("resumable per-class metrics changed")
    if not METRICS.exists():
        atomic_write_text(METRICS, expected_metrics)
    if not PER_CLASS.exists():
        atomic_write_text(PER_CLASS, expected_per_class)
    by_model = {str(row["model"]): row for row in metric_rows}
    if abs(float(by_model["source"]["AP"]) - HISTORICAL_SOURCE_AP) > 1e-12:
        raise CvbraDroneVehicleError("reused source validation AP anchor changed")
    comparisons = (("source", "CVBRA_v1"), ("STF", "CVBRA_v1"))
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method in comparisons:
        name = f"{method}_minus_{baseline}"
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        raw_result = paired_coco_ap_cluster_bootstrap_scopes(
            ANNOTATION,
            paths[baseline],
            paths[method],
            {
                "all_images": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_dronevehicle_validation",
                        "comparison": name,
                        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                        "split": "external_rgb_val",
                        "new_test_access": "prohibited",
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=100,
            accelerate_ap_only=True,
        )["all_images"]
        direct_delta = float(by_model[method]["AP"]) - float(by_model[baseline]["AP"])
        raw_observed_delta = float(raw_result["delta"])
        raw_deltas = _checkpoint_deltas(checkpoint)
        adjusted_deltas, offset = recenter_bootstrap_deltas(
            raw_deltas,
            direct_delta=direct_delta,
            raw_observed_delta=raw_observed_delta,
        )
        standard_deviation = stdev(adjusted_deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise CvbraDroneVehicleError(f"bootstrap variance is invalid: {name}")
        statistics_rows.append(
            {
                "comparison": name,
                "direct_delta": direct_delta,
                "raw_observed_delta": raw_observed_delta,
                "recenter_offset": offset,
                "delta": direct_delta,
                "raw_ci_low": float(raw_result["ci_low"]),
                "raw_ci_high": float(raw_result["ci_high"]),
                "ci_low": float(raw_result["ci_low"]) + offset,
                "ci_high": float(raw_result["ci_high"]) + offset,
                "resamples": RESAMPLES,
                "seed": SEED,
                "images": IMAGES,
                "raw_bootstrap_mean_delta": fmean(raw_deltas),
                "bootstrap_mean_delta": fmean(adjusted_deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": direct_delta / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(adjusted_deltas),
                "checkpoint": _relative(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        print(json.dumps({"bootstrap_complete": name}), flush=True)
    adjusted = holm_adjust([float(row["p_two_sided"]) for row in statistics_rows])
    for row, value in zip(statistics_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_significant_0p05"] = value < 0.05
    statistics = {
        "schema_version": 1,
        "status": "CVBRA_V1_DRONEVEHICLE_VALIDATION_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "unit": "image_via_singleton_cluster_encoding",
        "images": IMAGES,
        "resamples": RESAMPLES,
        "seed": SEED,
        "recenter_rule": "raw_delta_i_plus_direct_delta_minus_raw_observed_delta",
        "multiple_testing": "Holm across two within-dataset descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_DRONEVEHICLE_VALIDATION_AUDIT",
        "completed_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "analysis_lock_sha256": sha256_file(ANALYSIS_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "per_class": _relative(PER_CLASS),
        "per_class_sha256": sha256_file(PER_CLASS),
        "statistics": _relative(STATISTICS),
        "statistics_sha256": sha256_file(STATISTICS),
        "rows": metric_rows,
        "per_class_rows": per_class_rows,
        "AP_deltas": {
            "CVBRA_v1_minus_source": float(by_model["CVBRA_v1"]["AP"])
            - float(by_model["source"]["AP"]),
            "CVBRA_v1_minus_STF": float(by_model["CVBRA_v1"]["AP"])
            - float(by_model["STF"]["AP"]),
        },
        "paired_statistics": statistics_rows,
        "evidence_boundary": {
            "scope": "DroneVehicle official RGB validation only; 1469 images",
            "adaptive_supplementary_after_AU_AIR_results": True,
            "DroneVehicle_validation_labels_previously_accessed": True,
            "DroneVehicle_test_consumed_historically_for_prior_method": True,
            "new_test_access": "prohibited",
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "external_dataset_familywise_confirmatory_claim": False,
        },
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REPORT, report)
    atomic_write_json(
        COMPLETE,
        {
            "status": report["status"],
            "report_sha256": sha256_file(REPORT),
            "metrics_sha256": sha256_file(METRICS),
            "per_class_sha256": sha256_file(PER_CLASS),
            "statistics_sha256": sha256_file(STATISTICS),
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CVBRA-v1 on spent DroneVehicle RGB validation only"
    )
    parser.add_argument("--stage", choices=("preflight", "infer", "score"), default="score")
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        artifact = IMPLEMENTATION_LOCK
    elif args.stage == "infer":
        result = infer()
        artifact = PREDICTION_LOCK
    else:
        result = score()
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
