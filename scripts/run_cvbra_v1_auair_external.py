from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, cast

import yaml

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
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_auair_external_protocol.yaml"
OUTPUT = ROOT / "reports" / "external" / "cvbra_v1_auair"
PARENT_CONFIG = ROOT / "runs" / "auair1hz_yolo11n_b0_16a7c7c20c5d" / "config_resolved.yaml"
PARENT_MANIFEST = ROOT / "runs" / "auair1hz_yolo11n_b0_16a7c7c20c5d" / "data_manifest.json"
SOURCE_PREDICTION = (
    ROOT
    / "runs"
    / "auair1hz_yolo11n_b0_16a7c7c20c5d"
    / "predictions"
    / "final.coco.json"
)
SOURCE_TIMING = (
    ROOT
    / "runs"
    / "auair1hz_yolo11n_b0_16a7c7c20c5d"
    / "traces"
    / "timings.jsonl"
)
ANNOTATION = ROOT / "data" / "processed" / "auair_1hz" / "annotations" / "auair_1hz.coco.json"
SELECTION_MANIFEST = ROOT / "data" / "processed" / "auair_1hz" / "selection_manifest.json"
PARENT_JOINT_LOCK = (
    ROOT / "reports" / "external" / "auair_predictions" / "joint_prediction_lock.json"
)

PARENT_CONFIG_SHA256 = "731507f500ecebc8b244c0c01fd9d5e49aff3df25f67fd2b3040d34620eee3a5"
PARENT_MANIFEST_SHA256 = "29bd14d52b5dc191ac6ed4def48df78a02ff7d5e15c6608d1b0be8046f008d8e"
SOURCE_PREDICTION_SHA256 = "6f37ddf09f5f253d85f984e71d5b360f48ec3eb6430ee302c9f0d9167e892e71"
SOURCE_TIMING_SHA256 = "5267f91658508e462bfe734e33187ba0ab7a9b7474ca46d95fcd26b371292cec"
ANNOTATION_SHA256 = "f87191c18a87f591e0459ed9fa72306829b7a319d8230dbd92c797372f655e61"
SELECTION_MANIFEST_SHA256 = "1a1003813f192ac47b13778e163bd0d38ad8b7a0c5d4e81e03360f3aef6a6d27"
PARENT_JOINT_LOCK_SHA256 = "db07ea9675df74fab7d5569bd85c8a84b315c6e04427f76afdc308272ca0a698"
HISTORICAL_SOURCE_AP = 0.07862021830969958

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
    model: ROOT / "runs" / f"cvbra_v1_auair_external_{model.lower()}" for model in NEW_MODELS
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
REPORT = OUTPUT / "external_report.json"
COMPLETE = OUTPUT / "AUAIR_EXTERNAL_COMPLETE"

IMAGES = 6578
SEQUENCES = 8
MAX_DET = 500
RESAMPLES = 10000
SEED = 20260814
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class CvbraAuAirError(RuntimeError):
    """Raised when the registered AU-AIR audit cannot fail closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraAuAirError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CvbraAuAirError(f"expected mapping: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraAuAirError(f"cannot parse rows {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CvbraAuAirError(f"expected row list: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise CvbraAuAirError(f"locked {label} changed: {path}")


def _annotation_scope() -> tuple[tuple[int, ...], dict[str, list[int]]]:
    annotation = _load_mapping(ANNOTATION)
    raw_images = annotation.get("images")
    raw_categories = annotation.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise CvbraAuAirError("AU-AIR annotation structure changed")
    category_ids = [
        int(row["id"])
        for row in raw_categories
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    ]
    if category_ids != [1, 2, 3]:
        raise CvbraAuAirError("AU-AIR category IDs changed")
    image_ids: list[int] = []
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in raw_images:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), int)
            or not isinstance(row.get("sequence"), str)
        ):
            raise CvbraAuAirError("AU-AIR image/sequence identity changed")
        image_id = int(row["id"])
        image_ids.append(image_id)
        clusters[str(row["sequence"])].append(image_id)
    if (
        len(image_ids) != IMAGES
        or len(set(image_ids)) != IMAGES
        or len(clusters) != SEQUENCES
        or set().union(*map(set, clusters.values())) != set(image_ids)
    ):
        raise CvbraAuAirError("AU-AIR image or sequence scope changed")
    return tuple(image_ids), dict(clusters)


def locked_raw_config(model: str) -> dict[str, Any]:
    """Return the parent B0 config with only checkpoint and experiment name changed."""
    if model not in NEW_MODELS:
        raise CvbraAuAirError(f"unsupported new AU-AIR model: {model}")
    try:
        loaded = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CvbraAuAirError(f"cannot load parent AU-AIR config: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CvbraAuAirError("parent AU-AIR config is not a mapping")
    raw = cast(dict[str, Any], loaded)
    detector = raw.get("detector")
    experiment = raw.get("experiment")
    if not isinstance(detector, dict) or not isinstance(experiment, dict):
        raise CvbraAuAirError("parent AU-AIR config sections changed")
    detector["model"] = _relative(WEIGHTS[model][0])
    experiment["name"] = f"cvbra_v1_auair_external_{model.lower()}"
    return raw


def _render_config(model: str) -> str:
    return yaml.safe_dump(locked_raw_config(model), sort_keys=False, allow_unicode=True)


def _validate_parent() -> None:
    for path, digest, label in (
        (PARENT_CONFIG, PARENT_CONFIG_SHA256, "parent AU-AIR config"),
        (PARENT_MANIFEST, PARENT_MANIFEST_SHA256, "parent AU-AIR manifest"),
        (SOURCE_PREDICTION, SOURCE_PREDICTION_SHA256, "source AU-AIR prediction"),
        (SOURCE_TIMING, SOURCE_TIMING_SHA256, "source AU-AIR timing"),
        (ANNOTATION, ANNOTATION_SHA256, "AU-AIR annotation"),
        (SELECTION_MANIFEST, SELECTION_MANIFEST_SHA256, "AU-AIR selection manifest"),
        (PARENT_JOINT_LOCK, PARENT_JOINT_LOCK_SHA256, "parent joint prediction lock"),
    ):
        _assert_hash(path, digest, label=label)
    for model, (path, digest) in WEIGHTS.items():
        _assert_hash(path, digest, label=f"{model} checkpoint")
    image_ids, _ = _annotation_scope()
    manifest = _load_mapping(PARENT_MANIFEST)
    manifest_images = manifest.get("images")
    if (
        manifest.get("dataset") != "auair"
        or manifest.get("split") != "external_1hz"
        or not isinstance(manifest_images, list)
        or len(manifest_images) != IMAGES
        or tuple(
            int(row["image_id"])
            for row in manifest_images
            if isinstance(row, dict) and isinstance(row.get("image_id"), int)
        )
        != image_ids
    ):
        raise CvbraAuAirError("parent AU-AIR data manifest scope changed")
    source_rows = _load_rows(SOURCE_PREDICTION)
    if not source_rows or any(
        int(row.get("category_id", -1)) not in {1, 2, 3} for row in source_rows
    ):
        raise CvbraAuAirError("source AU-AIR category mapping changed")


def _registration() -> dict[str, Any]:
    _validate_parent()
    protocol_sha256 = sha256_file(PROTOCOL)
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
            raise CvbraAuAirError("AU-AIR registration is incomplete")
        lock = _load_mapping(REGISTRATION)
        marker = _load_mapping(REGISTRATION_MARKER)
        if (
            lock.get("protocol_sha256") != protocol_sha256
            or lock.get("new_prediction_or_metric_before_registration") is not False
            or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        ):
            raise CvbraAuAirError("AU-AIR registration changed")
        return lock
    if any(path.exists() for path in (PREDICTION_LOCK, ANALYSIS_LOCK, METRICS, REPORT)):
        raise CvbraAuAirError("AU-AIR output appeared before registration")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_AUAIR_EXTERNAL_REGISTERED_BEFORE_NEW_PREDICTIONS",
        "registered_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": protocol_sha256,
        "models": list(MODELS),
        "new_models": list(NEW_MODELS),
        "new_prediction_or_metric_before_registration": False,
        "existing_source_prediction_and_metric_observed": True,
        "AU_AIR_labels_previously_accessed_for_historical_methods": True,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
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
            raise CvbraAuAirError("AU-AIR implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("protocol_sha256") != sha256_file(PROTOCOL)
            or lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CvbraAuAirError("AU-AIR implementation lock changed")
        observed_config_hashes = lock.get("locked_config_sha256")
        if not isinstance(observed_config_hashes, dict):
            raise CvbraAuAirError("AU-AIR locked config hashes changed")
        for model in NEW_MODELS:
            path = _locked_config_path(model)
            _assert_hash(
                path,
                str(observed_config_hashes[model]),
                label=f"{model} AU-AIR config",
            )
            if path.read_text(encoding="utf-8") != _render_config(model):
                raise CvbraAuAirError(f"{model} AU-AIR rendered config changed")
        return lock
    if any(RUN_PATHS[model].exists() for model in NEW_MODELS):
        raise CvbraAuAirError("AU-AIR inference run appeared before implementation lock")
    created_config_hashes: dict[str, str] = {}
    for model in NEW_MODELS:
        path = _locked_config_path(model)
        atomic_write_text(path, _render_config(model))
        created_config_hashes[model] = sha256_file(path)
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_AUAIR_EXTERNAL_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "source_prediction_sha256": SOURCE_PREDICTION_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
        "locked_config_sha256": created_config_hashes,
        "weights": {
            model: {"path": _relative(path), "sha256": digest}
            for model, (path, digest) in WEIGHTS.items()
        },
        "models": list(MODELS),
        "images": IMAGES,
        "sequences": SEQUENCES,
        "python": platform.python_version(),
        "AU_AIR_labels_previously_accessed_for_historical_methods": True,
        "new_metrics_accessed": False,
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
        "status": "PASS_CVBRA_V1_AUAIR_EXTERNAL_PREFLIGHT",
        "models": list(MODELS),
        "new_inference_cells": len(NEW_MODELS),
        "images": IMAGES,
        "sequences": SEQUENCES,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "new_metrics_accessed": False,
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
        raise CvbraAuAirError(f"incomplete AU-AIR run for {model}: {missing}")
    if sha256_file(run_path / "config_resolved.yaml") != sha256_file(_locked_config_path(model)):
        raise CvbraAuAirError(f"AU-AIR config drifted for {model}")
    if sha256_file(run_path / "data_manifest.json") != PARENT_MANIFEST_SHA256:
        raise CvbraAuAirError(f"AU-AIR data manifest drifted for {model}")
    fingerprint = _load_mapping(run_path / "model_fingerprint.json")
    if fingerprint.get("sha256") != WEIGHTS[model][1]:
        raise CvbraAuAirError(f"AU-AIR checkpoint drifted for {model}")
    prediction = run_path / "predictions" / "final.coco.json"
    rows = _load_rows(prediction)
    annotation_ids = set(_annotation_scope()[0])
    if not rows or any(
        int(row.get("category_id", -1)) not in {1, 2, 3}
        or int(row.get("image_id", -1)) not in annotation_ids
        for row in rows
    ):
        raise CvbraAuAirError(f"AU-AIR prediction ontology drifted for {model}")
    forbidden = list(run_path.glob("metrics*.json")) + list(run_path.glob("metrics*.csv"))
    if forbidden:
        raise CvbraAuAirError(f"AU-AIR metrics appeared before joint lock: {forbidden}")
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
        raise CvbraAuAirError("AU-AIR joint prediction lock is incomplete")
    lock = _load_mapping(PREDICTION_LOCK)
    marker = _load_mapping(PREDICTION_MARKER)
    artifacts = lock.get("artifacts")
    if (
        lock.get("status") != "CVBRA_V1_AUAIR_ALL_PREDICTIONS_LOCKED_BEFORE_NEW_METRICS"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or lock.get("models") != list(MODELS)
        or not isinstance(artifacts, list)
        or len(artifacts) != len(MODELS)
        or marker.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
    ):
        raise CvbraAuAirError("AU-AIR joint prediction lock changed")
    for row in artifacts:
        if not isinstance(row, dict):
            raise CvbraAuAirError("AU-AIR prediction artifact changed")
        _assert_hash(
            ROOT / str(row["prediction"]),
            str(row["prediction_sha256"]),
            label=f"{row['model']} AU-AIR prediction",
        )
        _assert_hash(
            ROOT / str(row["timing"]),
            str(row["timing_sha256"]),
            label=f"{row['model']} AU-AIR timing",
        )
    return lock


def infer() -> dict[str, Any]:
    _implementation_lock()
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        return _validate_prediction_lock()
    artifacts: list[dict[str, Any]] = [
        {
            "model": "source",
            "run": _relative(PARENT_CONFIG.parent),
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
        "status": "CVBRA_V1_AUAIR_ALL_PREDICTIONS_LOCKED_BEFORE_NEW_METRICS",
        "locked_at_utc": _now(),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "parent_joint_prediction_lock_sha256": PARENT_JOINT_LOCK_SHA256,
        "models": list(MODELS),
        "images": IMAGES,
        "sequences": SEQUENCES,
        "artifacts": artifacts,
        "source_anchor_reused": True,
        "new_metric_accessed": False,
        "AU_AIR_labels_previously_accessed_for_historical_methods": True,
        "method_or_hyperparameter_selection": False,
        "independent_confirmation_claim": False,
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
            raise CvbraAuAirError("AU-AIR analysis authorization is incomplete")
        lock = _load_mapping(ANALYSIS_LOCK)
        marker = _load_mapping(ANALYSIS_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("prediction_lock_sha256") != sha256_file(PREDICTION_LOCK)
            or lock.get("annotation_sha256") != ANNOTATION_SHA256
            or marker.get("analysis_lock_sha256") != sha256_file(ANALYSIS_LOCK)
        ):
            raise CvbraAuAirError("AU-AIR analysis authorization changed")
        return lock
    if any(path.exists() for path in (METRICS, PER_CLASS, STATISTICS, REPORT)):
        raise CvbraAuAirError("AU-AIR metrics appeared before analysis authorization")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V1_AUAIR_ANALYSIS_AUTHORIZED_AFTER_PREDICTION_LOCK",
        "authorized_at_utc": _now(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "annotation_sha256": ANNOTATION_SHA256,
        "new_metric_accessed_before_authorization": False,
        "evaluation_image_scope": "all_6578_annotation_images",
        "bootstrap_unit": "eight_released_video_sequences",
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
        raise CvbraAuAirError("AU-AIR prediction artifacts changed")
    paths = {
        str(row["model"]): ROOT / str(row["prediction"])
        for row in artifacts
        if isinstance(row, dict)
    }
    if tuple(paths) != MODELS:
        raise CvbraAuAirError("AU-AIR prediction model coverage changed")
    return paths


def _mean_timing_ms(path: Path) -> float:
    values: list[float] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("total_ms"), (int, float)):
                raise CvbraAuAirError(f"invalid timing row: {path}")
            values.append(float(row["total_ms"]))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CvbraAuAirError(f"cannot parse timing {path}: {exc}") from exc
    if len(values) != IMAGES or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise CvbraAuAirError(f"AU-AIR timing coverage changed: {path}")
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
        raise CvbraAuAirError(f"AU-AIR bootstrap checkpoint is incomplete: {path}")
    return [float(value) for value in values]


def score() -> dict[str, Any]:
    prediction_lock = infer()
    _analysis_authorization()
    if REPORT.exists():
        report = _load_mapping(REPORT)
        marker = _load_mapping(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise CvbraAuAirError("existing AU-AIR report is not locked")
        return report
    paths = _prediction_paths(prediction_lock)
    artifacts = prediction_lock.get("artifacts")
    if not isinstance(artifacts, list):
        raise CvbraAuAirError("AU-AIR prediction metadata changed")
    timing_paths = {
        str(row["model"]): ROOT / str(row["timing"])
        for row in artifacts
        if isinstance(row, dict)
    }
    image_ids, clusters = _annotation_scope()
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model in MODELS:
        metrics = evaluate_coco(ANNOTATION, paths[model], max_det=MAX_DET, image_ids=image_ids)
        if int(metrics["images_evaluated"]) != IMAGES:
            raise CvbraAuAirError(f"{model} AU-AIR metrics omitted images")
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
        raise CvbraAuAirError("resumable AU-AIR metrics changed")
    if PER_CLASS.exists() and PER_CLASS.read_text(encoding="utf-8") != expected_per_class:
        raise CvbraAuAirError("resumable AU-AIR per-class metrics changed")
    if not METRICS.exists():
        atomic_write_text(METRICS, expected_metrics)
    if not PER_CLASS.exists():
        atomic_write_text(PER_CLASS, expected_per_class)
    by_model = {str(row["model"]): row for row in metric_rows}
    if abs(float(by_model["source"]["AP"]) - HISTORICAL_SOURCE_AP) > 1e-12:
        raise CvbraAuAirError("reused source AU-AIR AP anchor changed")
    comparisons = (("source", "CVBRA_v1"), ("STF", "CVBRA_v1"))
    statistics_rows: list[dict[str, Any]] = []
    for baseline, method in comparisons:
        name = f"{method}_minus_{baseline}"
        checkpoint = OUTPUT / "bootstrap" / f"{name}.json"
        result = paired_coco_ap_cluster_bootstrap_scopes(
            ANNOTATION,
            paths[baseline],
            paths[method],
            {
                "all_sequences": ClusterBootstrapScope(
                    clusters=clusters,
                    checkpoint_path=checkpoint,
                    checkpoint_identity={
                        "study": "cvbra_v1_auair_external",
                        "comparison": name,
                        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
                        "descriptive_post_freeze": True,
                    },
                )
            },
            resamples=RESAMPLES,
            seed=SEED,
            max_det=MAX_DET,
            workers=4,
            chunk_resamples=250,
            accelerate_ap_only=True,
        )["all_sequences"]
        expected = float(by_model[method]["AP"]) - float(by_model[baseline]["AP"])
        if abs(float(result["delta"]) - expected) > 1e-10:
            raise CvbraAuAirError(f"AU-AIR bootstrap point estimate drifted: {name}")
        deltas = _checkpoint_deltas(checkpoint)
        standard_deviation = stdev(deltas)
        if standard_deviation <= 0.0 or not math.isfinite(standard_deviation):
            raise CvbraAuAirError(f"AU-AIR bootstrap variance is invalid: {name}")
        statistics_rows.append(
            {
                "comparison": name,
                **result,
                "bootstrap_mean_delta": fmean(deltas),
                "bootstrap_standard_deviation": standard_deviation,
                "standardized_effect": expected / standard_deviation,
                "p_two_sided": bootstrap_sign_pvalue(deltas),
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
        "status": "CVBRA_V1_AUAIR_EXTERNAL_PAIRED_STATISTICS_COMPLETE",
        "completed_at_utc": _now(),
        "unit": "released_video_sequence",
        "sequences": SEQUENCES,
        "sequence_count_limitation": True,
        "resamples": RESAMPLES,
        "seed": SEED,
        "multiple_testing": "Holm across two descriptive AP contrasts",
        "rows": statistics_rows,
    }
    atomic_write_json(STATISTICS, statistics)
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_AUAIR_EXTERNAL_AUDIT",
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
            "scope": "AU-AIR 1-Hz external subset; 6578 frames from eight released sequences",
            "zero_tuning_after_CVBRA_freeze": True,
            "AU_AIR_labels_previously_accessed_for_historical_methods": True,
            "method_or_hyperparameter_selection": False,
            "independent_confirmation_claim": False,
            "sequence_count_limitation": True,
            "HazyDet_test_access": "prohibited",
            "UAV_OBB_test_access": "prohibited",
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
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 AU-AIR external audit")
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
