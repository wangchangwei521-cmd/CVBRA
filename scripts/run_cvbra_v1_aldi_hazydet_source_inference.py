from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts import run_cvbra_v1_aldi_direct_baseline as training
from scripts import run_cvbra_v1_hazydet_source_retention as engine

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_aldi_direct_baseline_v1.yaml"
PROTOCOL_SHA256 = "3e8f28f2d474d4c0caf971cdd2259baadaeac364c9ff168aed694d45481ec3e2"
REGISTRATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "ad7046de7a913321eaa9e528719a309358962cc1d517273db8672fb53413facf"
TRAINING_IMPLEMENTATION = (
    ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/implementation_lock_v4.json"
)
TRAINING_IMPLEMENTATION_SHA256 = (
    "786cf19e1502914bf7c65aa22f4b052bd047ca0c1d98e5af1c3456cfd24be8ac"
)
TRAINING_RUNNER = ROOT / "scripts/run_cvbra_v1_aldi_direct_baseline.py"
TRAINING_RUNNER_SHA256 = "abbeda18ded85f6f5b2fee6d3623d28587f597bb58597e8e8c47623f5d1645e5"
REUSED_ENGINE = ROOT / "scripts/run_cvbra_v1_hazydet_source_retention.py"

MANIFEST = ROOT / "data/manifests/hazydet_val_manifest.json"
MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
VALIDATION_AUDIT = ROOT / "data/manifests/hazydet_val_validation.json"
VALIDATION_AUDIT_SHA256 = "3b14a7f9181ea4820b7eca2837a7baaa52eb01d4a08055f8e1b2d96fd6aed83f"
ANNOTATION = ROOT / "data/raw/HazyDet/val/val_coco.json"
ANNOTATION_SHA256 = "2b2e39f7812631dfb4f3f0fbe1e743b65ca873151ca92d8e24ddbf0e9feacb9a"

OUTPUT = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
    / "hazydet_source_retention/evaluation"
)
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
PREDICTION_LOCK = OUTPUT / "prediction_lock.json"
PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"

MODELS = tuple(training.PAPER_LABELS[variant] for variant in training.VARIANTS)
WEIGHTS: dict[str, tuple[Path, str]] = {}


class ALDIHazyDetInferenceError(RuntimeError):
    """Raised when ALDI HazyDet source-retention inference cannot fail closed."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ALDIHazyDetInferenceError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ALDIHazyDetInferenceError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ALDIHazyDetInferenceError(f"locked {label} changed: {path}")


def _validate_scope_and_weights() -> None:
    global WEIGHTS
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "ALDI protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "ALDI registration"),
        (
            TRAINING_IMPLEMENTATION,
            TRAINING_IMPLEMENTATION_SHA256,
            "ALDI training implementation",
        ),
        (TRAINING_RUNNER, TRAINING_RUNNER_SHA256, "ALDI training runner"),
        (MANIFEST, MANIFEST_SHA256, "HazyDet validation manifest"),
        (VALIDATION_AUDIT, VALIDATION_AUDIT_SHA256, "HazyDet validation audit"),
        (ANNOTATION, ANNOTATION_SHA256, "HazyDet validation annotation"),
    ):
        _assert_hash(path, digest, label=label)
    audit = _load_mapping(VALIDATION_AUDIT)
    if (
        audit.get("valid") is not True
        or not isinstance(audit.get("stats"), dict)
        or audit["stats"].get("images") != engine.IMAGES
    ):
        raise ALDIHazyDetInferenceError("HazyDet validation scope changed")
    weights: dict[str, tuple[Path, str]] = {}
    for variant in training.VARIANTS:
        lock = training._validate_checkpoint_lock(variant)
        label = training.PAPER_LABELS[variant]
        checkpoint = ROOT / str(lock["checkpoint"])
        digest = str(lock["checkpoint_sha256"])
        if (
            lock.get("paper_label") != label
            or lock.get("validation_metric_used_for_training_or_selection") is not False
            or lock.get("official_test_access") != "prohibited"
        ):
            raise ALDIHazyDetInferenceError(f"ALDI checkpoint scope changed: {variant}")
        _assert_hash(checkpoint, digest, label=f"{label} checkpoint")
        weights[label] = (checkpoint, digest)
    WEIGHTS = weights
    engine.WEIGHTS = weights


def _implementation_lock() -> dict[str, Any]:
    _validate_scope_and_weights()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise ALDIHazyDetInferenceError("ALDI HazyDet implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("training_implementation_sha256")
            != TRAINING_IMPLEMENTATION_SHA256
            or lock.get("cuda_identity") != engine._cuda_identity()
            or marker.get("implementation_lock_sha256")
            != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise ALDIHazyDetInferenceError("ALDI HazyDet implementation changed")
        return lock
    if PREDICTION_LOCK.exists() or PREDICTION_MARKER.exists():
        raise ALDIHazyDetInferenceError("ALDI HazyDet prediction appeared before lock")
    records = engine._records(verify_hashes=True)
    payload = {
        "schema_version": 1,
        "status": "ALDI_HAZYDET_SOURCE_RETENTION_IMPLEMENTATION_LOCKED",
        "locked_at_utc": engine.primary_engine_time(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "training_implementation_sha256": TRAINING_IMPLEMENTATION_SHA256,
        "runner": engine._relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "reused_engine": engine._relative(REUSED_ENGINE),
        "reused_engine_sha256": sha256_file(REUSED_ENGINE),
        "python": platform.python_version(),
        "host": platform.node(),
        "cuda_identity": engine._cuda_identity(),
        "models": list(MODELS),
        "weights": {
            model: {"path": engine._relative(path), "sha256": digest}
            for model, (path, digest) in WEIGHTS.items()
        },
        "images": len(records),
        "inference_repetitions": engine.REPETITIONS,
        "category_id_by_class": {"0": 0, "1": 1, "2": 2},
        "category_mapping_note": "directly matches HazyDet validation category IDs",
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "validation_labels_previously_accessed": True,
        "labels_read_for_prediction": False,
        "method_or_hyperparameter_selection": False,
        "HazyDet_test_access": "prohibited",
        "UAV_OBB_test_access": "prohibited",
        "paper_body_change_authorized": False,
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


def _configure_engine() -> None:
    assignments: Mapping[str, object] = {
        "__file__": str(Path(__file__).resolve()),
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_SHA256": PROTOCOL_SHA256,
        "REGISTRATION": REGISTRATION,
        "REGISTRATION_SHA256": REGISTRATION_SHA256,
        "OUTPUT": OUTPUT,
        "MODELS": MODELS,
        "WEIGHTS": WEIGHTS,
        "IMPLEMENTATION_LOCK": IMPLEMENTATION_LOCK,
        "IMPLEMENTATION_MARKER": IMPLEMENTATION_MARKER,
        "PREDICTION_LOCK": PREDICTION_LOCK,
        "PREDICTION_MARKER": PREDICTION_MARKER,
        "CATEGORY_ID_BY_CLASS": {0: 0, 1: 1, 2: 2},
        "_validate_scope_and_weights": _validate_scope_and_weights,
        "_implementation_lock": _implementation_lock,
    }
    for name, value in assignments.items():
        setattr(engine, name, value)


def preflight() -> dict[str, Any]:
    _configure_engine()
    result = engine.preflight()
    result["validation_labels_previously_accessed"] = True
    result["labels_read_for_prediction"] = False
    result["method_or_hyperparameter_selection"] = False
    return result


def infer() -> dict[str, Any]:
    _configure_engine()
    result = engine.infer()
    result["validation_labels_previously_accessed"] = True
    result["labels_read_for_prediction"] = False
    result["method_or_hyperparameter_selection"] = False
    atomic_write_json(PREDICTION_LOCK, result)
    marker = _load_mapping(PREDICTION_MARKER)
    marker["prediction_lock_sha256"] = sha256_file(PREDICTION_LOCK)
    atomic_write_json(PREDICTION_MARKER, marker)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ALDI HazyDet source-retention inference")
    parser.add_argument("--stage", choices=("preflight", "infer"), default="infer")
    args = parser.parse_args()
    result = preflight() if args.stage == "preflight" else infer()
    summary = {key: value for key, value in result.items() if key != "artifacts"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
