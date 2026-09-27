from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from buse_uav.detectors.aldi_ultralytics import (
    ALDIRuntimeConfig,
    ALDITranslationDetectionTrainer,
    configure_aldi_runtime,
)
from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_aldi_direct_baseline_v1.yaml"
MODULE = ROOT / "src/buse_uav/detectors/aldi_ultralytics.py"
SOURCE_CHECKPOINT = ROOT / "weights/hazydet/yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
DATASET_SHA256 = "567ab0bb75434e3c735f8bda6b2aaae2b1c81a4ed6043c215c87900ec9c039d7"
MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"

ALDI_REPOSITORY = ROOT / "third_party/aldi_official"
ALDI_COMMIT = "774aaa299700e28c4a32a2cefec6a815c46053e0"
ALDI_FILES = (
    ALDI_REPOSITORY / "README.md",
    ALDI_REPOSITORY / "configs/Base-Yolo.yaml",
    ALDI_REPOSITORY / "configs/cityscapes/ALDI-Yolo-Cityscapes.yaml",
    ALDI_REPOSITORY / "aldi/aug.py",
    ALDI_REPOSITORY / "aldi/ema.py",
    ALDI_REPOSITORY / "aldi/pseudolabeler.py",
    ALDI_REPOSITORY / "aldi/yolo/distill.py",
)

OUTPUT = ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
RUN_ROOT = ROOT / "runs/cvbra_v1_aldi_direct_baseline_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REGISTRATION_MARKER = OUTPUT / "REGISTERED"
AMENDMENT_1 = OUTPUT / "AMENDMENT_1_BATCH_PATH_SEQUENCE.json"
AMENDMENT_1_MARKER = OUTPUT / "AMENDMENT_1_REGISTERED"
AMENDMENT_2 = OUTPUT / "AMENDMENT_2_DISABLE_UNREGISTERED_FINAL_VALIDATION.json"
AMENDMENT_2_MARKER = OUTPUT / "AMENDMENT_2_REGISTERED"
AMENDMENT_2_CORRECTION = OUTPUT / "AMENDMENT_2_CORRECTION_1.json"
AMENDMENT_2_CORRECTION_MARKER = OUTPUT / "AMENDMENT_2_CORRECTION_1_REGISTERED"
AMENDMENT_3 = OUTPUT / "AMENDMENT_3_RECOVER_COMPLETE_ENDPOINT.json"
AMENDMENT_3_MARKER = OUTPUT / "AMENDMENT_3_REGISTERED"
V1_IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
V1_IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
V2_IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock_v2.json"
V2_IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_V2_LOCKED"
V3_IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock_v3.json"
V3_IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_V3_LOCKED"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock_v4.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_V4_LOCKED"
FAILED_ATTEMPT_1_ARGS = OUTPUT / "failures/attempt_1_native_uda_run/raw_endpoint/fit/args.yaml"
FAILED_ATTEMPT_2 = OUTPUT / "failures/attempt_2_native_uda_final_validation"

VARIANTS = ("native_uda", "equal_supervision")
PAPER_LABELS = {
    "native_uda": "ALDIpp_AF_Y11_native_UDA",
    "equal_supervision": "ALDIpp_AF_Y11_equal_supervision",
}
SOURCE_IMAGES = 900
TARGET_IMAGES = 2700
TRAIN_IMAGES = 3600
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42


class ALDIDirectBaselineError(RuntimeError):
    """Raised when the preregistered ALDI direct comparison cannot fail closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if path.suffix.casefold() in {".yaml", ".yml"}
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ALDIDirectBaselineError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ALDIDirectBaselineError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise ALDIDirectBaselineError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ALDIDirectBaselineError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ALDI_REPOSITORY), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ALDIDirectBaselineError(
            f"cannot audit official ALDI repository: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _cuda_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ALDIDirectBaselineError("ALDI direct baseline training requires CUDA")
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    return {
        "device": "cuda:0",
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def _manifest_contract() -> tuple[dict[str, Any], tuple[Path, ...]]:
    _assert_hash(MANIFEST, MANIFEST_SHA256, label="CVBRA data manifest")
    manifest = _load_mapping(MANIFEST)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != TRAIN_IMAGES:
        raise ALDIDirectBaselineError("CVBRA manifest entry count changed")
    source_paths: list[Path] = []
    target_count = 0
    target_views: dict[str, int] = {"original": 0, "fog_0p6": 0, "fog_1p0": 0}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ALDIDirectBaselineError("CVBRA manifest row is malformed")
        image = _rooted(raw["image"])
        label = _rooted(raw["label"])
        _assert_hash(image, raw["image_sha256"], label="training image")
        _assert_hash(label, raw["label_sha256"], label="training label")
        if raw.get("role") == "source_haze_replay":
            source_paths.append(image.resolve())
        elif raw.get("role") == "target" and raw.get("view") in target_views:
            target_count += 1
            target_views[str(raw["view"])] += 1
        else:
            raise ALDIDirectBaselineError("CVBRA manifest role or target view changed")
    if (
        len(source_paths) != SOURCE_IMAGES
        or target_count != TARGET_IMAGES
        or set(target_views.values()) != {900}
        or len(set(source_paths)) != SOURCE_IMAGES
    ):
        raise ALDIDirectBaselineError("CVBRA source/target coverage changed")
    return (
        {
            "images": len(entries),
            "source_images": len(source_paths),
            "target_images": target_count,
            "target_views": target_views,
        },
        tuple(source_paths),
    )


def _validate_protocol() -> dict[str, Any]:
    for path, digest, label in (
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "source checkpoint"),
        (DATASET, DATASET_SHA256, "CVBRA dataset YAML"),
        (MANIFEST, MANIFEST_SHA256, "CVBRA data manifest"),
    ):
        _assert_hash(path, digest, label=label)
    protocol = _load_mapping(PROTOCOL)
    variants = protocol.get("variants")
    training = protocol.get("common_training_contract")
    source = protocol.get("recent_method_source")
    integrity = protocol.get("integrity")
    if not all(isinstance(value, dict) for value in (variants, training, source, integrity)):
        raise ALDIDirectBaselineError("ALDI protocol is incomplete")
    assert isinstance(variants, dict)
    assert isinstance(training, dict)
    assert isinstance(source, dict)
    assert isinstance(integrity, dict)
    if (
        protocol.get("status")
        != "PREREGISTERED_BEFORE_ALDI_TRANSLATION_TRAINING_OR_PREDICTION"
        or set(variants) != set(PAPER_LABELS.values())
        or source.get("locked_commit") != ALDI_COMMIT
        or training.get("images_per_epoch") != TRAIN_IMAGES
        or training.get("epochs") != EPOCHS
        or training.get("imgsz") != IMGSZ
        or training.get("batch") != BATCH
        or training.get("seed") != SEED
        or training.get("endpoint") != "fixed_last_epoch_EMA_teacher"
        or integrity.get("UAV_OBB_official_test_access") != "prohibited"
        or integrity.get("HazyDet_test_or_RDDTS_access") != "prohibited"
    ):
        raise ALDIDirectBaselineError("ALDI protocol fields changed")
    return protocol


def _registration_payload() -> dict[str, Any]:
    protocol = _validate_protocol()
    manifest, _ = _manifest_contract()
    if not ALDI_REPOSITORY.is_dir():
        raise ALDIDirectBaselineError("official ALDI repository is missing")
    observed_commit = _git_output("rev-parse", "HEAD")
    if observed_commit != ALDI_COMMIT:
        raise ALDIDirectBaselineError(
            f"official ALDI commit changed: {observed_commit} != {ALDI_COMMIT}"
        )
    if _git_output("status", "--porcelain"):
        raise ALDIDirectBaselineError("official ALDI repository has local changes")
    for path in ALDI_FILES:
        if not path.is_file():
            raise ALDIDirectBaselineError(f"official ALDI source file is missing: {path}")
    return {
        "schema_version": 1,
        "status": "ALDI_DIRECT_BASELINE_REGISTERED_BEFORE_TRAINING_OR_PREDICTION",
        "registered_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "translation_module": _relative(MODULE),
        "translation_module_sha256": sha256_file(MODULE),
        "official_repository": _relative(ALDI_REPOSITORY),
        "official_commit": observed_commit,
        "official_files": {
            _relative(path): sha256_file(path) for path in ALDI_FILES
        },
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "dataset_yaml_sha256": DATASET_SHA256,
        "data_manifest_sha256": MANIFEST_SHA256,
        "manifest_contract": manifest,
        "variants": list(VARIANTS),
        "paper_labels": PAPER_LABELS,
        "translation_is_official_author_code": False,
        "official_recipe_roles_preserved": [
            "source_strong_supervision",
            "EMA_teacher",
            "weak_to_strong_target_distillation",
            "MIC",
            "soft_confidence_and_class_terms",
            "pseudo_box_regression",
        ],
        "anchor_free_translation": (
            "YOLO11n_has_no_separate_objectness_head_so_confidence_map_distillation_"
            "replaces_the_YOLOv5_objectness_term"
        ),
        "validation_or_test_metric_used_for_registration": False,
        "official_test_access": "prohibited",
        "paper_body_change_before_result_lock": False,
        "protocol_snapshot": protocol["scientific_role"],
    }


def register() -> dict[str, Any]:
    if REGISTRATION.exists() or REGISTRATION_MARKER.exists():
        return _validate_registration()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ALDIDirectBaselineError("unregistered ALDI output already exists")
    if RUN_ROOT.exists():
        raise ALDIDirectBaselineError("unregistered ALDI run directory already exists")
    payload = _registration_payload()
    atomic_write_json(REGISTRATION, payload)
    atomic_write_json(
        REGISTRATION_MARKER,
        {
            "status": payload["status"],
            "registration_sha256": sha256_file(REGISTRATION),
        },
    )
    return payload


def _validate_registration() -> dict[str, Any]:
    if not REGISTRATION.is_file() or not REGISTRATION_MARKER.is_file():
        raise ALDIDirectBaselineError("ALDI registration lock is incomplete")
    lock = _load_mapping(REGISTRATION)
    marker = _load_mapping(REGISTRATION_MARKER)
    if (
        lock.get("status")
        != "ALDI_DIRECT_BASELINE_REGISTERED_BEFORE_TRAINING_OR_PREDICTION"
        or marker.get("registration_sha256") != sha256_file(REGISTRATION)
        or lock.get("official_commit") != ALDI_COMMIT
        or lock.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or lock.get("dataset_yaml_sha256") != DATASET_SHA256
        or lock.get("data_manifest_sha256") != MANIFEST_SHA256
        or lock.get("variants") != list(VARIANTS)
    ):
        raise ALDIDirectBaselineError("ALDI registration fields changed")
    _assert_hash(PROTOCOL, lock["protocol_sha256"], label="protocol_sha256")
    current_runner = sha256_file(Path(__file__))
    current_module = sha256_file(MODULE)
    if (
        current_runner != lock.get("runner_sha256")
        or current_module != lock.get("translation_module_sha256")
    ):
        _validate_amendments(lock)
    official = lock.get("official_files")
    if not isinstance(official, dict):
        raise ALDIDirectBaselineError("official ALDI file hashes are missing")
    for relative, digest in official.items():
        _assert_hash(_rooted(relative), digest, label="official ALDI source")
    if _git_output("rev-parse", "HEAD") != ALDI_COMMIT or _git_output(
        "status", "--porcelain"
    ):
        raise ALDIDirectBaselineError("official ALDI repository state changed")
    _validate_protocol()
    _manifest_contract()
    return lock


def _validate_amendments(registration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    for path in (
        AMENDMENT_1,
        AMENDMENT_1_MARKER,
        AMENDMENT_2,
        AMENDMENT_2_MARKER,
        AMENDMENT_2_CORRECTION,
        AMENDMENT_2_CORRECTION_MARKER,
        AMENDMENT_3,
        AMENDMENT_3_MARKER,
        V1_IMPLEMENTATION_LOCK,
        V1_IMPLEMENTATION_MARKER,
        V2_IMPLEMENTATION_LOCK,
        V2_IMPLEMENTATION_MARKER,
        V3_IMPLEMENTATION_LOCK,
        V3_IMPLEMENTATION_MARKER,
    ):
        if not path.is_file():
            raise ALDIDirectBaselineError(f"ALDI post-registration evidence is missing: {path}")
    amendment_1 = _load_mapping(AMENDMENT_1)
    marker_1 = _load_mapping(AMENDMENT_1_MARKER)
    if (
        amendment_1.get("status")
        != "REGISTERED_IMPLEMENTATION_ONLY_FIX_AFTER_ZERO_UPDATE_FAILURE"
        or amendment_1.get("registration_lock_sha256") != sha256_file(REGISTRATION)
        or amendment_1.get("v1_implementation_lock_sha256")
        != sha256_file(V1_IMPLEMENTATION_LOCK)
        or amendment_1.get("registered_runner_sha256") != registration.get("runner_sha256")
        or amendment_1.get("registered_module_sha256")
        != registration.get("translation_module_sha256")
        or amendment_1.get("amended_runner_sha256")
        != _load_mapping(V2_IMPLEMENTATION_LOCK).get("runner_sha256")
        or amendment_1.get("amended_module_sha256")
        != _load_mapping(V2_IMPLEMENTATION_LOCK).get("translation_module_sha256")
        or amendment_1.get("failed_attempt_args_sha256")
        != sha256_file(FAILED_ATTEMPT_1_ARGS)
        or amendment_1.get("optimizer_steps_before_failure") != 0
        or amendment_1.get("model_updates_before_failure") != 0
        or amendment_1.get("only_change")
        != "accept_list_or_tuple_for_Ultralytics_im_file_sequence_and_activate_v2_locks"
        or marker_1.get("amendment_sha256") != sha256_file(AMENDMENT_1)
    ):
        raise ALDIDirectBaselineError("ALDI amendment 1 fields changed")
    amendment_2 = _load_mapping(AMENDMENT_2)
    marker_2 = _load_mapping(AMENDMENT_2_MARKER)
    correction = _load_mapping(AMENDMENT_2_CORRECTION)
    correction_marker = _load_mapping(AMENDMENT_2_CORRECTION_MARKER)
    failed = amendment_2.get("failed_attempt")
    authorized = amendment_2.get("authorized_change")
    if not isinstance(failed, dict) or not isinstance(authorized, dict):
        raise ALDIDirectBaselineError("ALDI amendment 2 is incomplete")
    attempt_2_run = FAILED_ATTEMPT_2 / "run_native_uda/raw_endpoint/fit"
    attempt_2_report = FAILED_ATTEMPT_2 / "report_native_uda/train_diagnostics.json"
    for path, digest, label in (
        (attempt_2_run / "args.yaml", failed.get("args_sha256"), "attempt 2 args"),
        (attempt_2_run / "results.csv", failed.get("results_sha256"), "attempt 2 results"),
        (
            attempt_2_run / "weights/last.pt",
            failed.get("last_checkpoint_sha256"),
            "attempt 2 last checkpoint",
        ),
        (
            attempt_2_run / "weights/best.pt",
            failed.get("best_checkpoint_sha256"),
            "attempt 2 best checkpoint",
        ),
        (attempt_2_report, failed.get("diagnostics_sha256"), "attempt 2 diagnostics"),
    ):
        _assert_hash(path, digest, label=label)
    if (
        amendment_2.get("status")
        != "REGISTERED_IMPLEMENTATION_ONLY_FIX_AFTER_FINAL_VALIDATION_FAILURE"
        or amendment_2.get("registration_lock_sha256") != sha256_file(REGISTRATION)
        or amendment_2.get("v2_implementation_lock_sha256")
        != sha256_file(V2_IMPLEMENTATION_LOCK)
        or failed.get("attempt") != 2
        or failed.get("optimizer_updates_completed_before_failure")
        != EPOCHS * (TRAIN_IMAGES // BATCH)
        or failed.get("completed_checkpointed_epochs") != EPOCHS - 1
        or failed.get("validation_metrics_observed") is not False
        or authorized.get("scope") != "trainer_control_flow_only"
        or authorized.get("training_data_change") is not False
        or authorized.get("loss_change") is not False
        or authorized.get("optimizer_change") is not False
        or authorized.get("augmentation_change") is not False
        or authorized.get("schedule_change") is not False
        or authorized.get("checkpoint_endpoint_change") is not False
        or authorized.get("selection_change") is not False
        or marker_2.get("amendment_sha256") != sha256_file(AMENDMENT_2)
    ):
        raise ALDIDirectBaselineError("ALDI amendment 2 fields changed")
    if (
        correction.get("status") != "CORRECTION_REGISTERED_BEFORE_V3_IMPLEMENTATION_LOCK"
        or correction.get("amendment_2_sha256") != sha256_file(AMENDMENT_2)
        or correction.get("field")
        != "failed_attempt.optimizer_updates_completed_before_failure"
        or correction.get("original_value") != EPOCHS * (TRAIN_IMAGES // BATCH)
        or correction.get("correct_interpretation")
        != "training_batches_completed_before_failure"
        or correction.get("optimizer_update_count") != "not_claimed"
        or correction_marker.get("correction_sha256")
        != sha256_file(AMENDMENT_2_CORRECTION)
    ):
        raise ALDIDirectBaselineError("ALDI amendment 2 correction changed")
    amendment_3 = _load_mapping(AMENDMENT_3)
    marker_3 = _load_mapping(AMENDMENT_3_MARKER)
    failed_3 = amendment_3.get("failed_attempt")
    authorized_3 = amendment_3.get("authorized_change")
    if not isinstance(failed_3, dict) or not isinstance(authorized_3, dict):
        raise ALDIDirectBaselineError("ALDI amendment 3 is incomplete")
    attempt_3_fit = RUN_ROOT / "native_uda/raw_endpoint/fit"
    attempt_3_report = OUTPUT / "native_uda/train_diagnostics.json"
    for path, digest, label in (
        (attempt_3_fit / "args.yaml", failed_3.get("args_sha256"), "attempt 3 args"),
        (
            attempt_3_fit / "results.csv",
            failed_3.get("results_sha256"),
            "attempt 3 results",
        ),
        (
            attempt_3_fit / "weights/last.pt",
            failed_3.get("last_checkpoint_sha256"),
            "attempt 3 last checkpoint",
        ),
        (
            attempt_3_fit / "weights/best.pt",
            failed_3.get("best_checkpoint_sha256"),
            "attempt 3 best checkpoint",
        ),
        (attempt_3_report, failed_3.get("diagnostics_sha256"), "attempt 3 diagnostics"),
    ):
        _assert_hash(path, digest, label=label)
    observed_run_files = {
        path.relative_to(attempt_3_fit).as_posix()
        for path in attempt_3_fit.rglob("*")
        if path.is_file()
    }
    if observed_run_files != {
        "args.yaml",
        "results.csv",
        "weights/best.pt",
        "weights/last.pt",
    }:
        raise ALDIDirectBaselineError("ALDI attempt 3 run file set changed")
    diagnostic_3 = _load_mapping(attempt_3_report)
    checkpoint_3 = torch.load(
        attempt_3_fit / "weights/last.pt", map_location="cpu", weights_only=False
    )
    train_results_3 = checkpoint_3.get("train_results") if isinstance(checkpoint_3, dict) else None
    train_epochs_3 = train_results_3.get("epoch") if isinstance(train_results_3, dict) else None
    if (
        amendment_3.get("status")
        != "REGISTERED_RUNNER_ONLY_RECOVERY_AFTER_COMPLETE_ENDPOINT_SAVE"
        or amendment_3.get("registration_lock_sha256") != sha256_file(REGISTRATION)
        or amendment_3.get("v3_implementation_lock_sha256")
        != sha256_file(V3_IMPLEMENTATION_LOCK)
        or failed_3.get("attempt") != 3
        or failed_3.get("training_batches_completed") != EPOCHS * (TRAIN_IMAGES // BATCH)
        or failed_3.get("checkpoint_epoch_zero_based") != EPOCHS - 1
        or failed_3.get("checkpoint_train_result_epochs") != EPOCHS
        or failed_3.get("validation_metrics_observed") is not False
        or diagnostic_3.get("batches") != EPOCHS * (TRAIN_IMAGES // BATCH)
        or diagnostic_3.get("source_images") != SOURCE_IMAGES * EPOCHS
        or diagnostic_3.get("target_images") != TARGET_IMAGES * EPOCHS
        or diagnostic_3.get("implementation_lock_sha256")
        != sha256_file(V3_IMPLEMENTATION_LOCK)
        or not isinstance(checkpoint_3, dict)
        or checkpoint_3.get("epoch") != EPOCHS - 1
        or checkpoint_3.get("updates") != failed_3.get("checkpoint_updates")
        or not isinstance(train_epochs_3, list)
        or len(train_epochs_3) != EPOCHS
        or authorized_3.get("scope")
        != "runner_path_resolution_and_exact_artifact_recovery_only"
        or authorized_3.get("native_retraining") is not False
        or authorized_3.get("training_module_change") is not False
        or authorized_3.get("training_data_change") is not False
        or authorized_3.get("loss_change") is not False
        or authorized_3.get("selection_change") is not False
        or marker_3.get("amendment_sha256") != sha256_file(AMENDMENT_3)
    ):
        raise ALDIDirectBaselineError("ALDI amendment 3 or recoverable endpoint changed")
    return {
        "amendment_1": amendment_1,
        "amendment_2": amendment_2,
        "amendment_2_correction": correction,
        "amendment_3": amendment_3,
    }


def _implementation_lock() -> dict[str, Any]:
    registration = register()
    amendments = _validate_amendments(registration)
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise ALDIDirectBaselineError("ALDI implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("registration_sha256") != sha256_file(REGISTRATION)
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("translation_module_sha256") != sha256_file(MODULE)
            or lock.get("cuda_identity") != _cuda_identity()
            or marker.get("implementation_lock_sha256")
            != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise ALDIDirectBaselineError("ALDI implementation lock changed")
        return lock
    unexpected_runs = {path.name for path in RUN_ROOT.iterdir()} if RUN_ROOT.exists() else set()
    unexpected_outputs = {variant for variant in VARIANTS if (OUTPUT / variant).exists()}
    if not unexpected_runs.issubset({"native_uda"}) or not unexpected_outputs.issubset(
        {"native_uda"}
    ):
        raise ALDIDirectBaselineError("ALDI training output appeared before implementation lock")
    configure_ultralytics_environment(ROOT)
    import ultralytics

    lock = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_BASELINE_IMPLEMENTATION_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _utc_now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "amendment_1_sha256": sha256_file(AMENDMENT_1),
        "amendment_2_sha256": sha256_file(AMENDMENT_2),
        "amendment_2_correction_sha256": sha256_file(AMENDMENT_2_CORRECTION),
        "amendment_3_sha256": sha256_file(AMENDMENT_3),
        "v1_implementation_lock_sha256": sha256_file(V1_IMPLEMENTATION_LOCK),
        "v2_implementation_lock_sha256": sha256_file(V2_IMPLEMENTATION_LOCK),
        "v3_implementation_lock_sha256": sha256_file(V3_IMPLEMENTATION_LOCK),
        "protocol_sha256": registration["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "translation_module_sha256": sha256_file(MODULE),
        "official_commit": ALDI_COMMIT,
        "python": platform.python_version(),
        "ultralytics": str(ultralytics.__version__),
        "cuda_identity": _cuda_identity(),
        "variants": list(VARIANTS),
        "endpoint": "fixed_last_epoch_EMA_teacher",
        "implementation_change": amendments["amendment_3"]["authorized_change"]["change"],
        "final_epoch_validation": "disabled_label_blind_neutral_return",
        "final_eval": "disabled_no_op",
        "native_endpoint_recovery": "exact_hash_and_exposure_audited_in_place_without_retraining",
        "validation_metric_used_for_method_or_checkpoint_selection": False,
        "official_test_access": "prohibited",
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


def _variant_output(variant: str) -> Path:
    return OUTPUT / variant


def _variant_run_root(variant: str) -> Path:
    return RUN_ROOT / variant


def _training_lock_path(variant: str) -> Path:
    return _variant_output(variant) / "training_endpoint_lock.json"


def _training_marker_path(variant: str) -> Path:
    return _variant_output(variant) / "TRAINING_ENDPOINT_LOCKED"


def _diagnostics_path(variant: str) -> Path:
    return _variant_output(variant) / "train_diagnostics.json"


def _checkpoint_path(variant: str) -> Path:
    return _variant_run_root(variant) / f"{PAPER_LABELS[variant]}.pt"


def _checkpoint_lock_path(variant: str) -> Path:
    return _variant_output(variant) / "checkpoint_lock.json"


def _checkpoint_marker_path(variant: str) -> Path:
    return _variant_output(variant) / "CHECKPOINT_LOCKED"


def preflight() -> dict[str, Any]:
    implementation = _implementation_lock()
    manifest, source_paths = _manifest_contract()
    return {
        "status": "PASS_ALDI_DIRECT_BASELINE_PREFLIGHT",
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "registration_sha256": sha256_file(REGISTRATION),
        "official_commit": ALDI_COMMIT,
        "variants": list(VARIANTS),
        "manifest_contract": manifest,
        "source_path_count": len(source_paths),
        "cuda_identity": implementation["cuda_identity"],
        "official_test_access": "prohibited",
    }


def _validate_training_lock(variant: str) -> dict[str, Any]:
    _implementation_lock()
    lock_path = _training_lock_path(variant)
    marker_path = _training_marker_path(variant)
    if not lock_path.is_file() or not marker_path.is_file():
        raise ALDIDirectBaselineError(f"ALDI training lock is incomplete: {variant}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    if (
        lock.get("status") != "ALDI_DIRECT_BASELINE_RAW_ENDPOINT_LOCKED"
        or lock.get("variant") != variant
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("training_endpoint_lock_sha256") != sha256_file(lock_path)
    ):
        raise ALDIDirectBaselineError(f"ALDI training lock changed: {variant}")
    for key in ("last_checkpoint", "results", "args", "diagnostics"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=f"{variant} {key}")
    return lock


def train_variant(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ALDIDirectBaselineError(f"unknown ALDI variant: {variant}")
    preflight()
    lock_path = _training_lock_path(variant)
    marker_path = _training_marker_path(variant)
    if lock_path.exists() or marker_path.exists():
        return _validate_training_lock(variant)
    if _checkpoint_path(variant).exists() or _checkpoint_lock_path(variant).exists():
        raise ALDIDirectBaselineError("ALDI checkpoint appeared before raw endpoint lock")
    fit = _variant_run_root(variant) / "raw_endpoint/fit"
    recovery = variant == "native_uda" and fit.exists() and _variant_output(variant).exists()
    if (fit.exists() or _variant_output(variant).exists()) and not recovery:
        raise ALDIDirectBaselineError(
            f"incomplete ALDI output requires audit before retry: {variant}"
        )
    _, source_paths = _manifest_contract()
    diagnostics = _diagnostics_path(variant)
    execution_implementation_sha256 = (
        sha256_file(V3_IMPLEMENTATION_LOCK) if recovery else sha256_file(IMPLEMENTATION_LOCK)
    )
    if recovery:
        amendment_3 = _validate_amendments(_validate_registration())["amendment_3"]
        failed = amendment_3["failed_attempt"]
        if not isinstance(failed, dict):
            raise ALDIDirectBaselineError("ALDI native recovery evidence is incomplete")
        elapsed_seconds = float(failed["results_elapsed_seconds"])
        save_dir = fit
    else:
        configure_aldi_runtime(
            ALDIRuntimeConfig(
                variant=variant,  # type: ignore[arg-type]
                source_image_paths=source_paths,
                expected_source_images=SOURCE_IMAGES,
                diagnostics_path=diagnostics,
                protocol_sha256=sha256_file(PROTOCOL),
                registration_sha256=sha256_file(REGISTRATION),
                implementation_lock_sha256=execution_implementation_sha256,
                source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
                data_manifest_sha256=MANIFEST_SHA256,
                seed=SEED,
            )
        )
        configure_ultralytics_environment(ROOT)
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except (ImportError, OSError, PermissionError) as exc:
            raise ALDIDirectBaselineError(f"cannot import Ultralytics: {exc}") from exc
        model = YOLO(str(SOURCE_CHECKPOINT))
        started = time.perf_counter()
        model.train(
            trainer=ALDITranslationDetectionTrainer,
            data=str(DATASET.resolve()),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device="0",
            workers=4,
            project=str((_variant_run_root(variant) / "raw_endpoint").resolve()),
            name="fit",
            exist_ok=False,
            pretrained=True,
            freeze=None,
            val=False,
            optimizer="SGD",
            lr0=0.00075,
            lrf=0.10,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            translate=0.1,
            scale=0.3,
            close_mosaic=0,
            patience=EPOCHS,
            amp=True,
            seed=SEED,
            deterministic=True,
            max_det=500,
            cache=False,
            plots=False,
            verbose=True,
            save=True,
        )
        elapsed_seconds = time.perf_counter() - started
        save_dir = fit
    last = save_dir / "weights/last.pt"
    results_path = save_dir / "results.csv"
    args_path = save_dir / "args.yaml"
    for path in (last, results_path, args_path, diagnostics):
        if not path.is_file():
            raise ALDIDirectBaselineError(f"ALDI training output is incomplete: {path}")
    diagnostic = _load_mapping(diagnostics)
    expected_source_exposures = SOURCE_IMAGES * EPOCHS
    expected_target_exposures = TARGET_IMAGES * EPOCHS
    if (
        diagnostic.get("variant") != variant
        or diagnostic.get("source_images") != expected_source_exposures
        or diagnostic.get("target_images") != expected_target_exposures
        or diagnostic.get("implementation_lock_sha256")
        != execution_implementation_sha256
        or diagnostic.get("target_ground_truth_used_for_training")
        is not (variant == "equal_supervision")
    ):
        raise ALDIDirectBaselineError(f"ALDI training exposure audit failed: {variant}")
    payload = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_BASELINE_RAW_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "variant": variant,
        "paper_label": PAPER_LABELS[variant],
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "training_execution_implementation_lock_sha256": execution_implementation_sha256,
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_path),
        "results_sha256": sha256_file(results_path),
        "args": _relative(args_path),
        "args_sha256": sha256_file(args_path),
        "diagnostics": _relative(diagnostics),
        "diagnostics_sha256": sha256_file(diagnostics),
        "elapsed_seconds": elapsed_seconds,
        "recovered_complete_endpoint_without_retraining": recovery,
        "epochs": EPOCHS,
        "images_per_epoch": TRAIN_IMAGES,
        "source_exposures": expected_source_exposures,
        "target_exposures": expected_target_exposures,
        "target_ground_truth_used_for_training": variant == "equal_supervision",
        "checkpoint_selected_by_metric": False,
        "endpoint": "fixed_last_epoch_EMA_teacher",
        "official_validation_metric_used": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(lock_path, payload)
    atomic_write_json(
        marker_path,
        {
            "status": payload["status"],
            "training_endpoint_lock_sha256": sha256_file(lock_path),
        },
    )
    return payload


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ALDIDirectBaselineError(f"unsupported checkpoint payload: {path}")
    model = value.get("ema") or value.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ALDIDirectBaselineError(f"checkpoint has no model module: {path}")
    return value, model


def _build_checkpoint(variant: str, raw: Mapping[str, Any]) -> None:
    source_payload, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_teacher = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    source_state = source_model.state_dict()
    trained_state = trained_teacher.state_dict()
    if tuple(source_state) != tuple(trained_state):
        raise ALDIDirectBaselineError("ALDI source and teacher state schemas differ")
    output_model = copy.deepcopy(source_model)
    incompatible = output_model.load_state_dict(trained_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ALDIDirectBaselineError("ALDI teacher could not load into source architecture")
    output_model = output_model.half().eval()
    payload = copy.deepcopy(source_payload)
    payload.update(
        {
            "epoch": -1,
            "best_fitness": None,
            "model": None,
            "ema": output_model,
            "optimizer": None,
            "scaler": None,
            "updates": None,
            "date": _utc_now(),
            "train_metrics": {},
            "train_results": {},
            "cvbra_v1_recent_daod_baseline": {
                "paper_label": PAPER_LABELS[variant],
                "variant": variant,
                "method_source": "ALDI++ TMLR Featured 2025",
                "official_author_implementation": False,
                "translation": "registered_anchor_free_YOLO11n_translation",
                "target_ground_truth_used_for_training": variant == "equal_supervision",
                "all_layers_trainable": True,
                "epochs": EPOCHS,
                "endpoint": "fixed_last_epoch_EMA_teacher",
                "protocol_sha256": sha256_file(PROTOCOL),
                "registration_sha256": sha256_file(REGISTRATION),
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "training_execution_implementation_lock_sha256": raw[
                    "training_execution_implementation_lock_sha256"
                ],
                "training_endpoint_lock_sha256": sha256_file(_training_lock_path(variant)),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
                "metric_selected_epoch_or_checkpoint": False,
            },
        }
    )
    checkpoint = _checkpoint_path(variant)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)


def _verify_checkpoint(variant: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    _, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, raw_teacher = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    payload, output_model = _load_checkpoint(_checkpoint_path(variant))
    metadata = payload.get("cvbra_v1_recent_daod_baseline")
    if not isinstance(metadata, dict) or metadata.get("variant") != variant:
        raise ALDIDirectBaselineError("ALDI final checkpoint metadata is missing")
    source_state = source_model.state_dict()
    expected = raw_teacher.state_dict()
    observed = output_model.state_dict()
    if tuple(source_state) != tuple(expected) or tuple(expected) != tuple(observed):
        raise ALDIDirectBaselineError("ALDI final checkpoint state schema changed")
    changed = 0
    maximum_error = 0.0
    for name, expected_value in expected.items():
        observed_value = observed[name].to(dtype=expected_value.dtype)
        if expected_value.is_floating_point():
            maximum_error = max(
                maximum_error,
                float((observed_value.float() - expected_value.float()).abs().max()),
            )
            if not torch.equal(observed_value, source_state[name]):
                changed += 1
        elif not torch.equal(observed_value, expected_value):
            raise ALDIDirectBaselineError(f"ALDI nonfloating state differs: {name}")
    if maximum_error != 0.0 or changed == 0:
        raise ALDIDirectBaselineError(
            f"ALDI checkpoint verification failed: error={maximum_error}, changed={changed}"
        )
    return {
        "state_entries": len(observed),
        "changed_floating_states": changed,
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
        "class_names": getattr(output_model, "names", None),
        "metadata": metadata,
    }


def _validate_checkpoint_lock(variant: str) -> dict[str, Any]:
    raw = _validate_training_lock(variant)
    checkpoint = _checkpoint_path(variant)
    lock_path = _checkpoint_lock_path(variant)
    marker_path = _checkpoint_marker_path(variant)
    if not lock_path.is_file() or not marker_path.is_file():
        raise ALDIDirectBaselineError(f"ALDI checkpoint lock is incomplete: {variant}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    if (
        lock.get("status") != "ALDI_DIRECT_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED"
        or lock.get("variant") != variant
        or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        or lock.get("training_endpoint_lock_sha256") != sha256_file(_training_lock_path(variant))
        or marker.get("checkpoint_lock_sha256") != sha256_file(lock_path)
    ):
        raise ALDIDirectBaselineError(f"ALDI checkpoint lock changed: {variant}")
    _verify_checkpoint(variant, raw)
    return lock


def build_checkpoint(variant: str) -> dict[str, Any]:
    raw = train_variant(variant)
    checkpoint = _checkpoint_path(variant)
    lock_path = _checkpoint_lock_path(variant)
    marker_path = _checkpoint_marker_path(variant)
    if lock_path.exists() or marker_path.exists():
        return _validate_checkpoint_lock(variant)
    if checkpoint.exists():
        raise ALDIDirectBaselineError("unlocked ALDI checkpoint exists")
    _build_checkpoint(variant, raw)
    verification = _verify_checkpoint(variant, raw)
    payload = {
        "schema_version": 1,
        "status": "ALDI_DIRECT_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "variant": variant,
        "paper_label": PAPER_LABELS[variant],
        "protocol_sha256": sha256_file(PROTOCOL),
        "registration_sha256": sha256_file(REGISTRATION),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "training_endpoint_lock_sha256": sha256_file(_training_lock_path(variant)),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "verification": verification,
        "target_ground_truth_used_for_training": variant == "equal_supervision",
        "validation_metric_used_for_training_or_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(lock_path, payload)
    atomic_write_json(
        marker_path,
        {
            "status": payload["status"],
            "checkpoint_lock_sha256": sha256_file(lock_path),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered recent ALDI++ direct-comparison translations"
    )
    parser.add_argument(
        "--stage",
        choices=("register", "preflight", "train", "build-checkpoint", "all"),
        default="preflight",
    )
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "preflight":
        result = preflight()
    else:
        if args.variant is None:
            parser.error("--variant is required for training or checkpoint stages")
        if args.stage == "train":
            result = train_variant(str(args.variant))
        else:
            result = build_checkpoint(str(args.variant))
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"official_files", "protocol_snapshot", "verification"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
