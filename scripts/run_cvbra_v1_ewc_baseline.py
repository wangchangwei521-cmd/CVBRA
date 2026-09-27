from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from buse_uav.detectors.ewc_ultralytics import (
    EWCDetectionTrainer,
    EWCRuntimeConfig,
    configure_ewc_runtime,
)
from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_quality_upgrade_v2.yaml"
PROTOCOL_SHA256 = "14ce7f1edec4248080e545c5a4832a7f20959157c7ef8f84ae692f758aa980d7"
REGISTRATION = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/REGISTRATION_LOCK.json"
REGISTRATION_SHA256 = "1b218d9a585d82d01222be7990f749a6aa615887a8080e7b17f7994a26063dd7"
REGISTRATION_MARKER = REGISTRATION.parent / "REGISTERED"
REGISTRATION_MARKER_SHA256 = "6ef505f3a5431ba9a84886f5cb951e78eb9e3be3080a391aad27fe4c508b8343"

SOURCE_CHECKPOINT = ROOT / "weights/hazydet/yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
DATASET = ROOT / "data/processed/cvbra_v1/dataset.yaml"
DATASET_SHA256 = "567ab0bb75434e3c735f8bda6b2aaae2b1c81a4ed6043c215c87900ec9c039d7"
MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
DATA_LOCK_SOURCE = ROOT / "reports/development/cvbra_v1/generated_dataset_lock.json"
DATA_LOCK_SOURCE_SHA256 = "32ebc263674ef9b4fd123642e43b978cebcb04ec53f4eb9eef12c7e624fcbe4d"
EWC_MODULE = ROOT / "src/buse_uav/detectors/ewc_ultralytics.py"

OUTPUT = ROOT / "reports/development/cvbra_v1_quality_upgrade_v2/CVBRA_EWC"
RUN_ROOT = ROOT / "runs/cvbra_v1_quality_upgrade_v2/CVBRA_EWC"
FISHER_DATA_ROOT = ROOT / "data/processed/cvbra_v1_quality_upgrade_v2/CVBRA_EWC"
SOURCE_LIST = FISHER_DATA_ROOT / "source_replay_images.txt"
FISHER_DATA_LOCK = OUTPUT / "fisher_data_lock.json"
FISHER_DATA_MARKER = OUTPUT / "FISHER_DATA_LOCKED"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
FISHER_TENSOR = OUTPUT / "empirical_fisher.pt"
FISHER_REPORT = OUTPUT / "empirical_fisher_report.json"
FISHER_MARKER = OUTPUT / "FISHER_LOCKED"
TRAINING_LOCK = OUTPUT / "training_endpoint_lock.json"
TRAINING_MARKER = OUTPUT / "TRAINING_ENDPOINT_LOCKED"
CHECKPOINT = RUN_ROOT / "CVBRA_EWC.pt"
CHECKPOINT_LOCK = OUTPUT / "checkpoint_lock.json"
CHECKPOINT_MARKER = OUTPUT / "CHECKPOINT_LOCKED"

EXPECTED_STATUS = "REGISTERED_BEFORE_EWC_TRAINING_PREDICTION_OR_NEW_DERIVED_ANALYSIS"
SOURCE_IMAGES = 900
TRAIN_IMAGES = 3600
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
REFERENCE_DISPLACEMENT_FRACTION = 0.01
REFERENCE_LOSS_FRACTION = 0.05


class EWCBaselineError(RuntimeError):
    """Raised when the registered EWC retention baseline cannot fail closed."""


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
        raise EWCBaselineError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EWCBaselineError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise EWCBaselineError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise EWCBaselineError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _cuda_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise EWCBaselineError("CVBRA_EWC training requires CUDA")
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


def _validate_registration() -> dict[str, Any]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "quality-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "quality-upgrade registration"),
        (REGISTRATION_MARKER, REGISTRATION_MARKER_SHA256, "registration marker"),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "source checkpoint"),
        (DATASET, DATASET_SHA256, "CVBRA dataset YAML"),
        (MANIFEST, MANIFEST_SHA256, "CVBRA dataset manifest"),
        (DATA_LOCK_SOURCE, DATA_LOCK_SOURCE_SHA256, "CVBRA data lock"),
    ):
        _assert_hash(path, digest, label=label)
    protocol = _load_mapping(PROTOCOL)
    registration = _load_mapping(REGISTRATION)
    baseline = protocol.get("retention_baseline")
    boundary = protocol.get("evidence_boundary")
    state = registration.get("state_at_registration")
    if not all(isinstance(value, dict) for value in (baseline, boundary, state)):
        raise EWCBaselineError("quality-upgrade registration is incomplete")
    assert isinstance(baseline, dict)
    assert isinstance(boundary, dict)
    assert isinstance(state, dict)
    training = baseline.get("training")
    fisher = baseline.get("Fisher")
    strength = baseline.get("strength_calibration")
    if not all(isinstance(value, dict) for value in (training, fisher, strength)):
        raise EWCBaselineError("registered EWC contract is incomplete")
    assert isinstance(training, dict)
    assert isinstance(fisher, dict)
    assert isinstance(strength, dict)
    if (
        protocol.get("status") != EXPECTED_STATUS
        or registration.get("protocol_sha256") != PROTOCOL_SHA256
        or registration.get("retention_baseline") != "CVBRA_EWC"
        or baseline.get("name") != "CVBRA_EWC"
        or baseline.get("training_multiset") != "exact_CVBRA_v1_dataset"
        or baseline.get("all_layers_trainable") is not True
        or training.get("epochs") != EPOCHS
        or training.get("imgsz") != IMGSZ
        or training.get("batch") != BATCH
        or training.get("validation_during_training") is not False
        or fisher.get("samples") != "exact_900_fixed_source_replay_aliases"
        or strength.get("reference_displacement")
        != "one_percent_of_each_parameter_tensors_source_RMS"
        or strength.get("reference_penalty_fraction_of_mean_source_detection_loss")
        != REFERENCE_LOSS_FRACTION
        or boundary.get("UAV_OBB_official_test_access") != "prohibited"
        or any(bool(state.get(key)) for key in state if key != "UAV_OBB_official_test_accessed")
        or state.get("UAV_OBB_official_test_accessed") is not False
    ):
        raise EWCBaselineError("registered EWC fields changed")
    return protocol


def _source_entries() -> list[dict[str, Any]]:
    manifest = _load_mapping(MANIFEST)
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != TRAIN_IMAGES:
        raise EWCBaselineError("CVBRA manifest entry count changed")
    entries = [
        dict(row)
        for row in raw_entries
        if isinstance(row, dict) and row.get("role") == "source_haze_replay"
    ]
    if len(entries) != SOURCE_IMAGES:
        raise EWCBaselineError("CVBRA source replay coverage changed")
    return entries


def _validate_fisher_data_lock() -> dict[str, Any]:
    if not FISHER_DATA_LOCK.is_file() or not FISHER_DATA_MARKER.is_file():
        raise EWCBaselineError("EWC Fisher data lock is incomplete")
    lock = _load_mapping(FISHER_DATA_LOCK)
    marker = _load_mapping(FISHER_DATA_MARKER)
    if (
        lock.get("status") != "CVBRA_EWC_FISHER_SOURCE_DATA_LOCKED"
        or lock.get("source_list_sha256") != sha256_file(SOURCE_LIST)
        or marker.get("fisher_data_lock_sha256") != sha256_file(FISHER_DATA_LOCK)
    ):
        raise EWCBaselineError("EWC Fisher data lock changed")
    return lock


def build_fisher_data() -> dict[str, Any]:
    _validate_registration()
    if FISHER_DATA_LOCK.exists() or FISHER_DATA_MARKER.exists():
        return _validate_fisher_data_lock()
    if any(
        path.exists()
        for path in (
            IMPLEMENTATION_LOCK,
            FISHER_TENSOR,
            FISHER_REPORT,
            TRAINING_LOCK,
            CHECKPOINT,
            CHECKPOINT_LOCK,
        )
    ):
        raise EWCBaselineError("EWC output appeared before Fisher data lock")
    entries = _source_entries()
    paths: list[str] = []
    for row in entries:
        image = _rooted(row["image"])
        label = _rooted(row["label"])
        _assert_hash(image, row["image_sha256"], label="source replay image")
        _assert_hash(label, row["label_sha256"], label="source replay label")
        paths.append(str(image.resolve()))
    SOURCE_LIST.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(SOURCE_LIST, "\n".join(paths) + "\n")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_FISHER_SOURCE_DATA_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "source_manifest": _relative(MANIFEST),
        "source_manifest_sha256": MANIFEST_SHA256,
        "source_list": _relative(SOURCE_LIST),
        "source_list_sha256": sha256_file(SOURCE_LIST),
        "source_images": len(paths),
        "manifest_order_preserved": True,
        "validation_or_test_content_used": False,
    }
    atomic_write_json(FISHER_DATA_LOCK, payload)
    atomic_write_json(
        FISHER_DATA_MARKER,
        {"status": payload["status"], "fisher_data_lock_sha256": sha256_file(FISHER_DATA_LOCK)},
    )
    return payload


def _model_contract() -> dict[str, Any]:
    _, model = _load_checkpoint(SOURCE_CHECKPOINT)
    return {
        "state_entries": len(model.state_dict()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_names": {
            str(key): str(value) for key, value in dict(getattr(model, "names", {})).items()
        },
    }


def _validate_implementation_lock() -> dict[str, Any]:
    _validate_fisher_data_lock()
    if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
        raise EWCBaselineError("EWC implementation lock is incomplete")
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    marker = _load_mapping(IMPLEMENTATION_MARKER)
    if (
        lock.get("status") != "CVBRA_EWC_IMPLEMENTATION_LOCKED_BEFORE_FISHER_OR_TRAINING"
        or lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("ewc_module_sha256") != sha256_file(EWC_MODULE)
        or lock.get("fisher_data_lock_sha256") != sha256_file(FISHER_DATA_LOCK)
        or lock.get("model_contract") != _model_contract()
        or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise EWCBaselineError("EWC implementation lock changed")
    return lock


def implementation_lock() -> dict[str, Any]:
    build_fisher_data()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        return _validate_implementation_lock()
    if any(
        path.exists()
        for path in (
            FISHER_TENSOR,
            FISHER_REPORT,
            FISHER_MARKER,
            TRAINING_LOCK,
            TRAINING_MARKER,
            CHECKPOINT,
            CHECKPOINT_LOCK,
        )
    ):
        raise EWCBaselineError("EWC Fisher or training output appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_IMPLEMENTATION_LOCKED_BEFORE_FISHER_OR_TRAINING",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration": _relative(REGISTRATION),
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "ewc_module": _relative(EWC_MODULE),
        "ewc_module_sha256": sha256_file(EWC_MODULE),
        "fisher_data_lock_sha256": sha256_file(FISHER_DATA_LOCK),
        "model_contract": _model_contract(),
        "cuda_identity": _cuda_identity(),
        "all_layers_trainable": True,
        "Fisher_images": SOURCE_IMAGES,
        "reference_displacement_fraction": REFERENCE_DISPLACEMENT_FRACTION,
        "reference_loss_fraction": REFERENCE_LOSS_FRACTION,
        "validation_metric_feedback_allowed": False,
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
    lock = implementation_lock()
    return {
        "status": "PASS_CVBRA_EWC_TRAINING_PREFLIGHT",
        "source_Fisher_images": SOURCE_IMAGES,
        "training_images_per_epoch": TRAIN_IMAGES,
        "epochs": EPOCHS,
        "all_layers_trainable": True,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "validation_metric_feedback_allowed": False,
        "official_test_access": "prohibited",
    }


def _validate_fisher_lock() -> dict[str, Any]:
    for path in (FISHER_TENSOR, FISHER_REPORT, FISHER_MARKER):
        if not path.is_file():
            raise EWCBaselineError(f"EWC Fisher artifact is missing: {path}")
    report = _load_mapping(FISHER_REPORT)
    marker = _load_mapping(FISHER_MARKER)
    if (
        report.get("status") != "CVBRA_EWC_EMPIRICAL_FISHER_LOCKED_BEFORE_OPTIMIZATION"
        or report.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or report.get("fisher_tensor_sha256") != sha256_file(FISHER_TENSOR)
        or marker.get("fisher_report_sha256") != sha256_file(FISHER_REPORT)
        or report.get("source_images") != SOURCE_IMAGES
        or report.get("validation_metric_used_for_Fisher_or_strength") is not False
    ):
        raise EWCBaselineError("EWC Fisher lock changed")
    return report


def _validate_training_lock() -> dict[str, Any]:
    _validate_implementation_lock()
    _validate_fisher_lock()
    if not TRAINING_LOCK.is_file() or not TRAINING_MARKER.is_file():
        raise EWCBaselineError("EWC training endpoint lock is incomplete")
    lock = _load_mapping(TRAINING_LOCK)
    marker = _load_mapping(TRAINING_MARKER)
    if (
        lock.get("status") != "CVBRA_EWC_RAW_TRAINING_ENDPOINT_LOCKED"
        or lock.get("fisher_report_sha256") != sha256_file(FISHER_REPORT)
        or marker.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
    ):
        raise EWCBaselineError("EWC training endpoint lock changed")
    for key in ("last_checkpoint", "results", "args"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=f"EWC {key}")
    return lock


def train_raw_endpoint() -> dict[str, Any]:
    preflight()
    if TRAINING_LOCK.exists() or TRAINING_MARKER.exists():
        return _validate_training_lock()
    if CHECKPOINT.exists() or CHECKPOINT_LOCK.exists():
        raise EWCBaselineError("EWC checkpoint appeared before raw endpoint lock")
    if any(path.exists() for path in (FISHER_TENSOR, FISHER_REPORT, FISHER_MARKER)):
        raise EWCBaselineError("partial EWC Fisher output requires audit")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO
    except (ImportError, OSError, PermissionError) as exc:
        raise EWCBaselineError(f"cannot import Ultralytics: {exc}") from exc
    fit = RUN_ROOT / "raw_endpoint" / "fit"
    if fit.exists():
        raise EWCBaselineError(f"incomplete EWC training directory requires audit: {fit}")
    configure_ewc_runtime(
        EWCRuntimeConfig(
            source_image_list=SOURCE_LIST,
            expected_source_images=SOURCE_IMAGES,
            fisher_tensor_path=FISHER_TENSOR,
            fisher_report_path=FISHER_REPORT,
            fisher_marker_path=FISHER_MARKER,
            protocol_sha256=PROTOCOL_SHA256,
            registration_sha256=REGISTRATION_SHA256,
            implementation_lock_sha256=sha256_file(IMPLEMENTATION_LOCK),
            source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
            data_manifest_sha256=MANIFEST_SHA256,
            reference_displacement_fraction=REFERENCE_DISPLACEMENT_FRACTION,
            reference_loss_fraction=REFERENCE_LOSS_FRACTION,
        )
    )
    model = YOLO(str(SOURCE_CHECKPOINT))
    results = model.train(
        trainer=EWCDetectionTrainer,
        data=str(DATASET.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str((RUN_ROOT / "raw_endpoint").resolve()),
        name="fit",
        exist_ok=False,
        pretrained=True,
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
    save_dir = Path(results.save_dir)
    last = save_dir / "weights/last.pt"
    results_path = save_dir / "results.csv"
    args_path = save_dir / "args.yaml"
    for path in (last, results_path, args_path):
        if not path.is_file():
            raise EWCBaselineError(f"EWC training output is incomplete: {path}")
    _validate_fisher_lock()
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_RAW_TRAINING_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "fisher_report_sha256": sha256_file(FISHER_REPORT),
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_path),
        "results_sha256": sha256_file(results_path),
        "args": _relative(args_path),
        "args_sha256": sha256_file(args_path),
        "epochs": EPOCHS,
        "all_layers_trainable": True,
        "checkpoint_selected_by_metric": False,
        "external_validation_during_training": False,
        "official_validation_metric_used": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(TRAINING_LOCK, payload)
    atomic_write_json(
        TRAINING_MARKER,
        {"status": payload["status"], "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK)},
    )
    return payload


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise EWCBaselineError(f"unsupported checkpoint payload: {path}")
    model = value.get("ema") or value.get("model")
    if not isinstance(model, torch.nn.Module):
        raise EWCBaselineError(f"checkpoint has no model module: {path}")
    return value, model


def _fisher_penalty(
    source: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
) -> float:
    value = torch.load(FISHER_TENSOR, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not isinstance(value.get("fisher"), dict):
        raise EWCBaselineError("EWC Fisher tensor payload is invalid")
    fisher = value["fisher"]
    weighted = 0.0
    fisher_sum = 0.0
    for name, importance in fisher.items():
        if not isinstance(name, str) or not isinstance(importance, torch.Tensor):
            raise EWCBaselineError("EWC Fisher tensor row is invalid")
        weighted += float(
            (importance.double() * (trained[name].double() - source[name].double()).square()).sum()
        )
        fisher_sum += float(importance.double().sum())
    if fisher_sum <= 0.0:
        raise EWCBaselineError("EWC Fisher tensor has no positive mass")
    return weighted / fisher_sum


def _build_checkpoint(raw: Mapping[str, Any]) -> None:
    source_payload, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    source_state = source_model.state_dict()
    trained_state = trained_model.state_dict()
    if tuple(source_state) != tuple(trained_state):
        raise EWCBaselineError("source and EWC state schemas differ")
    output_model = copy.deepcopy(source_model)
    incompatible = output_model.load_state_dict(trained_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise EWCBaselineError("EWC final state could not load into the source architecture")
    output_model = output_model.half().eval()
    fisher_report = _load_mapping(FISHER_REPORT)
    final_penalty = _fisher_penalty(source_state, trained_state)
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
            "cvbra_v1_quality_retention_baseline": {
                "baseline": "CVBRA_EWC",
                "method_family": "diagonal empirical-Fisher elastic weight consolidation",
                "all_layers_trainable": True,
                "epochs": EPOCHS,
                "protocol_sha256": PROTOCOL_SHA256,
                "registration_sha256": REGISTRATION_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "fisher_report_sha256": sha256_file(FISHER_REPORT),
                "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
                "ewc_coefficient": fisher_report["ewc_coefficient"],
                "final_Fisher_weighted_mean_squared_displacement": final_penalty,
                "metric_selected_epoch_or_checkpoint": False,
            },
        }
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(CHECKPOINT)


def _verify_checkpoint(raw: Mapping[str, Any]) -> dict[str, Any]:
    _, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    checkpoint, output_model = _load_checkpoint(CHECKPOINT)
    metadata = checkpoint.get("cvbra_v1_quality_retention_baseline")
    if not isinstance(metadata, dict) or metadata.get("baseline") != "CVBRA_EWC":
        raise EWCBaselineError("EWC checkpoint metadata is missing")
    expected = trained_model.state_dict()
    observed = output_model.state_dict()
    source = source_model.state_dict()
    if tuple(expected) != tuple(observed) or tuple(source) != tuple(observed):
        raise EWCBaselineError("EWC checkpoint state schema changed")
    changed_states = 0
    maximum_error = 0.0
    for name, expected_value in expected.items():
        observed_value = observed[name].to(dtype=expected_value.dtype)
        if expected_value.is_floating_point():
            maximum_error = max(
                maximum_error,
                float((observed_value.float() - expected_value.float()).abs().max()),
            )
            if not torch.equal(observed_value, source[name]):
                changed_states += 1
        elif not torch.equal(observed_value, expected_value):
            raise EWCBaselineError(f"nonfloating EWC state differs: {name}")
    if maximum_error != 0.0 or changed_states == 0:
        raise EWCBaselineError(
            f"EWC checkpoint verification failed: error={maximum_error}, changed={changed_states}"
        )
    return {
        "state_entries": len(observed),
        "changed_floating_states": changed_states,
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
        "final_Fisher_weighted_mean_squared_displacement": _fisher_penalty(source, observed),
        "metadata": metadata,
    }


def _validate_checkpoint_lock() -> dict[str, Any]:
    raw = _validate_training_lock()
    if not CHECKPOINT_LOCK.is_file() or not CHECKPOINT_MARKER.is_file():
        raise EWCBaselineError("EWC checkpoint lock is incomplete")
    lock = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    if (
        lock.get("status") != "CVBRA_EWC_CHECKPOINT_VERIFIED_AND_LOCKED"
        or lock.get("checkpoint_sha256") != sha256_file(CHECKPOINT)
        or lock.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
    ):
        raise EWCBaselineError("EWC checkpoint lock changed")
    _verify_checkpoint(raw)
    return lock


def build_checkpoint() -> dict[str, Any]:
    raw = train_raw_endpoint()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT_MARKER.exists():
        return _validate_checkpoint_lock()
    if CHECKPOINT.exists():
        raise EWCBaselineError("unlocked EWC checkpoint exists")
    _build_checkpoint(raw)
    verification = _verify_checkpoint(raw)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_EWC_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "baseline": "CVBRA_EWC",
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "fisher_report_sha256": sha256_file(FISHER_REPORT),
        "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(CHECKPOINT_LOCK, payload)
    atomic_write_json(
        CHECKPOINT_MARKER,
        {"status": payload["status"], "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the registered CVBRA_EWC baseline")
    parser.add_argument(
        "--stage",
        choices=("preflight", "build-data", "train", "build-checkpoint", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "build-data":
        result = build_fisher_data()
    elif args.stage == "train":
        result = train_raw_endpoint()
    else:
        result = build_checkpoint()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
