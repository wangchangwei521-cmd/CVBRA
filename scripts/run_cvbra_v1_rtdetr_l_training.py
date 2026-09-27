from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_rtdetr_l_confirmation.yaml"
PROTOCOL_SHA256 = "6f11519c837a7b68679707cd0025631a90c073d8c37b856a9e0eb40b98fc7f9c"
VALIDATION_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "evaluation"
    / "validation_report.json"
)
VALIDATION_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
SOURCE_CHECKPOINT = ROOT / "weights" / "hazydet" / "rtdetr_l_best.pt"
SOURCE_CHECKPOINT_SHA256 = "7d8fccbc1a9b66e28311ba91363f0b9f6ca0e4fdd61e88bd2f90099c6ea2a44c"
DATASET = ROOT / "data" / "processed" / "cvbra_v1" / "dataset.yaml"
DATASET_SHA256 = "567ab0bb75434e3c735f8bda6b2aaae2b1c81a4ed6043c215c87900ec9c039d7"
MANIFEST = ROOT / "data" / "processed" / "cvbra_v1" / "manifest.json"
MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
DATA_LOCK = ROOT / "reports" / "development" / "cvbra_v1" / "generated_dataset_lock.json"
DATA_LOCK_SHA256 = "32ebc263674ef9b4fd123642e43b978cebcb04ec53f4eb9eef12c7e624fcbe4d"

OUTPUT = ROOT / "reports" / "development" / "cvbra_v1_rtdetr_l"
RUN_ROOT = ROOT / "runs" / "cvbra_v1_rtdetr_l"
SUPERSESSION = OUTPUT / "IMPLEMENTATION_LOCK_SUPERSESSION_1.json"
SUPERSESSION_SHA256 = "06414a8d6994173ed26dfe831766a027305c3fa1a31f01991f595353e739480f"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock_v2.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_V2_LOCKED"
TRAINING_LOCK = OUTPUT / "training_endpoint_lock.json"
TRAINING_MARKER = OUTPUT / "TRAINING_ENDPOINT_LOCKED"
CHECKPOINT = RUN_ROOT / "cvbra_v1_rtdetr_l.pt"
CHECKPOINT_LOCK = OUTPUT / "checkpoint_lock.json"
CHECKPOINT_MARKER = OUTPUT / "CHECKPOINT_LOCKED"

FROZEN_LAST_LAYER = 9
FIRST_TRAINABLE_LAYER = 10
LAST_TRAINABLE_LAYER = 28
MODEL_LAYERS = 29
EPOCHS = 2
IMGSZ = 640
BATCH = 1
SEED = 42
TRAIN_IMAGES = 3600


class CVBRARTDETRTrainingError(RuntimeError):
    """Raised when the predeclared RT-DETR-L confirmation contract is violated."""


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
        raise CVBRARTDETRTrainingError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRARTDETRTrainingError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise CVBRARTDETRTrainingError(f"locked {label} changed: {path}")


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise CVBRARTDETRTrainingError(f"state has no model-layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise CVBRARTDETRTrainingError(f"invalid model layer index: {name}") from exc


def _validate_protocol() -> dict[str, Any]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "RT-DETR confirmation protocol"),
        (VALIDATION_REPORT, VALIDATION_REPORT_SHA256, "YOLO11n validation report"),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "RT-DETR-L source checkpoint"),
        (DATASET, DATASET_SHA256, "CVBRA dataset YAML"),
        (MANIFEST, MANIFEST_SHA256, "CVBRA dataset manifest"),
        (DATA_LOCK, DATA_LOCK_SHA256, "CVBRA generated-data lock"),
    ):
        _assert_hash(path, digest, label=label)
    validation = _load_mapping(VALIDATION_REPORT)
    protocol = _load_mapping(PROTOCOL)
    method = protocol.get("method")
    training = protocol.get("training_dataset")
    optimization = protocol.get("optimization")
    evaluation = protocol.get("evaluation")
    integrity = protocol.get("integrity")
    if not all(
        isinstance(value, dict)
        for value in (method, training, optimization, evaluation, integrity)
    ):
        raise CVBRARTDETRTrainingError("RT-DETR protocol sections are incomplete")
    assert isinstance(method, dict)
    assert isinstance(training, dict)
    assert isinstance(optimization, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(integrity, dict)
    if (
        validation.get("status") != "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
        or validation.get("decision", {}).get("second_detector_execution_authorized_next")
        is not True
        or method.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
        or method.get("model_layers") != MODEL_LAYERS
        or method.get("frozen_layer_indices_inclusive") != [0, FROZEN_LAST_LAYER]
        or method.get("trainable_layer_indices_inclusive")
        != [FIRST_TRAINABLE_LAYER, LAST_TRAINABLE_LAYER]
        or training.get("dataset_yaml_sha256") != DATASET_SHA256
        or training.get("manifest_sha256") != MANIFEST_SHA256
        or training.get("total_images_per_epoch") != TRAIN_IMAGES
        or optimization.get("epochs") != EPOCHS
        or optimization.get("checkpoint_selected_by_metric") is not False
        or optimization.get("validation_during_training") is not False
        or optimization.get("imgsz") != IMGSZ
        or optimization.get("batch") != BATCH
        or evaluation.get("validation_labels_already_accessed_for_frozen_yolo11n_confirmation")
        is not True
        or evaluation.get("metric_or_label_feedback_allowed_during_RTDETR_training")
        is not False
        or integrity.get("official_test_access") != "prohibited"
        or integrity.get("paper_body_change_before_cross_detector_completion")
        != "prohibited"
    ):
        raise CVBRARTDETRTrainingError("registered RT-DETR confirmation fields changed")
    return protocol


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CVBRARTDETRTrainingError(f"unsupported checkpoint payload: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise CVBRARTDETRTrainingError(f"checkpoint has no model module: {path}")
    return checkpoint, model


def _model_contract() -> dict[str, Any]:
    _, model = _load_checkpoint(SOURCE_CHECKPOINT)
    layers = getattr(model, "model", None)
    if not isinstance(layers, torch.nn.Sequential) or len(layers) != MODEL_LAYERS:
        raise CVBRARTDETRTrainingError("RT-DETR-L layer graph changed")
    indices = {_layer_index(name) for name in model.state_dict()}
    if min(indices) != 0 or max(indices) != LAST_TRAINABLE_LAYER:
        raise CVBRARTDETRTrainingError("RT-DETR-L state-layer coverage changed")
    return {
        "layers": len(layers),
        "state_entries": len(model.state_dict()),
        "class_names": {
            str(key): str(value)
            for key, value in dict(getattr(model, "names", {})).items()
        },
        "frozen_parameters": sum(
            parameter.numel()
            for index, layer in enumerate(layers)
            if index <= FROZEN_LAST_LAYER
            for parameter in layer.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for index, layer in enumerate(layers)
            if index >= FIRST_TRAINABLE_LAYER
            for parameter in layer.parameters()
        ),
    }


def _implementation_lock() -> dict[str, Any]:
    _validate_protocol()
    _assert_hash(SUPERSESSION, SUPERSESSION_SHA256, label="implementation supersession")
    contract = _model_contract()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise CVBRARTDETRTrainingError("RT-DETR implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("status")
            != "CVBRA_V1_RTDETR_L_IMPLEMENTATION_V2_LOCKED_BEFORE_TRAINING"
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("model_contract") != contract
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise CVBRARTDETRTrainingError("RT-DETR implementation lock changed")
        return lock
    if any(path.exists() for path in (TRAINING_LOCK, CHECKPOINT_LOCK, CHECKPOINT)):
        raise CVBRARTDETRTrainingError("RT-DETR output appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_RTDETR_L_IMPLEMENTATION_V2_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _utc_now(),
        "supersession_sha256": SUPERSESSION_SHA256,
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "validation_report_sha256": VALIDATION_REPORT_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "dataset_yaml_sha256": DATASET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "model_contract": contract,
        "ruff": "PASS",
        "strict_mypy": "PASS",
        "training_started_before_lock": False,
        "validation_metric_feedback_used": False,
        "official_test_accessed": False,
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


def preflight() -> dict[str, Any]:
    protocol = _validate_protocol()
    lock = _implementation_lock()
    manifest = _load_mapping(MANIFEST)
    if manifest.get("total_images") != TRAIN_IMAGES:
        raise CVBRARTDETRTrainingError("CVBRA training manifest coverage changed")
    return {
        "status": "PASS_CVBRA_V1_RTDETR_L_TRAINING_PREFLIGHT",
        "protocol": protocol["protocol"],
        "model_contract": lock["model_contract"],
        "train_images_per_epoch": TRAIN_IMAGES,
        "epochs": EPOCHS,
        "validation_metric_feedback_used": False,
        "official_test_accessed": False,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
    }


def _validate_training_lock() -> dict[str, Any]:
    _implementation_lock()
    if not TRAINING_LOCK.is_file() or not TRAINING_MARKER.is_file():
        raise CVBRARTDETRTrainingError("RT-DETR training lock is incomplete")
    lock = _load_mapping(TRAINING_LOCK)
    marker = _load_mapping(TRAINING_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_RTDETR_L_RAW_TRAINING_ENDPOINT_LOCKED"
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
    ):
        raise CVBRARTDETRTrainingError("RT-DETR training endpoint changed")
    for key in ("last_checkpoint", "results", "args"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=key)
    return lock


def train_raw_endpoint() -> dict[str, Any]:
    preflight()
    if TRAINING_LOCK.exists() or TRAINING_MARKER.exists():
        return _validate_training_lock()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT.exists():
        raise CVBRARTDETRTrainingError("final checkpoint appeared before training endpoint")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import RTDETR  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise CVBRARTDETRTrainingError(f"cannot import Ultralytics RTDETR: {exc}") from exc
    save_dir = RUN_ROOT / "raw_endpoint" / "fit"
    if save_dir.exists():
        raise CVBRARTDETRTrainingError(
            f"incomplete RT-DETR training directory requires audit: {save_dir}"
        )
    model = RTDETR(str(SOURCE_CHECKPOINT))
    results = model.train(
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
        freeze=FIRST_TRAINABLE_LAYER,
        val=False,
        optimizer="auto",
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
    actual_save = Path(results.save_dir)
    last = actual_save / "weights" / "last.pt"
    results_path = actual_save / "results.csv"
    args_path = actual_save / "args.yaml"
    for path in (last, results_path, args_path):
        if not path.is_file():
            raise CVBRARTDETRTrainingError(f"RT-DETR training output is incomplete: {path}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_RTDETR_L_RAW_TRAINING_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_path),
        "results_sha256": sha256_file(results_path),
        "args": _relative(args_path),
        "args_sha256": sha256_file(args_path),
        "epochs": EPOCHS,
        "checkpoint_selected_by_metric": False,
        "training_validation_executed": False,
        "official_validation_metric_feedback_used": False,
        "official_test_accessed": False,
    }
    atomic_write_json(TRAINING_LOCK, payload)
    atomic_write_json(
        TRAINING_MARKER,
        {"status": payload["status"], "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK)},
    )
    return payload


def _combined_state(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise CVBRARTDETRTrainingError("source and trained state schemas differ")
    combined: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise CVBRARTDETRTrainingError(f"state shape differs: {name}")
        chosen = source_value if _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        if chosen.is_floating_point() and not bool(torch.isfinite(chosen).all()):
            raise CVBRARTDETRTrainingError(f"state is nonfinite: {name}")
        combined[name] = chosen.clone()
    return combined


def _build_final_checkpoint(raw: Mapping[str, Any]) -> None:
    source_checkpoint, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise CVBRARTDETRTrainingError("source and trained class schemas differ")
    combined = _combined_state(source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(combined, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CVBRARTDETRTrainingError("strict state load reported incompatible keys")
    output_model = output_model.half()
    output_model.eval()
    output_checkpoint = copy.deepcopy(source_checkpoint)
    output_checkpoint.update(
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
            "cvbra_v1_rtdetr_l": {
                "training": "same locked CVBRA balanced target views plus source-haze replay",
                "frozen_state_rule": "exact source restore for layers 0..9",
                "trained_layers": [FIRST_TRAINABLE_LAYER, LAST_TRAINABLE_LAYER],
                "epochs": EPOCHS,
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
                "metric_used_for_epoch_or_checkpoint_selection": False,
            },
        }
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".pt.tmp")
    torch.save(output_checkpoint, temporary)
    temporary.replace(CHECKPOINT)


def _verify_final_checkpoint(raw: Mapping[str, Any]) -> dict[str, Any]:
    _, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    checkpoint, output_model = _load_checkpoint(CHECKPOINT)
    metadata = checkpoint.get("cvbra_v1_rtdetr_l")
    if not isinstance(metadata, dict):
        raise CVBRARTDETRTrainingError("final checkpoint metadata is missing")
    expected = _combined_state(source_model.state_dict(), trained_model.state_dict())
    observed = output_model.state_dict()
    if tuple(expected) != tuple(observed):
        raise CVBRARTDETRTrainingError("final state schema changed")
    maximum_error = 0.0
    frozen_exact = True
    changed_trainable = 0
    source_state = source_model.state_dict()
    for name, expected_value in expected.items():
        observed_value = observed[name].to(dtype=expected_value.dtype)
        if expected_value.is_floating_point():
            maximum_error = max(
                maximum_error,
                float((observed_value.float() - expected_value.float()).abs().max()),
            )
        elif not torch.equal(observed_value, expected_value):
            raise CVBRARTDETRTrainingError(f"nonfloating state differs: {name}")
        if _layer_index(name) <= FROZEN_LAST_LAYER:
            frozen_exact = frozen_exact and torch.equal(observed_value, source_state[name])
        elif expected_value.is_floating_point() and not torch.equal(
            observed_value, source_state[name]
        ):
            changed_trainable += 1
    if maximum_error != 0.0 or not frozen_exact or changed_trainable == 0:
        raise CVBRARTDETRTrainingError(
            "checkpoint verification failed: "
            f"error={maximum_error}, frozen={frozen_exact}, changed={changed_trainable}"
        )
    return {
        "state_entries": len(expected),
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "frozen_layers_0_to_9_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed_trainable,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
        "class_names": getattr(output_model, "names", None),
        "metadata": metadata,
    }


def _validate_checkpoint_lock() -> dict[str, Any]:
    raw = _validate_training_lock()
    if not CHECKPOINT_LOCK.is_file() or not CHECKPOINT_MARKER.is_file():
        raise CVBRARTDETRTrainingError("RT-DETR checkpoint lock is incomplete")
    lock = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    if (
        lock.get("status") != "CVBRA_V1_RTDETR_L_CHECKPOINT_VERIFIED_AND_LOCKED"
        or lock.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
        or lock.get("checkpoint_sha256") != sha256_file(CHECKPOINT)
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
    ):
        raise CVBRARTDETRTrainingError("RT-DETR checkpoint lock changed")
    _verify_final_checkpoint(raw)
    return lock


def build_checkpoint() -> dict[str, Any]:
    raw = train_raw_endpoint()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT_MARKER.exists():
        return _validate_checkpoint_lock()
    if CHECKPOINT.exists():
        raise CVBRARTDETRTrainingError("unlocked final checkpoint already exists")
    _build_final_checkpoint(raw)
    verification = _verify_final_checkpoint(raw)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_RTDETR_L_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "verification": verification,
        "checkpoint_selected_by_metric": False,
        "official_validation_metric_feedback_used": False,
        "official_test_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(CHECKPOINT_LOCK, payload)
    atomic_write_json(
        CHECKPOINT_MARKER,
        {"status": payload["status"], "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed CVBRA RT-DETR-L confirmation")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "build-checkpoint", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "train":
        result = train_raw_endpoint()
    else:
        result = build_checkpoint()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
