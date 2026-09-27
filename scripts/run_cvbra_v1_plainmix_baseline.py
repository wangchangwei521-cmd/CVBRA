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

from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_acceptance_upgrade_v1.yaml"
PROTOCOL_SHA256 = "2b0dd70e4c67c2afcca2dafea3a4d5e885ce233e45681db691ce9422ca78e73b"
REGISTRATION = (
    ROOT
    / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "9bd0e3d383d23a8bfb14d18c4298fca8e445d82ae5a7a749acfe22405886fc85"
REGISTRATION_MARKER = (
    ROOT / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix/REGISTERED"
)
REGISTRATION_MARKER_SHA256 = (
    "ef40dbf860c498dc098026eaae7eccd9782ba01c3b4e7a3a0209eb2419d2c6a8"
)
SOURCE_CHECKPOINT = ROOT / "weights/hazydet/yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
)
DATASET = (
    ROOT / "data/processed/cvbra_v1_matched_baselines/CVBRA_noCV/dataset.yaml"
)
DATASET_SHA256 = "9ced2b1ef5a2e5759e29b6422a7fc3fad8088c48ebdb276b5c16c189ff8435f9"
MANIFEST = (
    ROOT / "data/processed/cvbra_v1_matched_baselines/CVBRA_noCV/manifest.json"
)
MANIFEST_SHA256 = "25d0f0ef90b0898befad6a2c8174f08972a2a691bdd94cd6185d86a73806cea6"
DATA_LOCK = (
    ROOT
    / "reports/development/cvbra_v1_matched_baselines/data_locks/CVBRA_noCV.json"
)
DATA_LOCK_SHA256 = "aa61989d889ca9693b39a0d0dffd388fc541b7f8c4577372760c393883583b0b"

OUTPUT = ROOT / "reports/development/cvbra_v1_acceptance_upgrade/PlainMix"
RUN_ROOT = ROOT / "runs/cvbra_v1_acceptance_upgrade/PlainMix"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
TRAINING_LOCK = OUTPUT / "training_endpoint_lock.json"
TRAINING_MARKER = OUTPUT / "TRAINING_ENDPOINT_LOCKED"
CHECKPOINT = RUN_ROOT / "PlainMix.pt"
CHECKPOINT_LOCK = OUTPUT / "checkpoint_lock.json"
CHECKPOINT_MARKER = OUTPUT / "CHECKPOINT_LOCKED"

TRAIN_IMAGES = 3600
TARGET_ORIGINAL_IMAGES = 2700
SOURCE_REPLAY_IMAGES = 900
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
MODEL_LAYERS = 24
PROTOCOL_STATUS = (
    "REGISTERED_BEFORE_NEW_BASELINE_TRAINING_NATURAL_VISIBILITY_METRICS_"
    "OR_MANUSCRIPT_RESULT_INTEGRATION"
)


class PlainMixError(RuntimeError):
    """Raised when the registered PlainMix baseline contract is violated."""


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
        raise PlainMixError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlainMixError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise PlainMixError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise PlainMixError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _validate_registration() -> None:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "acceptance-upgrade protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "PlainMix registration"),
        (REGISTRATION_MARKER, REGISTRATION_MARKER_SHA256, "registration marker"),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "source checkpoint"),
        (DATASET, DATASET_SHA256, "PlainMix dataset YAML"),
        (MANIFEST, MANIFEST_SHA256, "PlainMix dataset manifest"),
        (DATA_LOCK, DATA_LOCK_SHA256, "source data lock"),
    ):
        _assert_hash(path, digest, label=label)
    protocol = _load_mapping(PROTOCOL)
    registration = _load_mapping(REGISTRATION)
    marker = _load_mapping(REGISTRATION_MARKER)
    baseline = protocol.get("new_strong_baseline")
    test_guard = protocol.get("official_test")
    if not isinstance(baseline, dict) or not isinstance(test_guard, dict):
        raise PlainMixError("acceptance-upgrade protocol is incomplete")
    if (
        protocol.get("status") != PROTOCOL_STATUS
        or baseline.get("name") != "PlainMix"
        or baseline.get("images_per_epoch") != TRAIN_IMAGES
        or baseline.get("epochs") != EPOCHS
        or baseline.get("imgsz") != IMGSZ
        or baseline.get("batch") != BATCH
        or baseline.get("validation_during_training") is not False
        or test_guard.get("UAV_OBB_official_test_access") != "prohibited"
        or registration.get("status")
        != "PLAINMIX_REGISTERED_BEFORE_TRAINING_PREDICTION_OR_METRIC_ACCESS"
        or registration.get("protocol_sha256") != PROTOCOL_SHA256
        or registration.get("state_at_registration", {}).get("PlainMix_training_started")
        is not False
        or marker.get("status") != registration.get("status")
    ):
        raise PlainMixError("registered PlainMix fields changed")


def _validate_dataset() -> dict[str, Any]:
    _validate_registration()
    manifest = _load_mapping(MANIFEST)
    data_lock = _load_mapping(DATA_LOCK)
    entries = manifest.get("entries")
    composition = manifest.get("composition")
    if (
        manifest.get("status") != "MATCHED_BASELINE_DATASET_MATERIALIZED"
        or manifest.get("baseline") != "CVBRA_noCV"
        or not isinstance(entries, list)
        or len(entries) != TRAIN_IMAGES
        or composition
        != {
            "target_original": TARGET_ORIGINAL_IMAGES,
            "source_haze_replay_original_hazy": SOURCE_REPLAY_IMAGES,
        }
        or data_lock.get("status")
        != "CVBRA_V1_MATCHED_BASELINE_DATA_LOCKED_BEFORE_TRAINING"
        or data_lock.get("dataset_yaml_sha256") != DATASET_SHA256
        or data_lock.get("manifest_sha256") != MANIFEST_SHA256
        or data_lock.get("images_per_epoch") != TRAIN_IMAGES
    ):
        raise PlainMixError("locked PlainMix data contract changed")
    return manifest


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise PlainMixError(f"unsupported checkpoint payload: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise PlainMixError(f"checkpoint has no model module: {path}")
    return checkpoint, model


def _model_contract() -> dict[str, Any]:
    _, model = _load_checkpoint(SOURCE_CHECKPOINT)
    layers = getattr(model, "model", None)
    if not isinstance(layers, torch.nn.Sequential) or len(layers) != MODEL_LAYERS:
        raise PlainMixError("YOLO11n layer graph changed")
    return {
        "layers": len(layers),
        "state_entries": len(model.state_dict()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_names": {
            str(key): str(value)
            for key, value in dict(getattr(model, "names", {})).items()
        },
    }


def _validate_implementation_lock() -> dict[str, Any]:
    if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
        raise PlainMixError("PlainMix implementation lock is incomplete")
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    marker = _load_mapping(IMPLEMENTATION_MARKER)
    if (
        lock.get("status")
        != "PLAINMIX_IMPLEMENTATION_LOCKED_BEFORE_TRAINING"
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or lock.get("registration_sha256") != REGISTRATION_SHA256
        or lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("model_contract") != _model_contract()
        or marker.get("implementation_lock_sha256")
        != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise PlainMixError("PlainMix implementation lock changed")
    return lock


def implementation_lock() -> dict[str, Any]:
    _validate_dataset()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        return _validate_implementation_lock()
    if any(path.exists() for path in (RUN_ROOT, TRAINING_LOCK, CHECKPOINT_LOCK)):
        raise PlainMixError("PlainMix output appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_IMPLEMENTATION_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration": _relative(REGISTRATION),
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "dataset_yaml_sha256": DATASET_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "model_contract": _model_contract(),
        "all_layers_trainable": True,
        "training_started_before_lock": False,
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
    manifest = _validate_dataset()
    lock = implementation_lock()
    return {
        "status": "PASS_PLAINMIX_TRAINING_PREFLIGHT",
        "entries": len(manifest["entries"]),
        "composition": manifest["composition"],
        "all_layers_trainable": True,
        "epochs": EPOCHS,
        "validation_metric_feedback_allowed": False,
        "official_test_access": "prohibited",
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
    }


def _validate_training_lock() -> dict[str, Any]:
    _validate_implementation_lock()
    if not TRAINING_LOCK.is_file() or not TRAINING_MARKER.is_file():
        raise PlainMixError("PlainMix training lock is incomplete")
    lock = _load_mapping(TRAINING_LOCK)
    marker = _load_mapping(TRAINING_MARKER)
    if (
        lock.get("status") != "PLAINMIX_RAW_TRAINING_ENDPOINT_LOCKED"
        or lock.get("implementation_lock_sha256")
        != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
    ):
        raise PlainMixError("PlainMix training lock changed")
    for key in ("last_checkpoint", "results", "args"):
        _assert_hash(_rooted(lock[key]), str(lock[f"{key}_sha256"]), label=key)
    return lock


def train_raw_endpoint() -> dict[str, Any]:
    preflight()
    if TRAINING_LOCK.exists() or TRAINING_MARKER.exists():
        return _validate_training_lock()
    if CHECKPOINT.exists() or CHECKPOINT_LOCK.exists():
        raise PlainMixError("PlainMix checkpoint appeared before training lock")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise PlainMixError(f"cannot import Ultralytics: {exc}") from exc
    save_dir = RUN_ROOT / "raw_endpoint" / "fit"
    if save_dir.exists():
        raise PlainMixError(f"incomplete PlainMix training directory: {save_dir}")
    model = YOLO(str(SOURCE_CHECKPOINT))
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
    actual_save = Path(results.save_dir)
    last = actual_save / "weights" / "last.pt"
    results_path = actual_save / "results.csv"
    args_path = actual_save / "args.yaml"
    for path in (last, results_path, args_path):
        if not path.is_file():
            raise PlainMixError(f"PlainMix training output is incomplete: {path}")
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_RAW_TRAINING_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
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
        {
            "status": payload["status"],
            "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        },
    )
    return payload


def _build_checkpoint(raw: Mapping[str, Any]) -> None:
    source_checkpoint, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    if tuple(source_model.state_dict()) != tuple(trained_model.state_dict()):
        raise PlainMixError("source and trained state schemas differ")
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise PlainMixError("source and trained class schemas differ")
    for name, value in trained_model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise PlainMixError(f"nonfinite trained state: {name}")
    output_model = copy.deepcopy(trained_model).half().eval()
    payload = copy.deepcopy(source_checkpoint)
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
            "cvbra_v1_acceptance_baseline": {
                "baseline": "PlainMix",
                "training": "2700 target-original aliases plus 900 fixed source replay images",
                "all_layers_trainable": True,
                "epochs": EPOCHS,
                "protocol_sha256": PROTOCOL_SHA256,
                "registration_sha256": REGISTRATION_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
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
    metadata = checkpoint.get("cvbra_v1_acceptance_baseline")
    if not isinstance(metadata, dict) or metadata.get("baseline") != "PlainMix":
        raise PlainMixError("PlainMix checkpoint metadata is missing")
    expected = trained_model.state_dict()
    observed = output_model.state_dict()
    source = source_model.state_dict()
    if tuple(expected) != tuple(observed) or tuple(source) != tuple(observed):
        raise PlainMixError("PlainMix checkpoint state schema changed")
    maximum_error = 0.0
    changed_states = 0
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
            raise PlainMixError(f"nonfloating PlainMix state differs: {name}")
    if maximum_error != 0.0 or changed_states == 0:
        raise PlainMixError(
            "PlainMix checkpoint verification failed: "
            f"error={maximum_error}, changed={changed_states}"
        )
    return {
        "state_entries": len(observed),
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "changed_floating_states": changed_states,
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
        raise PlainMixError("PlainMix checkpoint lock is incomplete")
    lock = _load_mapping(CHECKPOINT_LOCK)
    marker = _load_mapping(CHECKPOINT_MARKER)
    if (
        lock.get("status") != "PLAINMIX_CHECKPOINT_VERIFIED_AND_LOCKED"
        or lock.get("checkpoint_sha256") != sha256_file(CHECKPOINT)
        or lock.get("training_endpoint_lock_sha256") != sha256_file(TRAINING_LOCK)
        or marker.get("checkpoint_lock_sha256") != sha256_file(CHECKPOINT_LOCK)
    ):
        raise PlainMixError("PlainMix checkpoint lock changed")
    _verify_checkpoint(raw)
    return lock


def build_checkpoint() -> dict[str, Any]:
    raw = train_raw_endpoint()
    if CHECKPOINT_LOCK.exists() or CHECKPOINT_MARKER.exists():
        return _validate_checkpoint_lock()
    if CHECKPOINT.exists():
        raise PlainMixError("unlocked PlainMix checkpoint exists")
    _build_checkpoint(raw)
    verification = _verify_checkpoint(raw)
    payload = {
        "schema_version": 1,
        "status": "PLAINMIX_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "baseline": "PlainMix",
        "protocol_sha256": PROTOCOL_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "training_endpoint_lock_sha256": sha256_file(TRAINING_LOCK),
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "official_test_access": "prohibited",
        "paper_result_integration_authorized_only_after_evaluation_lock": True,
    }
    atomic_write_json(CHECKPOINT_LOCK, payload)
    atomic_write_json(
        CHECKPOINT_MARKER,
        {
            "status": payload["status"],
            "checkpoint_lock_sha256": sha256_file(CHECKPOINT_LOCK),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run registered PlainMix baseline")
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
