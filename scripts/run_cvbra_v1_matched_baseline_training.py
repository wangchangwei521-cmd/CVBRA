from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml

from buse_uav.detectors.ultralytics_adapter import configure_ultralytics_environment
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_matched_baselines.yaml"
PROTOCOL_SHA256 = "1671b6f13b9f1ffad0af6b2bcb20c5ac2c680e5368174505be8c884bf3dd7b9b"
REGISTRATION = (
    ROOT / "reports" / "development" / "cvbra_v1_matched_baselines" / "REGISTRATION_LOCK.json"
)
REGISTRATION_SHA256 = "aaf2e18153b10605926c4c93593db071ee7f0e503f54ef5da89a13d8f7a26278"
PRIMARY_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "evaluation"
    / "validation_report.json"
)
PRIMARY_REPORT_SHA256 = "dbe2d7efb5f06d0251cde76fa6b8b9d3f4bf6f61e68652f0f9c9ed3bda81c29b"
SOURCE_CHECKPOINT = ROOT / "weights" / "hazydet" / "yolo11n_best.pt"
SOURCE_CHECKPOINT_SHA256 = "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430"
CVBRA_DATASET = ROOT / "data" / "processed" / "cvbra_v1" / "dataset.yaml"
CVBRA_DATASET_SHA256 = "567ab0bb75434e3c735f8bda6b2aaae2b1c81a4ed6043c215c87900ec9c039d7"
CVBRA_MANIFEST = ROOT / "data" / "processed" / "cvbra_v1" / "manifest.json"
CVBRA_MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
CVBRA_DATA_LOCK = ROOT / "reports" / "development" / "cvbra_v1" / "generated_dataset_lock.json"
CVBRA_DATA_LOCK_SHA256 = "32ebc263674ef9b4fd123642e43b978cebcb04ec53f4eb9eef12c7e624fcbe4d"

OUTPUT = ROOT / "reports" / "development" / "cvbra_v1_matched_baselines"
DATA_ROOT = ROOT / "data" / "processed" / "cvbra_v1_matched_baselines"
RUN_ROOT = ROOT / "runs" / "cvbra_v1_matched_baselines"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"

TARGET_SCENES = 900
SOURCE_REPLAY_IMAGES = 900
TRAIN_IMAGES = 3600
FROZEN_LAST_LAYER = 9
FIRST_TRAINABLE_LAYER = 10
LAST_LAYER = 23
MODEL_LAYERS = 24
EPOCHS = 8
IMGSZ = 1280
BATCH = 2
SEED = 42
REGISTERED_STATUS = (
    "REGISTERED_AFTER_PRIMARY_CONFIRMATION_BEFORE_BASELINE_DATA_GENERATION_"
    "TRAINING_OR_PREDICTION"
)


class BaselineSpec(NamedTuple):
    dataset_kind: str
    frozen_backbone: bool
    paper_label: str


BASELINES: dict[str, BaselineSpec] = {
    "STF": BaselineSpec("target_original_x4", False, "standard_target_finetuning"),
    "CVBRA_noCV": BaselineSpec(
        "target_original_x3_plus_replay", True, "cross_visibility_removal_control"
    ),
    "CVBRA_noReplay": BaselineSpec("balanced_target_x4", True, "source_replay_removal_control"),
    "CVBRA_noFreeze": BaselineSpec(
        "reuse_exact_cvbra_dataset", False, "frozen_backbone_removal_control"
    ),
}


class MatchedBaselineError(RuntimeError):
    """Raised when a frozen matched-baseline contract is violated."""


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
        raise MatchedBaselineError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedBaselineError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise MatchedBaselineError(f"missing locked {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise MatchedBaselineError(
            f"locked {label} changed: expected {expected}, observed {observed}"
        )


def _layer_index(name: str) -> int:
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "model":
        raise MatchedBaselineError(f"state has no YOLO model-layer index: {name}")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise MatchedBaselineError(f"invalid YOLO layer index: {name}") from exc


def _validate_registration() -> dict[str, Any]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "matched-baseline protocol"),
        (REGISTRATION, REGISTRATION_SHA256, "registration lock"),
        (PRIMARY_REPORT, PRIMARY_REPORT_SHA256, "primary validation report"),
        (SOURCE_CHECKPOINT, SOURCE_CHECKPOINT_SHA256, "source checkpoint"),
        (CVBRA_DATASET, CVBRA_DATASET_SHA256, "CVBRA dataset YAML"),
        (CVBRA_MANIFEST, CVBRA_MANIFEST_SHA256, "CVBRA manifest"),
        (CVBRA_DATA_LOCK, CVBRA_DATA_LOCK_SHA256, "CVBRA data lock"),
    ):
        _assert_hash(path, digest, label=label)
    protocol = _load_mapping(PROTOCOL)
    registration = _load_mapping(REGISTRATION)
    primary = _load_mapping(PRIMARY_REPORT)
    common = protocol.get("common_contract")
    baselines = protocol.get("baselines")
    integrity = protocol.get("integrity")
    if not all(isinstance(value, dict) for value in (common, baselines, integrity)):
        raise MatchedBaselineError("matched-baseline protocol sections are incomplete")
    assert isinstance(common, dict)
    assert isinstance(baselines, dict)
    assert isinstance(integrity, dict)
    if (
        protocol.get("status") != REGISTERED_STATUS
        or set(baselines) != set(BASELINES)
        or common.get("training_images_per_epoch") != TRAIN_IMAGES
        or common.get("epochs") != EPOCHS
        or common.get("imgsz") != IMGSZ
        or common.get("batch") != BATCH
        or common.get("metric_selected_epoch_or_checkpoint") is not False
        or integrity.get("UAV_OBB_official_test_access") != "prohibited"
        or integrity.get("validation_label_blind_claim_for_this_study") != "prohibited"
        or registration.get("protocol_sha256") != PROTOCOL_SHA256
        or registration.get("registered_baselines") != list(BASELINES)
        or registration.get("baseline_training_started_before_lock") is not False
        or registration.get("method_or_hyperparameter_selection_from_this_study") is not False
        or primary.get("status") != "PASS_CVBRA_V1_OFFICIAL_VALIDATION_CONFIRMATION"
        or primary.get("decision", {}).get("official_test_access_authorized") is not False
    ):
        raise MatchedBaselineError("registered matched-baseline fields changed")
    return protocol


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    configure_ultralytics_environment(ROOT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise MatchedBaselineError(f"unsupported checkpoint payload: {path}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise MatchedBaselineError(f"checkpoint has no model module: {path}")
    return checkpoint, model


def _model_contract() -> dict[str, Any]:
    _, model = _load_checkpoint(SOURCE_CHECKPOINT)
    layers = getattr(model, "model", None)
    if not isinstance(layers, torch.nn.Sequential) or len(layers) != MODEL_LAYERS:
        raise MatchedBaselineError("YOLO11n layer graph changed")
    indices = {_layer_index(name) for name in model.state_dict()}
    if min(indices) != 0 or max(indices) != LAST_LAYER:
        raise MatchedBaselineError("YOLO11n state-layer coverage changed")
    return {
        "layers": len(layers),
        "state_entries": len(model.state_dict()),
        "class_names": {
            str(key): str(value) for key, value in dict(getattr(model, "names", {})).items()
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _implementation_lock() -> dict[str, Any]:
    _validate_registration()
    contract = _model_contract()
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
            raise MatchedBaselineError("matched-baseline implementation lock is incomplete")
        lock = _load_mapping(IMPLEMENTATION_LOCK)
        marker = _load_mapping(IMPLEMENTATION_MARKER)
        if (
            lock.get("status")
            != "CVBRA_V1_MATCHED_BASELINE_IMPLEMENTATION_LOCKED_BEFORE_DATA_OR_TRAINING"
            or lock.get("protocol_sha256") != PROTOCOL_SHA256
            or lock.get("registration_sha256") != REGISTRATION_SHA256
            or lock.get("runner_sha256") != sha256_file(Path(__file__))
            or lock.get("model_contract") != contract
            or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        ):
            raise MatchedBaselineError("matched-baseline implementation lock changed")
        return lock
    if DATA_ROOT.exists() or RUN_ROOT.exists():
        raise MatchedBaselineError(
            "baseline data or run output appeared before implementation lock"
        )
    later = [
        OUTPUT / directory for directory in ("data_locks", "training_locks", "checkpoint_locks")
    ]
    if any(path.exists() for path in later):
        raise MatchedBaselineError("baseline evidence appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_IMPLEMENTATION_LOCKED_BEFORE_DATA_OR_TRAINING",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "registration": _relative(REGISTRATION),
        "registration_sha256": REGISTRATION_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_contract": contract,
        "baselines": list(BASELINES),
        "official_validation_labels_already_accessed": True,
        "baseline_data_or_training_before_lock": False,
        "metric_feedback_allowed_during_training": False,
        "official_test_access": "prohibited",
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


def _source_entries() -> list[dict[str, Any]]:
    manifest = _load_mapping(CVBRA_MANIFEST)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != TRAIN_IMAGES:
        raise MatchedBaselineError("CVBRA manifest entries changed")
    rows = [dict(row) for row in entries if isinstance(row, dict)]
    if len(rows) != TRAIN_IMAGES:
        raise MatchedBaselineError("CVBRA manifest contains invalid entries")
    counts = Counter(
        (
            str(row.get("role")),
            str(row.get("view", "original_hazy")),
        )
        for row in rows
    )
    expected = {
        ("target", "original"): 900,
        ("target", "fog_0p6"): 900,
        ("target", "fog_1p0"): 900,
        ("source_haze_replay", "original_hazy"): 900,
    }
    if dict(counts) != expected:
        raise MatchedBaselineError(f"CVBRA manifest composition changed: {counts}")
    return rows


def _dataset_paths(baseline: str) -> tuple[Path, Path]:
    if baseline == "CVBRA_noFreeze":
        return CVBRA_DATASET, CVBRA_MANIFEST
    root = DATA_ROOT / baseline
    return root / "dataset.yaml", root / "manifest.json"


def _data_lock_path(baseline: str) -> Path:
    return OUTPUT / "data_locks" / f"{baseline}.json"


def _data_marker_path(baseline: str) -> Path:
    return OUTPUT / "data_locks" / f"{baseline}.LOCKED"


def _training_lock_path(baseline: str) -> Path:
    return OUTPUT / "training_locks" / f"{baseline}.json"


def _training_marker_path(baseline: str) -> Path:
    return OUTPUT / "training_locks" / f"{baseline}.LOCKED"


def _checkpoint_path(baseline: str) -> Path:
    return RUN_ROOT / baseline / f"{baseline}.pt"


def _checkpoint_lock_path(baseline: str) -> Path:
    return OUTPUT / "checkpoint_locks" / f"{baseline}.json"


def _checkpoint_marker_path(baseline: str) -> Path:
    return OUTPUT / "checkpoint_locks" / f"{baseline}.LOCKED"


def _hardlink_or_validate(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or (
            not os.path.samefile(source, target) and sha256_file(source) != sha256_file(target)
        ):
            raise MatchedBaselineError(f"existing alias differs: {target}")
        return
    os.link(source, target)


def _alias_entry(
    row: Mapping[str, Any], *, baseline: str, alias: str, ordinal: int
) -> dict[str, Any]:
    source_image = _rooted(row["image"])
    source_label = _rooted(row["label"])
    if not source_image.is_file() or not source_label.is_file():
        raise MatchedBaselineError("locked CVBRA source data is incomplete")
    root = DATA_ROOT / baseline
    stem = f"{ordinal:04d}_{alias}"
    target_image = root / "images" / "train" / f"{stem}{source_image.suffix.casefold()}"
    target_label = root / "labels" / "train" / f"{stem}.txt"
    _hardlink_or_validate(source_image, target_image)
    _hardlink_or_validate(source_label, target_label)
    expected_image_sha = str(row["image_sha256"])
    expected_label_sha = str(row["label_sha256"])
    if (
        sha256_file(target_image) != expected_image_sha
        or sha256_file(target_label) != expected_label_sha
    ):
        raise MatchedBaselineError(f"alias hash mismatch: {target_image}")
    return {
        "ordinal": ordinal,
        "baseline": baseline,
        "alias": alias,
        "role": str(row["role"]),
        "view": str(row.get("view", "original_hazy")),
        "image_id": int(row["image_id"]),
        "image": _relative(target_image),
        "image_sha256": expected_image_sha,
        "label": _relative(target_label),
        "label_sha256": expected_label_sha,
        "source_image": str(row["image"]),
    }


def _planned_rows(baseline: str, rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    target_by_view: dict[str, dict[int, dict[str, Any]]] = {
        view: {} for view in ("original", "fog_0p6", "fog_1p0")
    }
    replay: list[dict[str, Any]] = []
    for row in rows:
        role = str(row["role"])
        view = str(row.get("view", "original_hazy"))
        if role == "target":
            target_by_view[view][int(row["image_id"])] = row
        elif role == "source_haze_replay":
            replay.append(row)
    if any(set(mapping) != set(range(1, TARGET_SCENES + 1)) for mapping in target_by_view.values()):
        raise MatchedBaselineError("target scene coverage changed")
    replay = sorted(replay, key=lambda row: int(row["image_id"]))
    result: list[tuple[dict[str, Any], str]] = []
    if baseline == "STF":
        for image_id in range(1, TARGET_SCENES + 1):
            for alias_index in range(4):
                result.append(
                    (
                        target_by_view["original"][image_id],
                        f"target_{image_id:04d}_original_a{alias_index}",
                    )
                )
    elif baseline == "CVBRA_noCV":
        for image_id in range(1, TARGET_SCENES + 1):
            for alias_index in range(3):
                result.append(
                    (
                        target_by_view["original"][image_id],
                        f"target_{image_id:04d}_original_a{alias_index}",
                    )
                )
        for replay_index, row in enumerate(replay, start=1):
            result.append((row, f"source_replay_{replay_index:04d}"))
    elif baseline == "CVBRA_noReplay":
        for image_id in range(1, TARGET_SCENES + 1):
            for view in ("original", "fog_0p6", "fog_1p0"):
                result.append((target_by_view[view][image_id], f"target_{image_id:04d}_{view}"))
            extra_view = ("original", "fog_0p6", "fog_1p0")[(image_id - 1) % 3]
            result.append(
                (target_by_view[extra_view][image_id], f"target_{image_id:04d}_{extra_view}_extra")
            )
    else:
        raise MatchedBaselineError(f"baseline does not require alias data: {baseline}")
    if len(result) != TRAIN_IMAGES:
        raise MatchedBaselineError(f"planned data count changed for {baseline}")
    return result


def _dataset_yaml(root: Path) -> str:
    return (
        f"path: {root.resolve()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
    )


def _validate_data_lock(baseline: str) -> dict[str, Any]:
    _implementation_lock()
    lock_path = _data_lock_path(baseline)
    marker_path = _data_marker_path(baseline)
    if not lock_path.is_file() or not marker_path.is_file():
        raise MatchedBaselineError(f"data lock is incomplete for {baseline}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    dataset, manifest = _dataset_paths(baseline)
    if (
        lock.get("status") != "CVBRA_V1_MATCHED_BASELINE_DATA_LOCKED_BEFORE_TRAINING"
        or lock.get("baseline") != baseline
        or lock.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or marker.get("data_lock_sha256") != sha256_file(lock_path)
        or lock.get("dataset_yaml_sha256") != sha256_file(dataset)
        or lock.get("manifest_sha256") != sha256_file(manifest)
        or lock.get("images_per_epoch") != TRAIN_IMAGES
    ):
        raise MatchedBaselineError(f"data lock changed for {baseline}")
    return lock


def build_data(baseline: str) -> dict[str, Any]:
    _implementation_lock()
    lock_path = _data_lock_path(baseline)
    marker_path = _data_marker_path(baseline)
    if lock_path.exists() or marker_path.exists():
        return _validate_data_lock(baseline)
    if _training_lock_path(baseline).exists() or _checkpoint_path(baseline).exists():
        raise MatchedBaselineError(f"later output appeared before data lock for {baseline}")
    spec = BASELINES[baseline]
    dataset, manifest = _dataset_paths(baseline)
    if baseline == "CVBRA_noFreeze":
        composition = {
            "target_original": 900,
            "target_fog_0p6": 900,
            "target_fog_1p0": 900,
            "source_replay": 900,
        }
    else:
        root = DATA_ROOT / baseline
        if root.exists():
            raise MatchedBaselineError(f"unlocked baseline data directory exists: {root}")
        planned = _planned_rows(baseline, _source_entries())
        entries = [
            _alias_entry(row, baseline=baseline, alias=alias, ordinal=index)
            for index, (row, alias) in enumerate(planned, start=1)
        ]
        composition = dict(Counter(f"{row['role']}_{row['view']}" for row in entries))
        atomic_write_text(dataset, _dataset_yaml(root))
        atomic_write_json(
            manifest,
            {
                "schema_version": 1,
                "status": "MATCHED_BASELINE_DATASET_MATERIALIZED",
                "baseline": baseline,
                "dataset_kind": spec.dataset_kind,
                "entries": entries,
                "composition": composition,
            },
        )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_DATA_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _utc_now(),
        "baseline": baseline,
        "paper_label": spec.paper_label,
        "dataset_kind": spec.dataset_kind,
        "frozen_backbone": spec.frozen_backbone,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "dataset_yaml": _relative(dataset),
        "dataset_yaml_sha256": sha256_file(dataset),
        "manifest": _relative(manifest),
        "manifest_sha256": sha256_file(manifest),
        "composition": composition,
        "images_per_epoch": TRAIN_IMAGES,
        "official_validation_metric_used": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(lock_path, payload)
    atomic_write_json(
        marker_path,
        {"status": payload["status"], "data_lock_sha256": sha256_file(lock_path)},
    )
    return payload


def _validate_training_lock(baseline: str) -> dict[str, Any]:
    _validate_data_lock(baseline)
    lock_path = _training_lock_path(baseline)
    marker_path = _training_marker_path(baseline)
    if not lock_path.is_file() or not marker_path.is_file():
        raise MatchedBaselineError(f"training lock is incomplete for {baseline}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    if (
        lock.get("status") != "CVBRA_V1_MATCHED_BASELINE_RAW_ENDPOINT_LOCKED"
        or lock.get("baseline") != baseline
        or lock.get("data_lock_sha256") != sha256_file(_data_lock_path(baseline))
        or marker.get("training_lock_sha256") != sha256_file(lock_path)
    ):
        raise MatchedBaselineError(f"training lock changed for {baseline}")
    for key in ("last_checkpoint", "results", "args"):
        _assert_hash(_rooted(lock[key]), lock[f"{key}_sha256"], label=f"{baseline} {key}")
    return lock


def train_raw_endpoint(baseline: str) -> dict[str, Any]:
    data_lock = build_data(baseline)
    lock_path = _training_lock_path(baseline)
    marker_path = _training_marker_path(baseline)
    if lock_path.exists() or marker_path.exists():
        return _validate_training_lock(baseline)
    if _checkpoint_path(baseline).exists() or _checkpoint_lock_path(baseline).exists():
        raise MatchedBaselineError(f"checkpoint appeared before training lock for {baseline}")
    configure_ultralytics_environment(ROOT)
    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]
    except (ImportError, OSError, PermissionError) as exc:
        raise MatchedBaselineError(f"cannot import Ultralytics: {exc}") from exc
    save_dir = RUN_ROOT / baseline / "raw_endpoint" / "fit"
    if save_dir.exists():
        raise MatchedBaselineError(f"incomplete training directory requires audit: {save_dir}")
    model = YOLO(str(SOURCE_CHECKPOINT))
    train_kwargs: dict[str, Any] = {
        "data": str(_rooted(data_lock["dataset_yaml"]).resolve()),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": "0",
        "workers": 4,
        "project": str((RUN_ROOT / baseline / "raw_endpoint").resolve()),
        "name": "fit",
        "exist_ok": False,
        "pretrained": True,
        "val": False,
        "optimizer": "SGD",
        "lr0": 0.00075,
        "lrf": 0.10,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.3,
        "close_mosaic": 0,
        "patience": EPOCHS,
        "amp": True,
        "seed": SEED,
        "deterministic": True,
        "max_det": 500,
        "cache": False,
        "plots": False,
        "verbose": True,
        "save": True,
    }
    if BASELINES[baseline].frozen_backbone:
        train_kwargs["freeze"] = FIRST_TRAINABLE_LAYER
    results = model.train(**train_kwargs)
    actual_save = Path(results.save_dir)
    last = actual_save / "weights" / "last.pt"
    results_path = actual_save / "results.csv"
    args_path = actual_save / "args.yaml"
    for path in (last, results_path, args_path):
        if not path.is_file():
            raise MatchedBaselineError(f"training output is incomplete: {path}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_RAW_ENDPOINT_LOCKED",
        "locked_at_utc": _utc_now(),
        "baseline": baseline,
        "data_lock_sha256": sha256_file(_data_lock_path(baseline)),
        "last_checkpoint": _relative(last),
        "last_checkpoint_sha256": sha256_file(last),
        "results": _relative(results_path),
        "results_sha256": sha256_file(results_path),
        "args": _relative(args_path),
        "args_sha256": sha256_file(args_path),
        "epochs": EPOCHS,
        "checkpoint_selected_by_metric": False,
        "external_validation_during_training": False,
        "official_validation_metric_used": False,
        "official_test_access": "prohibited",
    }
    atomic_write_json(lock_path, payload)
    atomic_write_json(
        marker_path,
        {"status": payload["status"], "training_lock_sha256": sha256_file(lock_path)},
    )
    return payload


def _endpoint_state(
    baseline: str,
    source: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise MatchedBaselineError(f"state schema differs for {baseline}")
    frozen = BASELINES[baseline].frozen_backbone
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, trained_value in trained.items():
        source_value = source[name]
        if source_value.shape != trained_value.shape:
            raise MatchedBaselineError(f"state shape differs for {baseline}: {name}")
        value = (
            source_value if frozen and _layer_index(name) <= FROZEN_LAST_LAYER else trained_value
        )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise MatchedBaselineError(f"nonfinite state for {baseline}: {name}")
        result[name] = value.clone()
    return result


def _build_checkpoint(baseline: str, raw: Mapping[str, Any]) -> None:
    source_checkpoint, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    if getattr(source_model, "names", None) != getattr(trained_model, "names", None):
        raise MatchedBaselineError(f"class schema differs for {baseline}")
    combined = _endpoint_state(baseline, source_model.state_dict(), trained_model.state_dict())
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(combined, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise MatchedBaselineError(f"strict state load failed for {baseline}")
    output_model = output_model.half()
    output_model.eval()
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
            "cvbra_v1_matched_baseline": {
                "baseline": baseline,
                "paper_label": BASELINES[baseline].paper_label,
                "dataset_kind": BASELINES[baseline].dataset_kind,
                "frozen_backbone": BASELINES[baseline].frozen_backbone,
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
                "data_lock_sha256": sha256_file(_data_lock_path(baseline)),
                "training_lock_sha256": sha256_file(_training_lock_path(baseline)),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "raw_endpoint_sha256": sha256_file(_rooted(raw["last_checkpoint"])),
                "metric_selected_epoch_or_checkpoint": False,
            },
        }
    )
    output = _checkpoint_path(baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def _verify_checkpoint(baseline: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    _, source_model = _load_checkpoint(SOURCE_CHECKPOINT)
    _, trained_model = _load_checkpoint(_rooted(raw["last_checkpoint"]))
    checkpoint, output_model = _load_checkpoint(_checkpoint_path(baseline))
    metadata = checkpoint.get("cvbra_v1_matched_baseline")
    if not isinstance(metadata, dict) or metadata.get("baseline") != baseline:
        raise MatchedBaselineError(f"checkpoint metadata missing for {baseline}")
    expected = _endpoint_state(baseline, source_model.state_dict(), trained_model.state_dict())
    observed = output_model.state_dict()
    if tuple(expected) != tuple(observed):
        raise MatchedBaselineError(f"checkpoint state schema changed for {baseline}")
    maximum_error = 0.0
    frozen_exact = True
    changed_states = 0
    for name, expected_value in expected.items():
        observed_value = observed[name].to(dtype=expected_value.dtype)
        if expected_value.is_floating_point():
            maximum_error = max(
                maximum_error,
                float((observed_value.float() - expected_value.float()).abs().max()),
            )
        elif not torch.equal(observed_value, expected_value):
            raise MatchedBaselineError(f"nonfloating state differs for {baseline}: {name}")
        source_value = source_model.state_dict()[name]
        if BASELINES[baseline].frozen_backbone and _layer_index(name) <= FROZEN_LAST_LAYER:
            frozen_exact = frozen_exact and torch.equal(observed_value, source_value)
        elif expected_value.is_floating_point() and not torch.equal(observed_value, source_value):
            changed_states += 1
    if maximum_error != 0.0 or not frozen_exact or changed_states == 0:
        raise MatchedBaselineError(
            f"checkpoint verification failed for {baseline}: "
            f"error={maximum_error}, frozen={frozen_exact}, changed={changed_states}"
        )
    return {
        "state_entries": len(expected),
        "maximum_absolute_state_error_after_serialization": maximum_error,
        "frozen_layers_0_to_9_exact_source": (
            frozen_exact if BASELINES[baseline].frozen_backbone else None
        ),
        "changed_floating_states": changed_states,
        "all_states_finite": all(
            bool(torch.isfinite(value).all())
            for value in observed.values()
            if value.is_floating_point()
        ),
        "class_names": getattr(output_model, "names", None),
        "metadata": metadata,
    }


def _validate_checkpoint_lock(baseline: str) -> dict[str, Any]:
    raw = _validate_training_lock(baseline)
    lock_path = _checkpoint_lock_path(baseline)
    marker_path = _checkpoint_marker_path(baseline)
    checkpoint = _checkpoint_path(baseline)
    if not lock_path.is_file() or not marker_path.is_file():
        raise MatchedBaselineError(f"checkpoint lock is incomplete for {baseline}")
    lock = _load_mapping(lock_path)
    marker = _load_mapping(marker_path)
    if (
        lock.get("status") != "CVBRA_V1_MATCHED_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED"
        or lock.get("baseline") != baseline
        or lock.get("training_lock_sha256") != sha256_file(_training_lock_path(baseline))
        or lock.get("checkpoint_sha256") != sha256_file(checkpoint)
        or marker.get("checkpoint_lock_sha256") != sha256_file(lock_path)
    ):
        raise MatchedBaselineError(f"checkpoint lock changed for {baseline}")
    _verify_checkpoint(baseline, raw)
    return lock


def build_checkpoint(baseline: str) -> dict[str, Any]:
    raw = train_raw_endpoint(baseline)
    lock_path = _checkpoint_lock_path(baseline)
    marker_path = _checkpoint_marker_path(baseline)
    checkpoint = _checkpoint_path(baseline)
    if lock_path.exists() or marker_path.exists():
        return _validate_checkpoint_lock(baseline)
    if checkpoint.exists():
        raise MatchedBaselineError(f"unlocked checkpoint exists for {baseline}")
    _build_checkpoint(baseline, raw)
    verification = _verify_checkpoint(baseline, raw)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_MATCHED_BASELINE_CHECKPOINT_VERIFIED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "baseline": baseline,
        "paper_label": BASELINES[baseline].paper_label,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "data_lock_sha256": sha256_file(_data_lock_path(baseline)),
        "training_lock_sha256": sha256_file(_training_lock_path(baseline)),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "verification": verification,
        "validation_metric_used_for_training_or_selection": False,
        "official_test_access": "prohibited",
        "paper_body_change_authorized": False,
    }
    atomic_write_json(lock_path, payload)
    atomic_write_json(
        marker_path,
        {"status": payload["status"], "checkpoint_lock_sha256": sha256_file(lock_path)},
    )
    return payload


def preflight() -> dict[str, Any]:
    lock = _implementation_lock()
    rows = _source_entries()
    return {
        "status": "PASS_CVBRA_V1_MATCHED_BASELINE_PREFLIGHT",
        "baselines": list(BASELINES),
        "source_manifest_entries": len(rows),
        "training_images_per_baseline_epoch": TRAIN_IMAGES,
        "epochs": EPOCHS,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "runner_sha256": lock["runner_sha256"],
        "validation_labels_already_accessed": True,
        "selection_from_this_study": False,
        "official_test_access": "prohibited",
    }


def _run_stage(baseline: str, stage: str) -> dict[str, Any]:
    if stage == "build-data":
        return build_data(baseline)
    if stage == "train":
        return train_raw_endpoint(baseline)
    return build_checkpoint(baseline)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen CVBRA-v1 matched baselines")
    parser.add_argument("--baseline", choices=(*BASELINES, "all"), default="all")
    parser.add_argument(
        "--stage",
        choices=("preflight", "build-data", "train", "build-checkpoint", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        result: object = preflight()
    else:
        selected = list(BASELINES) if args.baseline == "all" else [str(args.baseline)]
        effective_stage = "build-checkpoint" if args.stage == "all" else str(args.stage)
        result = {baseline: _run_stage(baseline, effective_stage) for baseline in selected}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
