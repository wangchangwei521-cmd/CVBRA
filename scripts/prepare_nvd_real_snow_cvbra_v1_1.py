from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/nvd_real_snow_cvbra_v1_1_amendment.yaml"
BASE_PROTOCOL = ROOT / "configs/experiment/nvd_real_snow_cvbra_v1.yaml"
BASE_OUTPUT = ROOT / "reports/development/nvd_real_snow_cvbra_v1"
BASE_REGISTRATION = BASE_OUTPUT / "REGISTRATION_LOCK.json"
BASE_DATA_LOCK = BASE_OUTPUT / "DATA_LOCK.json"
BASE_INCIDENT = BASE_OUTPUT / "INEFFECTIVE_SEED_REALIZATION_INCIDENT_1.json"
BASE_DATA_ROOT = ROOT / "data/processed/nvd_real_snow_cvbra_v1"
BASE_RUN_ROOT = ROOT / "runs/nvd_real_snow_cvbra_v1"
OUTPUT = ROOT / "reports/development/nvd_real_snow_cvbra_v1_1"
DATA_ROOT = ROOT / "data/processed/nvd_real_snow_cvbra_v1_1"
RUNNER = ROOT / "scripts/run_nvd_real_snow_cvbra_v1_1.py"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
DATA_LOCK = OUTPUT / "DATA_LOCK.json"
ULTRALYTICS_DATALOADER = (
    Path(__import__("importlib.util", fromlist=["find_spec"]).find_spec("ultralytics").origin).parent / "data" / "build.py"
)

METHODS = ("STF", "NoVisibility_L10", "CVBRA_L10")
REPLACEMENT_SEEDS = (27182, 31415)
TRAIN_IMAGES = 3600
BASE_INCIDENT_SHA256 = "61cee7b565b6ec89b3521f903bc48ab17eba23b85d398fa56b884dbdac684442"
BASE_REGISTRATION_SHA256 = "55450ea134895b666fe1a148bc051e7009e8f805aeb1088d5ece45718c4b8df1"
BASE_DATA_LOCK_SHA256 = "fafb2ff1ad6b70292a0a0532201b8515d1cc2551fdda1b58c69fb0c3b8655f68"


class NvdSeedOrderCorrectionError(RuntimeError):
    """Raised when the registered seed-order correction cannot fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NvdSeedOrderCorrectionError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NvdSeedOrderCorrectionError(f"expected JSON object: {path}")
    return value


def load_protocol() -> dict[str, Any]:
    try:
        value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise NvdSeedOrderCorrectionError(f"cannot parse protocol {PROTOCOL}: {exc}") from exc
    if not isinstance(value, dict):
        raise NvdSeedOrderCorrectionError("correction protocol must be a mapping")
    if (
        value.get("protocol") != "nvd_real_snow_cvbra_v1_1"
        or value.get("status")
        != "LOCK_BEFORE_CORRECTION_DATA_MATERIALIZATION_TRAINING_PREDICTION_OR_METRIC_ACCESS"
    ):
        raise NvdSeedOrderCorrectionError("correction protocol identity or status changed")
    return value


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise NvdSeedOrderCorrectionError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise NvdSeedOrderCorrectionError(
            f"{label} hash changed: expected {expected}, observed {observed}"
        )


def _base_seed_42_files() -> list[Path]:
    files = [
        BASE_RUN_ROOT / "source/source_yolo11n.pt",
        BASE_OUTPUT / "training_locks/source.json",
    ]
    for method in METHODS:
        files.extend(
            (
                BASE_RUN_ROOT / method / "seed_42" / f"{method}_seed_42.pt",
                BASE_OUTPUT / "training_locks" / method / "seed_42.json",
            )
        )
    return files


def _assert_pre_metric_boundary() -> dict[str, Any]:
    incident = load_json(BASE_INCIDENT)
    if (
        incident.get("status")
        != "CLOSED_BEFORE_ANY_METRIC_COMPUTATION_BY_REGISTERED_ORDER_REALIZATION_CORRECTION"
        or incident.get("scope", {}).get("target_metric_values_computed_or_accessed") is not False
        or incident.get("disposition", {}).get("correction_protocol")
        != "nvd_real_snow_cvbra_v1_1"
    ):
        raise NvdSeedOrderCorrectionError("base incident does not authorize this correction")
    forbidden = (
        BASE_OUTPUT / "PREDICTIONS_COMPLETE.json",
        BASE_OUTPUT / "metrics.csv",
        BASE_OUTPUT / "real_snow_report.json",
        BASE_OUTPUT / "COMPLETE.json",
    )
    appeared = [relative(path) for path in forbidden if path.exists()]
    if appeared:
        raise NvdSeedOrderCorrectionError(
            f"base metric or completion artifacts appeared after incident lock: {appeared}"
        )
    return incident


def register() -> dict[str, Any]:
    if REGISTRATION.exists():
        return validate_registration()
    if OUTPUT.exists() or DATA_ROOT.exists():
        raise NvdSeedOrderCorrectionError("correction output appeared before registration")
    load_protocol()
    _assert_hash(BASE_INCIDENT, BASE_INCIDENT_SHA256, label="base incident")
    _assert_hash(BASE_REGISTRATION, BASE_REGISTRATION_SHA256, label="base registration")
    _assert_hash(BASE_DATA_LOCK, BASE_DATA_LOCK_SHA256, label="base data lock")
    _assert_pre_metric_boundary()
    required = [
        BASE_PROTOCOL,
        BASE_REGISTRATION,
        BASE_DATA_LOCK,
        BASE_INCIDENT,
        ULTRALYTICS_DATALOADER,
        *_base_seed_42_files(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise NvdSeedOrderCorrectionError(f"registered correction inputs are missing: {missing}")
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_SEED_ORDER_CORRECTION_REGISTERED_BEFORE_MATERIALIZATION",
        "registered_at_utc": utc_now(),
        "protocol": relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "preparation_runner": relative(Path(__file__)),
        "preparation_runner_sha256": sha256_file(Path(__file__)),
        "experiment_runner": relative(RUNNER),
        "experiment_runner_sha256": sha256_file(RUNNER),
        "base_incident": relative(BASE_INCIDENT),
        "base_incident_sha256": sha256_file(BASE_INCIDENT),
        "base_registration_sha256": sha256_file(BASE_REGISTRATION),
        "base_data_lock_sha256": sha256_file(BASE_DATA_LOCK),
        "ultralytics_dataloader": str(ULTRALYTICS_DATALOADER),
        "ultralytics_dataloader_sha256": sha256_file(ULTRALYTICS_DATALOADER),
        "retained_inputs": [
            {"path": relative(path), "sha256": sha256_file(path)}
            for path in _base_seed_42_files()
        ],
        "replacement_methods": list(METHODS),
        "replacement_seeds": list(REPLACEMENT_SEEDS),
        "target_metric_values_computed_or_accessed_before_registration": False,
        "method_hyperparameter_or_checkpoint_selection_effect": False,
        "uavdt_used": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    atomic_write_json(REGISTRATION, payload)
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION.is_file():
        raise NvdSeedOrderCorrectionError("seed-order correction is not registered")
    lock = load_json(REGISTRATION)
    expected = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "preparation_runner_sha256": sha256_file(Path(__file__)),
        "experiment_runner_sha256": sha256_file(RUNNER),
        "base_incident_sha256": sha256_file(BASE_INCIDENT),
        "base_registration_sha256": sha256_file(BASE_REGISTRATION),
        "base_data_lock_sha256": sha256_file(BASE_DATA_LOCK),
        "ultralytics_dataloader_sha256": sha256_file(ULTRALYTICS_DATALOADER),
    }
    drifted = [key for key, value in expected.items() if lock.get(key) != value]
    if drifted:
        raise NvdSeedOrderCorrectionError(f"registered correction inputs changed: {drifted}")
    for row in lock.get("retained_inputs", []):
        if not isinstance(row, dict):
            raise NvdSeedOrderCorrectionError("malformed retained-input row")
        _assert_hash(rooted(str(row["path"])), str(row["sha256"]), label="retained input")
    return lock


def _base_entries(method: str) -> tuple[dict[str, Any], ...]:
    manifest = load_json(BASE_DATA_ROOT / "datasets" / method / "manifest.json")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != TRAIN_IMAGES:
        raise NvdSeedOrderCorrectionError(f"base dataset coverage changed: {method}")
    if not all(isinstance(row, dict) for row in entries):
        raise NvdSeedOrderCorrectionError(f"malformed base dataset manifest: {method}")
    return tuple(entries)


def _multiset_hash(entries: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    values = sorted(
        (
            str(row["image"]),
            str(row["label"]),
            str(row.get("role", "")),
            str(row.get("view", "")),
        )
        for row in entries
    )
    return stable_hash(values, length=64)


def ordered_rows(method: str, seed: int) -> tuple[dict[str, Any], ...]:
    if method not in METHODS or seed not in REPLACEMENT_SEEDS:
        raise NvdSeedOrderCorrectionError(f"unregistered order realization: {method}/{seed}")
    rows = list(_base_entries(method))
    random.Random(seed).shuffle(rows)
    return tuple(rows)


def _materialize_one(method: str, seed: int) -> dict[str, Any]:
    rows = ordered_rows(method, seed)
    base_rows = _base_entries(method)
    if _multiset_hash(rows) != _multiset_hash(base_rows):
        raise NvdSeedOrderCorrectionError(f"dataset multiset changed: {method}/{seed}")
    destination = DATA_ROOT / "datasets" / method / f"seed_{seed}"
    images = destination / "images/train"
    labels = destination / "labels/train"
    images.mkdir(parents=True, exist_ok=False)
    labels.mkdir(parents=True, exist_ok=False)
    materialized: list[dict[str, Any]] = []
    for slot, row in enumerate(rows, start=1):
        source_image = rooted(str(row["image"]))
        source_label = rooted(str(row["label"]))
        if not source_image.is_file() or not source_label.is_file():
            raise NvdSeedOrderCorrectionError(f"base alias is missing: {source_image}")
        destination_image = images / f"slot_{slot:04d}{source_image.suffix.lower()}"
        destination_label = labels / f"slot_{slot:04d}.txt"
        os.link(source_image, destination_image)
        os.link(source_label, destination_label)
        if not os.path.samefile(source_image, destination_image) or not os.path.samefile(
            source_label, destination_label
        ):
            raise NvdSeedOrderCorrectionError(f"hard-link identity failed: {method}/{seed}/{slot}")
        materialized.append(
            {
                "slot": slot,
                "source_image": relative(source_image),
                "source_label": relative(source_label),
                "image": relative(destination_image),
                "label": relative(destination_label),
                "role": str(row.get("role", "")),
                "view": str(row.get("view", "")),
                "frame_index": int(row.get("frame_index", -1)),
            }
        )
    manifest_path = destination / "manifest.json"
    base_manifest = BASE_DATA_ROOT / "datasets" / method / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "NVD_SEED_ORDER_REALIZATION_MATERIALIZED",
        "method": method,
        "seed": seed,
        "images": len(materialized),
        "order_algorithm": "python_random_mt19937_shuffle_of_locked_base_manifest_rows",
        "base_manifest": relative(base_manifest),
        "base_manifest_sha256": sha256_file(base_manifest),
        "base_multiset_sha256": _multiset_hash(base_rows),
        "realized_multiset_sha256": _multiset_hash(rows),
        "ordered_source_images_sha256": stable_hash(
            [str(row["image"]) for row in rows], length=64
        ),
        "rows": materialized,
    }
    atomic_write_json(manifest_path, manifest)
    yaml_path = destination / "dataset.yaml"
    yaml_text = "\n".join(
        (
            f"path: {destination.resolve().as_posix()}",
            "train: images/train",
            "val: images/train",
            "names:",
            "  0: car",
            "",
        )
    )
    atomic_write_text(yaml_path, yaml_text)
    return {
        "method": method,
        "seed": seed,
        "images": len(materialized),
        "dataset_yaml": relative(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "base_manifest_sha256": manifest["base_manifest_sha256"],
        "multiset_sha256": manifest["realized_multiset_sha256"],
        "order_sha256": manifest["ordered_source_images_sha256"],
        "hard_links_verified": True,
    }


def materialize() -> dict[str, Any]:
    registration = validate_registration()
    _assert_pre_metric_boundary()
    if DATA_ROOT.exists() or DATA_LOCK.exists():
        raise NvdSeedOrderCorrectionError("partial correction data requires audit")
    rows = [
        _materialize_one(method, seed)
        for method in METHODS
        for seed in REPLACEMENT_SEEDS
    ]
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if len({str(row["order_sha256"]) for row in method_rows}) != len(REPLACEMENT_SEEDS):
            raise NvdSeedOrderCorrectionError(f"order realizations are not distinct: {method}")
        if len({str(row["multiset_sha256"]) for row in method_rows}) != 1:
            raise NvdSeedOrderCorrectionError(f"method multiset drifted across seeds: {method}")
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_SEED_ORDER_DATA_LOCKED_BEFORE_CORRECTED_TRAINING",
        "locked_at_utc": utc_now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "base_data_lock_sha256": registration["base_data_lock_sha256"],
        "datasets": rows,
        "replacement_methods": list(METHODS),
        "replacement_seeds": list(REPLACEMENT_SEEDS),
        "sample_multisets_unchanged": True,
        "order_realizations_distinct_within_method": True,
        "target_test_accessed": False,
        "uavdt_used": False,
    }
    atomic_write_json(DATA_LOCK, payload)
    return payload


def validate_data_lock() -> dict[str, Any]:
    validate_registration()
    if not DATA_LOCK.is_file():
        raise NvdSeedOrderCorrectionError("correction data lock is missing")
    lock = load_json(DATA_LOCK)
    if (
        lock.get("status")
        != "NVD_REAL_SNOW_SEED_ORDER_DATA_LOCKED_BEFORE_CORRECTED_TRAINING"
        or lock.get("registration_sha256") != sha256_file(REGISTRATION)
        or lock.get("sample_multisets_unchanged") is not True
        or lock.get("order_realizations_distinct_within_method") is not True
    ):
        raise NvdSeedOrderCorrectionError("correction data lock changed")
    rows = lock.get("datasets")
    if not isinstance(rows, list) or len(rows) != len(METHODS) * len(REPLACEMENT_SEEDS):
        raise NvdSeedOrderCorrectionError("correction dataset coverage changed")
    for row in rows:
        if not isinstance(row, dict):
            raise NvdSeedOrderCorrectionError("malformed correction dataset row")
        _assert_hash(
            rooted(str(row["dataset_yaml"])),
            str(row["dataset_yaml_sha256"]),
            label="correction dataset YAML",
        )
        _assert_hash(
            rooted(str(row["manifest"])),
            str(row["manifest_sha256"]),
            label="correction dataset manifest",
        )
        manifest = load_json(rooted(str(row["manifest"])))
        if (
            manifest.get("images") != TRAIN_IMAGES
            or manifest.get("base_multiset_sha256") != manifest.get("realized_multiset_sha256")
            or manifest.get("ordered_source_images_sha256") != row.get("order_sha256")
        ):
            raise NvdSeedOrderCorrectionError(
                f"correction dataset invariants changed: {row.get('method')}/{row.get('seed')}"
            )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the registered NVD seed-order correction")
    parser.add_argument("--stage", choices=("register", "materialize", "validate"), required=True)
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "materialize":
        result = materialize()
    else:
        result = validate_data_lock()
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
