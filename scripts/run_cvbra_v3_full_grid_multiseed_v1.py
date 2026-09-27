from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_training_seed_robustness as seed_evidence
from scripts import run_cvbra_v3_metric_integrity_v1 as metric_integrity

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/development/cvbra_v3_full_grid_multiseed_v1"
DATA_ROOT = ROOT / "data/processed/cvbra_v3_full_grid_multiseed_v1"
RUN_ROOT = ROOT / "runs/cvbra_v3_full_grid_multiseed_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
DATA_LOCK = OUTPUT / "DATASETS_LOCKED.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
METRICS = OUTPUT / "full_grid_metrics.json"
REPORT = OUTPUT / "full_grid_multiseed_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

SOURCE = ROOT / "weights/hazydet/yolo11n_best.pt"
BASE_MANIFESTS = {
    "qS_0": (ROOT / "data/processed/cvbra_v1_matched_baselines/CVBRA_noReplay/manifest.json"),
    "qS_0p125": (ROOT / "data/processed/cvbra_v1_allocation_sensitivity_v1/qS_0p125/manifest.json"),
    "qS_0p25": ROOT / "data/processed/cvbra_v1/manifest.json",
}

SEEDS = (27182, 31415)
ALL_SEEDS = (42, 27182, 31415)
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
COORDINATES = (*TARGET_VIEWS, "HazyDet")
CLASS_NAMES = ("car", "truck", "bus")
TARGET_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}

EPOCHS = 8
IMGSZ = 1280
BATCH = 2
LOW_FLOOR = 0.001
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


@dataclass(frozen=True)
class Cell:
    allocation: str
    dataset_family: str
    boundary: int


CELLS = (
    Cell("qS_0_L10", "qS_0", 10),
    Cell("qS_0p125_L10", "qS_0p125", 10),
    Cell("qS_0p25_L0", "qS_0p25", 0),
    Cell("qS_0p25_L15", "qS_0p25", 15),
)
ALL_ALLOCATIONS = (
    "qS_0_L10",
    "qS_0p125_L10",
    "qS_0p25_L0",
    "qS_0p25_L5",
    "qS_0p25_L10",
    "qS_0p25_L15",
)


class FullGridError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(seed: int, family: str) -> Path:
    return DATA_ROOT / f"seed_{seed}" / family / "dataset.yaml"


def _checkpoint(seed: int, allocation: str) -> Path:
    return RUN_ROOT / f"seed_{seed}" / allocation / f"{allocation}_s{seed}.pt"


def _fit(seed: int, allocation: str) -> Path:
    return RUN_ROOT / f"seed_{seed}" / allocation / "raw_endpoint/fit"


def _training_lock(seed: int, allocation: str) -> Path:
    return OUTPUT / "training_locks" / f"{allocation}_s{seed}.json"


def _prediction(seed: int, allocation: str, domain: str, view: str) -> Path:
    return (
        OUTPUT
        / "predictions"
        / f"seed_{seed}"
        / allocation
        / domain
        / view
        / "predictions.coco.json"
    )


def register() -> dict[str, Any]:
    required = [SOURCE, metric_integrity.COMPLETE, *BASE_MANIFESTS.values(), Path(__file__)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FullGridError(f"full-grid inputs are missing: {missing}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_MULTISEED_REGISTERED_BEFORE_DATA_OR_TRAINING",
        "registered_at_utc": _now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "metric_integrity_complete_sha256": sha256_file(metric_integrity.COMPLETE),
        "base_manifest_sha256": {
            family: sha256_file(path) for family, path in BASE_MANIFESTS.items()
        },
        "new_seeds": list(SEEDS),
        "complete_seed_set": list(ALL_SEEDS),
        "new_cells": [
            {
                "allocation": cell.allocation,
                "dataset_family": cell.dataset_family,
                "first_trainable_layer": cell.boundary,
            }
            for cell in CELLS
        ],
        "epochs": EPOCHS,
        "candidate_score_floor": LOW_FLOOR,
        "fixed_last_epoch": True,
        "validation_during_training": False,
        "post_hoc_fixed_design_replication": True,
        "method_or_hyperparameter_selection": False,
        "experimental_conclusion_reselection": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = tuple(
            key for key in payload if key not in {"schema_version", "status", "registered_at_utc"}
        )
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise FullGridError("full-grid registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _manifest_entries(path: Path) -> list[dict[str, Any]]:
    document = _load(path)
    rows = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(rows, list) or len(rows) != 3600:
        raise FullGridError(f"base manifest does not contain 3,600 entries: {path}")
    entries = [dict(row) for row in rows if isinstance(row, dict)]
    if len(entries) != 3600:
        raise FullGridError(f"base manifest contains invalid entries: {path}")
    return entries


def _rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _permutation_key(seed: int, family: str, row: Mapping[str, Any]) -> str:
    identity = "|".join(
        (
            str(seed),
            family,
            str(row.get("ordinal")),
            str(row.get("image")),
            str(row.get("label")),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or sha256_file(target) != sha256_file(source):
            raise FullGridError(f"existing full-grid alias differs: {target}")
        return
    os.link(source, target)


def build_data() -> dict[str, Any]:
    register()
    if DATA_LOCK.exists():
        return _load(DATA_LOCK)
    locks: list[dict[str, Any]] = []
    for seed in SEEDS:
        for family, manifest in BASE_MANIFESTS.items():
            dataset = _dataset(seed, family)
            entries = sorted(
                _manifest_entries(manifest),
                key=lambda row: _permutation_key(seed, family, row),
            )
            mapping: list[dict[str, Any]] = []
            for index, row in enumerate(entries, 1):
                source_image = _rooted(row["image"])
                source_label = _rooted(row["label"])
                stem = f"{index:04d}_{Path(source_image).stem}"
                target_image = (
                    dataset.parent / "images/train" / f"{stem}{source_image.suffix.lower()}"
                )
                target_label = dataset.parent / "labels/train" / f"{stem}.txt"
                _hardlink(source_image, target_image)
                _hardlink(source_label, target_label)
                mapping.append(
                    {
                        "position": index,
                        "base_ordinal": int(row.get("ordinal", index)),
                        "source_image": _relative(source_image),
                        "source_label": _relative(source_label),
                        "image": _relative(target_image),
                        "label": _relative(target_label),
                        "permutation_key": _permutation_key(seed, family, row),
                    }
                )
            yaml_text = (
                f"path: {dataset.parent.resolve().as_posix()}\n"
                "train: images/train\n"
                "val: images/train\n"
                "names:\n"
                "  0: car\n"
                "  1: truck\n"
                "  2: bus\n"
            )
            atomic_write_text(dataset, yaml_text)
            mapping_path = dataset.parent / "mapping.json"
            atomic_write_json(
                mapping_path,
                {
                    "schema_version": 1,
                    "status": "CVBRA_V3_FULL_GRID_ORDER_MATERIALIZED",
                    "seed": seed,
                    "family": family,
                    "base_manifest": _relative(manifest),
                    "base_manifest_sha256": sha256_file(manifest),
                    "entries": mapping,
                },
            )
            locks.append(
                {
                    "seed": seed,
                    "family": family,
                    "dataset": _relative(dataset),
                    "dataset_sha256": sha256_file(dataset),
                    "mapping": _relative(mapping_path),
                    "mapping_sha256": sha256_file(mapping_path),
                    "images": len(mapping),
                    "distinct_order_key_sha256": hashlib.sha256(
                        "\n".join(row["permutation_key"] for row in mapping).encode("utf-8")
                    ).hexdigest(),
                }
            )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_DATASETS_LOCKED_BEFORE_TRAINING",
        "locked_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "datasets": locks,
        "all_datasets_have_3600_entries": all(row["images"] == 3600 for row in locks),
        "seed_orders_distinct_within_family": all(
            len({row["distinct_order_key_sha256"] for row in locks if row["family"] == family})
            == len(SEEDS)
            for family in BASE_MANIFESTS
        ),
    }
    atomic_write_json(DATA_LOCK, payload)
    return payload


def _endpoint_state(
    source: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    boundary: int,
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise FullGridError("source and trained state schemas differ")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in trained.items():
        source_value = source[name]
        selected = source_value if seed_evidence._layer_index(name) < boundary else value
        state[name] = selected.clone()
    return state


def train_cell(seed: int, cell: Cell) -> dict[str, Any]:
    build_data()
    lock_path = _training_lock(seed, cell.allocation)
    checkpoint = _checkpoint(seed, cell.allocation)
    if lock_path.exists():
        lock = _load(lock_path)
        if lock.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise FullGridError(f"full-grid checkpoint changed: {checkpoint}")
        return lock
    fit = _fit(seed, cell.allocation)
    if fit.exists() or checkpoint.exists():
        raise FullGridError(f"partial full-grid training output: {seed}/{cell.allocation}")
    configure_ultralytics_environment(ROOT)
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(SOURCE))
    kwargs: dict[str, Any] = {
        "data": str(_dataset(seed, cell.dataset_family).resolve()),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": "0",
        "workers": 4,
        "project": str(fit.parent.resolve()),
        "name": fit.name,
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
        "seed": seed,
        "deterministic": True,
        "max_det": MAX_DET,
        "cache": False,
        "plots": False,
        "verbose": False,
        "save": True,
    }
    if cell.boundary:
        kwargs["freeze"] = cell.boundary
    results = model.train(**kwargs)
    actual = Path(results.save_dir)
    raw = actual / "weights/last.pt"
    results_csv = actual / "results.csv"
    args_yaml = actual / "args.yaml"
    for path in (raw, results_csv, args_yaml):
        if not path.is_file():
            raise FullGridError(f"training output is missing: {path}")
    source_payload, source_model = seed_evidence._load_checkpoint(SOURCE)
    _, trained_model = seed_evidence._load_checkpoint(raw)
    state = _endpoint_state(source_model.state_dict(), trained_model.state_dict(), cell.boundary)
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FullGridError("strict full-grid endpoint load failed")
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
            "date": _now(),
            "train_metrics": {},
            "train_results": {},
            "cvbra_v3_full_grid_multiseed": {
                "allocation": cell.allocation,
                "dataset_family": cell.dataset_family,
                "first_trainable_layer": cell.boundary,
                "seed": seed,
                "endpoint": "fixed_last_epoch",
                "validation_selected": False,
                "registration_sha256": sha256_file(REGISTRATION),
            },
        }
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    _, observed = seed_evidence._load_checkpoint(checkpoint)
    source_state = source_model.state_dict()
    observed_state = observed.state_dict()
    frozen_exact = all(
        torch.equal(observed_state[name].to(dtype=source_state[name].dtype), source_state[name])
        for name in observed_state
        if seed_evidence._layer_index(name) < cell.boundary
    )
    changed = sum(
        1
        for name, value in observed_state.items()
        if seed_evidence._layer_index(name) >= cell.boundary
        and value.is_floating_point()
        and not torch.equal(value.to(dtype=source_state[name].dtype), source_state[name])
    )
    if not frozen_exact or changed == 0:
        raise FullGridError("full-grid endpoint verification failed")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_REPLICATE_TRAINED_AND_LOCKED",
        "completed_at_utc": _now(),
        "seed": seed,
        "allocation": cell.allocation,
        "dataset_family": cell.dataset_family,
        "first_trainable_layer": cell.boundary,
        "dataset_sha256": sha256_file(_dataset(seed, cell.dataset_family)),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "results_sha256": sha256_file(results_csv),
        "args_sha256": sha256_file(args_yaml),
        "frozen_state_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed,
        "validation_metric_used_for_training_or_selection": False,
    }
    atomic_write_json(lock_path, lock)
    del model, output_model, source_model, trained_model, observed
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"full_grid_training_complete": [seed, cell.allocation]}), flush=True)
    return lock


def train_all() -> dict[str, Any]:
    locks = [train_cell(seed, cell) for seed in SEEDS for cell in CELLS]
    return {"status": "CVBRA_V3_ALL_FULL_GRID_REPLICATES_TRAINED", "locks": locks}


def _write_prediction(
    *,
    seed: int,
    allocation: str,
    domain: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    path = _prediction(seed, allocation, domain, view)
    marker = path.parent / "SUCCESS.json"
    if marker.exists():
        value = _load(marker)
        if value.get("prediction_sha256") != sha256_file(path):
            raise FullGridError(f"full-grid prediction changed: {path}")
        return value
    if path.parent.exists():
        raise FullGridError(f"partial full-grid prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=LOW_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    rows = write_coco_predictions(path, batches, category_id_by_class=category_map)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_LOW_FLOOR_PREDICTION_COMPLETE",
        "seed": seed,
        "allocation": allocation,
        "domain": domain,
        "view": view,
        "images": len(records),
        "detections": len(rows),
        "minimum_score": min(float(row["score"]) for row in rows),
        "checkpoint_sha256": sha256_file(_checkpoint(seed, allocation)),
        "prediction_sha256": sha256_file(path),
    }
    atomic_write_json(marker, payload)
    return payload


def infer() -> dict[str, Any]:
    train_all()
    if PREDICTION_LOCK.exists():
        return _load(PREDICTION_LOCK)
    target_records, _ = sensitivity._target_records()
    hazy_records = seed_evidence._hazy_records(verify_hashes=False)
    artifacts: list[dict[str, Any]] = []
    configure_ultralytics_environment(ROOT)
    for seed in SEEDS:
        for cell in CELLS:
            detector = UltralyticsDetector(
                _checkpoint(seed, cell.allocation),
                model_name="yolo11n",
                device="cuda:0",
                expected_class_names=CLASS_NAMES,
                project_root=ROOT,
                stream_chunk_records=CHUNK_SIZE,
                release_cuda_cache_between_chunks=False,
            )
            detector.predict(
                target_records["original"][:WARMUP_IMAGES],
                imgsz=IMGSZ,
                conf=LOW_FLOOR,
                iou=NMS_IOU,
                max_det=MAX_DET,
                fp16=True,
            )
            for view in TARGET_VIEWS:
                artifacts.append(
                    _write_prediction(
                        seed=seed,
                        allocation=cell.allocation,
                        domain="UAV_OBB_validation",
                        view=view,
                        detector=detector,
                        records=target_records[view],
                        category_map=TARGET_CATEGORY_ID_BY_CLASS,
                    )
                )
            artifacts.append(
                _write_prediction(
                    seed=seed,
                    allocation=cell.allocation,
                    domain="HazyDet_validation",
                    view="hazy",
                    detector=detector,
                    records=hazy_records,
                    category_map=HAZY_CATEGORY_ID_BY_CLASS,
                )
            )
            del detector
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({"full_grid_inference_complete": [seed, cell.allocation]}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_PREDICTIONS_LOCKED",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "artifacts": artifacts,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def metrics() -> dict[str, Any]:
    infer()
    if METRICS.exists():
        return _load(METRICS)
    _, primary_ids = sensitivity._target_records()
    hazy_ids = [int(record.image_id) for record in seed_evidence._hazy_records(verify_hashes=False)]
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for cell in CELLS:
            for view in TARGET_VIEWS:
                result = evaluate_coco(
                    metric_integrity.TARGET_ANNOTATION,
                    _prediction(seed, cell.allocation, "UAV_OBB_validation", view),
                    max_det=MAX_DET,
                    image_ids=primary_ids,
                )
                rows.append(
                    {
                        "seed": seed,
                        "allocation": cell.allocation,
                        "coordinate": view,
                        **{key: float(result[key]) for key in METRIC_KEYS},
                        "images_evaluated": int(result["images_evaluated"]),
                    }
                )
            result = evaluate_coco(
                metric_integrity.HAZY_ANNOTATION,
                _prediction(seed, cell.allocation, "HazyDet_validation", "hazy"),
                max_det=MAX_DET,
                image_ids=hazy_ids,
            )
            rows.append(
                {
                    "seed": seed,
                    "allocation": cell.allocation,
                    "coordinate": "HazyDet",
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(result["images_evaluated"]),
                }
            )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V3_FULL_GRID_LOW_FLOOR_METRICS_COMPLETE",
        "candidate_score_floor": LOW_FLOOR,
        "rows": rows,
    }
    atomic_write_json(METRICS, payload)
    return payload


def _metric_integrity_lookup() -> dict[tuple[str, str], float]:
    report = _load(metric_integrity.POINT_REPORT)
    lookup = {
        (str(row["model"]), str(row["view"])): float(row["AP"]) for row in report["target_rows"]
    }
    lookup.update(
        {(str(row["model"]), "HazyDet"): float(row["AP"]) for row in report["hazydet_rows"]}
    )
    return lookup


def _seed42_model(allocation: str) -> str:
    return {
        "qS_0_L10": "CVBRA_noReplay",
        "qS_0p125_L10": "qS_0p125_L10",
        "qS_0p25_L0": "CVBRA_noFreeze",
        "qS_0p25_L5": "CVBRA_L5_s42",
        "qS_0p25_L10": "CVBRA_L10_s42",
        "qS_0p25_L15": "qS_0p25_L15",
    }[allocation]


def _nondominated(profiles: Mapping[str, Mapping[str, float]]) -> list[str]:
    result: list[str] = []
    for name, profile in profiles.items():
        if not any(
            other != name
            and all(other_profile[key] >= profile[key] for key in COORDINATES)
            and any(other_profile[key] > profile[key] for key in COORDINATES)
            for other, other_profile in profiles.items()
        ):
            result.append(name)
    return result


def analyze() -> dict[str, Any]:
    metric_report = metrics()
    if REPORT.exists():
        return _load(REPORT)
    base = _metric_integrity_lookup()
    profiles: dict[int, dict[str, dict[str, float]]] = {seed: {} for seed in ALL_SEEDS}
    for allocation in ALL_ALLOCATIONS:
        profiles[42][allocation] = {
            coordinate: base[(_seed42_model(allocation), coordinate)] for coordinate in COORDINATES
        }
    for row in metric_report["rows"]:
        seed = int(row["seed"])
        allocation = str(row["allocation"])
        profiles[seed].setdefault(allocation, {})[str(row["coordinate"])] = float(row["AP"])
    for seed in SEEDS:
        for boundary in (5, 10):
            allocation = f"qS_0p25_L{boundary}"
            model = f"CVBRA_L{boundary}_s{seed}"
            profiles[seed][allocation] = {
                coordinate: base[(model, coordinate)] for coordinate in COORDINATES
            }
    if any(
        set(seed_profiles) != set(ALL_ALLOCATIONS)
        or any(set(profile) != set(COORDINATES) for profile in seed_profiles.values())
        for seed_profiles in profiles.values()
    ):
        raise FullGridError("full-grid profile coverage is incomplete")
    summaries: dict[str, dict[str, Any]] = {}
    mean_profiles: dict[str, dict[str, float]] = {}
    for allocation in ALL_ALLOCATIONS:
        mean_profiles[allocation] = {}
        summaries[allocation] = {}
        for coordinate in COORDINATES:
            values = [profiles[seed][allocation][coordinate] for seed in ALL_SEEDS]
            mean_profiles[allocation][coordinate] = fmean(values)
            summaries[allocation][coordinate] = {
                "values": values,
                "mean": fmean(values),
                "sample_standard_deviation": stdev(values),
            }
    nondominated_by_seed = {str(seed): _nondominated(profiles[seed]) for seed in ALL_SEEDS}
    mean_nondominated = _nondominated(mean_profiles)
    retention_breakpoints = sorted({mean_profiles[name]["HazyDet"] for name in mean_nondominated})
    selections: list[dict[str, Any]] = []
    for retention in retention_breakpoints:
        feasible = [
            name for name in mean_nondominated if mean_profiles[name]["HazyDet"] >= retention
        ]
        selected = max(
            feasible,
            key=lambda name: min(mean_profiles[name][view] for view in TARGET_VIEWS),
        )
        selections.append(
            {
                "retention_floor": retention,
                "selected_allocation": selected,
                "worst_view_target_AP": min(mean_profiles[selected][view] for view in TARGET_VIEWS),
            }
        )
    report = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V3_FULL_GRID_MULTISEED_ANALYSIS",
        "completed_at_utc": _now(),
        "candidate_score_floor": LOW_FLOOR,
        "profiles_by_seed": profiles,
        "summary_by_allocation": summaries,
        "mean_profiles": mean_profiles,
        "nondominated_by_seed": nondominated_by_seed,
        "mean_nondominated_allocations": mean_nondominated,
        "retention_constrained_mean_selection": selections,
        "all_six_allocations_have_three_trajectories": True,
        "post_hoc_fixed_design_replication": True,
        "validation_metric_used_for_training_or_checkpoint_selection": False,
    }
    atomic_write_json(REPORT, report)
    atomic_write_json(
        COMPLETE,
        {
            "schema_version": 1,
            "status": report["status"],
            "report_sha256": sha256_file(REPORT),
            "all_six_allocations_have_three_trajectories": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "build-data", "train", "infer", "metrics", "analyze"),
    )
    args = parser.parse_args()
    result = {
        "register": register,
        "build-data": build_data,
        "train": train_all,
        "infer": infer,
        "metrics": metrics,
        "analyze": analyze,
    }[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
