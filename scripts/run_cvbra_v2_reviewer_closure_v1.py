from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import statistics
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_final_test_v1 as final_test
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v2_reviewer_closure_v1.yaml"
SOURCE = ROOT / "weights/hazydet/yolo11n_best.pt"
DATASETS = {
    27182: ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_a/dataset.yaml",
    31415: ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_b/dataset.yaml",
}
PRIMARY = {
    5: ROOT / "runs/cvbra_v1_allocation_sensitivity_v1/first_trainable_5/first_trainable_5.pt",
    10: ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt",
}
SEEDS = (42, 27182, 31415)
NEW_SEEDS = (27182, 31415)
BOUNDARIES = (5, 10)
OUTPUT = ROOT / "reports/development/cvbra_v2_reviewer_closure_v1"
RUN_ROOT = ROOT / "runs/cvbra_v2_reviewer_closure_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
METRICS = OUTPUT / "replication_metrics.csv"
REPORT = OUTPUT / "replication_report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

EPOCHS = 8
IMGSZ = 1280
BATCH = 2
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
WARMUP_IMAGES = 16
CLASS_NAMES = ("car", "truck", "bus")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
TARGET_VIEWS = ("original", "fog_0p6", "fog_1p0")
TARGET_CATEGORY_ID_BY_CLASS = {0: 1, 1: 2, 2: 3}
HAZY_CATEGORY_ID_BY_CLASS = {0: 0, 1: 1, 2: 2}


class ReviewerClosureError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewerClosureError(f"expected JSON object: {path}")
    return value


def _checkpoint(boundary: int, seed: int) -> Path:
    if seed == 42:
        return PRIMARY[boundary]
    return RUN_ROOT / f"layer_{boundary}" / f"seed_{seed}" / f"cvbra_l{boundary}_s{seed}.pt"


def _fit(boundary: int, seed: int) -> Path:
    return RUN_ROOT / f"layer_{boundary}" / f"seed_{seed}" / "raw_endpoint" / "fit"


def _training_lock(boundary: int, seed: int) -> Path:
    return OUTPUT / "training_locks" / f"layer_{boundary}_seed_{seed}.json"


def _prediction(boundary: int, seed: int, dataset: str, view: str) -> Path:
    return (
        OUTPUT
        / "predictions"
        / f"layer_{boundary}"
        / f"seed_{seed}"
        / dataset
        / view
        / "predictions.coco.json"
    )


def _state_from_boundary(
    source: Mapping[str, torch.Tensor], trained: Mapping[str, torch.Tensor], boundary: int
) -> OrderedDict[str, torch.Tensor]:
    if tuple(source) != tuple(trained):
        raise ReviewerClosureError("source and trained state schemas differ")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in trained.items():
        if source[name].shape != value.shape:
            raise ReviewerClosureError(f"state shape changed: {name}")
        selected = source[name] if evidence_base._layer_index(name) < boundary else value
        state[name] = selected.clone()
    return state


def register() -> dict[str, Any]:
    required = [PROTOCOL, SOURCE, *DATASETS.values(), *PRIMARY.values()]
    for path in required:
        if not path.is_file():
            raise ReviewerClosureError(f"required artifact is missing: {path}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V2_REVIEWER_CLOSURE_REGISTERED",
        "locked_at_utc": _now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "seeds": list(SEEDS),
        "boundaries": list(BOUNDARIES),
        "dataset_sha256": {str(seed): sha256_file(path) for seed, path in DATASETS.items()},
        "existing_checkpoint_sha256": {
            str(boundary): sha256_file(path) for boundary, path in PRIMARY.items()
        },
        "validation_and_test_labels_previously_accessed": True,
        "new_test_results_are_post_hoc_fixed_checkpoint_checks": True,
        "primary_endpoint_replacement": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable_fields = (
            "protocol_sha256",
            "runner_sha256",
            "source_checkpoint_sha256",
            "seeds",
            "boundaries",
            "dataset_sha256",
            "existing_checkpoint_sha256",
        )
        if any(existing.get(field) != payload.get(field) for field in stable_fields):
            raise ReviewerClosureError("reviewer-closure registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _validate_checkpoint(boundary: int, seed: int) -> dict[str, Any]:
    checkpoint = _checkpoint(boundary, seed)
    lock = _training_lock(boundary, seed)
    if not checkpoint.is_file() or not lock.is_file():
        raise ReviewerClosureError(f"checkpoint is incomplete: layer {boundary}, seed {seed}")
    value = _load(lock)
    if value.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ReviewerClosureError(f"checkpoint lock changed: {checkpoint}")
    return value


def train_one(boundary: int, seed: int) -> dict[str, Any]:
    register()
    if boundary not in BOUNDARIES or seed not in NEW_SEEDS:
        raise ReviewerClosureError("only registered new seed/boundary cells may be trained")
    if _training_lock(boundary, seed).exists():
        return _validate_checkpoint(boundary, seed)
    checkpoint = _checkpoint(boundary, seed)
    fit = _fit(boundary, seed)
    if checkpoint.exists() or fit.exists():
        raise ReviewerClosureError(f"partial training output requires audit: {boundary}/{seed}")
    configure_ultralytics_environment(ROOT)
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(SOURCE))
    results = model.train(
        data=str(DATASETS[seed].resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="0",
        workers=4,
        project=str(fit.parent.resolve()),
        name=fit.name,
        exist_ok=False,
        pretrained=True,
        freeze=boundary,
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
        seed=seed,
        deterministic=True,
        max_det=MAX_DET,
        cache=False,
        plots=False,
        verbose=False,
        save=True,
    )
    actual = Path(results.save_dir)
    raw = actual / "weights/last.pt"
    results_csv = actual / "results.csv"
    args_yaml = actual / "args.yaml"
    for path in (raw, results_csv, args_yaml):
        if not path.is_file():
            raise ReviewerClosureError(f"training output is missing: {path}")
    source_payload, source_model = evidence_base._load_checkpoint(SOURCE)
    _, trained_model = evidence_base._load_checkpoint(raw)
    state = _state_from_boundary(source_model.state_dict(), trained_model.state_dict(), boundary)
    output_model = copy.deepcopy(trained_model)
    incompatible = output_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ReviewerClosureError("strict endpoint load failed")
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
            "cvbra_v2_reviewer_closure": {
                "boundary": boundary,
                "seed": seed,
                "dataset_order": _relative(DATASETS[seed]),
                "endpoint": "fixed_last_epoch",
                "registration_sha256": sha256_file(REGISTRATION),
                "validation_selected": False,
            },
        }
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    _, observed = evidence_base._load_checkpoint(checkpoint)
    source_state = source_model.state_dict()
    observed_state = observed.state_dict()
    frozen_exact = all(
        torch.equal(observed_state[name].to(dtype=source_state[name].dtype), source_state[name])
        for name in observed_state
        if evidence_base._layer_index(name) < boundary
    )
    changed = sum(
        1
        for name, value in observed_state.items()
        if evidence_base._layer_index(name) >= boundary
        and value.is_floating_point()
        and not torch.equal(value.to(dtype=source_state[name].dtype), source_state[name])
    )
    if not frozen_exact or changed == 0:
        raise ReviewerClosureError("replicate endpoint verification failed")
    lock = {
        "schema_version": 1,
        "status": "CVBRA_V2_REPLICATE_TRAINED_AND_LOCKED",
        "completed_at_utc": _now(),
        "boundary": boundary,
        "seed": seed,
        "dataset_yaml": _relative(DATASETS[seed]),
        "dataset_yaml_sha256": sha256_file(DATASETS[seed]),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "raw_last_sha256": sha256_file(raw),
        "results_sha256": sha256_file(results_csv),
        "args_sha256": sha256_file(args_yaml),
        "frozen_state_exact_source": frozen_exact,
        "changed_trainable_floating_states": changed,
        "validation_metric_used_for_training_or_selection": False,
    }
    atomic_write_json(_training_lock(boundary, seed), lock)
    print(json.dumps({"trained": [boundary, seed]}), flush=True)
    return lock


def train_all() -> dict[str, Any]:
    locks: dict[str, Any] = {}
    for boundary in BOUNDARIES:
        for seed in NEW_SEEDS:
            locks[f"layer_{boundary}_seed_{seed}"] = train_one(boundary, seed)
    return {"status": "ALL_REVIEWER_CLOSURE_REPLICATES_TRAINED", "locks": locks}


def _filtered(batches: Sequence[DetectionBatch]) -> tuple[DetectionBatch, ...]:
    return tuple(
        DetectionBatch(
            image_id=batch.image_id,
            boxes=tuple(box for box in batch.boxes if box.score >= PUBLISH_CONF),
            latency_ms=batch.latency_ms,
            meta={**batch.meta, "publish_conf": PUBLISH_CONF},
        )
        for batch in batches
    )


def _predict(
    boundary: int,
    seed: int,
    dataset: str,
    view: str,
    detector: UltralyticsDetector,
    records: Sequence[ImageRecord],
    category_map: Mapping[int, int],
) -> dict[str, Any]:
    path = _prediction(boundary, seed, dataset, view)
    marker = path.parent / "SUCCESS.json"
    if marker.exists():
        value = _load(marker)
        if value.get("prediction_sha256") != sha256_file(path):
            raise ReviewerClosureError(f"prediction changed: {path}")
        return value
    if path.parent.exists():
        raise ReviewerClosureError(f"partial prediction output: {path.parent}")
    batches = detector.predict(
        records,
        imgsz=IMGSZ,
        conf=PROBE_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        fp16=True,
    )
    filtered = _filtered(batches)
    write_coco_predictions(path, filtered, category_id_by_class=category_map)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V2_REPLICATE_PREDICTION_COMPLETE",
        "boundary": boundary,
        "seed": seed,
        "dataset": dataset,
        "view": view,
        "images": len(filtered),
        "checkpoint_sha256": sha256_file(_checkpoint(boundary, seed)),
        "prediction_sha256": sha256_file(path),
    }
    atomic_write_json(marker, payload)
    return payload


def infer() -> dict[str, Any]:
    register()
    for boundary in BOUNDARIES:
        for seed in NEW_SEEDS:
            _validate_checkpoint(boundary, seed)
    if PREDICTION_LOCK.exists():
        return _load(PREDICTION_LOCK)
    target_validation, _ = sensitivity._target_records()
    hazy_validation = evidence_base._hazy_records(verify_hashes=False)
    target_test = final_test._uav_records()
    hazy_test = final_test._hazy_records()
    artifacts: list[dict[str, Any]] = []
    for boundary in BOUNDARIES:
        for seed in SEEDS:
            detector = UltralyticsDetector(
                _checkpoint(boundary, seed),
                model_name="yolo11n",
                device="cuda:0",
                expected_class_names=CLASS_NAMES,
                project_root=ROOT,
                stream_chunk_records=CHUNK_SIZE,
                release_cuda_cache_between_chunks=False,
            )
            detector.predict(
                target_validation["original"][:WARMUP_IMAGES],
                imgsz=IMGSZ,
                conf=PROBE_CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                fp16=True,
            )
            for view in TARGET_VIEWS:
                artifacts.append(
                    _predict(
                        boundary,
                        seed,
                        "UAV_OBB_validation",
                        view,
                        detector,
                        target_validation[view],
                        TARGET_CATEGORY_ID_BY_CLASS,
                    )
                )
            artifacts.append(
                _predict(
                    boundary,
                    seed,
                    "HazyDet_validation",
                    "hazy",
                    detector,
                    hazy_validation,
                    HAZY_CATEGORY_ID_BY_CLASS,
                )
            )
            for view in TARGET_VIEWS:
                artifacts.append(
                    _predict(
                        boundary,
                        seed,
                        "UAV_OBB_test",
                        view,
                        detector,
                        target_test[view],
                        TARGET_CATEGORY_ID_BY_CLASS,
                    )
                )
            artifacts.append(
                _predict(
                    boundary,
                    seed,
                    "HazyDet_test",
                    "hazy",
                    detector,
                    hazy_test,
                    HAZY_CATEGORY_ID_BY_CLASS,
                )
            )
            del detector
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({"inferred": [boundary, seed]}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V2_REVIEWER_CLOSURE_PREDICTIONS_COMPLETE",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "artifacts": artifacts,
        "validation_and_test_labels_previously_accessed": True,
        "new_test_predictions_are_post_hoc_fixed_checkpoint_checks": True,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def _row(
    boundary: int,
    seed: int,
    dataset: str,
    view: str,
    annotation: Path,
    image_ids: Sequence[int | str],
) -> dict[str, Any]:
    result = evaluate_coco(
        annotation,
        _prediction(boundary, seed, dataset, view),
        max_det=MAX_DET,
        image_ids=image_ids,
    )
    return {
        "boundary": boundary,
        "seed": seed,
        "dataset": dataset,
        "view": view,
        **{key: float(result[key]) for key in METRIC_KEYS},
        "images_evaluated": int(result["images_evaluated"]),
    }


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("boundary", "seed", "dataset", "view", *METRIC_KEYS, "images_evaluated")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue()


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for boundary in BOUNDARIES:
        result[str(boundary)] = {}
        for dataset, view in sorted({(str(row["dataset"]), str(row["view"])) for row in rows}):
            selected = [
                float(row["AP"])
                for row in rows
                if int(row["boundary"]) == boundary
                and row["dataset"] == dataset
                and row["view"] == view
            ]
            if len(selected) != len(SEEDS):
                raise ReviewerClosureError(
                    f"incomplete replication cell: {boundary}/{dataset}/{view}"
                )
            result[str(boundary)][f"{dataset}/{view}"] = {
                "mean_AP": statistics.mean(selected),
                "sample_sd_AP": statistics.stdev(selected),
                "minimum_AP": min(selected),
                "maximum_AP": max(selected),
            }
    paired: dict[str, Any] = {}
    for dataset, view in sorted({(str(row["dataset"]), str(row["view"])) for row in rows}):
        deltas = []
        for seed in SEEDS:
            lookup = {
                int(row["boundary"]): float(row["AP"])
                for row in rows
                if int(row["seed"]) == seed and row["dataset"] == dataset and row["view"] == view
            }
            deltas.append(lookup[5] - lookup[10])
        paired[f"{dataset}/{view}"] = {
            "layer5_minus_layer10_by_seed": dict(zip(map(str, SEEDS), deltas, strict=True)),
            "mean_difference": statistics.mean(deltas),
            "layer5_higher_in_all_three": all(value > 0 for value in deltas),
        }
    return {"by_boundary": result, "paired_layer5_minus_layer10": paired}


def _state_diversity() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for boundary in BOUNDARIES:
        states = {}
        for seed in SEEDS:
            _, model = evidence_base._load_checkpoint(_checkpoint(boundary, seed))
            states[seed] = model.state_dict()
        comparisons = {}
        for left, right in ((42, 27182), (42, 31415), (27182, 31415)):
            differing = sum(
                1
                for name in states[left]
                if evidence_base._layer_index(name) >= boundary
                and not torch.equal(states[left][name], states[right][name])
            )
            comparisons[f"{left}_vs_{right}"] = differing
        output[str(boundary)] = {
            "differing_trainable_state_entries": comparisons,
            "all_pairs_distinct": all(value > 0 for value in comparisons.values()),
        }
    return output


def score() -> dict[str, Any]:
    infer()
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        complete = _load(COMPLETE)
        if complete.get("report_sha256") != sha256_file(REPORT):
            raise ReviewerClosureError("replication report changed")
        return report
    target_conversion = _load(evidence_base.CONVERSION_LOCK)
    target_validation_annotation = ROOT / str(target_conversion["annotation"])
    _, primary_ids = sensitivity._target_records()
    hazy_validation_records = evidence_base._hazy_records(verify_hashes=False)
    hazy_validation_ids = [record.image_id for record in hazy_validation_records]
    hazy_test_document = _load(final_test.HAZY_ANNOTATION)
    hazy_test_ids = [int(image["id"]) for image in hazy_test_document["images"]]
    rows: list[dict[str, Any]] = []
    for boundary in BOUNDARIES:
        for seed in SEEDS:
            for view in TARGET_VIEWS:
                rows.append(
                    _row(
                        boundary,
                        seed,
                        "UAV_OBB_validation",
                        view,
                        target_validation_annotation,
                        primary_ids,
                    )
                )
            rows.append(
                _row(
                    boundary,
                    seed,
                    "HazyDet_validation",
                    "hazy",
                    evidence_base.HAZY_ANNOTATION,
                    hazy_validation_ids,
                )
            )
            for view in TARGET_VIEWS:
                rows.append(
                    _row(
                        boundary,
                        seed,
                        "UAV_OBB_test",
                        view,
                        final_test.UAV_ANNOTATION,
                        list(range(1, final_test.UAV_IMAGES + 1)),
                    )
                )
            rows.append(
                _row(
                    boundary,
                    seed,
                    "HazyDet_test",
                    "hazy",
                    final_test.HAZY_ANNOTATION,
                    hazy_test_ids,
                )
            )
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["view"]),
            int(row["boundary"]),
            int(row["seed"]),
        )
    )
    atomic_write_text(METRICS, _csv(rows))
    state_diversity = _state_diversity()
    summaries = _summaries(rows)
    all_distinct = all(value["all_pairs_distinct"] for value in state_diversity.values())
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V2_REVIEWER_CLOSURE_REPLICATION",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "seeds": list(SEEDS),
        "boundaries": list(BOUNDARIES),
        "rows": rows,
        "summaries": summaries,
        "state_diversity": state_diversity,
        "all_seed_boundary_trajectories_effectively_distinct": all_distinct,
        "interpretation": {
            "layer10_remains_original_confirmatory_endpoint": True,
            "layer5_is_response_refined_fixed_checkpoint": True,
            "new_test_predictions_are_post_hoc": True,
            "population_level_claim_from_16_image_test": False,
            "negative_results_retained": True,
        },
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "report_sha256": sha256_file(REPORT),
            "metrics_sha256": sha256_file(METRICS),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA reviewer-closure replications")
    parser.add_argument(
        "--stage", choices=("register", "train", "infer", "score", "all"), default="all"
    )
    parser.add_argument("--boundary", type=int, choices=BOUNDARIES)
    parser.add_argument("--seed", type=int, choices=NEW_SEEDS)
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "train":
        if (args.boundary is None) != (args.seed is None):
            raise ReviewerClosureError("--boundary and --seed must be supplied together")
        result = train_one(args.boundary, args.seed) if args.boundary is not None else train_all()
    elif args.stage == "infer":
        result = infer()
    elif args.stage == "score":
        result = score()
    else:
        train_all()
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
