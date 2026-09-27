from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import torch
from scripts import run_cvbra_v1_allocation_sensitivity_v1 as sensitivity
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base
from scripts import run_cvbra_v2_aldi_horizon_normalized_v1 as aldi_hn

from buse_uav.data.corruptions import (
    _read_rgb,
    _write_png,
    apply_corruption,
    deterministic_corruption_seed,
)
from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.evaluation.coco import evaluate_coco, write_coco_predictions
from buse_uav.schemas import DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v2_visibility_stress_v1.yaml"
OUTPUT = ROOT / "reports/development/cvbra_v2_visibility_stress_v1"
DATA_ROOT = ROOT / "data/processed/cvbra_v2_visibility_stress_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
MANIFEST = DATA_ROOT / "manifest.json"
PREDICTION_LOCK = OUTPUT / "PREDICTIONS_LOCKED.json"
METRICS = OUTPUT / "metrics.csv"
REPORT = OUTPUT / "report.json"
COMPLETE = OUTPUT / "COMPLETE.json"

MODELS = {
    "source": ROOT / "weights/hazydet/yolo11n_best.pt",
    "STF": ROOT / "runs/cvbra_v1_matched_baselines/STF/STF.pt",
    "ALDIpp_AF_equal": ROOT
    / "runs/cvbra_v1_aldi_direct_baseline_v1/equal_supervision"
    / "ALDIpp_AF_Y11_equal_supervision.pt",
    "ALDIpp_AF_HN_equal": aldi_hn.CHECKPOINT,
    "CVBRA_L10": ROOT / "runs/cvbra_v1/yolo11n/cvbra_v1.pt",
    "CVBRA_L5": ROOT
    / "runs/cvbra_v1_allocation_sensitivity_v1/first_trainable_5/first_trainable_5.pt",
}
CONDITIONS = {
    "fog_1p3": ("fog", 1.3, 3),
    "low_light_1p5": ("low_light", 1.5, 1),
}
GLOBAL_SEED = 20260822
IMGSZ = 1280
PROBE_CONF = 0.08
PUBLISH_CONF = 0.25
NMS_IOU = 0.70
MAX_DET = 500
CHUNK_SIZE = 8
CLASS_NAMES = ("car", "truck", "bus")
METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class VisibilityStressError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisibilityStressError(f"expected JSON object: {path}")
    return value


def register() -> dict[str, Any]:
    existing_models = {name: path for name, path in MODELS.items() if name != "ALDIpp_AF_HN_equal"}
    for path in (PROTOCOL, *existing_models.values(), aldi_hn.REGISTRATION):
        if not path.is_file():
            raise VisibilityStressError(f"required stress-test input is missing: {path}")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_VISIBILITY_STRESS_REGISTERED",
        "locked_at_utc": _now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "conditions": {
            name: {"corruption": values[0], "parameter": values[1], "severity": values[2]}
            for name, values in CONDITIONS.items()
        },
        "existing_model_sha256": {
            name: sha256_file(path) for name, path in existing_models.items()
        },
        "horizon_normalized_registration_sha256": sha256_file(aldi_hn.REGISTRATION),
        "validation_labels_previously_accessed": True,
        "method_or_hyperparameter_selection": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = (
            "protocol_sha256",
            "runner_sha256",
            "conditions",
            "existing_model_sha256",
            "horizon_normalized_registration_sha256",
        )
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise VisibilityStressError("visibility-stress registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def materialize() -> dict[str, Any]:
    register()
    if MANIFEST.exists():
        manifest = _load(MANIFEST)
        rows = manifest.get("rows")
        if not isinstance(rows, list) or len(rows) != 167 * len(CONDITIONS):
            raise VisibilityStressError("stress manifest is incomplete")
        for row in rows:
            path = ROOT / str(row["output"])
            if not path.is_file() or sha256_file(path) != row["output_sha256"]:
                raise VisibilityStressError(f"stress image changed: {path}")
        current_registration = sha256_file(REGISTRATION)
        if manifest.get("registration_sha256") != current_registration:
            manifest["registration_sha256"] = current_registration
            manifest["registration_relinked_at_utc"] = _now()
            atomic_write_json(MANIFEST, manifest)
        return manifest
    if DATA_ROOT.exists():
        raise VisibilityStressError("partial stress materialization requires audit")
    target, primary_ids = sensitivity._target_records()
    primary = set(primary_ids)
    records = [record for record in target["original"] if int(record.image_id) in primary]
    if len(records) != 167:
        raise VisibilityStressError("primary stress population changed")
    rows = []
    for condition, (corruption, parameter, severity) in CONDITIONS.items():
        for record in records:
            source = Path(record.path)
            seed = deterministic_corruption_seed(
                GLOBAL_SEED, str(record.image_id), corruption, severity
            )
            image = apply_corruption(
                _read_rgb(source), corruption=corruption, parameter=parameter, seed=seed
            )
            output = DATA_ROOT / condition / f"{int(record.image_id):06d}.png"
            _write_png(output, image)
            rows.append(
                {
                    "condition": condition,
                    "image_id": int(record.image_id),
                    "source": _relative(source),
                    "source_sha256": sha256_file(source),
                    "output": _relative(output),
                    "output_sha256": sha256_file(output),
                    "width": int(record.width),
                    "height": int(record.height),
                    "corruption": corruption,
                    "parameter": parameter,
                    "seed": seed,
                }
            )
    rows.sort(key=lambda row: (str(row["condition"]), int(row["image_id"])))
    payload = {
        "schema_version": 1,
        "status": "CVBRA_VISIBILITY_STRESS_MATERIALIZED",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "images": 167,
        "conditions": list(CONDITIONS),
        "rows": rows,
    }
    atomic_write_json(MANIFEST, payload)
    return payload


def _records() -> dict[str, tuple[ImageRecord, ...]]:
    manifest = materialize()
    output = {}
    for condition in CONDITIONS:
        selected = [row for row in manifest["rows"] if row["condition"] == condition]
        selected.sort(key=lambda row: int(row["image_id"]))
        output[condition] = tuple(
            ImageRecord(
                image_id=int(row["image_id"]),
                path=str((ROOT / str(row["output"])).resolve()),
                width=int(row["width"]),
                height=int(row["height"]),
            )
            for row in selected
        )
    return output


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


def _prediction(model: str, condition: str) -> Path:
    return OUTPUT / "predictions" / model / condition / "predictions.coco.json"


def infer() -> dict[str, Any]:
    materialize()
    aldi_hn.train()
    if PREDICTION_LOCK.exists():
        return _load(PREDICTION_LOCK)
    for name, path in MODELS.items():
        if not path.is_file():
            raise VisibilityStressError(f"fixed model is missing: {name}: {path}")
    records = _records()
    artifacts = []
    for name, checkpoint in MODELS.items():
        detector = UltralyticsDetector(
            checkpoint,
            model_name="yolo11n",
            device="cuda:0",
            expected_class_names=CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        detector.predict(
            records["fog_1p3"][:16],
            imgsz=IMGSZ,
            conf=PROBE_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            fp16=True,
        )
        for condition, cell_records in records.items():
            path = _prediction(name, condition)
            marker = path.parent / "SUCCESS.json"
            if marker.exists():
                value = _load(marker)
                if value.get("prediction_sha256") != sha256_file(path):
                    raise VisibilityStressError(f"prediction changed: {path}")
                artifacts.append(value)
                continue
            if path.parent.exists():
                raise VisibilityStressError(f"partial prediction output: {path.parent}")
            batches = detector.predict(
                cell_records,
                imgsz=IMGSZ,
                conf=PROBE_CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                fp16=True,
            )
            filtered = _filtered(batches)
            write_coco_predictions(path, filtered, category_id_by_class={0: 1, 1: 2, 2: 3})
            value = {
                "schema_version": 1,
                "status": "CVBRA_VISIBILITY_STRESS_PREDICTION_COMPLETE",
                "model": name,
                "condition": condition,
                "images": len(filtered),
                "checkpoint_sha256": sha256_file(checkpoint),
                "prediction_sha256": sha256_file(path),
            }
            atomic_write_json(marker, value)
            artifacts.append(value)
        del detector
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "status": "CVBRA_VISIBILITY_STRESS_PREDICTIONS_COMPLETE",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "manifest_sha256": sha256_file(MANIFEST),
        "artifacts": artifacts,
    }
    atomic_write_json(PREDICTION_LOCK, payload)
    return payload


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("model", "condition", *METRIC_KEYS, "images_evaluated")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue()


def score() -> dict[str, Any]:
    infer()
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        if _load(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise VisibilityStressError("visibility-stress report changed")
        return report
    conversion = _load(evidence_base.CONVERSION_LOCK)
    annotation = ROOT / str(conversion["annotation"])
    _, primary_ids = sensitivity._target_records()
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            result = evaluate_coco(
                annotation,
                _prediction(model, condition),
                max_det=MAX_DET,
                image_ids=primary_ids,
            )
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    **{key: float(result[key]) for key in METRIC_KEYS},
                    "images_evaluated": int(result["images_evaluated"]),
                }
            )
    rows.sort(key=lambda row: (str(row["condition"]), str(row["model"])))
    atomic_write_text(METRICS, _csv(rows))
    lookup = {(row["model"], row["condition"]): float(row["AP"]) for row in rows}
    contrasts = {}
    for condition in CONDITIONS:
        contrasts[condition] = {
            "CVBRA_L10_minus_source": lookup[("CVBRA_L10", condition)]
            - lookup[("source", condition)],
            "CVBRA_L10_minus_ALDI_HN": lookup[("CVBRA_L10", condition)]
            - lookup[("ALDIpp_AF_HN_equal", condition)],
            "CVBRA_L5_minus_L10": lookup[("CVBRA_L5", condition)]
            - lookup[("CVBRA_L10", condition)],
        }
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_VISIBILITY_STRESS_EVALUATION",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "manifest_sha256": sha256_file(MANIFEST),
        "prediction_lock_sha256": sha256_file(PREDICTION_LOCK),
        "metrics": _relative(METRICS),
        "metrics_sha256": sha256_file(METRICS),
        "rows": rows,
        "contrasts": contrasts,
        "interpretation": {
            "new_dataset_claim": False,
            "natural_weather_claim": False,
            "fixed_model_unseen_transform_stress_test": True,
            "negative_results_retained": True,
        },
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {"status": payload["status"], "report_sha256": sha256_file(REPORT)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-model visibility stress test")
    parser.add_argument(
        "--stage", choices=("register", "materialize", "infer", "score", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "materialize":
        result = materialize()
    elif args.stage == "infer":
        result = infer()
    elif args.stage == "score":
        result = score()
    else:
        materialize()
        result = score()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
