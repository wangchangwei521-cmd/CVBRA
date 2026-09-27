from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from itertools import permutations
from pathlib import Path
from statistics import fmean, median
from typing import Any

import torch
from PIL import Image

from buse_uav.detectors.ultralytics_adapter import (
    UltralyticsDetector,
    configure_ultralytics_environment,
)
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_interleaved_latency_audit.yaml"
PROTOCOL_SHA256 = "66389e19acad82d1ac2dcd3c0e0d73e9043735e674555fe4a89361522f0439bf"
MANIFEST = ROOT / "data" / "manifests" / "hazydet_val_manifest.json"
MANIFEST_SHA256 = "e8e5ea234dd098348a1906c8aff93d0d773d6ed1c828c09ee98d7cca6b60f916"
IMAGE_ROOT = ROOT / "data" / "raw" / "HazyDet"
OUTPUT = ROOT / "reports" / "development" / "cvbra_v1_interleaved_latency_audit"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
IMPLEMENTATION_MARKER = OUTPUT / "IMPLEMENTATION_LOCKED"
OBSERVATIONS = OUTPUT / "interleaved_chunk_observations.csv"
REPORT = OUTPUT / "latency_report.json"
COMPLETE = OUTPUT / "LATENCY_AUDIT_COMPLETE"

MODELS = ("source", "CVBRA_v1", "STF")
WEIGHTS: dict[str, tuple[Path, str]] = {
    "source": (
        ROOT / "weights" / "hazydet" / "yolo11n_best.pt",
        "16ad9de39a1ff86a1826eb5cda680166196ff33ecc5d3dcb297070a166e42430",
    ),
    "CVBRA_v1": (
        ROOT / "runs" / "cvbra_v1" / "yolo11n" / "cvbra_v1.pt",
        "d44f0926696e93b5f2e0ec5c9201f1e6360c43d4d40fccc68646e1ced633bd42",
    ),
    "STF": (
        ROOT / "runs" / "cvbra_v1_matched_baselines" / "STF" / "STF.pt",
        "90556b62756d0b1a944c42a7cb0e5e86935e2684058c88621ed92d7bfd78b6d2",
    ),
}

CLASS_NAMES = ("car", "truck", "bus")
IMAGES = 192
WARMUP_IMAGES = 32
CHUNK_SIZE = 8
IMAGE_SIZE = 1280
CONF = 0.08
IOU = 0.70
MAX_DET = 500
EQUIVALENCE_LOW = 0.85
EQUIVALENCE_HIGH = 1.15
FEASIBILITY_CEILING = 1.50


class InterleavedLatencyAuditError(RuntimeError):
    """Raised when the frozen latency audit cannot fail closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InterleavedLatencyAuditError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InterleavedLatencyAuditError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise InterleavedLatencyAuditError(f"locked {label} changed: {path}")


def counterbalanced_orders() -> tuple[tuple[str, ...], ...]:
    """Return every order once so each model occupies every position twice."""
    return tuple(permutations(MODELS))


def _records(*, verify_hashes: bool) -> tuple[ImageRecord, ...]:
    manifest = _load_mapping(MANIFEST)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise InterleavedLatencyAuditError("HazyDet validation manifest has no files")
    files = [
        row
        for row in raw_files
        if isinstance(row, dict)
        and str(row.get("path", "")).replace("\\", "/").startswith("val/hazy_images/")
    ]
    if manifest.get("dataset") != "hazydet" or len(files) != 1000:
        raise InterleavedLatencyAuditError("HazyDet validation manifest changed")
    selected = sorted(files, key=lambda row: int(Path(str(row["path"])).stem))[:IMAGES]
    records: list[ImageRecord] = []
    for row in selected:
        path = IMAGE_ROOT / str(row["path"])
        if not path.is_file() or (verify_hashes and sha256_file(path) != str(row["sha256"])):
            raise InterleavedLatencyAuditError(f"latency image changed: {path}")
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as exc:
            raise InterleavedLatencyAuditError(f"cannot inspect latency image: {path}") from exc
        records.append(
            ImageRecord(
                image_id=int(path.stem),
                path=str(path.resolve()),
                width=int(width),
                height=int(height),
            )
        )
    if len(records) != IMAGES or len({record.image_id for record in records}) != IMAGES:
        raise InterleavedLatencyAuditError("latency image selection changed")
    return tuple(records)


def _checkpoint_architecture(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise InterleavedLatencyAuditError(f"unsupported checkpoint: {path}")
    model = payload.get("ema") or payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise InterleavedLatencyAuditError(f"checkpoint has no model: {path}")
    schema = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.state_dict().items()
    ]
    names = getattr(model, "names", None)
    return {
        "parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
        "state_entries": len(schema),
        "state_schema_sha256": stable_hash(schema, length=64),
        "class_names": names,
    }


def _cuda_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise InterleavedLatencyAuditError("latency audit requires CUDA")
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


def _validate_inputs() -> dict[str, dict[str, Any]]:
    _assert_hash(PROTOCOL, PROTOCOL_SHA256, label="latency protocol")
    _assert_hash(MANIFEST, MANIFEST_SHA256, label="HazyDet validation manifest")
    configure_ultralytics_environment(ROOT)
    architecture: dict[str, dict[str, Any]] = {}
    for model, (path, digest) in WEIGHTS.items():
        _assert_hash(path, digest, label=f"{model} checkpoint")
        architecture[model] = _checkpoint_architecture(path)
    schemas = {str(row["state_schema_sha256"]) for row in architecture.values()}
    parameters = {int(row["parameters"]) for row in architecture.values()}
    names = {json.dumps(row["class_names"], sort_keys=True) for row in architecture.values()}
    if len(schemas) != 1 or len(parameters) != 1 or len(names) != 1:
        raise InterleavedLatencyAuditError("checkpoint architecture identities differ")
    return architecture


def _validate_existing_lock() -> dict[str, Any]:
    if not IMPLEMENTATION_LOCK.is_file() or not IMPLEMENTATION_MARKER.is_file():
        raise InterleavedLatencyAuditError("latency implementation lock is incomplete")
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    marker = _load_mapping(IMPLEMENTATION_MARKER)
    if (
        lock.get("runner_sha256") != sha256_file(Path(__file__))
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or marker.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
    ):
        raise InterleavedLatencyAuditError("latency implementation lock changed")
    return lock


def preflight() -> dict[str, Any]:
    architecture = _validate_inputs()
    records = _records(verify_hashes=True)
    if IMPLEMENTATION_LOCK.exists() or IMPLEMENTATION_MARKER.exists():
        return _validate_existing_lock()
    if any(path.exists() for path in (OBSERVATIONS, REPORT, COMPLETE)):
        raise InterleavedLatencyAuditError(
            "latency observation appeared before implementation lock"
        )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_INTERLEAVED_LATENCY_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": MANIFEST_SHA256,
        "image_ids_sha256": stable_hash([record.image_id for record in records], length=64),
        "images": len(records),
        "orders": [list(order) for order in counterbalanced_orders()],
        "architecture": architecture,
        "weights": {
            model: {"path": _relative(path), "sha256": digest, "bytes": path.stat().st_size}
            for model, (path, digest) in WEIGHTS.items()
        },
        "cuda_identity": _cuda_identity(),
        "python": platform.python_version(),
        "accuracy_labels_accessed": False,
        "test_content_accessed": False,
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


def summarize_observations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = len(counterbalanced_orders()) * (IMAGES // CHUNK_SIZE) * len(MODELS)
    if len(rows) != expected:
        raise InterleavedLatencyAuditError(
            f"expected {expected} interleaved chunks, observed {len(rows)}"
        )
    cycle_means: dict[str, list[float]] = {model: [] for model in MODELS}
    for cycle in range(1, len(counterbalanced_orders()) + 1):
        for model in MODELS:
            values = [
                float(row["mean_ms"])
                for row in rows
                if int(row["cycle"]) == cycle and str(row["model"]) == model
            ]
            if len(values) != IMAGES // CHUNK_SIZE:
                raise InterleavedLatencyAuditError(f"incomplete cycle {cycle}/{model}")
            cycle_means[model].append(fmean(values))
    model_summary = {
        model: {
            "cycle_means_ms": values,
            "median_cycle_mean_ms": median(values),
            "mean_cycle_mean_ms": fmean(values),
            "min_cycle_mean_ms": min(values),
            "max_cycle_mean_ms": max(values),
        }
        for model, values in cycle_means.items()
    }
    source = cycle_means["source"]
    contrasts: dict[str, Any] = {}
    for model in MODELS[1:]:
        ratios = [
            value / baseline
            for value, baseline in zip(cycle_means[model], source, strict=True)
        ]
        median_ratio = median(ratios)
        contrasts[f"{model}_over_source"] = {
            "cycle_ratios": ratios,
            "median_cycle_ratio": median_ratio,
            "min_cycle_ratio": min(ratios),
            "max_cycle_ratio": max(ratios),
            "median_practically_equivalent_0p85_to_1p15": (
                EQUIVALENCE_LOW <= median_ratio <= EQUIVALENCE_HIGH
            ),
            "all_cycles_within_equivalence_band": (
                min(ratios) >= EQUIVALENCE_LOW and max(ratios) <= EQUIVALENCE_HIGH
            ),
            "below_1p50_feasibility_ceiling": median_ratio <= FEASIBILITY_CEILING,
        }
    return {"models": model_summary, "contrasts": contrasts}


def _write_observations(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("cycle", "order", "position", "chunk", "model", "images", "mean_ms")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    atomic_write_text(OBSERVATIONS, buffer.getvalue())


def _validate_existing_report() -> dict[str, Any]:
    if not REPORT.is_file() or not OBSERVATIONS.is_file() or not COMPLETE.is_file():
        raise InterleavedLatencyAuditError("latency report is incomplete")
    report = _load_mapping(REPORT)
    marker = _load_mapping(COMPLETE)
    if (
        report.get("implementation_lock_sha256") != sha256_file(IMPLEMENTATION_LOCK)
        or report.get("observations_sha256") != sha256_file(OBSERVATIONS)
        or marker.get("latency_report_sha256") != sha256_file(REPORT)
    ):
        raise InterleavedLatencyAuditError("latency report changed")
    return report


def run() -> dict[str, Any]:
    lock = preflight()
    if REPORT.exists() or OBSERVATIONS.exists() or COMPLETE.exists():
        return _validate_existing_report()
    records = _records(verify_hashes=False)
    detectors = {
        model: UltralyticsDetector(
            path,
            model_name="yolo11n",
            device="cuda:0",
            expected_class_names=CLASS_NAMES,
            project_root=ROOT,
            stream_chunk_records=CHUNK_SIZE,
            release_cuda_cache_between_chunks=False,
        )
        for model, (path, _) in WEIGHTS.items()
    }
    warmup = records[:WARMUP_IMAGES]
    for model in MODELS:
        detectors[model].predict(
            warmup,
            imgsz=IMAGE_SIZE,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            fp16=True,
        )
    rows: list[dict[str, Any]] = []
    for cycle, order in enumerate(counterbalanced_orders(), start=1):
        for chunk_index, start in enumerate(range(0, len(records), CHUNK_SIZE), start=1):
            chunk = records[start : start + CHUNK_SIZE]
            for position, model in enumerate(order, start=1):
                batches = detectors[model].predict(
                    chunk,
                    imgsz=IMAGE_SIZE,
                    conf=CONF,
                    iou=IOU,
                    max_det=MAX_DET,
                    fp16=True,
                )
                rows.append(
                    {
                        "cycle": cycle,
                        "order": ">".join(order),
                        "position": position,
                        "chunk": chunk_index,
                        "model": model,
                        "images": len(batches),
                        "mean_ms": fmean(float(batch.latency_ms) for batch in batches),
                    }
                )
        print(json.dumps({"latency_cycle_complete": cycle, "order": order}), flush=True)
    del detectors
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    summary = summarize_observations(rows)
    _write_observations(rows)
    cvbra = summary["contrasts"]["CVBRA_v1_over_source"]
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_INTERLEAVED_LATENCY_AUDIT",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "observations": _relative(OBSERVATIONS),
        "observations_sha256": sha256_file(OBSERVATIONS),
        "images_per_cycle": IMAGES,
        "cycles": len(counterbalanced_orders()),
        "chunk_size": CHUNK_SIZE,
        "architecture": lock["architecture"],
        "weights": lock["weights"],
        "summary": summary,
        "checks": {
            "same_parameter_and_state_schema": True,
            "CVBRA_median_within_0p85_to_1p15": cvbra[
                "median_practically_equivalent_0p85_to_1p15"
            ],
            "CVBRA_median_below_1p50": cvbra["below_1p50_feasibility_ceiling"],
        },
        "all_primary_checks_pass": (
            bool(cvbra["median_practically_equivalent_0p85_to_1p15"])
            and bool(cvbra["below_1p50_feasibility_ceiling"])
        ),
        "accuracy_labels_accessed": False,
        "test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "latency_report_sha256": sha256_file(REPORT),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVBRA-v1 interleaved latency audit")
    parser.add_argument("--stage", choices=("preflight", "run"), default="run")
    args = parser.parse_args()
    result = preflight() if args.stage == "preflight" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
