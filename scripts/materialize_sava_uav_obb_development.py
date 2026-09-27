from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from buse_uav.data.corruptions import apply_corruption, deterministic_corruption_seed
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "sava_v1_fresh_scope_protocol.yaml"
PROTOCOL_SHA256 = "5332649233d63cb8e0356f2e82bd87b461f2e0d6db5d10b60e9f8808b1aeb9d1"
PARTITION_LOCK = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "train_partition_lock.json"
PARTITION_LOCK_SHA256 = "9c4a1052d0030f6b909305cbbb953e96baa5384e2d7b6133200828ff5a767e27"
CORRUPTION_IMPLEMENTATION = ROOT / "src" / "buse_uav" / "data" / "corruptions.py"
CORRUPTION_IMPLEMENTATION_SHA256 = (
    "8f6a70204d123219a2d411285bb6c90452b20be9447c0e7e5193897419824c7e"
)
OUTPUT_ROOT = ROOT / "data" / "processed" / "sava_v1" / "uav_obb_development_A"
REPORT_ROOT = ROOT / "reports" / "development" / "sava_v1" / "uav_obb_development_A"
LOCK = REPORT_ROOT / "view_materialization_lock.json"
MARKER = REPORT_ROOT / "VIEWS_MATERIALIZED"
ERROR = REPORT_ROOT / "MATERIALIZATION_ERROR.json"

IMAGES = 900
GLOBAL_SEED = 42
FOG_LEVELS = ((1, 0.6, "fog_0p6"), (2, 1.0, "fog_1p0"))
WORKERS = 6


class SAVAViewError(RuntimeError):
    """Raised when label-blind SAVA development-view generation fails closed."""


@dataclass(frozen=True)
class ViewTask:
    image_id: int
    member: str
    source: str
    source_sha256: str
    width: int
    height: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SAVAViewError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SAVAViewError(f"expected mapping: {path}")
    return value


def _validate_prerequisites() -> dict[str, Any]:
    for path, expected, label in (
        (PROTOCOL, PROTOCOL_SHA256, "SAVA protocol"),
        (PARTITION_LOCK, PARTITION_LOCK_SHA256, "UAV-OBB train partition"),
        (
            CORRUPTION_IMPLEMENTATION,
            CORRUPTION_IMPLEMENTATION_SHA256,
            "controlled-corruption implementation",
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise SAVAViewError(f"locked prerequisite changed: {label}")
    partition = _load_mapping(PARTITION_LOCK)
    assignments = partition.get("assignments")
    if (
        partition.get("pass") is not True
        or partition.get("annotation_content_accessed") is not False
        or partition.get("prediction_or_metric_accessed") is not False
        or partition.get("validation_or_test_content_accessed") is not False
        or not isinstance(assignments, list)
        or sum(isinstance(row, dict) and row.get("role") == "development_A" for row in assignments)
        != IMAGES
    ):
        raise SAVAViewError("UAV-OBB train partition evidence changed")
    return partition


def _tasks(partition: Mapping[str, Any]) -> list[ViewTask]:
    assignments = partition["assignments"]
    assert isinstance(assignments, list)
    rows = sorted(
        (
            row
            for row in assignments
            if isinstance(row, dict) and row.get("role") == "development_A"
        ),
        key=lambda row: str(row["member"]).casefold(),
    )
    tasks: list[ViewTask] = []
    for image_id, row in enumerate(rows, start=1):
        path = ROOT / str(row["path"])
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise SAVAViewError(f"locked development source image changed: {path}")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if (width, height) != (1920, 1080):
            raise SAVAViewError(f"development source geometry changed: {path}")
        tasks.append(
            ViewTask(
                image_id=image_id,
                member=str(row["member"]),
                source=str(path.resolve()),
                source_sha256=str(row["sha256"]),
                width=width,
                height=height,
            )
        )
    if len(tasks) != IMAGES:
        raise SAVAViewError("development task coverage changed")
    return tasks


def _generate_task(task: ViewTask) -> list[dict[str, Any]]:
    with Image.open(task.source) as image:
        rgb = np.asarray(image.convert("RGB"))
    rows: list[dict[str, Any]] = []
    for severity, beta, view in FOG_LEVELS:
        seed = deterministic_corruption_seed(GLOBAL_SEED, str(task.image_id), "fog", severity)
        output = apply_corruption(
            rgb,
            corruption="fog",
            parameter=beta,
            seed=seed,
        )
        if output.shape != rgb.shape or output.dtype != np.uint8:
            raise SAVAViewError(f"fog output properties changed: {task.member}/{view}")
        destination = OUTPUT_ROOT / "views" / view / f"{task.image_id:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.tmp")
        Image.fromarray(output, mode="RGB").save(
            temporary, format="PNG", compress_level=3, optimize=False
        )
        os.replace(temporary, destination)
        with Image.open(destination) as written:
            written.load()
            if written.size != (task.width, task.height) or written.mode != "RGB":
                raise SAVAViewError(f"written fog image cannot be verified: {destination}")
        rows.append(
            {
                "image_id": task.image_id,
                "member": task.member,
                "view": view,
                "severity": severity,
                "beta": beta,
                "seed": seed,
                "path": _relative(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "width": task.width,
                "height": task.height,
            }
        )
    return rows


def _validate_lock() -> dict[str, Any]:
    if not LOCK.is_file() or not MARKER.is_file():
        raise SAVAViewError("SAVA view materialization lock is incomplete")
    lock = _load_mapping(LOCK)
    marker = _load_mapping(MARKER)
    if (
        marker.get("view_materialization_lock_sha256") != sha256_file(LOCK)
        or lock.get("pass") is not True
        or lock.get("annotation_content_accessed") is not False
        or lock.get("prediction_or_metric_accessed") is not False
    ):
        raise SAVAViewError("SAVA view materialization lock changed")
    return lock


def materialize() -> dict[str, Any]:
    partition = _validate_prerequisites()
    if LOCK.exists() or MARKER.exists():
        return _validate_lock()
    tasks = _tasks(partition)
    fog_rows: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            for position, rows in enumerate(
                executor.map(_generate_task, tasks, chunksize=2), start=1
            ):
                fog_rows.extend(rows)
                if position % 50 == 0:
                    print(json.dumps({"development_fog_images_completed": position}), flush=True)
    except Exception as exc:
        atomic_write_json(
            ERROR,
            {
                "status": "SAVA_UAV_OBB_DEVELOPMENT_MATERIALIZATION_ERROR",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "annotation_content_accessed": False,
                "validation_or_test_content_accessed": False,
            },
        )
        raise
    fog_rows.sort(key=lambda row: (int(row["image_id"]), int(row["severity"])))
    clean_rows = [
        {
            "image_id": task.image_id,
            "member": task.member,
            "view": "original",
            "path": _relative(Path(task.source)),
            "sha256": task.source_sha256,
            "width": task.width,
            "height": task.height,
            "source_bytes_reused": True,
        }
        for task in tasks
    ]
    counts = Counter(str(row["view"]) for row in fog_rows)
    passed = (
        len(clean_rows) == IMAGES
        and len(fog_rows) == IMAGES * len(FOG_LEVELS)
        and counts == Counter({"fog_0p6": IMAGES, "fog_1p0": IMAGES})
        and all((ROOT / str(row["path"])).is_file() for row in fog_rows)
    )
    payload = {
        "schema_version": 1,
        "status": "SAVA_UAV_OBB_DEVELOPMENT_A_VIEWS_MATERIALIZED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "partition_lock_sha256": PARTITION_LOCK_SHA256,
        "corruption_implementation_sha256": CORRUPTION_IMPLEMENTATION_SHA256,
        "workers": WORKERS,
        "images": IMAGES,
        "clean_rows": clean_rows,
        "clean_rows_payload_sha256": stable_hash(clean_rows, length=64),
        "fog_rows": fog_rows,
        "fog_rows_payload_sha256": stable_hash(fog_rows, length=64),
        "fog_outputs": len(fog_rows),
        "fog_outputs_by_view": dict(sorted(counts.items())),
        "fog_output_bytes": sum(int(row["bytes"]) for row in fog_rows),
        "all_geometry_preserved": True,
        "fog_outputs_lossless_PNG": True,
        "clean_source_bytes_reused": True,
        "pass": passed,
        "annotation_content_accessed": False,
        "prediction_or_metric_accessed": False,
        "reserve_B_content_accessed": False,
        "validation_or_test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    if not passed:
        raise SAVAViewError("SAVA development view materialization gate failed")
    atomic_write_json(LOCK, payload)
    atomic_write_json(
        MARKER,
        {
            "status": payload["status"],
            "view_materialization_lock_sha256": sha256_file(LOCK),
        },
    )
    return payload


def main() -> int:
    argparse.ArgumentParser(description="Materialize SAVA UAV-OBB development views").parse_args()
    result = materialize()
    summary = {key: value for key, value in result.items() if key not in {"clean_rows", "fog_rows"}}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
