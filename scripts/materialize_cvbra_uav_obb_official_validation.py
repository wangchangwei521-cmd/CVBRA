from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from buse_uav.data.corruptions import apply_corruption, deterministic_corruption_seed
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "cvbra_v1_official_validation_protocol.yaml"
PROTOCOL_SHA256 = "820984d022cdb5583484ce1edaab718f428e0001ead5f320653cf1199dcbe4a7"
ACCESS_LOCK = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_official_validation"
    / "ACCESS_SCOPE_LOCK.json"
)
ACCESS_LOCK_SHA256 = "d14707ef941da07b75ac5f2ddfff5f09d5fd75888abee9bc657802c7fd8b084b"
DECONTAMINATION = ACCESS_LOCK.parent / "SPLIT_DECONTAMINATION_AMENDMENT_1.json"
DECONTAMINATION_SHA256 = "02a355fc1d3bc231439d2a0652f0d1d5031ad4b7b480fa0024672e296066eb14"
SELECTION_REPORT = (
    ROOT
    / "reports"
    / "development"
    / "cvbra_v1"
    / "uav_obb_reserve_B"
    / "evaluation"
    / "selection_report.json"
)
SELECTION_REPORT_SHA256 = "4bbef9c0fa47f4c0fb15fca7b34b774f4a463a81fbac1e2ef02110063b669c53"
ARCHIVE = ROOT / "data" / "raw" / "UAV_OBB_v4" / "downloads" / "UAV-OBB-dlaCi7.zip"
ARCHIVE_SHA256 = "bf65b0d6a00bc1a320002012146bf44913e2b3f97b9e6acbd7954775d61f69e2"
INVENTORY = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "archive_inventory_lock.json"
INVENTORY_SHA256 = "4b2d0830a0e7f76f86a391b51c715f92155fee9a44b91dac6b37ad88b5157dbc"
PARTITION = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "train_partition_lock.json"
PARTITION_SHA256 = "9c4a1052d0030f6b909305cbbb953e96baa5384e2d7b6133200828ff5a767e27"
CORRUPTIONS = ROOT / "src" / "buse_uav" / "data" / "corruptions.py"
CORRUPTIONS_SHA256 = "8f6a70204d123219a2d411285bb6c90452b20be9447c0e7e5193897419824c7e"

RAW_ROOT = ROOT / "data" / "raw" / "UAV_OBB_v4" / "sealed_valid_images"
OUTPUT_ROOT = ROOT / "data" / "processed" / "cvbra_v1" / "uav_obb_official_validation"
LOCK = ACCESS_LOCK.parent / "view_materialization_lock.json"
MARKER = ACCESS_LOCK.parent / "VIEWS_MATERIALIZED"
ERROR = ACCESS_LOCK.parent / "MATERIALIZATION_ERROR.json"

IMAGES = 218
PRIMARY_IMAGES = 167
PRIMARY_GROUPS = 141
GLOBAL_SEED = 42
FOG_LEVELS = ((1, 0.6, "fog_0p6"), (2, 1.0, "fog_1p0"))
WORKERS = 4
_ROBOFLOW_SUFFIX = re.compile(r"(?:_jpg)?\.rf\.[^.]+\.jpg$", re.IGNORECASE)


class CVBRAValidationViewError(RuntimeError):
    """Raised when official-validation materialization cannot fail closed."""


@dataclass(frozen=True)
class ViewTask:
    image_id: int
    member: str
    source_key: str
    source: str
    source_sha256: str
    width: int
    height: int
    primary_eligible: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CVBRAValidationViewError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CVBRAValidationViewError(f"expected mapping: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise CVBRAValidationViewError(f"locked prerequisite changed: {label}")


def _source_key(member: str) -> str:
    return _ROBOFLOW_SUFFIX.sub("", PurePosixPath(member).name)


def _payload_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_prerequisites() -> tuple[list[dict[str, Any]], set[str], set[str]]:
    for path, digest, label in (
        (PROTOCOL, PROTOCOL_SHA256, "official-validation protocol"),
        (ACCESS_LOCK, ACCESS_LOCK_SHA256, "official-validation access lock"),
        (DECONTAMINATION, DECONTAMINATION_SHA256, "split decontamination amendment"),
        (SELECTION_REPORT, SELECTION_REPORT_SHA256, "reserve-B selection report"),
        (ARCHIVE, ARCHIVE_SHA256, "UAV-OBB archive"),
        (INVENTORY, INVENTORY_SHA256, "archive inventory"),
        (PARTITION, PARTITION_SHA256, "official-train partition"),
        (CORRUPTIONS, CORRUPTIONS_SHA256, "corruption implementation"),
    ):
        _assert_hash(path, digest, label=label)
    selection = _load_mapping(SELECTION_REPORT)
    access = _load_mapping(ACCESS_LOCK)
    amendment = _load_mapping(DECONTAMINATION)
    inventory = _load_mapping(INVENTORY)
    partition = _load_mapping(PARTITION)
    if (
        selection.get("status") != "PASS_CVBRA_V1_RESERVE_B_SELECTION"
        or selection.get("decision", {}).get("official_validation_access_authorized_next")
        is not True
        or access.get("official_validation_image_or_label_content_accessed_before_lock")
        is not False
        or amendment.get("validation_image_or_label_content_accessed") is not False
        or inventory.get("validation_or_test_content_accessed") is not False
        or partition.get("validation_or_test_content_accessed") is not False
    ):
        raise CVBRAValidationViewError("official-validation authorization changed")
    member_rows = inventory.get("member_rows")
    assignments = partition.get("assignments")
    if not isinstance(member_rows, list) or not isinstance(assignments, list):
        raise CVBRAValidationViewError("locked dataset registries are invalid")
    validation = sorted(
        (
            row
            for row in member_rows
            if isinstance(row, dict) and row.get("role") == "valid_image"
        ),
        key=lambda row: str(row["member"]).casefold(),
    )
    train_keys = {
        _source_key(str(row["member"]))
        for row in assignments
        if isinstance(row, dict) and row.get("member")
    }
    train_hashes = {
        str(row["sha256"])
        for row in assignments
        if isinstance(row, dict) and row.get("sha256")
    }
    members = [str(row["member"]) for row in validation]
    excluded = [member for member in members if _source_key(member) in train_keys]
    primary = [member for member in members if _source_key(member) not in train_keys]
    if (
        len(validation) != IMAGES
        or len(primary) != PRIMARY_IMAGES
        or len({_source_key(member) for member in primary}) != PRIMARY_GROUPS
        or _payload_sha256(members) != amendment.get("full_members_payload_sha256")
        or _payload_sha256(excluded) != amendment.get("excluded_members_payload_sha256")
        or _payload_sha256(primary)
        != amendment.get("decontaminated_members_payload_sha256")
    ):
        raise CVBRAValidationViewError("decontamination registry changed")
    return validation, train_keys, train_hashes


def _extract_images(
    validation: Sequence[Mapping[str, Any]], train_keys: set[str], train_hashes: set[str]
) -> list[ViewTask]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    tasks: list[ViewTask] = []
    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = set(bundle.namelist())
        for image_id, row in enumerate(validation, start=1):
            member = str(row["member"])
            if member not in names or not member.startswith("UAV-OBB/valid/images/"):
                raise CVBRAValidationViewError(f"official-validation image missing: {member}")
            raw = bundle.read(member)
            if len(raw) != int(row["uncompressed_size"]):
                raise CVBRAValidationViewError(f"official-validation image size changed: {member}")
            destination = RAW_ROOT / f"{image_id:04d}{PurePosixPath(member).suffix.lower()}"
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, destination)
            digest = sha256_file(destination)
            with Image.open(destination) as image:
                image.load()
                width, height = image.size
            source_key = _source_key(member)
            primary_eligible = source_key not in train_keys
            if primary_eligible and digest in train_hashes:
                raise CVBRAValidationViewError(
                    f"exact train image entered decontaminated validation: {member}"
                )
            tasks.append(
                ViewTask(
                    image_id=image_id,
                    member=member,
                    source_key=source_key,
                    source=str(destination.resolve()),
                    source_sha256=digest,
                    width=width,
                    height=height,
                    primary_eligible=primary_eligible,
                )
            )
    return tasks


def _generate_task(task: ViewTask) -> list[dict[str, Any]]:
    with Image.open(task.source) as image:
        rgb = np.asarray(image.convert("RGB"))
    rows: list[dict[str, Any]] = []
    for severity, beta, view in FOG_LEVELS:
        seed = deterministic_corruption_seed(GLOBAL_SEED, str(task.image_id), "fog", severity)
        output = apply_corruption(rgb, corruption="fog", parameter=beta, seed=seed)
        if output.shape != rgb.shape or output.dtype != np.uint8:
            raise CVBRAValidationViewError(f"fog output changed: {task.member}/{view}")
        destination = OUTPUT_ROOT / "views" / view / f"{task.image_id:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.tmp")
        Image.fromarray(output, mode="RGB").save(
            temporary, format="PNG", compress_level=3, optimize=False
        )
        os.replace(temporary, destination)
        rows.append(
            {
                "image_id": task.image_id,
                "member": task.member,
                "source_key": task.source_key,
                "primary_eligible": task.primary_eligible,
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
        raise CVBRAValidationViewError("official-validation materialization lock is incomplete")
    lock = _load_mapping(LOCK)
    marker = _load_mapping(MARKER)
    if (
        marker.get("view_materialization_lock_sha256") != sha256_file(LOCK)
        or lock.get("pass") is not True
        or lock.get("official_validation_labels_accessed") is not False
        or lock.get("official_test_content_accessed") is not False
    ):
        raise CVBRAValidationViewError("official-validation materialization lock changed")
    return lock


def materialize() -> dict[str, Any]:
    validation, train_keys, train_hashes = _validate_prerequisites()
    if LOCK.exists() or MARKER.exists():
        return _validate_lock()
    tasks = _extract_images(validation, train_keys, train_hashes)
    try:
        fog_rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            for position, rows in enumerate(
                executor.map(_generate_task, tasks, chunksize=2), start=1
            ):
                fog_rows.extend(rows)
                if position % 50 == 0:
                    print(json.dumps({"validation_fog_images_completed": position}), flush=True)
    except Exception as exc:
        atomic_write_json(
            ERROR,
            {
                "status": "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_MATERIALIZATION_ERROR",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_validation_labels_accessed": False,
                "official_test_content_accessed": False,
            },
        )
        raise
    fog_rows.sort(key=lambda row: (int(row["image_id"]), int(row["severity"])))
    clean_rows = [
        {
            "image_id": task.image_id,
            "member": task.member,
            "source_key": task.source_key,
            "primary_eligible": task.primary_eligible,
            "view": "original",
            "path": _relative(Path(task.source)),
            "sha256": task.source_sha256,
            "width": task.width,
            "height": task.height,
        }
        for task in tasks
    ]
    counts = Counter(str(row["view"]) for row in fog_rows)
    primary_ids = [
        int(str(row["image_id"])) for row in clean_rows if row["primary_eligible"]
    ]
    exact_train_overlap_all = sum(str(row["sha256"]) in train_hashes for row in clean_rows)
    passed = (
        len(clean_rows) == IMAGES
        and len(primary_ids) == PRIMARY_IMAGES
        and len({str(row["source_key"]) for row in clean_rows if row["primary_eligible"]})
        == PRIMARY_GROUPS
        and len(fog_rows) == IMAGES * 2
        and counts == Counter({"fog_0p6": IMAGES, "fog_1p0": IMAGES})
        and all((ROOT / str(row["path"])).is_file() for row in fog_rows)
        and not any(
            str(row["sha256"]) in train_hashes
            for row in clean_rows
            if row["primary_eligible"]
        )
    )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_UAV_OBB_OFFICIAL_VALIDATION_VIEWS_MATERIALIZED_BEFORE_LABEL_ACCESS",
        "locked_at_utc": _utc_now(),
        "runner": _relative(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": PROTOCOL_SHA256,
        "access_scope_lock_sha256": ACCESS_LOCK_SHA256,
        "decontamination_amendment_sha256": DECONTAMINATION_SHA256,
        "selection_report_sha256": SELECTION_REPORT_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "inventory_lock_sha256": INVENTORY_SHA256,
        "train_partition_lock_sha256": PARTITION_SHA256,
        "corruption_implementation_sha256": CORRUPTIONS_SHA256,
        "images": IMAGES,
        "primary_images": PRIMARY_IMAGES,
        "primary_source_groups": PRIMARY_GROUPS,
        "excluded_images": IMAGES - PRIMARY_IMAGES,
        "exact_train_image_sha256_overlap_full": exact_train_overlap_all,
        "exact_train_image_sha256_overlap_primary": 0,
        "clean_rows": clean_rows,
        "clean_rows_payload_sha256": stable_hash(clean_rows, length=64),
        "primary_image_ids": primary_ids,
        "primary_image_ids_payload_sha256": stable_hash(primary_ids, length=64),
        "fog_rows": fog_rows,
        "fog_rows_payload_sha256": stable_hash(fog_rows, length=64),
        "fog_outputs": len(fog_rows),
        "fog_outputs_by_view": dict(sorted(counts.items())),
        "pass": passed,
        "official_validation_images_accessed": True,
        "official_validation_labels_accessed": False,
        "prediction_or_metric_accessed": False,
        "official_test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    if not passed:
        raise CVBRAValidationViewError("official-validation materialization gate failed")
    atomic_write_json(LOCK, payload)
    atomic_write_json(
        MARKER,
        {"status": payload["status"], "view_materialization_lock_sha256": sha256_file(LOCK)},
    )
    return payload


def main() -> int:
    argparse.ArgumentParser(
        description="Materialize image-only CVBRA official-validation views"
    ).parse_args()
    result = materialize()
    summary = {key: value for key, value in result.items() if key not in {"clean_rows", "fog_rows"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
