from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiment" / "sava_v1_fresh_scope_protocol.yaml"
PROTOCOL_SHA256 = "5332649233d63cb8e0356f2e82bd87b461f2e0d6db5d10b60e9f8808b1aeb9d1"
ARCHIVE = ROOT / "data" / "raw" / "UAV_OBB_v4" / "downloads" / "UAV-OBB-dlaCi7.zip"
ARCHIVE_SHA256 = "bf65b0d6a00bc1a320002012146bf44913e2b3f97b9e6acbd7954775d61f69e2"
INVENTORY = ROOT / "reports" / "data_metadata" / "uav_obb_v4" / "archive_inventory_lock.json"
INVENTORY_SHA256 = "4b2d0830a0e7f76f86a391b51c715f92155fee9a44b91dac6b37ad88b5157dbc"
OUTPUT_ROOT = ROOT / "data" / "raw" / "UAV_OBB_v4" / "sealed_train"
IMAGE_ROOT = OUTPUT_ROOT / "images"
REPORT_ROOT = ROOT / "reports" / "data_metadata" / "uav_obb_v4"
EXTRACTION_LOCK = REPORT_ROOT / "train_image_extraction_lock.json"
EXTRACTION_MARKER = REPORT_ROOT / "TRAIN_IMAGES_EXTRACTED"
PARTITION_LOCK = REPORT_ROOT / "train_partition_lock.json"
PARTITION_MARKER = REPORT_ROOT / "TRAIN_PARTITION_LOCKED_BEFORE_LABEL_OR_METRIC_ACCESS"

TRAIN_IMAGES = 1383
TARGET_DEVELOPMENT = 900
TARGET_RESERVE = 483
MINIMUM_ALLOWED_HAMMING = 5
PARTITION_SEED = "sava-v1-uav-obb-train-partition-2026-08-14"
HASH_WORKERS = 6
TRANSFORM_NAMES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "mirror",
    "mirror_rotate_90",
    "mirror_rotate_180",
    "mirror_rotate_270",
)
ROBOFLOW_SUFFIX = re.compile(r"_jpg\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
SEGMENTS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))


class UavObbPreparationError(RuntimeError):
    """Raised when label-blind UAV-OBB train preparation fails closed."""


@dataclass(frozen=True)
class HashTask:
    member: str
    path: str
    sha256: str
    source_group: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UavObbPreparationError(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UavObbPreparationError(f"expected mapping: {path}")
    return value


def _source_group(member: str) -> str:
    stem = Path(member).stem
    stripped = ROBOFLOW_SUFFIX.sub("", stem)
    if not stripped:
        raise UavObbPreparationError(f"cannot derive source group: {member}")
    return stripped.casefold()


def _validate_prerequisites() -> dict[str, Any]:
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise UavObbPreparationError("SAVA protocol changed")
    if not INVENTORY.is_file() or sha256_file(INVENTORY) != INVENTORY_SHA256:
        raise UavObbPreparationError("UAV-OBB archive inventory changed")
    if not ARCHIVE.is_file() or sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise UavObbPreparationError("UAV-OBB archive changed")
    inventory = _load_mapping(INVENTORY)
    if (
        inventory.get("pass") is not True
        or inventory.get("image_content_accessed") is not False
        or inventory.get("annotation_content_accessed") is not False
        or inventory.get("validation_or_test_content_accessed") is not False
    ):
        raise UavObbPreparationError("UAV-OBB inventory evidence boundary changed")
    return inventory


def _train_member_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = inventory.get("member_rows")
    if not isinstance(raw, list):
        raise UavObbPreparationError("archive inventory has no member registry")
    rows = [row for row in raw if isinstance(row, dict) and row.get("role") == "train_image"]
    rows.sort(key=lambda row: str(row["member"]).casefold())
    if (
        len(rows) != TRAIN_IMAGES
        or len({str(row["member"]).casefold() for row in rows}) != TRAIN_IMAGES
    ):
        raise UavObbPreparationError("official train image inventory changed")
    return rows


def _extract_member(archive: zipfile.ZipFile, row: Mapping[str, Any]) -> dict[str, Any]:
    member = str(row["member"])
    output = IMAGE_ROOT / Path(member).name
    if output.exists():
        raise UavObbPreparationError(f"unlocked extracted image already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    crc = 0
    size = 0
    try:
        with archive.open(member, "r") as source, temporary.open("wb") as destination:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
                crc = zlib.crc32(block, crc)
                size += len(block)
        if size != int(row["uncompressed_size"]) or f"{crc & 0xFFFFFFFF:08x}" != row["crc32"]:
            raise UavObbPreparationError(f"extracted train image failed CRC/size: {member}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    with Image.open(output) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    if dimensions != [1920, 1080] or mode != "RGB":
        raise UavObbPreparationError(
            f"official train image properties changed: {member} -> {dimensions}/{mode}"
        )
    return {
        "member": member,
        "output": output.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(output),
        "output_crc32": f"{crc & 0xFFFFFFFF:08x}",
        "size_bytes": size,
        "dimensions": dimensions,
        "mode": mode,
        "source_group": _source_group(member),
    }


def _validate_extraction_lock() -> dict[str, Any]:
    if not EXTRACTION_LOCK.is_file() or not EXTRACTION_MARKER.is_file():
        raise UavObbPreparationError("train-image extraction lock is incomplete")
    lock = _load_mapping(EXTRACTION_LOCK)
    marker = _load_mapping(EXTRACTION_MARKER)
    rows = lock.get("image_rows")
    if (
        lock.get("status") != "UAV_OBB_V4_TRAIN_IMAGES_ONLY_EXTRACTED_CRC_DECODED_AND_LOCKED"
        or marker.get("train_image_extraction_lock_sha256") != sha256_file(EXTRACTION_LOCK)
        or lock.get("archive_inventory_sha256") != INVENTORY_SHA256
        or not isinstance(rows, list)
        or len(rows) != TRAIN_IMAGES
    ):
        raise UavObbPreparationError("train-image extraction lock changed")
    for row in rows:
        if not isinstance(row, dict):
            raise UavObbPreparationError("invalid extracted image row")
        path = ROOT / str(row["output"])
        if not path.is_file() or sha256_file(path) != row.get("output_sha256"):
            raise UavObbPreparationError(f"locked extracted image changed: {path}")
    return lock


def extract_train_images() -> dict[str, Any]:
    inventory = _validate_prerequisites()
    if EXTRACTION_LOCK.exists() or EXTRACTION_MARKER.exists():
        return _validate_extraction_lock()
    if PARTITION_LOCK.exists() or PARTITION_MARKER.exists():
        raise UavObbPreparationError("partition output appeared before extraction lock")
    rows = _train_member_rows(inventory)
    outputs: list[dict[str, Any]] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for position, row in enumerate(rows, start=1):
            outputs.append(_extract_member(archive, row))
            if position % 100 == 0:
                print(json.dumps({"train_images_extracted": position}), flush=True)
    payload = {
        "schema_version": 1,
        "status": "UAV_OBB_V4_TRAIN_IMAGES_ONLY_EXTRACTED_CRC_DECODED_AND_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_inventory_sha256": INVENTORY_SHA256,
        "images": len(outputs),
        "unique_source_groups": len({str(row["source_group"]) for row in outputs}),
        "image_rows": outputs,
        "image_rows_payload_sha256": stable_hash(outputs, length=64),
        "annotation_content_accessed": False,
        "validation_or_test_content_accessed": False,
        "prediction_or_metric_accessed": False,
    }
    atomic_write_json(EXTRACTION_LOCK, payload)
    atomic_write_json(
        EXTRACTION_MARKER,
        {
            "status": payload["status"],
            "train_image_extraction_lock_sha256": sha256_file(EXTRACTION_LOCK),
        },
    )
    return payload


def _dhash(image: Image.Image, *, size: int = 8) -> int:
    values = np.asarray(image.resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _phash(image: Image.Image, *, size: int = 8, high_frequency_factor: int = 4) -> int:
    sample_size = size * high_frequency_factor
    values = np.asarray(
        image.resize((sample_size, sample_size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    coordinates = np.arange(sample_size, dtype=np.float64)
    basis = np.cos(np.pi * (2.0 * coordinates[:, None] + 1.0) * coordinates / (2.0 * sample_size))
    basis[:, 0] *= np.sqrt(1.0 / sample_size)
    basis[:, 1:] *= np.sqrt(2.0 / sample_size)
    low = (basis.T @ values @ basis)[:size, :size]
    median = float(np.median(low.reshape(-1)[1:]))
    result = 0
    for bit in (low > median).reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _hash_task(task: HashTask) -> dict[str, Any]:
    with Image.open(task.path) as image:
        base = image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    mirror = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    variants = (
        base,
        base.transpose(Image.Transpose.ROTATE_90),
        base.transpose(Image.Transpose.ROTATE_180),
        base.transpose(Image.Transpose.ROTATE_270),
        mirror,
        mirror.transpose(Image.Transpose.ROTATE_90),
        mirror.transpose(Image.Transpose.ROTATE_180),
        mirror.transpose(Image.Transpose.ROTATE_270),
    )
    return {
        "member": task.member,
        "path": Path(task.path).resolve().relative_to(ROOT.resolve()).as_posix(),
        "sha256": task.sha256,
        "source_group": task.source_group,
        "transform_names": list(TRANSFORM_NAMES),
        "phash64_variants": [f"{_phash(value):016x}" for value in variants],
        "dhash64_variants": [f"{_dhash(value):016x}" for value in variants],
    }


def _hash_values(row: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = row.get(key)
    if not isinstance(values, list) or len(values) != len(TRANSFORM_NAMES):
        raise UavObbPreparationError(f"invalid perceptual hashes: {key}")
    return tuple(int(str(value), 16) for value in values)


def _candidate_pairs(values: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for image_index, variants in enumerate(values):
        for value in variants:
            for segment_index, (low, high) in enumerate(SEGMENTS):
                width = high - low
                segment = (value >> (64 - high)) & ((1 << width) - 1)
                buckets[(segment_index, segment)].add(image_index)
    candidates: set[tuple[int, int]] = set()
    for indices in buckets.values():
        ordered = sorted(indices)
        for first_position, first in enumerate(ordered):
            for second in ordered[first_position + 1 :]:
                candidates.add((first, second))
    return candidates


def _minimum_distance(first: Sequence[int], second: Sequence[int]) -> int:
    return min((a ^ b).bit_count() for a in first for b in second)


def _components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[list[int]], list[dict[str, Any]], int]:
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        parent[upper] = lower

    source_groups: dict[str, list[int]] = defaultdict(list)
    exact_hashes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source_groups[str(row["source_group"])].append(index)
        exact_hashes[str(row["sha256"])].append(index)
    for group in (*source_groups.values(), *exact_hashes.values()):
        for index in group[1:]:
            union(group[0], index)
    phashes = [_hash_values(row, "phash64_variants") for row in rows]
    dhashes = [_hash_values(row, "dhash64_variants") for row in rows]
    candidates = _candidate_pairs(phashes) | _candidate_pairs(dhashes)
    edges: list[dict[str, Any]] = []
    for first, second in sorted(candidates):
        p_distance = _minimum_distance(phashes[first], phashes[second])
        d_distance = _minimum_distance(dhashes[first], dhashes[second])
        if p_distance < MINIMUM_ALLOWED_HAMMING or d_distance < MINIMUM_ALLOWED_HAMMING:
            union(first, second)
            edges.append(
                {
                    "first_member": str(rows[first]["member"]),
                    "second_member": str(rows[second]["member"]),
                    "minimum_pHash_distance": p_distance,
                    "minimum_dHash_distance": d_distance,
                }
            )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    components = sorted(
        grouped.values(),
        key=lambda indices: min(str(rows[index]["sha256"]) for index in indices),
    )
    return components, edges, len(candidates)


def _component_id(indices: Sequence[int], rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "|".join(sorted(str(rows[index]["sha256"]) for index in indices))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _select_development(
    components: Sequence[Sequence[int]], rows: Sequence[Mapping[str, Any]]
) -> tuple[set[str], int, int]:
    ordered = sorted(
        (
            (
                hashlib.sha256(
                    f"{PARTITION_SEED}:{_component_id(component, rows)}".encode()
                ).hexdigest(),
                _component_id(component, rows),
                len(component),
            )
            for component in components
        ),
        key=lambda value: value[0],
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for _, component_id, size in ordered:
        additions: dict[int, tuple[str, ...]] = {}
        for total, selected in sorted(reachable.items(), reverse=True):
            candidate = total + size
            if (
                candidate <= TRAIN_IMAGES
                and candidate not in reachable
                and candidate not in additions
            ):
                additions[candidate] = (*selected, component_id)
        reachable.update(additions)
    selected_total = min(
        reachable,
        key=lambda value: (abs(value - TARGET_DEVELOPMENT), value < TARGET_DEVELOPMENT, value),
    )
    return set(reachable[selected_total]), selected_total, len(reachable)


def _validate_partition_lock() -> dict[str, Any]:
    if not PARTITION_LOCK.is_file() or not PARTITION_MARKER.is_file():
        raise UavObbPreparationError("UAV-OBB train partition lock is incomplete")
    lock = _load_mapping(PARTITION_LOCK)
    marker = _load_mapping(PARTITION_MARKER)
    if (
        marker.get("train_partition_lock_sha256") != sha256_file(PARTITION_LOCK)
        or lock.get("pass") is not True
        or lock.get("annotation_content_accessed") is not False
        or lock.get("prediction_or_metric_accessed") is not False
    ):
        raise UavObbPreparationError("UAV-OBB train partition lock changed")
    return lock


def partition_train_images() -> dict[str, Any]:
    extraction = extract_train_images()
    if PARTITION_LOCK.exists() or PARTITION_MARKER.exists():
        return _validate_partition_lock()
    raw_rows = extraction.get("image_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != TRAIN_IMAGES:
        raise UavObbPreparationError("extracted train image registry changed")
    tasks = [
        HashTask(
            member=str(row["member"]),
            path=str((ROOT / str(row["output"])).resolve()),
            sha256=str(row["output_sha256"]),
            source_group=str(row["source_group"]),
        )
        for row in raw_rows
        if isinstance(row, dict)
    ]
    if len(tasks) != TRAIN_IMAGES:
        raise UavObbPreparationError("invalid extracted train image rows")
    with ProcessPoolExecutor(max_workers=HASH_WORKERS) as executor:
        hash_rows = list(executor.map(_hash_task, tasks, chunksize=4))
    hash_rows.sort(key=lambda row: str(row["member"]).casefold())
    components, edges, candidate_pairs = _components(hash_rows)
    development_components, development_count, reachable_totals = _select_development(
        components, hash_rows
    )
    assignments: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for component in components:
        component_id = _component_id(component, hash_rows)
        role = "development_A" if component_id in development_components else "reserve_B"
        members = sorted(str(hash_rows[index]["member"]) for index in component)
        component_rows.append(
            {
                "component_id": component_id,
                "role": role,
                "images": len(component),
                "members": members,
            }
        )
        for index in component:
            row = hash_rows[index]
            assignments.append(
                {
                    "role": role,
                    "component_id": component_id,
                    **row,
                }
            )
    assignments.sort(key=lambda row: (str(row["role"]), str(row["member"])))
    role_counts = Counter(str(row["role"]) for row in assignments)
    development_hashes = {
        str(row["sha256"]) for row in assignments if row["role"] == "development_A"
    }
    reserve_hashes = {str(row["sha256"]) for row in assignments if row["role"] == "reserve_B"}
    components_whole = all(
        len(
            {
                str(row["role"])
                for row in assignments
                if row["component_id"] == component["component_id"]
            }
        )
        == 1
        for component in component_rows
    )
    passed = (
        role_counts["development_A"] == TARGET_DEVELOPMENT
        and role_counts["reserve_B"] == TARGET_RESERVE
        and not development_hashes.intersection(reserve_hashes)
        and components_whole
    )
    if not passed:
        raise UavObbPreparationError(
            f"cannot form registered 900/483 leakage-safe roles: {dict(role_counts)}"
        )
    payload = {
        "schema_version": 1,
        "status": "UAV_OBB_V4_TRAIN_A_B_PARTITION_LOCKED_BEFORE_LABEL_PREDICTION_OR_METRIC_ACCESS",
        "locked_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "train_image_extraction_lock_sha256": sha256_file(EXTRACTION_LOCK),
        "images": TRAIN_IMAGES,
        "role_counts": dict(sorted(role_counts.items())),
        "source_groups": len({str(row["source_group"]) for row in assignments}),
        "near_duplicate_components": len(components),
        "component_size_distribution": dict(
            sorted(Counter(len(component) for component in components).items())
        ),
        "threshold_candidate_pairs_examined": candidate_pairs,
        "near_duplicate_edges": edges,
        "minimum_allowed_pHash_or_dHash_hamming_distance": MINIMUM_ALLOWED_HAMMING,
        "threshold_search": (
            "exhaustive for distance <=4 by five disjoint 64-bit segments; any pair with "
            "Hamming distance <=4 shares at least one segment"
        ),
        "components_kept_whole": components_whole,
        "exact_cross_role_SHA256_overlap": sorted(development_hashes.intersection(reserve_hashes)),
        "reachable_component_size_totals": reachable_totals,
        "selected_development_images": development_count,
        "component_rows": component_rows,
        "assignments": assignments,
        "assignments_payload_sha256": stable_hash(assignments, length=64),
        "pass": passed,
        "annotation_content_accessed": False,
        "prediction_or_metric_accessed": False,
        "validation_or_test_content_accessed": False,
    }
    atomic_write_json(PARTITION_LOCK, payload)
    atomic_write_json(
        PARTITION_MARKER,
        {
            "status": payload["status"],
            "train_partition_lock_sha256": sha256_file(PARTITION_LOCK),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare label-blind UAV-OBB v4 train roles")
    parser.add_argument("--stage", choices=("extract", "partition", "all"), default="all")
    args = parser.parse_args()
    result = extract_train_images() if args.stage == "extract" else partition_train_images()
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"image_rows", "assignments", "component_rows", "near_duplicate_edges"}
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
