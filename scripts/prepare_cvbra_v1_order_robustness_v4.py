from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts import prepare_cvbra_v1_order_robustness_v3 as base

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = base.ROOT
OUTPUT = ROOT / "data/processed/cvbra_v1_order_robustness_v4"
MANIFEST = OUTPUT / "manifest.json"
V3_FAILURE = ROOT / "data/processed/cvbra_v1_order_robustness_v3/FAILURE.json"
V3_FAILURE_SHA256 = "839a57396b15f27595b45715dab2267c70b4d3f3b781edf784350f9e8ce6e690"


class MaterializationV4Error(RuntimeError):
    pass


def _validate_existing() -> dict[str, Any]:
    manifest = base._load_mapping(MANIFEST)
    orders = manifest.get("orders")
    if not isinstance(orders, dict) or len(orders) != 2:
        raise MaterializationV4Error("existing v4 manifest is malformed")
    for order, row in orders.items():
        if not isinstance(row, dict):
            raise MaterializationV4Error(f"invalid v4 order row: {order}")
        for field in ("dataset_yaml", "mapping"):
            path = ROOT / str(row[field])
            if not path.is_file() or sha256_file(path) != row[f"{field}_sha256"]:
                raise MaterializationV4Error(f"existing v4 artifact changed: {path}")
    return manifest


def prepare() -> dict[str, Any]:
    if MANIFEST.exists():
        return _validate_existing()
    if OUTPUT.exists():
        raise MaterializationV4Error("partial v4 materialization requires audit")
    if sha256_file(base.BASE_MANIFEST) != base.BASE_MANIFEST_SHA256:
        raise MaterializationV4Error("base CVBRA-v1 manifest changed")
    if sha256_file(base.V2_MANIFEST) != base.V2_MANIFEST_SHA256:
        raise MaterializationV4Error("registered v2 order manifest changed")
    if sha256_file(V3_FAILURE) != V3_FAILURE_SHA256:
        raise MaterializationV4Error("v3 materialization failure audit changed")
    base_manifest = base._load_mapping(base.BASE_MANIFEST)
    entries = base_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != base.EXPECTED_IMAGES:
        raise MaterializationV4Error("base manifest membership is incomplete")
    entry_by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise MaterializationV4Error("invalid base manifest entry")
        required = ("image", "label", "image_sha256", "label_sha256", "role")
        if any(field not in entry for field in required):
            raise MaterializationV4Error("base manifest entry is missing a required field")
        entry_by_name[Path(str(entry["image"])).name] = entry
    if len(entry_by_name) != base.EXPECTED_IMAGES:
        raise MaterializationV4Error("base image names are not unique")

    plans: dict[str, list[tuple[Path, Path, dict[str, Any]]]] = {}
    for order, expected_list_hash in base.V2_LIST_HASHES.items():
        source_list = base.V2_ROOT / f"train_{order}.txt"
        if sha256_file(source_list) != expected_list_hash:
            raise MaterializationV4Error(f"registered source order changed: {order}")
        source_paths = [Path(line) for line in source_list.read_text(encoding="utf-8").splitlines()]
        if len(source_paths) != base.EXPECTED_IMAGES:
            raise MaterializationV4Error(f"registered source order length changed: {order}")
        order_plan: list[tuple[Path, Path, dict[str, Any]]] = []
        for source_image in source_paths:
            entry = entry_by_name.get(source_image.name)
            if entry is None or ROOT / str(entry["image"]) != source_image:
                raise MaterializationV4Error(f"unregistered source image: {source_image}")
            source_label = ROOT / str(entry["label"])
            if not source_image.is_file() or not source_label.is_file():
                raise MaterializationV4Error(f"source image or label is missing: {source_image}")
            order_plan.append((source_image, source_label, entry))
        plans[order] = order_plan

    order_rows: dict[str, dict[str, Any]] = {}
    for order, order_plan in plans.items():
        order_root = OUTPUT / order
        image_root = order_root / "images" / "train"
        label_root = order_root / "labels" / "train"
        image_root.mkdir(parents=True, exist_ok=False)
        label_root.mkdir(parents=True, exist_ok=False)
        mapping_rows: list[dict[str, Any]] = []
        for position, (source_image, source_label, entry) in enumerate(order_plan):
            target_image = image_root / base._materialized_name(position, source_image)
            target_label = label_root / f"{target_image.stem}.txt"
            os.link(source_image, target_image)
            os.link(source_label, target_label)
            if not os.path.samefile(source_image, target_image):
                raise MaterializationV4Error(f"image is not a hard link: {target_image}")
            if not os.path.samefile(source_label, target_label):
                raise MaterializationV4Error(f"label is not a hard link: {target_label}")
            mapping_rows.append(
                {
                    "position": position,
                    "materialized_image": target_image.relative_to(ROOT).as_posix(),
                    "materialized_label": target_label.relative_to(ROOT).as_posix(),
                    "source_image": source_image.relative_to(ROOT).as_posix(),
                    "source_label": source_label.relative_to(ROOT).as_posix(),
                    "image_sha256": entry["image_sha256"],
                    "label_sha256": entry["label_sha256"],
                    "role": entry["role"],
                    "view": entry.get("view"),
                }
            )
        expected_names = [
            Path(str(row["materialized_image"])).name for row in mapping_rows
        ]
        if [path.name for path in sorted(image_root.iterdir())] != expected_names:
            raise MaterializationV4Error(f"lexical backend order verification failed: {order}")
        mapping = order_root / "mapping.json"
        dataset_yaml = order_root / "dataset.yaml"
        atomic_write_json(mapping, mapping_rows)
        atomic_write_text(dataset_yaml, base._yaml(order_root))
        order_rows[order] = {
            "root": order_root.relative_to(ROOT).as_posix(),
            "images": base.EXPECTED_IMAGES,
            "mapping": mapping.relative_to(ROOT).as_posix(),
            "mapping_sha256": sha256_file(mapping),
            "dataset_yaml": dataset_yaml.relative_to(ROOT).as_posix(),
            "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "source_order_list_sha256": base.V2_LIST_HASHES[order],
            "lexically_sorted_materialization_matches_registered_order": True,
            "all_images_and_labels_are_hard_links": True,
        }
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_EFFECTIVE_ORDER_ROBUSTNESS_V4_DATA_PREPARED",
        "base_dataset_manifest": base.BASE_MANIFEST.relative_to(ROOT).as_posix(),
        "base_dataset_manifest_sha256": base.BASE_MANIFEST_SHA256,
        "base_entries_payload_sha256": base_manifest.get("entries_payload_sha256"),
        "images_per_order": base.EXPECTED_IMAGES,
        "orders": order_rows,
        "same_training_multiset": True,
        "image_or_label_bytes_modified": False,
        "storage_strategy": "NTFS hard links with permutation encoded in lexical filenames",
        "backend_sorted_order_verified_before_training": True,
        "source_replay_view_encoded_as_null": True,
        "supersedes_failed_materialization": V3_FAILURE.relative_to(ROOT).as_posix(),
    }
    atomic_write_json(MANIFEST, payload)
    return payload


def main() -> int:
    result = prepare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
