from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "data/processed/cvbra_v1/manifest.json"
BASE_MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
V2_ROOT = ROOT / "data/processed/cvbra_v1_order_robustness_v2"
V2_MANIFEST = V2_ROOT / "manifest.json"
V2_MANIFEST_SHA256 = "12b972972d56aae7bd27c8d46e5d2477c9c2552979acd40876aaa0f25df074de"
V2_LIST_HASHES = {
    "hash_a": "d91942aa3411277b5d9f157ac937d935ff7a1121f08e12d4f85808907d0dd96d",
    "hash_b": "dd07b36b5b50ccb18ef3aa2b95813eb180d3cd834ee842972647d1e561c1bb8e",
}
OUTPUT = ROOT / "data/processed/cvbra_v1_order_robustness_v3"
MANIFEST = OUTPUT / "manifest.json"
EXPECTED_IMAGES = 3600


class MaterializationError(RuntimeError):
    pass


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def _materialized_name(position: int, source: Path) -> str:
    return f"{position:04d}__{source.stem}{source.suffix.lower()}"


def _yaml(order_root: Path) -> str:
    return (
        f"path: {order_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
    )


def _validate_existing() -> dict[str, Any]:
    manifest = _load_mapping(MANIFEST)
    orders = manifest.get("orders")
    if not isinstance(orders, dict) or len(orders) != 2:
        raise MaterializationError("existing v3 manifest is malformed")
    for order, row in orders.items():
        if not isinstance(row, dict):
            raise MaterializationError(f"invalid v3 order row: {order}")
        for field in ("dataset_yaml", "mapping"):
            path = ROOT / str(row[field])
            if not path.is_file() or sha256_file(path) != row[f"{field}_sha256"]:
                raise MaterializationError(f"existing v3 artifact changed: {path}")
    return manifest


def prepare() -> dict[str, Any]:
    if MANIFEST.exists():
        return _validate_existing()
    if OUTPUT.exists():
        raise MaterializationError("partial v3 materialization requires audit")
    if sha256_file(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
        raise MaterializationError("base CVBRA-v1 manifest changed")
    if sha256_file(V2_MANIFEST) != V2_MANIFEST_SHA256:
        raise MaterializationError("registered v2 order manifest changed")
    base = _load_mapping(BASE_MANIFEST)
    entries = base.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_IMAGES:
        raise MaterializationError("base manifest membership is incomplete")
    entry_by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise MaterializationError("invalid base manifest entry")
        name = Path(str(entry["image"])).name
        entry_by_name[name] = entry
    if len(entry_by_name) != EXPECTED_IMAGES:
        raise MaterializationError("base image names are not unique")

    order_rows: dict[str, dict[str, Any]] = {}
    for order, expected_list_hash in V2_LIST_HASHES.items():
        source_list = V2_ROOT / f"train_{order}.txt"
        if sha256_file(source_list) != expected_list_hash:
            raise MaterializationError(f"registered source order changed: {order}")
        source_paths = [Path(line) for line in source_list.read_text(encoding="utf-8").splitlines()]
        if len(source_paths) != EXPECTED_IMAGES:
            raise MaterializationError(f"registered source order length changed: {order}")
        order_root = OUTPUT / order
        image_root = order_root / "images" / "train"
        label_root = order_root / "labels" / "train"
        image_root.mkdir(parents=True, exist_ok=False)
        label_root.mkdir(parents=True, exist_ok=False)
        mapping_rows: list[dict[str, Any]] = []
        for position, source_image in enumerate(source_paths):
            entry = entry_by_name.get(source_image.name)
            if entry is None or ROOT / str(entry["image"]) != source_image:
                raise MaterializationError(f"unregistered source image: {source_image}")
            source_label = ROOT / str(entry["label"])
            target_image = image_root / _materialized_name(position, source_image)
            target_label = label_root / f"{target_image.stem}.txt"
            os.link(source_image, target_image)
            os.link(source_label, target_label)
            if not os.path.samefile(source_image, target_image):
                raise MaterializationError(f"image is not a hard link: {target_image}")
            if not os.path.samefile(source_label, target_label):
                raise MaterializationError(f"label is not a hard link: {target_label}")
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
                    "view": entry["view"],
                }
            )
        sorted_names = [path.name for path in sorted(image_root.iterdir())]
        expected_names = [
            Path(str(row["materialized_image"])).name for row in mapping_rows
        ]
        if sorted_names != expected_names:
            raise MaterializationError(f"lexical backend order verification failed: {order}")
        mapping = order_root / "mapping.json"
        dataset_yaml = order_root / "dataset.yaml"
        atomic_write_json(mapping, mapping_rows)
        atomic_write_text(dataset_yaml, _yaml(order_root))
        order_rows[order] = {
            "root": order_root.relative_to(ROOT).as_posix(),
            "images": EXPECTED_IMAGES,
            "mapping": mapping.relative_to(ROOT).as_posix(),
            "mapping_sha256": sha256_file(mapping),
            "dataset_yaml": dataset_yaml.relative_to(ROOT).as_posix(),
            "dataset_yaml_sha256": sha256_file(dataset_yaml),
            "source_order_list_sha256": expected_list_hash,
            "lexically_sorted_materialization_matches_registered_order": True,
            "all_images_and_labels_are_hard_links": True,
        }
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_EFFECTIVE_ORDER_ROBUSTNESS_V3_DATA_PREPARED",
        "base_dataset_manifest": BASE_MANIFEST.relative_to(ROOT).as_posix(),
        "base_dataset_manifest_sha256": BASE_MANIFEST_SHA256,
        "base_entries_payload_sha256": base.get("entries_payload_sha256"),
        "images_per_order": EXPECTED_IMAGES,
        "orders": order_rows,
        "same_training_multiset": True,
        "image_or_label_bytes_modified": False,
        "storage_strategy": "NTFS hard links with permutation encoded in lexical filenames",
        "backend_sorted_order_verified_before_training": True,
    }
    atomic_write_json(MANIFEST, payload)
    return payload


def main() -> int:
    result = prepare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
