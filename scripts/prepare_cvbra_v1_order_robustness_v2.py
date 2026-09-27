from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "cvbra_v1"
IMAGE_DIR = BASE / "images" / "train"
LABEL_DIR = BASE / "labels" / "train"
OUTPUT = ROOT / "data" / "processed" / "cvbra_v1_order_robustness_v2"
BASE_MANIFEST = BASE / "manifest.json"
BASE_MANIFEST_SHA256 = "4d802e360563d5d8fd85be4a62b1a6c25930df07f78dc31fc57330c10b2325f1"
EXPECTED_IMAGES = 3600
ORDER_KEYS = ("hash_a", "hash_b")


class OrderPreparationError(RuntimeError):
    pass


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _order_key(path: Path, order: str) -> bytes:
    return hashlib.sha256(f"{order}\0{path.name}".encode()).digest()


def _dataset_yaml(list_path: Path) -> str:
    root = BASE.resolve().as_posix()
    train = list_path.resolve().as_posix()
    return (
        f"path: {root}\n"
        f"train: {train}\n"
        f"val: {train}\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
    )


def prepare() -> dict[str, Any]:
    if not BASE_MANIFEST.is_file() or sha256_file(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
        raise OrderPreparationError("locked CVBRA-v1 dataset manifest changed")
    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise OrderPreparationError("unexpected CVBRA-v1 image membership")
    missing_labels = [
        path.name for path in images if not (LABEL_DIR / f"{path.stem}.txt").is_file()
    ]
    if missing_labels:
        raise OrderPreparationError(f"missing labels for {len(missing_labels)} images")
    membership = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "label_size_bytes": (LABEL_DIR / f"{path.stem}.txt").stat().st_size,
        }
        for path in images
    ]
    order_artifacts: dict[str, dict[str, Any]] = {}
    sequences: dict[str, list[str]] = {}
    for order in ORDER_KEYS:
        ordered = sorted(images, key=lambda path: _order_key(path, order))
        sequence = [path.name for path in ordered]
        sequences[order] = sequence
        list_path = OUTPUT / f"train_{order}.txt"
        yaml_path = OUTPUT / f"dataset_{order}.yaml"
        atomic_write_text(
            list_path,
            "\n".join(path.resolve().as_posix() for path in ordered) + "\n",
        )
        atomic_write_text(yaml_path, _dataset_yaml(list_path))
        order_artifacts[order] = {
            "train_list": list_path.relative_to(ROOT).as_posix(),
            "train_list_sha256": sha256_file(list_path),
            "dataset_yaml": yaml_path.relative_to(ROOT).as_posix(),
            "dataset_yaml_sha256": sha256_file(yaml_path),
            "sequence_sha256": _sha256_json(sequence),
            "images": len(sequence),
        }
    if sequences["hash_a"] == sequences["hash_b"]:
        raise OrderPreparationError("registered order perturbations are identical")
    positional_matches = sum(
        left == right for left, right in zip(sequences["hash_a"], sequences["hash_b"], strict=True)
    )
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_ORDER_ROBUSTNESS_V2_DATA_PREPARED",
        "base_dataset_manifest": BASE_MANIFEST.relative_to(ROOT).as_posix(),
        "base_dataset_manifest_sha256": BASE_MANIFEST_SHA256,
        "images": EXPECTED_IMAGES,
        "membership_sha256": _sha256_json(membership),
        "membership_definition": "sorted image name, image size, and matched label size",
        "orders": order_artifacts,
        "positional_matches_between_orders": positional_matches,
        "same_training_multiset": True,
        "image_or_label_bytes_modified": False,
    }
    manifest = OUTPUT / "manifest.json"
    atomic_write_json(manifest, payload)
    return payload


def main() -> int:
    payload = prepare()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
