from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from buse_uav.data.corruptions import apply_corruption, deterministic_corruption_seed
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/nvd_real_snow_cvbra_v1.yaml"
RUNNER = ROOT / "scripts/run_nvd_real_snow_cvbra_v1.py"
OUTPUT = ROOT / "reports/development/nvd_real_snow_cvbra_v1"
DATA_ROOT = ROOT / "data/processed/nvd_real_snow_cvbra_v1"
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
DATA_LOCK = OUTPUT / "DATA_LOCK.json"

SPLIT_NAMES = (
    "source_train",
    "source_retention",
    "target_adaptation",
    "target_validation",
    "target_test",
)
ADAPTATION_METHODS = ("STF", "NoVisibility_L10", "CVBRA_L10")
TARGET_SCENES = 900
TRAIN_IMAGES = 3600
FOG_LEVELS = (0.6, 1.0)
FOG_GLOBAL_SEED = 20260825


class NvdProtocolError(RuntimeError):
    """Raised when the registered real-snow protocol cannot fail closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def rooted(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NvdProtocolError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NvdProtocolError(f"expected JSON object: {path}")
    return value


def load_protocol() -> dict[str, Any]:
    try:
        value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise NvdProtocolError(f"cannot read protocol: {exc}") from exc
    if not isinstance(value, dict):
        raise NvdProtocolError("NVD real-snow protocol must be a mapping")
    return value


def evenly_spaced_indices(start: int, stop: int, count: int) -> tuple[int, ...]:
    """Return deterministic inclusive integer positions without duplicated frames."""
    if start < 0 or stop < start or count <= 0 or count > stop - start + 1:
        raise ValueError("invalid deterministic frame-selection request")
    if count == 1:
        return (start,)
    span = stop - start
    indices = tuple(start + (position * span) // (count - 1) for position in range(count))
    if len(set(indices)) != count or indices[0] != start or indices[-1] != stop:
        raise RuntimeError("deterministic frame selection is not unique and endpoint complete")
    return indices


def rotated_hbb(
    xtl: float,
    ytl: float,
    xbr: float,
    ybr: float,
    rotation_degrees: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Project a CVAT center-rotated rectangle to its clipped enclosing HBB."""
    if image_width <= 0 or image_height <= 0 or xbr <= xtl or ybr <= ytl:
        return None
    width = xbr - xtl
    height = ybr - ytl
    center_x = (xtl + xbr) / 2.0
    center_y = (ytl + ybr) / 2.0
    angle = math.radians(rotation_degrees)
    enclosing_width = abs(width * math.cos(angle)) + abs(height * math.sin(angle))
    enclosing_height = abs(width * math.sin(angle)) + abs(height * math.cos(angle))
    left = max(0.0, center_x - enclosing_width / 2.0)
    top = max(0.0, center_y - enclosing_height / 2.0)
    right = min(float(image_width), center_x + enclosing_width / 2.0)
    bottom = min(float(image_height), center_y + enclosing_height / 2.0)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _assert_hash(path: Path, digest: str, *, label: str) -> None:
    if not path.is_file():
        raise NvdProtocolError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise NvdProtocolError(
            f"{label} changed: expected {digest}, observed {observed}: {path}"
        )


def _validate_protocol_fields(protocol: Mapping[str, Any]) -> None:
    integrity = protocol.get("integrity")
    adaptation = protocol.get("adaptation")
    evaluation = protocol.get("evaluation")
    source = protocol.get("source_training")
    if not all(isinstance(row, dict) for row in (integrity, adaptation, evaluation, source)):
        raise NvdProtocolError("required protocol sections are incomplete")
    assert isinstance(integrity, dict)
    assert isinstance(adaptation, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(source, dict)
    if (
        protocol.get("status")
        != "LOCK_BEFORE_MATERIALIZATION_TRAINING_PREDICTION_OR_METRIC_ACCESS"
        or tuple(adaptation.get("methods", ())) != ADAPTATION_METHODS
        or tuple(adaptation.get("seeds", ())) != (42, 27182, 31415)
        or adaptation.get("target_scenes") != TARGET_SCENES
        or adaptation.get("total_images_per_epoch") != TRAIN_IMAGES
        or source.get("checkpoint_selected_by_metric") is not False
        or adaptation.get("checkpoint_selected_by_metric") is not False
        or evaluation.get("target_test_temporal_blocks") != 32
        or evaluation.get("temporal_block_bootstrap_resamples") != 10000
        or integrity.get("target_test_predictions_before_registration") is not False
        or integrity.get("target_test_metrics_before_registration") is not False
        or integrity.get("target_validation_can_change_protocol") is not False
        or integrity.get("uavdt") != "permanently_withdrawn_and_prohibited"
    ):
        raise NvdProtocolError("registered NVD protocol fields changed")


def _raw_file_rows(protocol: Mapping[str, Any]) -> list[tuple[Path, str, str]]:
    raw = protocol.get("raw_integrity")
    splits = protocol.get("splits")
    if not isinstance(raw, dict) or not isinstance(splits, dict):
        raise NvdProtocolError("raw-integrity or split mapping is missing")
    rows: list[tuple[Path, str, str]] = []
    source_root = rooted(raw.get("source_root"))
    for group_name in ("provenance_files", "archives"):
        group = raw.get(group_name)
        if not isinstance(group, dict):
            raise NvdProtocolError(f"raw integrity group is missing: {group_name}")
        rows.extend(
            (source_root / str(name), str(digest), f"NVD {group_name}/{name}")
            for name, digest in group.items()
        )
    for split_name in SPLIT_NAMES:
        specification = splits.get(split_name)
        if not isinstance(specification, dict):
            raise NvdProtocolError(f"split is missing: {split_name}")
        rows.append(
            (
                rooted(specification["xml"]),
                str(specification["xml_sha256"]),
                f"{split_name} XML",
            )
        )
        media = rooted(specification["media"])
        if media.is_file():
            rows.append(
                (
                    media,
                    str(specification["media_sha256"]),
                    f"{split_name} media",
                )
            )
        elif not media.is_dir():
            raise NvdProtocolError(f"missing {split_name} media directory: {media}")
    return rows


def _validate_test_frame_inventory(protocol: Mapping[str, Any]) -> dict[str, Any]:
    splits = protocol["splits"]
    specification = splits["target_test"]
    media = rooted(specification["media"])
    start = int(specification["frame_start"])
    stop = int(specification["frame_stop"])
    expected = [f"frame_{frame:06d}.PNG" for frame in range(start, stop + 1)]
    observed = sorted(path.name for path in media.glob("*.PNG"))
    if observed != expected:
        raise NvdProtocolError("held-out target-test extracted-frame inventory changed")
    return {
        "directory": relative(media),
        "frames": len(observed),
        "first": observed[0],
        "last": observed[-1],
        "names_sha256": stable_hash(observed, length=64),
    }


def register() -> dict[str, Any]:
    protocol = load_protocol()
    _validate_protocol_fields(protocol)
    if not RUNNER.is_file():
        raise NvdProtocolError(f"runner is missing before registration: {RUNNER}")
    for path, digest, label in _raw_file_rows(protocol):
        _assert_hash(path, digest, label=label)
    pretrained = ROOT / str(protocol["source_training"]["initialization"])
    if not pretrained.is_file():
        raise NvdProtocolError(f"source initialization is missing: {pretrained}")
    inventory = _validate_test_frame_inventory(protocol)
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_CVBRA_REGISTERED_BEFORE_DERIVED_DATA_OR_MODEL_ACCESS",
        "registered_at_utc": utc_now(),
        "protocol": relative(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "preparation_runner": relative(Path(__file__)),
        "preparation_runner_sha256": sha256_file(Path(__file__)),
        "experiment_runner": relative(RUNNER),
        "experiment_runner_sha256": sha256_file(RUNNER),
        "source_initialization": relative(pretrained),
        "source_initialization_sha256": sha256_file(pretrained),
        "raw_files": [
            {
                "path": relative(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "role": label,
            }
            for path, digest, label in _raw_file_rows(protocol)
        ],
        "target_test_inventory": inventory,
        "target_test_label_access_before_registration": "structural_schema_count_audit_only",
        "target_test_image_access_before_registration": "file_inventory_only",
        "target_test_predictions_before_registration": False,
        "target_test_metrics_before_registration": False,
        "method_or_hyperparameter_selection_from_validation_or_test": False,
        "all_results_must_be_retained": True,
        "uavdt_used": False,
    }
    if REGISTRATION.exists():
        existing = load_json(REGISTRATION)
        stable_keys = tuple(
            key for key in payload if key not in {"schema_version", "status", "registered_at_utc"}
        )
        if any(existing.get(key) != payload.get(key) for key in stable_keys):
            raise NvdProtocolError("NVD real-snow registration changed")
        return existing
    if DATA_ROOT.exists() or DATA_LOCK.exists():
        raise NvdProtocolError("derived NVD data appeared before registration")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION.is_file():
        raise NvdProtocolError("NVD real-snow experiment is not registered")
    lock = load_json(REGISTRATION)
    if (
        lock.get("protocol_sha256") != sha256_file(PROTOCOL)
        or lock.get("preparation_runner_sha256") != sha256_file(Path(__file__))
        or lock.get("experiment_runner_sha256") != sha256_file(RUNNER)
    ):
        raise NvdProtocolError("registered protocol or implementation changed")
    for row in lock.get("raw_files", []):
        if not isinstance(row, dict):
            raise NvdProtocolError("malformed raw-file registration row")
        _assert_hash(rooted(row["path"]), str(row["sha256"]), label=str(row["role"]))
    return lock


def _split_indices(specification: Mapping[str, Any]) -> tuple[int, ...]:
    start = int(specification["frame_start"])
    stop = int(specification["frame_stop"])
    count = int(specification["selected_frames"])
    selection = str(specification["selection"])
    if selection in {"all_frames", "all_extracted_frames"}:
        indices = tuple(range(start, stop + 1))
    elif selection == "deterministic_even_spacing":
        indices = evenly_spaced_indices(start, stop, count)
    else:
        raise NvdProtocolError(f"unsupported frame selection: {selection}")
    if len(indices) != count:
        raise NvdProtocolError("registered frame-selection count changed")
    return indices


def _parse_annotations(
    path: Path,
    *,
    expected_start: int,
    expected_stop: int,
) -> tuple[int, int, dict[int, list[dict[str, Any]]], dict[str, int]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise NvdProtocolError(f"cannot parse NVD XML {path}: {exc}") from exc
    width_text = root.findtext("./meta/original_size/width") or root.findtext(
        "./meta/task/original_size/width"
    )
    height_text = root.findtext("./meta/original_size/height") or root.findtext(
        "./meta/task/original_size/height"
    )
    if width_text is None or height_text is None:
        raise NvdProtocolError(f"NVD XML has no original size: {path}")
    width, height = int(width_text), int(height_text)
    labels = {str(track.attrib.get("label")) for track in root.findall("track")}
    if labels != {"car"}:
        raise NvdProtocolError(f"unexpected NVD label set in {path}: {sorted(labels)}")
    boxes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        for element in track.findall("box"):
            counts["xml_boxes"] += 1
            frame = int(element.attrib["frame"])
            if element.attrib.get("outside", "0") == "1":
                counts["outside_excluded"] += 1
                continue
            if frame < expected_start or frame > expected_stop:
                counts["visible_outside_registered_range"] += 1
                continue
            values = rotated_hbb(
                float(element.attrib["xtl"]),
                float(element.attrib["ytl"]),
                float(element.attrib["xbr"]),
                float(element.attrib["ybr"]),
                float(element.attrib.get("rotation", "0")),
                width,
                height,
            )
            if values is None:
                counts["degenerate_excluded"] += 1
                continue
            left, top, right, bottom = values
            boxes[frame].append(
                {
                    "track_id": track_id,
                    "bbox": [left, top, right - left, bottom - top],
                    "rotation_degrees": float(element.attrib.get("rotation", "0")),
                    "occluded": int(element.attrib.get("occluded", "0")),
                }
            )
            counts["visible_projected"] += 1
    return width, height, boxes, dict(counts)


def _hardlink_or_validate(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise NvdProtocolError(f"derived path is not a file: {destination}")
        if os.path.samefile(source, destination) or sha256_file(source) == sha256_file(destination):
            return
        raise NvdProtocolError(f"existing derived file differs: {destination}")
    os.link(source, destination)


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise NvdProtocolError(f"refusing to overwrite derived image: {path}")
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise NvdProtocolError(f"cannot encode derived JPEG: {path}")
    temporary.replace(path)


def _materialize_video_frames(
    media: Path,
    indices: Sequence[int],
    output: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> dict[int, Path]:
    capture = cv2.VideoCapture(str(media))
    if not capture.isOpened():
        raise NvdProtocolError(f"cannot open NVD video: {media}")
    reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (reported_width, reported_height) != (expected_width, expected_height):
        capture.release()
        raise NvdProtocolError(f"video/XML geometry mismatch: {media}")
    wanted = set(indices)
    last = max(indices)
    materialized: dict[int, Path] = {}
    for frame_index in range(last + 1):
        ok, image = capture.read()
        if not ok or image is None:
            capture.release()
            raise NvdProtocolError(f"video ended before registered frame {frame_index}: {media}")
        if frame_index not in wanted:
            continue
        destination = output / f"frame_{frame_index:06d}.jpg"
        _write_jpeg(destination, image)
        materialized[frame_index] = destination
    capture.release()
    if set(materialized) != wanted:
        raise NvdProtocolError(f"video frame materialization is incomplete: {media}")
    if reported_frames <= last:
        raise NvdProtocolError(f"reported video frame count is shorter than registered: {media}")
    return materialized


def _materialize_extracted_frames(
    media: Path,
    indices: Sequence[int],
    output: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for frame_index in indices:
        source = media / f"frame_{frame_index:06d}.PNG"
        if not source.is_file():
            raise NvdProtocolError(f"missing registered extracted frame: {source}")
        destination = output / source.name
        _hardlink_or_validate(source, destination)
        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (expected_height, expected_width):
            raise NvdProtocolError(f"extracted frame geometry changed: {source}")
        result[frame_index] = destination
    return result


def _temporal_blocks(indices: Sequence[int], count: int) -> dict[int, int]:
    if count <= 0 or len(indices) < count:
        raise NvdProtocolError("invalid temporal-block request")
    result: dict[int, int] = {}
    for block, positions in enumerate(np.array_split(np.arange(len(indices)), count)):
        for position in positions.tolist():
            result[int(indices[int(position)])] = block
    if len(result) != len(indices) or set(result.values()) != set(range(count)):
        raise NvdProtocolError("temporal-block construction is incomplete")
    return result


def _label_text(rows: Sequence[Mapping[str, Any]], width: int, height: int) -> str:
    lines = []
    for row in rows:
        x, y, box_width, box_height = (float(value) for value in row["bbox"])
        values = (
            (x + box_width / 2.0) / width,
            (y + box_height / 2.0) / height,
            box_width / width,
            box_height / height,
        )
        if not all(0.0 <= value <= 1.0 for value in values):
            raise NvdProtocolError("projected YOLO label escaped normalized extent")
        lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def _materialize_split(name: str, specification: Mapping[str, Any]) -> dict[str, Any]:
    split_root = DATA_ROOT / "splits" / name
    if split_root.exists():
        raise NvdProtocolError(f"partial split output requires audit: {split_root}")
    indices = _split_indices(specification)
    xml_path = rooted(specification["xml"])
    width, height, boxes, annotation_counts = _parse_annotations(
        xml_path,
        expected_start=int(specification["frame_start"]),
        expected_stop=int(specification["frame_stop"]),
    )
    media = rooted(specification["media"])
    images_root = split_root / "images"
    materialized = (
        _materialize_video_frames(
            media,
            indices,
            images_root,
            expected_width=width,
            expected_height=height,
        )
        if media.is_file()
        else _materialize_extracted_frames(
            media,
            indices,
            images_root,
            expected_width=width,
            expected_height=height,
        )
    )
    block_count = 32 if name == "target_test" else 0
    temporal = _temporal_blocks(indices, block_count) if block_count else {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, frame_index in enumerate(indices, start=1):
        image_path = materialized[frame_index]
        label_path = split_root / "labels" / f"frame_{frame_index:06d}.txt"
        frame_boxes = boxes.get(frame_index, [])
        atomic_write_text(label_path, _label_text(frame_boxes, width, height))
        image_row: dict[str, Any] = {
            "id": image_id,
            "file_name": image_path.relative_to(split_root).as_posix(),
            "width": width,
            "height": height,
            "frame_index": frame_index,
            "sequence": name,
        }
        if block_count:
            image_row["temporal_block"] = temporal[frame_index]
        images.append(image_row)
        for box in frame_boxes:
            bbox = [float(value) for value in box["bbox"]]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "attributes": {
                        "track_id": int(box["track_id"]),
                        "source_rotation_degrees": float(box["rotation_degrees"]),
                        "occluded": int(box["occluded"]),
                    },
                }
            )
            annotation_id += 1
        manifest_rows.append(
            {
                "image_id": image_id,
                "frame_index": frame_index,
                "image": relative(image_path),
                "image_sha256": sha256_file(image_path),
                "label": relative(label_path),
                "label_sha256": sha256_file(label_path),
                "annotations": len(frame_boxes),
                "temporal_block": temporal.get(frame_index),
            }
        )
    coco_path = split_root / "annotations.coco.json"
    manifest_path = split_root / "manifest.json"
    coco = {
        "info": {
            "description": f"NVD {name}: registered rotated-to-HBB projection",
            "protocol": "nvd_real_snow_cvbra_v1",
        },
        "licenses": [{"id": 1, "name": "CC-BY-NC"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}],
    }
    atomic_write_json(coco_path, coco)
    manifest = {
        "schema_version": 1,
        "split": name,
        "role": specification["role"],
        "location": specification["location"],
        "date": specification["date"],
        "snow_cover": specification["snow_cover"],
        "selection": specification["selection"],
        "frame_indices": list(indices),
        "frame_indices_sha256": stable_hash(list(indices), length=64),
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": sum(row["annotations"] == 0 for row in manifest_rows),
        "annotation_projection": (
            "enclosing axis-aligned box of each CVAT center-rotated rectangle, clipped"
        ),
        "annotation_audit": annotation_counts,
        "temporal_blocks": block_count,
        "rows": manifest_rows,
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "split": name,
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "annotation": relative(coco_path),
        "annotation_sha256": sha256_file(coco_path),
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": manifest["empty_images"],
        "frame_indices_sha256": manifest["frame_indices_sha256"],
        "temporal_blocks": block_count,
    }


def _dataset_yaml(path: Path) -> str:
    return (
        f"path: {path.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "names:\n"
        "  0: car\n"
    )


def _alias(source: Path, label: Path, root: Path, stem: str) -> tuple[Path, Path]:
    image_target = root / "images" / "train" / f"{stem}{source.suffix.casefold()}"
    label_target = root / "labels" / "train" / f"{stem}.txt"
    _hardlink_or_validate(source, image_target)
    _hardlink_or_validate(label, label_target)
    return image_target, label_target


def _load_manifest_rows(split: str) -> list[dict[str, Any]]:
    value = load_json(DATA_ROOT / "splits" / split / "manifest.json").get("rows")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise NvdProtocolError(f"malformed split manifest: {split}")
    return value


def _materialize_fog_views(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    view_root = DATA_ROOT / "target_views"
    if view_root.exists():
        raise NvdProtocolError(f"partial target-view output requires audit: {view_root}")
    rows = []
    for target in target_rows:
        frame = int(target["frame_index"])
        source = rooted(target["image"])
        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise NvdProtocolError(f"cannot read target frame for fog materialization: {source}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        for level_index, beta in enumerate(FOG_LEVELS, start=1):
            view = f"fog_beta_{str(beta).replace('.', 'p')}"
            seed = deterministic_corruption_seed(
                FOG_GLOBAL_SEED,
                f"nvd-target-adaptation-{frame:06d}",
                "fog",
                level_index,
            )
            fog_rgb = apply_corruption(
                image_rgb,
                corruption="fog",
                parameter=beta,
                seed=seed,
            )
            destination = view_root / view / f"frame_{frame:06d}.jpg"
            _write_jpeg(destination, cv2.cvtColor(fog_rgb, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "frame_index": frame,
                    "view": view,
                    "beta": beta,
                    "seed": seed,
                    "image": relative(destination),
                    "image_sha256": sha256_file(destination),
                    "source_image_sha256": target["image_sha256"],
                }
            )
    manifest_path = view_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "operator": "buse_uav.data.corruptions.apply_corruption/fog",
        "global_seed": FOG_GLOBAL_SEED,
        "betas": list(FOG_LEVELS),
        "source_scenes": len(target_rows),
        "generated_images": len(rows),
        "rows": rows,
    }
    atomic_write_json(manifest_path, payload)
    return {
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "generated_images": len(rows),
        "rows": rows,
    }


def _build_training_datasets(fog: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_rows = _load_manifest_rows("source_train")
    target_rows = _load_manifest_rows("target_adaptation")
    if len(source_rows) != TARGET_SCENES or len(target_rows) != TARGET_SCENES:
        raise NvdProtocolError("registered source/target training coverage changed")
    fog_rows = fog.get("rows")
    if not isinstance(fog_rows, list):
        raise NvdProtocolError("fog manifest rows are missing")
    fog_by_key = {
        (int(row["frame_index"]), str(row["view"])): row
        for row in fog_rows
        if isinstance(row, dict)
    }
    outputs: list[dict[str, Any]] = []

    source_dataset = DATA_ROOT / "datasets" / "source"
    source_entries = []
    for position, row in enumerate(source_rows, start=1):
        image, label = _alias(
            rooted(row["image"]),
            rooted(row["label"]),
            source_dataset,
            f"source_{position:04d}",
        )
        source_entries.append(
            {
                "role": "source_train",
                "frame_index": row["frame_index"],
                "image": relative(image),
                "label": relative(label),
            }
        )
    source_yaml = source_dataset / "dataset.yaml"
    source_manifest = source_dataset / "manifest.json"
    atomic_write_text(source_yaml, _dataset_yaml(source_dataset))
    atomic_write_json(
        source_manifest,
        {"schema_version": 1, "dataset": "source", "entries": source_entries},
    )
    outputs.append(
        {
            "dataset": "source",
            "images": len(source_entries),
            "yaml": relative(source_yaml),
            "yaml_sha256": sha256_file(source_yaml),
            "manifest": relative(source_manifest),
            "manifest_sha256": sha256_file(source_manifest),
        }
    )

    for method in ADAPTATION_METHODS:
        dataset_root = DATA_ROOT / "datasets" / method
        entries = []
        for position, row in enumerate(target_rows, start=1):
            frame = int(row["frame_index"])
            target_image = rooted(row["image"])
            target_label = rooted(row["label"])
            if method == "CVBRA_L10":
                views = (
                    ("original", target_image),
                    ("fog_beta_0p6", rooted(fog_by_key[(frame, "fog_beta_0p6")]["image"])),
                    ("fog_beta_1p0", rooted(fog_by_key[(frame, "fog_beta_1p0")]["image"])),
                )
            else:
                repeats = 4 if method == "STF" else 3
                views = tuple(
                    (f"original_repeat_{index + 1}", target_image)
                    for index in range(repeats)
                )
            for view, image_source in views:
                image, label = _alias(
                    image_source,
                    target_label,
                    dataset_root,
                    f"target_{position:04d}_{view}",
                )
                entries.append(
                    {
                        "role": "target",
                        "frame_index": frame,
                        "view": view,
                        "image": relative(image),
                        "label": relative(label),
                    }
                )
        if method != "STF":
            for position, row in enumerate(source_rows, start=1):
                image, label = _alias(
                    rooted(row["image"]),
                    rooted(row["label"]),
                    dataset_root,
                    f"source_replay_{position:04d}",
                )
                entries.append(
                    {
                        "role": "source_replay",
                        "frame_index": row["frame_index"],
                        "view": "source_replay",
                        "image": relative(image),
                        "label": relative(label),
                    }
                )
        if len(entries) != TRAIN_IMAGES:
            raise NvdProtocolError(f"matched training budget changed for {method}")
        dataset_yaml = dataset_root / "dataset.yaml"
        manifest_path = dataset_root / "manifest.json"
        atomic_write_text(dataset_yaml, _dataset_yaml(dataset_root))
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "dataset": method,
                "images_per_epoch": len(entries),
                "roles": dict(Counter(str(row["role"]) for row in entries)),
                "views": dict(Counter(str(row["view"]) for row in entries)),
                "entries": entries,
            },
        )
        outputs.append(
            {
                "dataset": method,
                "images": len(entries),
                "yaml": relative(dataset_yaml),
                "yaml_sha256": sha256_file(dataset_yaml),
                "manifest": relative(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return outputs


def materialize() -> dict[str, Any]:
    registration = validate_registration()
    if DATA_LOCK.exists():
        return validate_data_lock()
    if DATA_ROOT.exists():
        raise NvdProtocolError(f"partial derived data requires audit: {DATA_ROOT}")
    protocol = load_protocol()
    splits = protocol.get("splits")
    if not isinstance(splits, dict):
        raise NvdProtocolError("protocol split mapping is missing")
    split_rows = []
    for name in SPLIT_NAMES:
        specification = splits.get(name)
        if not isinstance(specification, dict):
            raise NvdProtocolError(f"split specification is missing: {name}")
        split_rows.append(_materialize_split(name, specification))
    fog = _materialize_fog_views(_load_manifest_rows("target_adaptation"))
    datasets = _build_training_datasets(fog)
    payload = {
        "schema_version": 1,
        "status": "NVD_REAL_SNOW_DERIVED_DATA_LOCKED_BEFORE_TRAINING_OR_PREDICTION",
        "locked_at_utc": utc_now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "protocol_sha256": registration["protocol_sha256"],
        "splits": split_rows,
        "fog_views": {key: value for key, value in fog.items() if key != "rows"},
        "training_datasets": datasets,
        "training_started_before_data_lock": False,
        "target_test_predictions_before_data_lock": False,
        "target_test_metrics_before_data_lock": False,
    }
    atomic_write_json(DATA_LOCK, payload)
    return payload


def validate_data_lock() -> dict[str, Any]:
    validate_registration()
    lock = load_json(DATA_LOCK)
    if (
        lock.get("status")
        != "NVD_REAL_SNOW_DERIVED_DATA_LOCKED_BEFORE_TRAINING_OR_PREDICTION"
        or lock.get("registration_sha256") != sha256_file(REGISTRATION)
    ):
        raise NvdProtocolError("derived-data lock changed")
    for row in lock.get("splits", []):
        if not isinstance(row, dict):
            raise NvdProtocolError("malformed split lock")
        _assert_hash(rooted(row["manifest"]), str(row["manifest_sha256"]), label="split manifest")
        _assert_hash(
            rooted(row["annotation"]),
            str(row["annotation_sha256"]),
            label="COCO annotation",
        )
    fog = lock.get("fog_views")
    if not isinstance(fog, dict):
        raise NvdProtocolError("fog-view lock is missing")
    _assert_hash(rooted(fog["manifest"]), str(fog["manifest_sha256"]), label="fog manifest")
    for row in lock.get("training_datasets", []):
        if not isinstance(row, dict):
            raise NvdProtocolError("malformed training-dataset lock")
        _assert_hash(rooted(row["yaml"]), str(row["yaml_sha256"]), label="dataset YAML")
        _assert_hash(rooted(row["manifest"]), str(row["manifest_sha256"]), label="dataset manifest")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the registered NVD real-snow study")
    parser.add_argument("--stage", choices=("register", "materialize", "validate"), required=True)
    args = parser.parse_args()
    if args.stage == "register":
        result = register()
    elif args.stage == "materialize":
        result = materialize()
    else:
        result = validate_data_lock()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
