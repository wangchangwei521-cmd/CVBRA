from __future__ import annotations

import io
import json
import math
import shutil
import uuid
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from buse_uav.data.coco import CocoDocument, load_coco
from buse_uav.data.common import DataError, image_size, validation_result
from buse_uav.schemas import ImageRecord
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

DRONEVEHICLE_CLASS_NAMES = ("car", "truck", "bus")
DRONEVEHICLE_TARGET_CATEGORIES = {1: "car", 2: "truck", 3: "bus"}
DRONEVEHICLE_SOURCE_LABELS = (
    "car",
    "truck",
    "bus",
    "van",
    "feright car",
    "feright_car",
)
DRONEVEHICLE_SOURCE_TO_TARGET = {"car": 1, "truck": 2, "bus": 3}
DRONEVEHICLE_IGNORED_LABELS = ("van", "feright car", "feright_car")
DRONEVEHICLE_VAL_SHA256 = "043b7944ebb8ce076c1e5cfd37c33de6a59a9f62cf47c0f028387f703d4f5250"
DRONEVEHICLE_VAL_BYTES = 723_321_423
DRONEVEHICLE_HF_COMMIT = "947bd13f002857eb79a3e5b4feb2c6aca453ade7"
DRONEVEHICLE_OFFICIAL_COMMIT = "a1cf9289288a486b759c5d2798d2dcdc3bfc4683"
DRONEVEHICLE_VAL_IMAGES = 1469
SOURCE_WIDTH = 840
SOURCE_HEIGHT = 712
BORDER = 100
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 512


class DroneVehicleAdapter:
    """Adapter for the frozen RGB-only DroneVehicle validation conversion."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        image_directory: Path | None = None,
        annotation_file: Path | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self._image_directory = image_directory
        self._annotation_file = annotation_file

    @property
    def class_names(self) -> Sequence[str]:
        return DRONEVEHICLE_CLASS_NAMES

    def image_dir(self) -> Path:
        path = self._image_directory or self.root / "images"
        if not path.is_dir():
            raise DataError(f"DroneVehicle RGB image directory does not exist: {path}")
        return path

    def annotation_file(self) -> Path:
        path = self._annotation_file or self.root / "annotations" / "dronevehicle_rgb_val.coco.json"
        if not path.is_file():
            raise DataError(f"DroneVehicle COCO annotation file does not exist: {path}")
        return path

    def document(self) -> CocoDocument:
        return load_coco(self.annotation_file())

    def image_path(self, file_name: str) -> Path:
        root = self.image_dir().resolve()
        path = (root / file_name).resolve()
        if path != root and root not in path.parents:
            raise DataError(f"DroneVehicle image path escapes the configured root: {file_name}")
        return path

    def records(self) -> Sequence[ImageRecord]:
        records: list[ImageRecord] = []
        for image in self.document().images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")
            if (
                not isinstance(image_id, int)
                or not isinstance(file_name, str)
                or width != OUTPUT_WIDTH
                or height != OUTPUT_HEIGHT
            ):
                raise DataError(f"invalid DroneVehicle COCO image entry: {image}")
            path = self.image_path(file_name)
            if not path.is_file():
                raise DataError(f"DroneVehicle RGB image file does not exist: {path}")
            records.append(
                ImageRecord(
                    image_id=image_id,
                    path=str(path),
                    width=OUTPUT_WIDTH,
                    height=OUTPUT_HEIGHT,
                )
            )
        return tuple(records)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            document = self.document()
        except DataError as exc:
            return validation_result(
                dataset="dronevehicle",
                split=self.split,
                errors=[str(exc)],
                warnings=[],
                stats={},
            )
        if document.category_names_by_id != DRONEVEHICLE_TARGET_CATEGORIES:
            errors.append("DroneVehicle categories must be exactly car=1, truck=2, bus=3")
        if self.split == "external_rgb_val" and len(document.images) != DRONEVEHICLE_VAL_IMAGES:
            errors.append("DroneVehicle RGB validation image count changed")
        image_ids: set[int] = set()
        missing = 0
        for image in document.images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            if not isinstance(image_id, int) or image_id in image_ids:
                errors.append(f"invalid or duplicate DroneVehicle image ID: {image_id}")
                continue
            image_ids.add(image_id)
            if not isinstance(file_name, str):
                errors.append(f"DroneVehicle image {image_id} has no file name")
                continue
            path = self.image_path(file_name)
            if not path.is_file():
                missing += 1
                errors.append(f"missing DroneVehicle RGB image: {path}")
                continue
            if image_size(path) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
                errors.append(f"DroneVehicle RGB image has the wrong cropped size: {path}")
        class_counts: Counter[str] = Counter()
        annotation_ids: set[int] = set()
        for annotation in document.annotations:
            annotation_id = annotation.get("id")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if not isinstance(annotation_id, int) or annotation_id in annotation_ids:
                errors.append(f"invalid or duplicate DroneVehicle annotation ID: {annotation_id}")
                continue
            annotation_ids.add(annotation_id)
            if image_id not in image_ids or category_id not in DRONEVEHICLE_TARGET_CATEGORIES:
                errors.append(f"DroneVehicle annotation {annotation_id} has an invalid reference")
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"DroneVehicle annotation {annotation_id} has an invalid bbox")
                continue
            x, y, width, height = (float(value) for value in bbox)
            if (
                not all(math.isfinite(value) for value in (x, y, width, height))
                or x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > OUTPUT_WIDTH + 1e-6
                or y + height > OUTPUT_HEIGHT + 1e-6
            ):
                errors.append(f"DroneVehicle annotation {annotation_id} exceeds image bounds")
                continue
            class_counts[DRONEVEHICLE_TARGET_CATEGORIES[int(category_id)]] += 1
        return validation_result(
            dataset="dronevehicle",
            split=self.split,
            errors=errors,
            warnings=warnings,
            stats={
                "images": len(document.images),
                "instances": len(document.annotations),
                "classes": dict(sorted(class_counts.items())),
                "missing_images": missing,
                "modality": "RGB",
                "geometry": (
                    "source polygon or bndbox envelope to clipped AABB after 100-pixel crop"
                ),
            },
        )


def prepare_dronevehicle_rgb_val(
    archive_path: Path,
    output_root: Path,
    *,
    report_output: Path | None = None,
    expected_archive_sha256: str = DRONEVEHICLE_VAL_SHA256,
    expected_archive_bytes: int = DRONEVEHICLE_VAL_BYTES,
    expected_images: int = DRONEVEHICLE_VAL_IMAGES,
    source_metadata: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate and convert the official RGB validation split without metric access."""
    if not archive_path.is_file():
        raise DataError(f"DroneVehicle validation archive does not exist: {archive_path}")
    if archive_path.stat().st_size != expected_archive_bytes:
        raise DataError("DroneVehicle validation archive byte size differs from the source lock")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != expected_archive_sha256.casefold():
        raise DataError("DroneVehicle validation archive SHA256 differs from the source lock")
    metadata_hashes = _validate_source_metadata(source_metadata)
    if output_root.exists():
        return _reuse_prepared_dronevehicle(
            output_root,
            archive_sha256=archive_sha256,
            metadata_hashes=metadata_hashes,
            report_output=report_output,
        )

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataError(f"cannot open DroneVehicle validation ZIP: {exc}") from exc
    with archive:
        infos = archive.infolist()
        if any(_unsafe_member(info.filename) for info in infos):
            raise DataError("DroneVehicle ZIP contains an unsafe member path")
        rgb_images = _indexed_members(infos, "val/valimg", ".jpg")
        rgb_labels = _indexed_members(infos, "val/vallabel", ".xml")
        ir_images = _indexed_members(infos, "val/valimgr", ".jpg")
        ir_labels = _indexed_members(infos, "val/vallabelr", ".xml")
        expected_ids = {f"{index:05d}" for index in range(1, expected_images + 1)}
        for label, members in (
            ("RGB images", rgb_images),
            ("RGB labels", rgb_labels),
            ("IR images", ir_images),
            ("IR labels", ir_labels),
        ):
            if set(members) != expected_ids:
                raise DataError(f"DroneVehicle {label} membership differs from the source lock")

        temporary = output_root.parent / f".{output_root.name}.preparing-{uuid.uuid4().hex}"
        if temporary.exists():
            raise DataError(f"refusing to reuse DroneVehicle temporary directory: {temporary}")
        try:
            images_dir = temporary / "images"
            annotations_dir = temporary / "annotations"
            images_dir.mkdir(parents=True)
            annotations_dir.mkdir(parents=True)
            coco_images: list[dict[str, Any]] = []
            coco_annotations: list[dict[str, Any]] = []
            image_manifest: list[dict[str, Any]] = []
            source_counts: Counter[str] = Counter()
            target_counts: Counter[str] = Counter()
            ignored_counts: Counter[str] = Counter()
            clipped_counts: Counter[str] = Counter()
            dropped_outside_counts: Counter[str] = Counter()
            dropped_unresolvable_counts: Counter[str] = Counter()
            source_geometry_counts: Counter[str] = Counter()
            source_out_of_bounds_counts: Counter[str] = Counter()
            xml_depth_counts: Counter[int] = Counter()
            decoded_mode_counts: Counter[str] = Counter()
            xml_depth_mismatch_count = 0
            border_means: list[float] = []
            border_white_fractions: list[float] = []
            annotation_id = 1
            for image_id, source_id in enumerate(sorted(expected_ids), start=1):
                xml_member = rgb_labels[source_id]
                xml_root = _parse_xml(archive.read(xml_member), member=xml_member)
                declared_width, declared_height, declared_depth = _declared_image(
                    xml_root, member=xml_member
                )
                if (declared_width, declared_height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise DataError(
                        "DroneVehicle RGB XML size changed for "
                        f"{source_id}: {(declared_width, declared_height)}"
                    )
                xml_depth_counts[declared_depth] += 1
                image_member = rgb_images[source_id]
                try:
                    with Image.open(io.BytesIO(archive.read(image_member))) as source_image:
                        source_image.load()
                        decoded_mode_counts[source_image.mode] += 1
                        if source_image.mode != "RGB" or source_image.getbands() != (
                            "R",
                            "G",
                            "B",
                        ):
                            raise DataError(
                                "DroneVehicle valimg member is not a decoded RGB image: "
                                f"{image_member} mode={source_image.mode!r}"
                            )
                        if declared_depth != len(source_image.getbands()):
                            xml_depth_mismatch_count += 1
                        rgb = source_image.convert("RGB")
                except DataError:
                    raise
                except (OSError, ValueError) as exc:
                    raise DataError(
                        f"cannot decode DroneVehicle RGB image {image_member}: {exc}"
                    ) from exc
                if rgb.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise DataError(f"DroneVehicle RGB image size changed: {image_member}")
                border_mean, border_white_fraction = _white_border_stats(rgb)
                if border_mean < 254.5 or border_white_fraction < 0.995:
                    raise DataError(
                        f"DroneVehicle official white border check failed for {image_member}: "
                        f"mean={border_mean:.6f}, fraction={border_white_fraction:.6f}"
                    )
                border_means.append(border_mean)
                border_white_fractions.append(border_white_fraction)
                destination = images_dir / f"{source_id}.png"
                rgb.crop((BORDER, BORDER, SOURCE_WIDTH - BORDER, SOURCE_HEIGHT - BORDER)).save(
                    destination, format="PNG", compress_level=3
                )
                if image_size(destination) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
                    raise DataError(f"DroneVehicle cropped image has the wrong size: {destination}")
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": destination.name,
                        "width": OUTPUT_WIDTH,
                        "height": OUTPUT_HEIGHT,
                        "source_id": source_id,
                        "source_image_member": image_member,
                        "source_label_member": xml_member,
                        "modality": "RGB",
                        "crop_border": BORDER,
                    }
                )
                image_manifest.append(
                    {
                        "image_id": image_id,
                        "source_id": source_id,
                        "source_member": image_member,
                        "source_crc32": f"{archive.getinfo(image_member).CRC:08x}",
                        "output": destination.name,
                        "output_sha256": sha256_file(destination),
                        "border_mean": border_mean,
                        "border_white_fraction": border_white_fraction,
                    }
                )
                for object_element in xml_root.findall("object"):
                    source_name = _required_text(object_element, "name", member=xml_member)
                    if source_name not in DRONEVEHICLE_SOURCE_LABELS:
                        raise DataError(
                            f"unknown DroneVehicle source label {source_name!r} in {xml_member}"
                        )
                    source_counts[source_name] += 1
                    polygon, source_geometry = _object_geometry(object_element, member=xml_member)
                    source_geometry_counts[source_geometry] += 1
                    if polygon is None:
                        dropped_unresolvable_counts[source_name] += 1
                        continue
                    if _source_geometry_out_of_bounds(polygon):
                        source_out_of_bounds_counts[source_name] += 1
                    if source_name not in DRONEVEHICLE_SOURCE_TO_TARGET:
                        ignored_counts[source_name] += 1
                        continue
                    bbox, clipped = _cropped_envelope(polygon)
                    if bbox is None:
                        dropped_outside_counts[source_name] += 1
                        continue
                    if clipped:
                        clipped_counts[source_name] += 1
                    x, y, width, height = bbox
                    category_id = DRONEVEHICLE_SOURCE_TO_TARGET[source_name]
                    target_counts[source_name] += 1
                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x, y, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                            "source_category_name": source_name,
                            "source_polygon": [list(point) for point in polygon],
                            "source_geometry": source_geometry,
                            "geometry": ("source_geometry_envelope_then_crop_translate_and_clip"),
                        }
                    )
                    annotation_id += 1

            if set(source_counts) != set(DRONEVEHICLE_SOURCE_LABELS):
                raise DataError(f"DroneVehicle source label set changed: {sorted(source_counts)}")
            annotation_output = annotations_dir / "dronevehicle_rgb_val.coco.json"
            atomic_write_json(
                annotation_output,
                {
                    "info": {
                        "description": "DroneVehicle RGB validation exact three-class conversion",
                        "modality": "RGB only",
                        "crop": "remove official 100-pixel border on every side",
                        "geometry": (
                            "source polygon or bndbox axis-aligned envelope, translated and clipped"
                        ),
                        "mapping": (
                            "exact car/truck/bus only; van and the source freight-car spelling "
                            "variants are ignored"
                        ),
                    },
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": [
                        {"id": category_id, "name": name}
                        for category_id, name in DRONEVEHICLE_TARGET_CATEGORIES.items()
                    ],
                },
            )
            image_manifest_path = temporary / "image_manifest.json"
            atomic_write_json(
                image_manifest_path,
                {
                    "schema_version": 1,
                    "images": image_manifest,
                    "source_modality": "RGB",
                    "modality_basis": (
                        "official UA-CMDet validation img_prefix=val/valimg plus decoded JPEG "
                        "RGB bands; XML depth is audited but not trusted for modality"
                    ),
                    "source_size": [SOURCE_WIDTH, SOURCE_HEIGHT],
                    "output_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
                    "crop_border": BORDER,
                },
            )
            core: dict[str, Any] = {
                "schema_version": 1,
                "status": "PASS",
                "dataset": "dronevehicle",
                "split": "external_rgb_val",
                "source": {
                    "official_repository": "https://github.com/VisDrone/DroneVehicle",
                    "official_commit": DRONEVEHICLE_OFFICIAL_COMMIT,
                    "mirror_repository": "https://huggingface.co/datasets/McCheng/DroneVehicle",
                    "mirror_commit": DRONEVEHICLE_HF_COMMIT,
                    "archive_bytes": expected_archive_bytes,
                    "archive_sha256": archive_sha256,
                    "metadata_sha256": metadata_hashes,
                    "license_record": (
                        "official repository publishes download links and requests citation but "
                        "declares no machine-readable license; local research use only, "
                        "no redistribution"
                    ),
                },
                "archive": {
                    "entries": len(infos),
                    "unsafe_members": 0,
                    "rgb_images": len(rgb_images),
                    "rgb_labels": len(rgb_labels),
                    "ir_images_ignored": len(ir_images),
                    "ir_labels_ignored": len(ir_labels),
                },
                "conversion": {
                    "images": len(coco_images),
                    "annotations": len(coco_annotations),
                    "source_class_counts": dict(sorted(source_counts.items())),
                    "target_class_counts": dict(sorted(target_counts.items())),
                    "ignored_class_counts": dict(sorted(ignored_counts.items())),
                    "source_geometry_counts": dict(sorted(source_geometry_counts.items())),
                    "source_geometry_out_of_bounds_counts": dict(
                        sorted(source_out_of_bounds_counts.items())
                    ),
                    "dropped_unresolvable_source_geometry": dict(
                        sorted(dropped_unresolvable_counts.items())
                    ),
                    "clipped_mapped_boxes": dict(sorted(clipped_counts.items())),
                    "dropped_fully_outside_mapped_boxes": dict(
                        sorted(dropped_outside_counts.items())
                    ),
                    "xml_declared_depth_counts": {
                        str(key): value for key, value in sorted(xml_depth_counts.items())
                    },
                    "decoded_image_mode_counts": dict(sorted(decoded_mode_counts.items())),
                    "xml_depth_mismatch_with_decoded_rgb_images": xml_depth_mismatch_count,
                    "modality_basis": (
                        "official UA-CMDet validation img_prefix=val/valimg and decoded RGB "
                        "JPEG content; erroneous XML depth metadata retained as an audit field"
                    ),
                    "minimum_border_mean": min(border_means),
                    "minimum_border_white_fraction": min(border_white_fractions),
                    "annotation_sha256": sha256_file(annotation_output),
                    "image_manifest_sha256": sha256_file(image_manifest_path),
                },
                "formal_metrics_calculated_or_viewed": False,
                "training_tuning_or_calibration": False,
                "source_archive_modified": False,
                "reused_existing": False,
            }
            core["evidence_sha256"] = stable_hash(core, length=64)
            atomic_write_json(temporary / "conversion_manifest.json", core)
            output_root.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(output_root)
        except BaseException:
            if temporary.is_dir():
                shutil.rmtree(temporary)
            raise

    report = _prepared_report(output_root, core, reused=False)
    if report_output is not None:
        atomic_write_json(report_output, report)
    return report


def _validate_source_metadata(source_metadata: Mapping[str, Path] | None) -> dict[str, str]:
    if source_metadata is None:
        return {}
    hashes: dict[str, str] = {}
    for label, path in source_metadata.items():
        if not path.is_file():
            raise DataError(f"DroneVehicle source metadata is missing: {path}")
        hashes[label] = sha256_file(path)
    readme = source_metadata.get("official_readme")
    if readme is not None:
        text = readme.read_text(encoding="utf-8")
        required = ("56,878", "width of 100 pixels", "840 x 712", "Validation", "cite")
        if any(value not in text for value in required):
            raise DataError("DroneVehicle official README no longer states the frozen data rules")
    repo = source_metadata.get("official_repo")
    if repo is not None:
        value = json.loads(repo.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("full_name") != "VisDrone/DroneVehicle":
            raise DataError("DroneVehicle official repository metadata is invalid")
        if value.get("license") is not None:
            raise DataError("DroneVehicle license state changed; review before continuing")
    commit = source_metadata.get("official_commit")
    if commit is not None:
        value = json.loads(commit.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("sha") != DRONEVEHICLE_OFFICIAL_COMMIT:
            raise DataError("DroneVehicle official repository commit changed")
    uacmdet_config = source_metadata.get("official_uacmdet_config")
    if uacmdet_config is not None:
        text = uacmdet_config.read_text(encoding="utf-8")
        uacmdet_required = (
            "img_prefix=data_root + 'DroneVehicle/val/valimg'",
            "img_scale=(640, 512)",
        )
        if any(value not in text for value in uacmdet_required):
            raise DataError(
                "DroneVehicle UA-CMDet config no longer confirms the frozen RGB validation path"
            )
    return dict(sorted(hashes.items()))


def _reuse_prepared_dronevehicle(
    output_root: Path,
    *,
    archive_sha256: str,
    metadata_hashes: Mapping[str, str],
    report_output: Path | None,
) -> dict[str, Any]:
    manifest_path = output_root / "conversion_manifest.json"
    annotation_path = output_root / "annotations" / "dronevehicle_rgb_val.coco.json"
    image_manifest_path = output_root / "image_manifest.json"
    for path in (manifest_path, annotation_path, image_manifest_path):
        if not path.is_file():
            raise DataError(f"existing DroneVehicle conversion is incomplete: {path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise DataError("existing DroneVehicle conversion manifest is invalid")
    source = value.get("source")
    conversion = value.get("conversion")
    if (
        not isinstance(source, dict)
        or source.get("archive_sha256") != archive_sha256
        or source.get("metadata_sha256") != dict(metadata_hashes)
        or not isinstance(conversion, dict)
        or conversion.get("annotation_sha256") != sha256_file(annotation_path)
        or conversion.get("image_manifest_sha256") != sha256_file(image_manifest_path)
    ):
        raise DataError("existing DroneVehicle conversion hashes do not verify")
    report = _prepared_report(output_root, value, reused=True)
    if report_output is not None:
        atomic_write_json(report_output, report)
    return report


def _prepared_report(output_root: Path, core: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        **core,
        "reused_existing": reused,
        "output_root": str(output_root.resolve()),
        "annotation": str(
            (output_root / "annotations" / "dronevehicle_rgb_val.coco.json").resolve()
        ),
        "image_manifest": str((output_root / "image_manifest.json").resolve()),
        "conversion_manifest": str((output_root / "conversion_manifest.json").resolve()),
        "conversion_manifest_sha256": sha256_file(output_root / "conversion_manifest.json"),
    }


def _unsafe_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts or bool(path.parts and ":" in path.parts[0])


def _indexed_members(
    infos: Sequence[zipfile.ZipInfo], directory: str, suffix: str
) -> dict[str, str]:
    prefix = f"{directory.rstrip('/')}/"
    output: dict[str, str] = {}
    for info in infos:
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        path = PurePosixPath(info.filename)
        if path.parent.as_posix() != directory or path.suffix.casefold() != suffix:
            continue
        if path.stem in output:
            raise DataError(f"duplicate DroneVehicle archive member ID: {path.stem}")
        output[path.stem] = info.filename
    return output


def _parse_xml(payload: bytes, *, member: str) -> ElementTree.Element:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise DataError(f"cannot parse DroneVehicle XML {member}: {exc}") from exc
    if root.tag != "annotation":
        raise DataError(f"unexpected DroneVehicle XML root in {member}: {root.tag}")
    return root


def _declared_image(root: ElementTree.Element, *, member: str) -> tuple[int, int, int]:
    size = root.find("size")
    if size is None:
        raise DataError(f"DroneVehicle XML has no size element: {member}")
    try:
        return tuple(
            int(_required_text(size, key, member=member)) for key in ("width", "height", "depth")
        )  # type: ignore[return-value]
    except ValueError as exc:
        raise DataError(f"DroneVehicle XML has a non-integer size: {member}") from exc


def _required_text(element: ElementTree.Element, key: str, *, member: str) -> str:
    child = element.find(key)
    if child is None or child.text is None or not child.text.strip():
        raise DataError(f"DroneVehicle XML {member} lacks {key}")
    return child.text.strip()


def _polygon(
    object_element: ElementTree.Element, *, member: str
) -> tuple[tuple[float, float], ...]:
    polygon = object_element.find("polygon")
    if polygon is None:
        raise DataError(f"DroneVehicle object has no polygon: {member}")
    points: list[tuple[float, float]] = []
    try:
        for index in range(1, 5):
            x = float(_required_text(polygon, f"x{index}", member=member))
            y = float(_required_text(polygon, f"y{index}", member=member))
            if not math.isfinite(x) or not math.isfinite(y):
                raise DataError(f"DroneVehicle polygon contains a nonfinite point: {member}")
            points.append((x, y))
    except ValueError as exc:
        raise DataError(f"DroneVehicle polygon contains a nonnumeric point: {member}") from exc
    return tuple(points)


def _object_geometry(
    object_element: ElementTree.Element, *, member: str
) -> tuple[tuple[tuple[float, float], ...] | None, str]:
    polygon = object_element.find("polygon")
    bndbox = object_element.find("bndbox")
    point = object_element.find("point")
    present = sum(value is not None for value in (polygon, bndbox, point))
    if present != 1:
        raise DataError(f"DroneVehicle object has ambiguous or missing geometry: {member}")
    if polygon is not None:
        return _polygon(object_element, member=member), "polygon"
    if bndbox is not None:
        try:
            xmin = float(_required_text(bndbox, "xmin", member=member))
            ymin = float(_required_text(bndbox, "ymin", member=member))
            xmax = float(_required_text(bndbox, "xmax", member=member))
            ymax = float(_required_text(bndbox, "ymax", member=member))
        except ValueError as exc:
            raise DataError(f"DroneVehicle bndbox contains a nonnumeric point: {member}") from exc
        if (
            not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax))
            or xmax <= xmin
            or ymax <= ymin
        ):
            raise DataError(f"DroneVehicle bndbox is invalid: {member}")
        return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), "bndbox"
    assert point is not None
    try:
        x = float(_required_text(point, "x", member=member))
        y = float(_required_text(point, "y", member=member))
    except ValueError as exc:
        raise DataError(f"DroneVehicle point contains a nonnumeric coordinate: {member}") from exc
    if (
        not math.isfinite(x)
        or not math.isfinite(y)
        or not 0 <= x <= SOURCE_WIDTH
        or not 0 <= y <= SOURCE_HEIGHT
    ):
        raise DataError(f"DroneVehicle point is outside source bounds: {member}")
    return None, "point_unresolvable"


def _source_geometry_out_of_bounds(polygon: Sequence[tuple[float, float]]) -> bool:
    return any(x < 0 or x > SOURCE_WIDTH or y < 0 or y > SOURCE_HEIGHT for x, y in polygon)


def _cropped_envelope(
    polygon: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float, float, float] | None, bool]:
    raw_x1 = min(point[0] for point in polygon) - BORDER
    raw_y1 = min(point[1] for point in polygon) - BORDER
    raw_x2 = max(point[0] for point in polygon) - BORDER
    raw_y2 = max(point[1] for point in polygon) - BORDER
    x1 = min(max(raw_x1, 0.0), float(OUTPUT_WIDTH))
    y1 = min(max(raw_y1, 0.0), float(OUTPUT_HEIGHT))
    x2 = min(max(raw_x2, 0.0), float(OUTPUT_WIDTH))
    y2 = min(max(raw_y2, 0.0), float(OUTPUT_HEIGHT))
    if x2 <= x1 or y2 <= y1:
        return None, True
    clipped = (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2)
    return (x1, y1, x2 - x1, y2 - y1), clipped


def _white_border_stats(image: Image.Image) -> tuple[float, float]:
    array = np.asarray(image, dtype=np.uint8)
    mask = np.ones((SOURCE_HEIGHT, SOURCE_WIDTH), dtype=bool)
    mask[BORDER : SOURCE_HEIGHT - BORDER, BORDER : SOURCE_WIDTH - BORDER] = False
    border = array[mask]
    return float(border.mean()), float(np.mean(np.all(border >= 250, axis=1)))
