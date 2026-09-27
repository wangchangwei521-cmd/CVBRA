from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import stat
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json

DATASET_ROOT = "UAV-benchmark-M"
TOOLKIT_ROOT = "UAV-benchmark-MOTD_v1.0"
ATTRIBUTES_ROOT = "M_attr"
TEST_SPLIT_SCRIPT = f"{TOOLKIT_ROOT}/utils/CalculateDetectionPR_seq.m"
TOOLKIT_README = f"{TOOLKIT_ROOT}/README.md"

DATASET_ARCHIVE_EXPECTATION = (
    6809272793,
    "3494eba22e6ed3ca319fe93eba9f3f4682a2b0ecd9fc1bd6f915b37b3503b421",
)
TOOLKIT_ARCHIVE_EXPECTATION = (
    245719325,
    "1e565da8c2a035bf1e56b1322de76c74b73702ea3b0f918f45debf69c2b26799",
)
ATTRIBUTES_ARCHIVE_EXPECTATION = (
    9503,
    "ac6f85e355db3f4808cd64148b1f3c33933906ebd47c63939bb7eba6974824d1",
)
OFFICIAL_README_EXPECTATION = (
    2290,
    "72d63451b602af6d6dd1c938e29fe94d8ebc3809facc8dd4d9e5f24e44754b5b",
)

EXPECTED_CATEGORY_MAPPING = {1: "car", 2: "truck", 3: "bus"}
EXPECTED_TRAIN_SEQUENCES = 30
EXPECTED_TEST_SEQUENCES = 20
EXPECTED_DATASET_MEMBERS = 41106
EXPECTED_DATASET_IMAGES = 40735
EXPECTED_TOOLKIT_MEMBERS = 1094

_SEQUENCE_RE = re.compile(r"M\d{4}\Z")
_IMAGE_RE = re.compile(
    rf"{re.escape(DATASET_ROOT)}/(?P<sequence>M\d{{4}})/img1/img(?P<frame>\d{{6}})\.jpg\Z"
)
_DATASET_GT_RE = re.compile(
    rf"{re.escape(DATASET_ROOT)}/(?P<sequence>M\d{{4}})/gt/"
    r"(?P<kind>gt|gt_ignore|gt_whole)\.txt\Z"
)
_DATASET_DET_RE = re.compile(rf"{re.escape(DATASET_ROOT)}/(?P<sequence>M\d{{4}})/det/det\.txt\Z")
_TOOLKIT_GT_RE = re.compile(
    rf"{re.escape(TOOLKIT_ROOT)}/GT/(?P<sequence>M\d{{4}})_"
    r"(?P<kind>gt|gt_ignore|gt_whole)\.txt\Z"
)
_ATTRIBUTE_RE = re.compile(
    rf"{re.escape(ATTRIBUTES_ROOT)}/(?P<split>train|test)/"
    r"(?P<sequence>M\d{4})\s*_attr\.txt\Z"
)


class UavdtSourceError(RuntimeError):
    """Raised when an ordered UAVDT source gate fails closed."""


@dataclass(frozen=True)
class ArchiveExpectation:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArchiveSummary:
    path: str
    bytes: int
    sha256: str
    members: int
    files: int
    directories: int
    encrypted_members: int
    symlink_members: int
    unsafe_members: int
    duplicate_members: int
    crc_status: str


@dataclass(frozen=True)
class DatasetInventory:
    sequences: tuple[str, ...]
    frame_counts: dict[str, int]
    image_count: int
    gt_paths: dict[tuple[str, str], str]
    detection_sequences: tuple[str, ...]


@dataclass(frozen=True)
class ToolkitInventory:
    sequences: tuple[str, ...]
    test_sequences: tuple[str, ...]
    test_frame_counts: dict[str, int]
    gt_paths: dict[tuple[str, str], str]
    category_mapping: dict[int, str]


@dataclass(frozen=True)
class AttributesInventory:
    train_sequences: tuple[str, ...]
    test_sequences: tuple[str, ...]
    normalized_filename_count: int
    values: dict[str, tuple[int, ...]]


def _expectation(value: tuple[int, str]) -> ArchiveExpectation:
    return ArchiveExpectation(bytes=value[0], sha256=value[1])


OFFICIAL_DATASET = _expectation(DATASET_ARCHIVE_EXPECTATION)
OFFICIAL_TOOLKIT = _expectation(TOOLKIT_ARCHIVE_EXPECTATION)
OFFICIAL_ATTRIBUTES = _expectation(ATTRIBUTES_ARCHIVE_EXPECTATION)
OFFICIAL_README = _expectation(OFFICIAL_README_EXPECTATION)


def _display_path(path: Path, project_root: Path | None) -> str:
    resolved = path.resolve()
    if project_root is not None:
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return str(resolved)


def _safe_member_name(name: str, *, expected_root: str) -> str:
    if not name or "\x00" in name or "\\" in name or "//" in name:
        raise UavdtSourceError(f"unsafe archive member path: {name!r}")
    normalized = name.rstrip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in normalized
        or not path.parts
        or path.parts[0] != expected_root
    ):
        raise UavdtSourceError(f"unsafe archive member path: {name!r}")
    windows_reserved = {"aux", "con", "nul", "prn"}
    windows_reserved.update(f"com{index}" for index in range(1, 10))
    windows_reserved.update(f"lpt{index}" for index in range(1, 10))
    if any(
        part.rstrip(". ").split(".", 1)[0].casefold() in windows_reserved for part in path.parts
    ):
        raise UavdtSourceError(f"unsafe Windows archive member path: {name!r}")
    return normalized


def _is_symlink(entry: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK


def _inspect_archive(
    path: Path,
    *,
    expected_root: str,
    expectation: ArchiveExpectation,
) -> ArchiveSummary:
    path = path.resolve()
    if not path.is_file():
        raise UavdtSourceError(f"required archive is unavailable: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expectation.bytes:
        raise UavdtSourceError(
            f"archive byte size mismatch for {path.name}: expected {expectation.bytes}, "
            f"observed {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expectation.sha256.lower():
        raise UavdtSourceError(
            f"archive SHA256 mismatch for {path.name}: expected {expectation.sha256}, "
            f"observed {actual_sha256}"
        )

    try:
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.infolist()
            normalized: list[str] = []
            encrypted = 0
            symlinks = 0
            for entry in entries:
                normalized.append(_safe_member_name(entry.filename, expected_root=expected_root))
                encrypted += int(bool(entry.flag_bits & 0x1))
                symlinks += int(_is_symlink(entry))
            duplicate_count = len(normalized) - len({name.casefold() for name in normalized})
            if duplicate_count:
                raise UavdtSourceError(
                    f"archive contains {duplicate_count} duplicate or case-colliding members: "
                    f"{path.name}"
                )
            if encrypted:
                raise UavdtSourceError(
                    f"archive contains {encrypted} encrypted members: {path.name}"
                )
            if symlinks:
                raise UavdtSourceError(
                    f"archive contains {symlinks} symbolic-link members: {path.name}"
                )
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise UavdtSourceError(
                    f"archive CRC validation failed for {path.name}: {bad_member}"
                )
            files = sum(not entry.is_dir() for entry in entries)
    except UavdtSourceError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise UavdtSourceError(f"cannot validate archive {path.name}: {exc}") from exc

    return ArchiveSummary(
        path=str(path),
        bytes=actual_bytes,
        sha256=actual_sha256,
        members=len(entries),
        files=files,
        directories=len(entries) - files,
        encrypted_members=0,
        symlink_members=0,
        unsafe_members=0,
        duplicate_members=0,
        crc_status="PASS",
    )


def _archive_files(bundle: zipfile.ZipFile, *, expected_root: str) -> set[str]:
    return {
        _safe_member_name(entry.filename, expected_root=expected_root)
        for entry in bundle.infolist()
        if not entry.is_dir()
    }


def _inventory_dataset(bundle: zipfile.ZipFile) -> DatasetInventory:
    files = _archive_files(bundle, expected_root=DATASET_ROOT)
    sequence_images: dict[str, list[int]] = {}
    gt_paths: dict[tuple[str, str], str] = {}
    detection_sequences: set[str] = set()
    unexpected: list[str] = []

    for name in sorted(files):
        image_match = _IMAGE_RE.fullmatch(name)
        if image_match is not None:
            sequence = image_match.group("sequence")
            sequence_images.setdefault(sequence, []).append(int(image_match.group("frame")))
            continue
        gt_match = _DATASET_GT_RE.fullmatch(name)
        if gt_match is not None:
            key = (gt_match.group("sequence"), gt_match.group("kind"))
            if key in gt_paths:
                raise UavdtSourceError(f"duplicate Dataset GT identity: {key}")
            gt_paths[key] = name
            continue
        det_match = _DATASET_DET_RE.fullmatch(name)
        if det_match is not None:
            detection_sequences.add(det_match.group("sequence"))
            continue
        unexpected.append(name)

    if unexpected:
        raise UavdtSourceError(f"Dataset archive contains unexpected files: {unexpected[:5]}")
    sequences = tuple(sorted(sequence_images))
    if any(_SEQUENCE_RE.fullmatch(sequence) is None for sequence in sequences):
        raise UavdtSourceError("Dataset archive contains an invalid sequence identifier")

    frame_counts: dict[str, int] = {}
    required_kinds = {"gt", "gt_ignore", "gt_whole"}
    for sequence in sequences:
        frames = sorted(sequence_images[sequence])
        expected_frames = list(range(1, len(frames) + 1))
        if frames != expected_frames:
            raise UavdtSourceError(f"Dataset frames are not contiguous from one for {sequence}")
        observed_kinds = {kind for seq, kind in gt_paths if seq == sequence}
        if observed_kinds != required_kinds:
            raise UavdtSourceError(
                f"Dataset GT files are incomplete for {sequence}: {sorted(observed_kinds)}"
            )
        frame_counts[sequence] = len(frames)

    gt_sequences = {sequence for sequence, _ in gt_paths}
    if gt_sequences != set(sequences):
        raise UavdtSourceError("Dataset GT sequence set differs from image sequence set")
    if not detection_sequences.issubset(set(sequences)):
        raise UavdtSourceError("Dataset detection files reference unknown sequences")

    return DatasetInventory(
        sequences=sequences,
        frame_counts=frame_counts,
        image_count=sum(frame_counts.values()),
        gt_paths=gt_paths,
        detection_sequences=tuple(sorted(detection_sequences)),
    )


def _parse_matlab_test_split(text: str) -> tuple[tuple[str, ...], dict[str, int]]:
    sequence_match = re.search(r"seqDirs\s*=\s*\{(?P<body>.*?)\}\s*;", text, re.DOTALL)
    lengths_match = re.search(r"seqLens\s*=\s*\[(?P<body>.*?)\]\s*;", text, re.DOTALL)
    if sequence_match is None or lengths_match is None:
        raise UavdtSourceError("Toolkit split script lacks seqDirs or seqLens")

    sequence_body = sequence_match.group("body")
    sequences = tuple(re.findall(r"'(M\d{4})'", sequence_body))
    residue = re.sub(r"'M\d{4}'|,|\s", "", sequence_body)
    if residue or not sequences or len(sequences) != len(set(sequences)):
        raise UavdtSourceError("Toolkit seqDirs definition is malformed or duplicated")

    lengths_body = lengths_match.group("body")
    length_tokens = re.findall(r"\d+", lengths_body)
    residue = re.sub(r"\d+|,|\s", "", lengths_body)
    if residue or not length_tokens:
        raise UavdtSourceError("Toolkit seqLens definition is malformed")
    lengths = tuple(int(token) for token in length_tokens)
    if len(sequences) != len(lengths) or any(length <= 0 for length in lengths):
        raise UavdtSourceError("Toolkit test sequence names and lengths are inconsistent")
    return sequences, dict(zip(sequences, lengths, strict=True))


def _parse_category_mapping(readme: str) -> dict[int, str]:
    normalized = " ".join(readme.lower().split())
    anchors = {
        1: re.search(r"car\s*\(\s*1\s*\)", normalized),
        2: re.search(r"truck\s*\(\s*2\s*\)", normalized),
        3: re.search(r"bus\s*\(\s*3\s*\)", normalized),
    }
    if any(match is None for match in anchors.values()):
        raise UavdtSourceError("Toolkit README lacks the registered car/truck/bus ID mapping")
    return EXPECTED_CATEGORY_MAPPING.copy()


def _inventory_toolkit(bundle: zipfile.ZipFile) -> ToolkitInventory:
    files = _archive_files(bundle, expected_root=TOOLKIT_ROOT)
    required = {TOOLKIT_README, TEST_SPLIT_SCRIPT}
    missing = sorted(required - files)
    if missing:
        raise UavdtSourceError(f"Toolkit archive lacks required files: {missing}")

    gt_paths: dict[tuple[str, str], str] = {}
    for name in sorted(files):
        match = _TOOLKIT_GT_RE.fullmatch(name)
        if match is None:
            continue
        key = (match.group("sequence"), match.group("kind"))
        if key in gt_paths:
            raise UavdtSourceError(f"duplicate Toolkit GT identity: {key}")
        gt_paths[key] = name

    sequences = tuple(sorted({sequence for sequence, _ in gt_paths}))
    required_kinds = {"gt", "gt_ignore", "gt_whole"}
    for sequence in sequences:
        observed_kinds = {kind for seq, kind in gt_paths if seq == sequence}
        if observed_kinds != required_kinds:
            raise UavdtSourceError(
                f"Toolkit GT files are incomplete for {sequence}: {sorted(observed_kinds)}"
            )

    try:
        split_text = bundle.read(TEST_SPLIT_SCRIPT).decode("ascii")
        readme = bundle.read(TOOLKIT_README).decode("utf-8", errors="replace")
    except (KeyError, UnicodeError, OSError) as exc:
        raise UavdtSourceError(f"cannot read Toolkit split or category evidence: {exc}") from exc
    test_sequences, test_frame_counts = _parse_matlab_test_split(split_text)
    category_mapping = _parse_category_mapping(readme)
    return ToolkitInventory(
        sequences=sequences,
        test_sequences=test_sequences,
        test_frame_counts=test_frame_counts,
        gt_paths=gt_paths,
        category_mapping=category_mapping,
    )


def _parse_attribute_values(data: bytes, *, name: str) -> tuple[int, ...]:
    try:
        rows = list(csv.reader(io.StringIO(data.decode("ascii")), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise UavdtSourceError(f"cannot parse official attribute file {name}: {exc}") from exc
    if len(rows) != 1 or len(rows[0]) != 10:
        raise UavdtSourceError(f"official attribute file must contain ten fields: {name}")
    try:
        values = tuple(int(value.strip()) for value in rows[0])
    except ValueError as exc:
        raise UavdtSourceError(f"official attribute file is not integral: {name}") from exc
    if any(value not in {0, 1} for value in values):
        raise UavdtSourceError(f"official attribute file is not binary: {name}")
    if sum(values[0:3]) != 1 or sum(values[3:6]) != 1 or sum(values[6:9]) < 1:
        raise UavdtSourceError(f"official attribute groups are inconsistent: {name}")
    return values


def _inventory_attributes(bundle: zipfile.ZipFile) -> AttributesInventory:
    files = _archive_files(bundle, expected_root=ATTRIBUTES_ROOT)
    readme = f"{ATTRIBUTES_ROOT}/readme.txt"
    if readme not in files:
        raise UavdtSourceError("Attributes archive lacks readme.txt")
    unexpected: list[str] = []
    splits: dict[str, set[str]] = {"train": set(), "test": set()}
    values: dict[str, tuple[int, ...]] = {}
    normalized_filename_count = 0

    for name in sorted(files - {readme}):
        match = _ATTRIBUTE_RE.fullmatch(name)
        if match is None:
            unexpected.append(name)
            continue
        split = match.group("split")
        sequence = match.group("sequence")
        if sequence in values:
            raise UavdtSourceError(f"duplicate Attributes sequence identity: {sequence}")
        splits[split].add(sequence)
        values[sequence] = _parse_attribute_values(bundle.read(name), name=name)
        canonical = f"{ATTRIBUTES_ROOT}/{split}/{sequence}_attr.txt"
        normalized_filename_count += int(name != canonical)
    if unexpected:
        raise UavdtSourceError(f"Attributes archive contains unexpected files: {unexpected[:5]}")
    if splits["train"] & splits["test"]:
        raise UavdtSourceError("Attributes train and test sequence sets overlap")
    return AttributesInventory(
        train_sequences=tuple(sorted(splits["train"])),
        test_sequences=tuple(sorted(splits["test"])),
        normalized_filename_count=normalized_filename_count,
        values=values,
    )


def _validate_gt_file(
    data: bytes,
    *,
    name: str,
    kind: str,
    frame_count: int,
) -> tuple[int, set[int]]:
    try:
        text = data.decode("ascii")
        rows = csv.reader(text.splitlines(), strict=True)
        row_count = 0
        categories: set[int] = set()
        for line_number, row in enumerate(rows, start=1):
            if len(row) != 9:
                raise UavdtSourceError(f"GT row must contain nine columns: {name}:{line_number}")
            try:
                values = tuple(int(value.strip()) for value in row)
            except ValueError as exc:
                raise UavdtSourceError(f"GT row must be integral: {name}:{line_number}") from exc
            if values[0] < 1 or values[0] > frame_count:
                raise UavdtSourceError(
                    f"GT frame index is outside the image sequence: {name}:{line_number}"
                )
            if kind == "gt_whole":
                categories.add(values[8])
            row_count += 1
    except UnicodeError as exc:
        raise UavdtSourceError(f"GT file is not ASCII: {name}") from exc
    except csv.Error as exc:
        raise UavdtSourceError(f"GT CSV parsing failed for {name}: {exc}") from exc
    return row_count, categories


def _compare_and_validate_ground_truth(
    dataset_bundle: zipfile.ZipFile,
    toolkit_bundle: zipfile.ZipFile,
    dataset: DatasetInventory,
    toolkit: ToolkitInventory,
) -> dict[str, Any]:
    dataset_keys = set(dataset.gt_paths)
    toolkit_keys = set(toolkit.gt_paths)
    if dataset_keys != toolkit_keys:
        missing_dataset = sorted(toolkit_keys - dataset_keys)
        missing_toolkit = sorted(dataset_keys - toolkit_keys)
        raise UavdtSourceError(
            "Dataset and Toolkit GT identities differ: "
            f"missing_dataset={missing_dataset[:5]}, missing_toolkit={missing_toolkit[:5]}"
        )

    digest = hashlib.sha256()
    row_counts = {"gt": 0, "gt_ignore": 0, "gt_whole": 0}
    observed_categories: set[int] = set()
    compared_bytes = 0
    for sequence, kind in sorted(dataset_keys):
        dataset_name = dataset.gt_paths[(sequence, kind)]
        toolkit_name = toolkit.gt_paths[(sequence, kind)]
        dataset_bytes = dataset_bundle.read(dataset_name)
        toolkit_bytes = toolkit_bundle.read(toolkit_name)
        if dataset_bytes != toolkit_bytes:
            raise UavdtSourceError(
                f"Dataset GT bytes differ from the official Toolkit for {sequence}/{kind}"
            )
        rows, categories = _validate_gt_file(
            dataset_bytes,
            name=dataset_name,
            kind=kind,
            frame_count=dataset.frame_counts[sequence],
        )
        row_counts[kind] += rows
        observed_categories.update(categories)
        compared_bytes += len(dataset_bytes)
        digest.update(f"{sequence}/{kind}\0".encode("ascii"))
        digest.update(dataset_bytes)

    expected_categories = set(EXPECTED_CATEGORY_MAPPING)
    if observed_categories != expected_categories:
        raise UavdtSourceError(
            "GT category IDs differ from the registered mapping: "
            f"expected {sorted(expected_categories)}, observed {sorted(observed_categories)}"
        )
    return {
        "dataset_toolkit_byte_identical": True,
        "files_compared": len(dataset_keys),
        "bytes_compared": compared_bytes,
        "combined_sha256": digest.hexdigest(),
        "rows_by_kind": row_counts,
        "all_nonempty_rows_have_nine_columns": True,
        "observed_gt_whole_category_ids": sorted(observed_categories),
    }


def _check_plain_file(
    path: Path,
    expectation: ArchiveExpectation,
    *,
    project_root: Path | None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise UavdtSourceError(f"required official file is unavailable: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expectation.bytes or actual_sha256.lower() != expectation.sha256.lower():
        raise UavdtSourceError(
            f"official file hash evidence changed for {path.name}: "
            f"bytes={actual_bytes}, sha256={actual_sha256}"
        )
    return {
        "path": _display_path(path, project_root),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def _archive_report(summary: ArchiveSummary, project_root: Path | None) -> dict[str, Any]:
    result = asdict(summary)
    result["path"] = _display_path(Path(summary.path), project_root)
    return result


def uavdt_archive_evidence_sha256(report: dict[str, Any]) -> str:
    """Hash immutable source evidence while excluding timestamps and report locations."""

    required = (
        "gates",
        "archives",
        "dataset",
        "official_split",
        "attributes",
        "category_mapping",
        "ground_truth",
        "safety",
    )
    missing = [key for key in required if key not in report]
    if missing:
        raise UavdtSourceError(f"UAVDT archive evidence is incomplete: {missing}")
    archives = report["archives"]
    if not isinstance(archives, dict):
        raise UavdtSourceError("UAVDT archive evidence has malformed archive summaries")
    stable_archives: dict[str, Any] = {}
    for name, value in archives.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise UavdtSourceError("UAVDT archive evidence has malformed archive entries")
        stable_archives[name] = {key: item for key, item in value.items() if key != "path"}
    payload = {
        "schema_version": 1,
        "gates": report["gates"],
        "archives": stable_archives,
        "dataset": report["dataset"],
        "official_split": report["official_split"],
        "attributes": report["attributes"],
        "category_mapping": report["category_mapping"],
        "ground_truth": report["ground_truth"],
        "safety": report["safety"],
    }
    return stable_hash(payload, length=64)


def validate_uavdt_source(
    dataset_archive: Path,
    toolkit_archive: Path,
    attributes_archive: Path,
    *,
    dataset_expectation: ArchiveExpectation = OFFICIAL_DATASET,
    toolkit_expectation: ArchiveExpectation = OFFICIAL_TOOLKIT,
    attributes_expectation: ArchiveExpectation = OFFICIAL_ATTRIBUTES,
    official_readme: Path | None = None,
    official_readme_expectation: ArchiveExpectation = OFFICIAL_README,
    project_root: Path | None = None,
    protocol_path: Path | None = None,
    output: Path | None = None,
    download_manifest_output: Path | None = None,
    expected_train_sequences: int = EXPECTED_TRAIN_SEQUENCES,
    expected_test_sequences: int = EXPECTED_TEST_SEQUENCES,
    expected_dataset_members: int | None = EXPECTED_DATASET_MEMBERS,
    expected_dataset_images: int | None = EXPECTED_DATASET_IMAGES,
    expected_toolkit_members: int | None = EXPECTED_TOOLKIT_MEMBERS,
) -> dict[str, Any]:
    """Validate official UAVDT bytes without extraction, inference, or metric access."""

    dataset_archive = dataset_archive.resolve()
    toolkit_archive = toolkit_archive.resolve()
    attributes_archive = attributes_archive.resolve()
    dataset_summary = _inspect_archive(
        dataset_archive,
        expected_root=DATASET_ROOT,
        expectation=dataset_expectation,
    )
    toolkit_summary = _inspect_archive(
        toolkit_archive,
        expected_root=TOOLKIT_ROOT,
        expectation=toolkit_expectation,
    )
    attributes_summary = _inspect_archive(
        attributes_archive,
        expected_root=ATTRIBUTES_ROOT,
        expectation=attributes_expectation,
    )
    if expected_dataset_members is not None and dataset_summary.members != expected_dataset_members:
        raise UavdtSourceError(
            f"Dataset member count changed: expected {expected_dataset_members}, "
            f"observed {dataset_summary.members}"
        )
    if expected_toolkit_members is not None and toolkit_summary.members != expected_toolkit_members:
        raise UavdtSourceError(
            f"Toolkit member count changed: expected {expected_toolkit_members}, "
            f"observed {toolkit_summary.members}"
        )

    try:
        with (
            zipfile.ZipFile(dataset_archive) as dataset_bundle,
            zipfile.ZipFile(toolkit_archive) as toolkit_bundle,
            zipfile.ZipFile(attributes_archive) as attributes_bundle,
        ):
            dataset = _inventory_dataset(dataset_bundle)
            toolkit = _inventory_toolkit(toolkit_bundle)
            attributes = _inventory_attributes(attributes_bundle)

            if len(dataset.sequences) != expected_train_sequences + expected_test_sequences:
                raise UavdtSourceError(
                    f"Dataset must contain {expected_train_sequences + expected_test_sequences} "
                    f"sequences, observed {len(dataset.sequences)}"
                )
            if (
                expected_dataset_images is not None
                and dataset.image_count != expected_dataset_images
            ):
                raise UavdtSourceError(
                    f"Dataset image count changed: expected {expected_dataset_images}, "
                    f"observed {dataset.image_count}"
                )
            if set(dataset.sequences) != set(toolkit.sequences):
                raise UavdtSourceError("Dataset and Toolkit sequence sets differ")
            if len(attributes.train_sequences) != expected_train_sequences:
                raise UavdtSourceError(
                    f"official train split must contain {expected_train_sequences} sequences"
                )
            if len(attributes.test_sequences) != expected_test_sequences:
                raise UavdtSourceError(
                    f"official test split must contain {expected_test_sequences} sequences"
                )
            if set(attributes.test_sequences) != set(toolkit.test_sequences):
                raise UavdtSourceError("Toolkit and Attributes official test sets differ")
            expected_train = set(dataset.sequences) - set(toolkit.test_sequences)
            if set(attributes.train_sequences) != expected_train:
                raise UavdtSourceError("Attributes train set differs from the Dataset complement")
            if set(attributes.values) != set(dataset.sequences):
                raise UavdtSourceError("Attributes sequence union differs from the Dataset")
            if set(dataset.detection_sequences) != set(toolkit.test_sequences):
                raise UavdtSourceError("Dataset detection-file sequences differ from official test")
            for sequence, expected_frames in toolkit.test_frame_counts.items():
                if dataset.frame_counts.get(sequence) != expected_frames:
                    raise UavdtSourceError(
                        f"Toolkit frame count differs from Dataset for {sequence}: "
                        f"expected {expected_frames}, observed {dataset.frame_counts.get(sequence)}"
                    )
            if toolkit.category_mapping != EXPECTED_CATEGORY_MAPPING:
                raise UavdtSourceError("Toolkit category mapping differs from the preregistration")

            ground_truth = _compare_and_validate_ground_truth(
                dataset_bundle,
                toolkit_bundle,
                dataset,
                toolkit,
            )
    except UavdtSourceError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise UavdtSourceError(f"cannot inspect official UAVDT archive content: {exc}") from exc

    protocol: dict[str, Any] | None = None
    if protocol_path is not None:
        protocol_path = protocol_path.resolve()
        if not protocol_path.is_file():
            raise UavdtSourceError(f"protocol file is unavailable: {protocol_path}")
        protocol = {
            "path": _display_path(protocol_path, project_root),
            "sha256": sha256_file(protocol_path),
        }
    readme_report = None
    if official_readme is not None:
        readme_report = _check_plain_file(
            official_readme,
            official_readme_expectation,
            project_root=project_root,
        )

    observed_at = datetime.now(timezone.utc).isoformat()
    archive_reports = {
        "dataset": _archive_report(dataset_summary, project_root),
        "toolkit": _archive_report(toolkit_summary, project_root),
        "attributes": _archive_report(attributes_summary, project_root),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "protocol": "uavdt_zero_shot_external_source_validation",
        "validated_at_utc": observed_at,
        "protocol_lock": protocol,
        "gates": {
            "download_hashes": "PASS",
            "archive_structure": "PASS",
            "official_split": "PASS",
            "category_mapping": "PASS",
        },
        "archives": archive_reports,
        "dataset": {
            "root": DATASET_ROOT,
            "sequence_count": len(dataset.sequences),
            "image_count": dataset.image_count,
            "continuous_frame_numbering": True,
            "frame_counts": dataset.frame_counts,
            "detection_files_restricted_to_test_sequences": True,
        },
        "official_split": {
            "authority": TEST_SPLIT_SCRIPT,
            "train_count": len(attributes.train_sequences),
            "test_count": len(attributes.test_sequences),
            "train_sequences": list(attributes.train_sequences),
            "test_sequences": list(attributes.test_sequences),
            "disjoint": True,
            "union_equals_dataset": True,
            "toolkit_test_lengths_match_dataset": True,
        },
        "attributes": {
            "sequence_files": len(attributes.values),
            "binary_fields_per_sequence": 10,
            "normalized_source_filenames": attributes.normalized_filename_count,
            "source_files_modified": False,
        },
        "category_mapping": {
            "authority": TOOLKIT_README,
            "mapping": {str(key): value for key, value in toolkit.category_mapping.items()},
            "observed_ids": ground_truth["observed_gt_whole_category_ids"],
        },
        "ground_truth": ground_truth,
        "safety": {
            "archives_extracted": False,
            "formal_metrics_calculated_or_viewed": False,
            "training_tuning_or_inference_run": False,
            "published_detection_result_values_parsed": False,
            "fail_closed": True,
        },
    }
    result["evidence_sha256"] = uavdt_archive_evidence_sha256(result)

    download_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_COMPLETE_OFFICIAL_FILES_HASHED",
        "created_at_utc": observed_at,
        "protocol_lock": protocol,
        "files": {
            "dataset": {
                **archive_reports["dataset"],
                "transport": "official_baidu_share",
            },
            "toolkit": {
                **archive_reports["toolkit"],
                "transport": "official_google_drive",
            },
            "attributes": {
                **archive_reports["attributes"],
                "transport": "official_google_drive",
            },
            "readme": readme_report,
        },
        "excluded": {
            "google_drive_dataset_partial_combined": False,
            "google_drive_dataset_partial_admitted": False,
            "zenodo_fallback_used": False,
        },
        "safety": result["safety"],
    }
    if output is not None:
        atomic_write_json(output, result)
    if download_manifest_output is not None:
        atomic_write_json(download_manifest_output, download_manifest)
    return result


def _load_archive_gate(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UavdtSourceError(f"cannot read UAVDT archive-gate report: {exc}") from exc
    if not isinstance(value, dict):
        raise UavdtSourceError("UAVDT archive-gate report must be a mapping")
    gates = value.get("gates")
    if value.get("status") != "PASS" or not isinstance(gates, dict):
        raise UavdtSourceError("UAVDT archive-gate report is not PASS")
    required = {"download_hashes", "archive_structure", "official_split", "category_mapping"}
    if any(gates.get(name) != "PASS" for name in required):
        raise UavdtSourceError("UAVDT ordered archive gates are incomplete")
    safety = value.get("safety")
    if (
        not isinstance(safety, dict)
        or safety.get("formal_metrics_calculated_or_viewed") is not False
    ):
        raise UavdtSourceError("UAVDT archive-gate safety boundary is not clean")
    return value


def _existing_extraction(destination: Path, archive_sha256: str) -> dict[str, Any] | None:
    if not destination.exists():
        return None
    marker = destination / ".uavdt_extraction.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UavdtSourceError(
            f"extraction destination exists without a valid marker: {destination}: {exc}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "PASS"
        or value.get("archive_sha256") != archive_sha256
        or not (destination / DATASET_ROOT).is_dir()
    ):
        raise UavdtSourceError(
            f"extraction destination exists but does not match the admitted archive: {destination}"
        )
    return {**value, "reused_existing": True}


def extract_validated_uavdt_dataset(
    archive: Path,
    archive_gate_report: Path,
    destination: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically extract the already validated Dataset archive without overwriting data."""

    archive = archive.resolve()
    archive_gate_report = archive_gate_report.resolve()
    destination = destination.resolve()
    if project_root is not None:
        root = project_root.resolve()
        if destination == root or root not in destination.parents:
            raise UavdtSourceError(
                f"extraction destination is outside the project root: {destination}"
            )
    gate = _load_archive_gate(archive_gate_report)
    archives = gate.get("archives")
    if not isinstance(archives, dict) or not isinstance(archives.get("dataset"), dict):
        raise UavdtSourceError("UAVDT archive-gate report lacks Dataset evidence")
    dataset_evidence = archives["dataset"]
    expected_sha256 = dataset_evidence.get("sha256")
    expected_bytes = dataset_evidence.get("bytes")
    expected_members = dataset_evidence.get("members")
    expected_files = dataset_evidence.get("files")
    if (
        not isinstance(expected_sha256, str)
        or not isinstance(expected_bytes, int)
        or not isinstance(expected_members, int)
        or not isinstance(expected_files, int)
    ):
        raise UavdtSourceError("UAVDT archive-gate Dataset evidence is incomplete")
    if not archive.is_file() or archive.stat().st_size != expected_bytes:
        raise UavdtSourceError("admitted UAVDT Dataset archive is missing or has changed size")
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise UavdtSourceError("admitted UAVDT Dataset archive has changed since validation")

    existing = _existing_extraction(destination, expected_sha256)
    if existing is not None:
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.parent / f".{destination.name}.extracting-{uuid.uuid4().hex}"
    if temporary.exists():
        raise UavdtSourceError(f"refusing to reuse extraction temporary directory: {temporary}")
    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if len(entries) != expected_members:
                raise UavdtSourceError("Dataset archive member count changed before extraction")
            total_uncompressed = sum(entry.file_size for entry in entries if not entry.is_dir())
            free_bytes = shutil.disk_usage(destination.parent).free
            if free_bytes < total_uncompressed + 1024**3:
                raise UavdtSourceError(
                    "insufficient free space for atomic UAVDT extraction with a 1 GiB reserve"
                )
            temporary.mkdir()
            normalized: set[str] = set()
            extracted_files = 0
            extracted_bytes = 0
            for entry in entries:
                name = _safe_member_name(entry.filename, expected_root=DATASET_ROOT)
                folded = name.casefold()
                if folded in normalized:
                    raise UavdtSourceError("Dataset archive member collision during extraction")
                normalized.add(folded)
                if entry.flag_bits & 0x1 or _is_symlink(entry):
                    raise UavdtSourceError("Dataset archive contains a prohibited member type")
                target = temporary.joinpath(*PurePosixPath(name).parts)
                resolved_target = target.resolve()
                if temporary not in resolved_target.parents:
                    raise UavdtSourceError(f"Dataset member escapes extraction root: {name}")
                if entry.is_dir():
                    resolved_target.mkdir(parents=True, exist_ok=True)
                    continue
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, resolved_target.open("xb") as output_stream:
                    shutil.copyfileobj(source, output_stream, length=1024 * 1024)
                if resolved_target.stat().st_size != entry.file_size:
                    raise UavdtSourceError(f"extracted Dataset member has wrong size: {name}")
                extracted_files += 1
                extracted_bytes += entry.file_size
            if extracted_files != expected_files or extracted_bytes != total_uncompressed:
                raise UavdtSourceError("extracted Dataset totals differ from the admitted archive")

        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_path": _display_path(archive, project_root),
            "archive_sha256": actual_sha256,
            "archive_gate_report": _display_path(archive_gate_report, project_root),
            "destination": _display_path(destination, project_root),
            "dataset_root": f"{_display_path(destination, project_root)}/{DATASET_ROOT}",
            "members": expected_members,
            "files": extracted_files,
            "uncompressed_bytes": extracted_bytes,
            "atomic_promotion": True,
            "source_files_modified": False,
            "formal_metrics_calculated_or_viewed": False,
            "reused_existing": False,
        }
        atomic_write_json(temporary / ".uavdt_extraction.json", result)
        temporary.replace(destination)
        return result
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
