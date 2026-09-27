from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from buse_uav.data.common import DataError, image_size
from buse_uav.data.visdrone import VisDroneAdapter
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_bytes, atomic_write_json

REQUIRED_CORRUPTIONS = (
    "fog",
    "low_light",
    "gaussian_blur",
    "motion_blur",
    "rain",
    "jpeg",
)


def deterministic_corruption_seed(
    global_seed: int,
    image_id: str,
    corruption: str,
    severity: int,
) -> int:
    payload = f"{global_seed}|{image_id}|{corruption}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def build_visdrone_c(
    *,
    source_root: Path,
    output_root: Path,
    splits: Sequence[str],
    corruption_levels: Mapping[str, Sequence[float]],
    global_seed: int,
    max_images: int | None = None,
) -> dict[str, Any]:
    _validate_corruption_levels(corruption_levels)
    _validate_existing_manifest_identity(
        output_root,
        global_seed=global_seed,
        corruption_levels=corruption_levels,
    )
    rows: list[dict[str, Any]] = []
    for split in splits:
        adapter = VisDroneAdapter(source_root, split=split)
        validation = adapter.validate()
        if not validation["valid"]:
            raise DataError(
                f"cannot build VisDrone-C from invalid {split} data: "
                + "; ".join(validation["errors"][:10])
            )
        images = adapter.image_paths()
        if max_images is not None:
            images = images[:max_images]
        for source in images:
            rgb = _read_rgb(source)
            height, width = rgb.shape[:2]
            for corruption in REQUIRED_CORRUPTIONS:
                for severity, parameter in enumerate(corruption_levels[corruption], start=1):
                    seed = deterministic_corruption_seed(
                        global_seed,
                        source.stem,
                        corruption,
                        severity,
                    )
                    output = apply_corruption(
                        rgb,
                        corruption=corruption,
                        parameter=float(parameter),
                        seed=seed,
                    )
                    if output.shape != rgb.shape:
                        raise DataError(f"{corruption}/{severity} changed geometry for {source}")
                    destination = (
                        output_root / split / corruption / str(severity) / f"{source.stem}.png"
                    )
                    _write_png(destination, output)
                    rows.append(
                        {
                            "source_split": split,
                            "image_id": source.stem,
                            "corruption": corruption,
                            "severity": severity,
                            "seed": seed,
                            "parameter": float(parameter),
                            "parameters_json": json.dumps(
                                {
                                    "name": corruption,
                                    "value": float(parameter),
                                    "seed": seed,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "source_path": str(source.resolve()),
                            "output_path": destination.relative_to(output_root).as_posix(),
                            "width": width,
                            "height": height,
                            "source_sha256": sha256_file(source),
                            "output_sha256": sha256_file(destination),
                        }
                    )

    if not rows:
        raise DataError("no images were found for VisDrone-C generation")
    new_frame = pd.DataFrame(rows).sort_values(
        ["source_split", "image_id", "corruption", "severity"],
        kind="stable",
    )
    frame = _merge_existing_manifest(
        output_root,
        new_frame,
        replaced_splits=set(splits),
        global_seed=global_seed,
        corruption_levels=corruption_levels,
    )
    _write_parquet_atomic(output_root / "manifest.parquet", frame)
    json_rows = frame.to_dict(orient="records")
    split_order = {"val": 0, "test-dev": 1}
    manifest_splits = sorted(
        {str(value) for value in frame["source_split"].tolist()},
        key=lambda value: (split_order.get(value, len(split_order)), value),
    )
    manifest = {
        "schema_version": 1,
        "dataset": "VisDrone2019-DET-C",
        "global_seed": global_seed,
        "splits": manifest_splits,
        "corruptions": {
            key: [float(value) for value in corruption_levels[key]] for key in REQUIRED_CORRUPTIONS
        },
        "rows": json_rows,
    }
    manifest["manifest_sha256"] = stable_hash(manifest, length=64)
    atomic_write_json(output_root / "manifest.json", manifest)
    return {
        "images": len({(row["source_split"], row["image_id"]) for row in rows}),
        "generated": len(rows),
        "manifest_rows": len(frame),
        "manifest": str(output_root / "manifest.parquet"),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _merge_existing_manifest(
    output_root: Path,
    new_frame: pd.DataFrame,
    *,
    replaced_splits: set[str],
    global_seed: int,
    corruption_levels: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    parquet_path = output_root / "manifest.parquet"
    json_path = output_root / "manifest.json"
    if not parquet_path.is_file() and not json_path.is_file():
        return new_frame
    if not parquet_path.is_file() or not json_path.is_file():
        raise DataError("existing VisDrone-C root has only one manifest format")
    try:
        existing_document = json.loads(json_path.read_text(encoding="utf-8"))
        existing_frame = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise DataError(f"cannot read existing VisDrone-C manifest: {exc}") from exc
    expected_levels = {
        key: [float(value) for value in corruption_levels[key]] for key in REQUIRED_CORRUPTIONS
    }
    if existing_document.get("global_seed") != global_seed:
        raise DataError("cannot merge VisDrone-C manifests with different global seeds")
    if existing_document.get("corruptions") != expected_levels:
        raise DataError("cannot merge VisDrone-C manifests with different corruption levels")
    required_columns = set(new_frame.columns)
    if not required_columns.issubset(existing_frame.columns):
        missing = sorted(required_columns - set(existing_frame.columns))
        raise DataError(f"existing VisDrone-C manifest is missing columns: {missing}")
    retained = existing_frame[~existing_frame["source_split"].isin(replaced_splits)]
    combined = pd.concat([retained, new_frame], ignore_index=True).sort_values(
        ["source_split", "image_id", "corruption", "severity"],
        kind="stable",
    )
    identity_columns = ["source_split", "image_id", "corruption", "severity"]
    if bool(combined.duplicated(identity_columns).any()):
        raise DataError("merged VisDrone-C manifest contains duplicate corruption identities")
    return cast(pd.DataFrame, combined.reset_index(drop=True))


def _validate_existing_manifest_identity(
    output_root: Path,
    *,
    global_seed: int,
    corruption_levels: Mapping[str, Sequence[float]],
) -> None:
    parquet_path = output_root / "manifest.parquet"
    json_path = output_root / "manifest.json"
    if not parquet_path.is_file() and not json_path.is_file():
        return
    if not parquet_path.is_file() or not json_path.is_file():
        raise DataError("existing VisDrone-C root has only one manifest format")
    try:
        existing_document = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataError(f"cannot read existing VisDrone-C manifest identity: {exc}") from exc
    expected_levels = {
        key: [float(value) for value in corruption_levels[key]] for key in REQUIRED_CORRUPTIONS
    }
    if existing_document.get("global_seed") != global_seed:
        raise DataError("cannot merge VisDrone-C manifests with different global seeds")
    if existing_document.get("corruptions") != expected_levels:
        raise DataError("cannot merge VisDrone-C manifests with different corruption levels")


def apply_corruption(
    image: np.ndarray,
    *,
    corruption: str,
    parameter: float,
    seed: int,
) -> np.ndarray:
    _validate_rgb(image)
    rng = np.random.Generator(np.random.PCG64(seed))
    if corruption == "fog":
        return _fog(image, beta=parameter, rng=rng)
    if corruption == "low_light":
        return _low_light(image, gamma=parameter)
    if corruption == "gaussian_blur":
        return _gaussian_blur(image, sigma=parameter)
    if corruption == "motion_blur":
        return _motion_blur(image, kernel_size=round(parameter), rng=rng)
    if corruption == "rain":
        return _rain(image, density=parameter, rng=rng)
    if corruption == "jpeg":
        return _jpeg(image, quality=round(parameter))
    raise DataError(f"unsupported corruption: {corruption}")


def verify_visdrone_c(
    root: Path,
    *,
    contact_output: Path,
    samples_per_severity: int = 8,
) -> dict[str, Any]:
    manifest_path = root / "manifest.parquet"
    json_path = root / "manifest.json"
    if not manifest_path.is_file() or not json_path.is_file():
        raise DataError(f"VisDrone-C manifest pair does not exist below: {root}")
    try:
        frame = pd.read_parquet(manifest_path)
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataError(f"cannot read corruption manifests below {root}: {exc}") from exc

    errors: list[str] = []
    json_rows = document.get("rows")
    recorded_content_hash = document.get("manifest_sha256")
    unhashed_document = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if recorded_content_hash != stable_hash(unhashed_document, length=64):
        errors.append("JSON corruption manifest content hash mismatch")
    if not isinstance(json_rows, list) or len(json_rows) != len(frame):
        errors.append("JSON/Parquet corruption manifest row count mismatch")
    elif stable_hash(json_rows, length=64) != stable_hash(
        frame.to_dict(orient="records"), length=64
    ):
        errors.append("JSON/Parquet corruption manifest rows differ")
    combinations = {
        (str(row.corruption), int(cast(Any, row.severity))) for row in frame.itertuples(index=False)
    }
    required = {
        (corruption, severity) for corruption in REQUIRED_CORRUPTIONS for severity in (1, 2, 3)
    }
    missing = sorted(required - combinations)
    if missing:
        errors.append(f"missing corruption/severity combinations: {missing}")
    identity_columns = ["source_split", "image_id", "corruption", "severity"]
    if bool(frame.duplicated(identity_columns).any()):
        errors.append("corruption manifest contains duplicate image cells")

    verified_outputs = 0
    latest_output_mtime_ns = 0
    for row in frame.itertuples(index=False):
        path = root / str(row.output_path)
        if not path.is_file():
            errors.append(f"generated image is missing: {path}")
            continue
        verified_outputs += 1
        latest_output_mtime_ns = max(latest_output_mtime_ns, path.stat().st_mtime_ns)
        try:
            size = image_size(path)
        except DataError as exc:
            errors.append(str(exc))
            continue
        expected_width = int(cast(Any, row.width))
        expected_height = int(cast(Any, row.height))
        if size != (expected_width, expected_height):
            errors.append(
                f"geometry mismatch for {path}: expected "
                f"{(expected_width, expected_height)}, found {size}"
            )
        if sha256_file(path) != str(row.output_sha256):
            errors.append(f"hash mismatch for generated image: {path}")

    contact_sheets: list[str] = []
    if not errors:
        for corruption in REQUIRED_CORRUPTIONS:
            path = contact_output / f"{corruption}.png"
            _build_contact_sheet(
                frame[frame["corruption"] == corruption],
                root=root,
                output=path,
                samples_per_severity=samples_per_severity,
            )
            contact_sheets.append(str(path))
    report = {
        "valid": not errors,
        "rows": len(frame),
        "splits": {
            str(split): int(count)
            for split, count in frame.groupby("source_split", sort=True).size().items()
        },
        "manifest_content_sha256": recorded_content_hash,
        "manifest_json_sha256": sha256_file(json_path),
        "manifest_parquet_sha256": sha256_file(manifest_path),
        "verified_outputs": verified_outputs,
        "latest_output_mtime_ns": latest_output_mtime_ns,
        "errors": errors,
        "contact_sheets": contact_sheets,
    }
    atomic_write_json(root / "verification.json", report)
    return report


def _fog(
    image: np.ndarray,
    *,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    small_height = max(2, math.ceil(height / 32))
    small_width = max(2, math.ceil(width / 32))
    noise = rng.random((small_height, small_width), dtype=np.float32)
    depth = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    sigma = max(height, width) / 24
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=max(1.0, sigma))
    depth -= depth.min()
    depth = depth / np.float32(max(float(depth.max()), 1e-6))
    depth = 0.35 + 0.65 * depth
    transmission = np.exp(-beta * depth).astype(np.float32)[..., None]
    normalized = image.astype(np.float32) / 255.0
    atmospheric_light = np.float32(0.92)
    result = normalized * transmission + atmospheric_light * (1.0 - transmission)
    return _to_uint8(result * 255.0)


def _low_light(image: np.ndarray, *, gamma: float) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return _to_uint8(np.power(normalized, gamma) * 255.0)


def _gaussian_blur(image: np.ndarray, *, sigma: float) -> np.ndarray:
    kernel = max(3, math.ceil(sigma * 6) | 1)
    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _motion_blur(
    image: np.ndarray,
    *,
    kernel_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    kernel_size = max(3, kernel_size | 1)
    angle = float(rng.uniform(-90.0, 90.0))
    radius = (kernel_size - 1) / 2
    radians = math.radians(angle)
    center = kernel_size // 2
    dx = radius * math.cos(radians)
    dy = radius * math.sin(radians)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    start = (round(center - dx), round(center - dy))
    end = (round(center + dx), round(center + dy))
    cv2.line(kernel, start, end, color=1.0, thickness=1)
    kernel /= max(float(kernel.sum()), 1.0)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _rain(
    image: np.ndarray,
    *,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    count = max(1, round(height * width * density))
    overlay = np.zeros_like(image)
    length = max(5, round(min(height, width) * 0.06))
    angle = float(rng.uniform(-20.0, 20.0))
    dx = round(length * math.sin(math.radians(angle)))
    dy = round(length * math.cos(math.radians(angle)))
    xs = rng.integers(0, width, size=count)
    ys = rng.integers(-length, height, size=count)
    intensities = rng.integers(160, 256, size=count)
    for x, y, intensity in zip(xs, ys, intensities, strict=True):
        cv2.line(
            overlay,
            (int(x), int(y)),
            (int(x + dx), int(y + dy)),
            (int(intensity),) * 3,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    overlay = cv2.GaussianBlur(overlay, (3, 3), sigmaX=0.6)
    return cv2.addWeighted(image, 1.0, overlay, 0.35, 0.0)


def _jpeg(image: np.ndarray, *, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise DataError("OpenCV failed to encode a JPEG corruption")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise DataError("OpenCV failed to decode a JPEG corruption")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise DataError(f"OpenCV cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_png(path: Path, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise DataError(f"OpenCV cannot encode PNG: {path}")
    atomic_write_bytes(path, encoded.tobytes())


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.parquet",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build_contact_sheet(
    frame: pd.DataFrame,
    *,
    root: Path,
    output: Path,
    samples_per_severity: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tile_width = 224
    tile_height = 160
    header_height = 24
    severities = (1, 2, 3)
    columns = max(
        1,
        min(
            samples_per_severity,
            max(len(frame[frame["severity"] == severity]) for severity in severities),
        ),
    )
    canvas = Image.new(
        "RGB",
        (columns * tile_width, len(severities) * (tile_height + header_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, severity in enumerate(severities):
        subset = frame[frame["severity"] == severity].sort_values(
            ["source_split", "image_id"], kind="stable"
        )
        subset = subset.head(samples_per_severity)
        y_offset = row_index * (tile_height + header_height)
        draw.text((4, y_offset + 4), f"severity {severity}", fill="black")
        for column, row in enumerate(subset.itertuples(index=False)):
            with Image.open(root / str(row.output_path)) as image:
                tile = image.convert("RGB")
                tile.thumbnail((tile_width, tile_height))
                x = column * tile_width + (tile_width - tile.width) // 2
                y = y_offset + header_height + (tile_height - tile.height) // 2
                canvas.paste(tile, (x, y))
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp.png",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        canvas.save(temporary, format="PNG")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_corruption_levels(
    corruption_levels: Mapping[str, Sequence[float]],
) -> None:
    missing = [name for name in REQUIRED_CORRUPTIONS if name not in corruption_levels]
    if missing:
        raise DataError(f"corruption configuration is missing: {missing}")
    invalid = [name for name in REQUIRED_CORRUPTIONS if len(corruption_levels[name]) != 3]
    if invalid:
        raise DataError(f"each corruption must define exactly three severities: {invalid}")


def _validate_rgb(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise DataError(
            f"corruption input must be uint8 RGB HWC, found {image.dtype} {image.shape}"
        )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return result
