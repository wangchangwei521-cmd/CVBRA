from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from buse_uav.data.common import DataError
from buse_uav.data.corruptions import (
    REQUIRED_CORRUPTIONS,
    apply_corruption,
    build_visdrone_c,
    deterministic_corruption_seed,
    verify_visdrone_c,
)
from buse_uav.utils.hashing import sha256_file

LEVELS = {
    "fog": [0.6, 1.0, 1.5],
    "low_light": [1.5, 2.0, 2.8],
    "gaussian_blur": [0.8, 1.5, 2.5],
    "motion_blur": [7, 13, 21],
    "rain": [0.004, 0.008, 0.015],
    "jpeg": [50, 25, 10],
}


def test_all_corruptions_preserve_shape_dtype_and_are_deterministic() -> None:
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    for corruption in REQUIRED_CORRUPTIONS:
        seed = deterministic_corruption_seed(42, "image-1", corruption, 1)
        first = apply_corruption(
            image,
            corruption=corruption,
            parameter=float(LEVELS[corruption][0]),
            seed=seed,
        )
        second = apply_corruption(
            image,
            corruption=corruption,
            parameter=float(LEVELS[corruption][0]),
            seed=seed,
        )
        assert first.shape == image.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)


def test_visdrone_c_build_is_hash_deterministic_and_verifiable(
    synthetic_visdrone: Path,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "corruptions-a"
    second_root = tmp_path / "corruptions-b"
    first = build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=first_root,
        splits=["val"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )
    second = build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=second_root,
        splits=["val"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )

    assert first["generated"] == 18
    assert second["generated"] == 18
    first_images = sorted(path for path in first_root.rglob("*.png"))
    second_images = sorted(path for path in second_root.rglob("*.png"))
    assert len(first_images) == len(second_images) == 18
    for first_path, second_path in zip(first_images, second_images, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert Image.open(first_path).size == (96, 72)

    report = verify_visdrone_c(
        first_root,
        contact_output=tmp_path / "contact-sheets",
        samples_per_severity=1,
    )
    assert report["valid"], report["errors"]
    assert report["manifest_json_sha256"] == sha256_file(first_root / "manifest.json")
    assert report["manifest_parquet_sha256"] == sha256_file(first_root / "manifest.parquet")
    assert report["splits"] == {"val": 18}
    assert report["verified_outputs"] == 18
    assert report["latest_output_mtime_ns"] > 0
    assert len(report["contact_sheets"]) == 6
    assert all(Path(path).is_file() for path in report["contact_sheets"])


def test_visdrone_c_build_preserves_existing_split_manifest(
    synthetic_visdrone: Path,
    tmp_path: Path,
) -> None:
    shutil.copytree(
        synthetic_visdrone / "VisDrone2019-DET-val",
        synthetic_visdrone / "VisDrone2019-DET-test-dev",
    )
    output = tmp_path / "combined"
    build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=output,
        splits=["val"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )
    result = build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=output,
        splits=["test-dev"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )

    frame = pd.read_parquet(output / "manifest.parquet")
    report = verify_visdrone_c(
        output,
        contact_output=tmp_path / "combined-contacts",
        samples_per_severity=1,
    )
    assert result["generated"] == 18
    assert result["manifest_rows"] == 36
    assert set(frame["source_split"]) == {"val", "test-dev"}
    assert len(frame) == 36
    assert report["valid"] is True
    assert report["rows"] == 36
    assert report["splits"] == {"test-dev": 18, "val": 18}
    assert report["verified_outputs"] == 36


def test_visdrone_c_verify_rejects_divergent_json_manifest(
    synthetic_visdrone: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "corruptions"
    build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=output,
        splits=["val"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )
    document = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    document["rows"][0]["severity"] = 3
    (output / "manifest.json").write_text(json.dumps(document), encoding="utf-8")

    report = verify_visdrone_c(output, contact_output=tmp_path / "contacts")

    assert report["valid"] is False
    assert any("content hash mismatch" in error for error in report["errors"])


def test_visdrone_c_merge_rejects_different_seed(
    synthetic_visdrone: Path,
    tmp_path: Path,
) -> None:
    shutil.copytree(
        synthetic_visdrone / "VisDrone2019-DET-val",
        synthetic_visdrone / "VisDrone2019-DET-test-dev",
    )
    output = tmp_path / "combined"
    build_visdrone_c(
        source_root=synthetic_visdrone,
        output_root=output,
        splits=["val"],
        corruption_levels=LEVELS,
        global_seed=42,
        max_images=1,
    )

    with pytest.raises(DataError, match="different global seeds"):
        build_visdrone_c(
            source_root=synthetic_visdrone,
            output_root=output,
            splits=["test-dev"],
            corruption_levels=LEVELS,
            global_seed=43,
            max_images=1,
        )
    assert not (output / "test-dev").exists()
