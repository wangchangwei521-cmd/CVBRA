from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np

from buse_uav.schemas import Region

FEATURE_NAMES = ("luminance", "contrast", "blur", "haze", "entropy")
EPSILON = 1e-12
_DARK_CHANNEL_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))


@dataclass(frozen=True)
class CalibrationStatistic:
    median: float
    mad: float


@dataclass(frozen=True)
class QualityCalibration:
    statistics: Mapping[str, CalibrationStatistic]
    dataset: str
    source_split: str
    image_variant: str

    @classmethod
    def from_file(cls, path: Path) -> QualityCalibration:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            statistics = {
                name: CalibrationStatistic(
                    median=float(document["statistics"][name]["median"]),
                    mad=float(document["statistics"][name]["mad"]),
                )
                for name in FEATURE_NAMES
            }
            calibration = cls(
                statistics=statistics,
                dataset=str(document["dataset"]),
                source_split=str(document["source_split"]),
                image_variant=str(document["image_variant"]),
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(f"invalid quality calibration {path}: {exc}") from exc
        if calibration.source_split.casefold() != "train":
            raise ValueError("quality calibration must come from the train split")
        if calibration.image_variant.casefold() != "clean":
            raise ValueError("quality calibration must come from clean images")
        return calibration


class DegradationScore(TypedDict):
    region_id: int
    degradation: float
    components: dict[str, float]
    raw_components: dict[str, float]


def raw_degradation_features(image: np.ndarray, region: Region) -> dict[str, float]:
    _validate_rgb(image)
    x1, y1, x2, y2 = region.core_xyxy
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("region core is outside the image")
    core_u8 = image[y1:y2, x1:x2]
    red = core_u8[..., 0].astype(np.float64)
    green = core_u8[..., 1].astype(np.float64)
    blue = core_u8[..., 2].astype(np.float64)
    red /= 255.0
    green /= 255.0
    blue /= 255.0
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    laplacian = cv2.Laplacian(luminance, cv2.CV_64F)
    dark_u8 = np.minimum(np.minimum(core_u8[..., 0], core_u8[..., 1]), core_u8[..., 2])
    dark_channel_u8 = cv2.erode(
        dark_u8,
        _DARK_CHANNEL_KERNEL,
        borderType=cv2.BORDER_REPLICATE,
    )
    gray_u8 = np.clip(np.rint(luminance * 255.0), 0, 255).astype(np.uint8)
    histogram = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray_u8.size
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "luminance": float(1.0 - np.mean(luminance)),
        "contrast": float(1.0 - np.std(luminance)),
        "blur": float(-math.log(float(np.var(laplacian)) + EPSILON)),
        "haze": float(np.mean(dark_channel_u8.astype(np.float64) / 255.0)),
        "entropy": float(1.0 - entropy_bits / 8.0),
    }


def raw_feature_matrix(
    image: np.ndarray,
    regions: Sequence[Region],
    *,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    if workers < 0:
        raise ValueError("degradation workers must be nonnegative")
    if not regions:
        return {name: np.empty(0, dtype=np.float64) for name in FEATURE_NAMES}
    if workers <= 1 or len(regions) == 1:
        rows = [raw_degradation_features(image, region) for region in regions]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(regions))) as executor:
            rows = list(executor.map(partial(raw_degradation_features, image), regions))
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in FEATURE_NAMES
    }


def score_degradation(
    image: np.ndarray,
    regions: Sequence[Region],
    *,
    calibration: QualityCalibration | None,
    rank_mix: float,
    weights: Mapping[str, float],
    allow_rank_only: bool = False,
    workers: int = 1,
) -> tuple[DegradationScore, ...]:
    """Score all regions according to specification section 6.2."""
    if not 0.0 <= rank_mix <= 1.0:
        raise ValueError("rank_mix must be in [0, 1]")
    if set(weights) != set(FEATURE_NAMES):
        raise ValueError(f"degradation weights must be exactly {FEATURE_NAMES}")
    if any(value < 0 for value in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0, abs_tol=1e-6
    ):
        raise ValueError("degradation weights must be nonnegative and sum to one")
    if calibration is None and not allow_rank_only:
        raise ValueError("quality calibration is required outside smoke rank-only mode")

    raw = raw_feature_matrix(image, regions, workers=workers)
    normalized: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        ranks = _rank01(raw[name])
        if calibration is None:
            normalized[name] = ranks
            continue
        statistic = calibration.statistics[name]
        denominator = 1.4826 * statistic.mad + EPSILON
        robust_z = np.clip((raw[name] - statistic.median) / denominator, -60.0, 60.0)
        absolute = 1.0 / (1.0 + np.exp(-robust_z))
        normalized[name] = rank_mix * ranks + (1.0 - rank_mix) * absolute

    scores: list[DegradationScore] = []
    for index, region in enumerate(regions):
        components = {name: float(normalized[name][index]) for name in FEATURE_NAMES}
        score = sum(float(weights[name]) * components[name] for name in FEATURE_NAMES)
        scores.append(
            {
                "region_id": region.id,
                "degradation": float(np.clip(score, 0.0, 1.0)),
                "components": components,
                "raw_components": {name: float(raw[name][index]) for name in FEATURE_NAMES},
            }
        )
    return tuple(scores)


def calibration_statistics(
    feature_rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if not feature_rows:
        raise ValueError("at least one feature row is required for calibration")
    statistics: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = np.asarray([float(row[name]) for row in feature_rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite calibration feature: {name}")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        statistics[name] = {"median": median, "mad": mad}
    return statistics


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return cast(np.ndarray, values.copy())
    if values.size == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = sorted(range(values.size), key=lambda index: float(values[index]))
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return cast(np.ndarray, ranks / (values.size - 1.0))


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("degradation scoring expects uint8 RGB HWC")
