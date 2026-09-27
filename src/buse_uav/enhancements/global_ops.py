from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from buse_uav.data.common import DataError

GLOBAL_OPERATIONS = ("gamma", "clahe", "unsharp")


def apply_global_enhancement(
    image: np.ndarray,
    *,
    operation: str,
    gamma: float,
    clahe_clip_limit: float,
    clahe_tile_grid: tuple[int, int],
    unsharp_amount: float,
    unsharp_sigma: float,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("global enhancement expects one uint8 BGR image")
    if operation == "gamma":
        table = np.array(
            [round(255.0 * ((index / 255.0) ** gamma)) for index in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)
    if operation == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    if operation == "unsharp":
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=unsharp_sigma)
        return cv2.addWeighted(
            image,
            1.0 + unsharp_amount,
            blurred,
            -unsharp_amount,
            0,
        )
    raise ValueError(f"unsupported global enhancement {operation!r}; choose {GLOBAL_OPERATIONS}")


def enhance_file(
    source: Path,
    destination: Path,
    *,
    operation: str,
    gamma: float,
    clahe_clip_limit: float,
    clahe_tile_grid: tuple[int, int],
    unsharp_amount: float,
    unsharp_sigma: float,
    horizontal_flip: bool = False,
) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise DataError(f"OpenCV cannot read image {source}")
    if operation != "identity":
        image = apply_global_enhancement(
            image,
            operation=operation,
            gamma=gamma,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
            unsharp_amount=unsharp_amount,
            unsharp_sigma=unsharp_sigma,
        )
    if horizontal_flip:
        image = cv2.flip(image, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise DataError(f"OpenCV cannot write image {destination}")
