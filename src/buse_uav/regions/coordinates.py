from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from buse_uav.schemas import Region

XYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    scale: float
    pad_left: int
    pad_top: int

    @classmethod
    def create(
        cls,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
    ) -> LetterboxTransform:
        if len(source_shape) < 2 or len(target_shape) < 2:
            raise ValueError("source_shape and target_shape must contain height and width")
        source_height, source_width = int(source_shape[0]), int(source_shape[1])
        target_height, target_width = int(target_shape[0]), int(target_shape[1])
        if min(source_height, source_width, target_height, target_width) <= 0:
            raise ValueError("letterbox dimensions must be positive")
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = min(target_width, max(1, round(source_width * scale)))
        resized_height = min(target_height, max(1, round(source_height * scale)))
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        return cls(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            resized_width=resized_width,
            resized_height=resized_height,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    def to_letterbox(self, box: XYXY) -> XYXY:
        x1, y1, x2, y2 = _finite_xyxy(box)
        return (
            x1 * self.scale_x + self.pad_left,
            y1 * self.scale_y + self.pad_top,
            x2 * self.scale_x + self.pad_left,
            y2 * self.scale_y + self.pad_top,
        )

    def from_letterbox(self, box: XYXY, *, clip: bool = True) -> XYXY:
        x1, y1, x2, y2 = _finite_xyxy(box)
        restored = (
            (x1 - self.pad_left) / self.scale_x,
            (y1 - self.pad_top) / self.scale_y,
            (x2 - self.pad_left) / self.scale_x,
            (y2 - self.pad_top) / self.scale_y,
        )
        if not clip:
            return restored
        return clip_xyxy(restored, width=self.source_width, height=self.source_height)


def letterbox_image(
    image: np.ndarray,
    target_shape: tuple[int, int],
    *,
    fill: int = 114,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize with aspect ratio preserved and symmetric integer padding."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("letterbox_image expects uint8 HWC with three channels")
    if not 0 <= fill <= 255:
        raise ValueError("fill must be in [0, 255]")
    transform = LetterboxTransform.create(image.shape, target_shape)
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full(
        (transform.target_height, transform.target_width, 3),
        fill,
        dtype=np.uint8,
    )
    y1, x1 = transform.pad_top, transform.pad_left
    output[
        y1 : y1 + transform.resized_height,
        x1 : x1 + transform.resized_width,
    ] = resized
    return output, transform


def crop_letterbox_to_original(
    box: XYXY,
    *,
    region: Region,
    transform: LetterboxTransform,
    image_width: int,
    image_height: int,
) -> XYXY:
    """Map a detector box from a letterboxed context crop to the original image."""
    local = transform.from_letterbox(box)
    crop_x1, crop_y1, _, _ = region.crop_xyxy
    original = (
        local[0] + crop_x1,
        local[1] + crop_y1,
        local[2] + crop_x1,
        local[3] + crop_y1,
    )
    return clip_xyxy(original, width=image_width, height=image_height)


def original_to_crop_letterbox(
    box: XYXY,
    *,
    region: Region,
    transform: LetterboxTransform,
) -> XYXY:
    """Map an original-image box into a region's letterboxed crop."""
    x1, y1, x2, y2 = _finite_xyxy(box)
    crop_x1, crop_y1, _, _ = region.crop_xyxy
    return transform.to_letterbox((x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1))


def clip_xyxy(box: XYXY, *, width: int, height: int) -> XYXY:
    if width <= 0 or height <= 0:
        raise ValueError("clip dimensions must be positive")
    x1, y1, x2, y2 = _finite_xyxy(box)
    return (
        min(float(width), max(0.0, x1)),
        min(float(height), max(0.0, y1)),
        min(float(width), max(0.0, x2)),
        min(float(height), max(0.0, y2)),
    )


def _finite_xyxy(box: XYXY) -> XYXY:
    values = tuple(float(value) for value in box)
    if len(values) != 4 or not all(np.isfinite(value) for value in values):
        raise ValueError("box must contain four finite xyxy coordinates")
    return values
