from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import cv2
import numpy as np


class Enhancer(Protocol):
    @property
    def name(self) -> str:
        """Return the stable operation name."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return a new uint8 RGB HWC image with unchanged dimensions."""

    def params(self) -> dict[str, Any]:
        """Return serializable operation parameters."""


@dataclass(frozen=True)
class IdentityEnhancer:
    name: str = "identity"

    def apply(self, image: np.ndarray) -> np.ndarray:
        _validate_rgb(image)
        return cast(np.ndarray, image.copy())

    def params(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class GammaEnhancer:
    gamma: float = 0.65
    name: str = "gamma"

    def apply(self, image: np.ndarray) -> np.ndarray:
        _validate_rgb(image)
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        normalized = image.astype(np.float32) / 255.0
        return cast(
            np.ndarray,
            np.clip(
                np.rint(np.power(normalized, self.gamma) * 255.0),
                0,
                255,
            ).astype(np.uint8),
        )

    def params(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


@dataclass(frozen=True)
class ClaheEnhancer:
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    name: str = "clahe"

    def apply(self, image: np.ndarray) -> np.ndarray:
        _validate_rgb(image)
        if self.clip_limit <= 0.0 or min(self.tile_grid_size) <= 0:
            raise ValueError("CLAHE parameters must be positive")
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def params(self) -> dict[str, Any]:
        return {
            "clip_limit": self.clip_limit,
            "tile_grid_size": list(self.tile_grid_size),
        }


@dataclass(frozen=True)
class UnsharpEnhancer:
    amount: float = 0.8
    sigma: float = 1.2
    name: str = "unsharp"

    def apply(self, image: np.ndarray) -> np.ndarray:
        _validate_rgb(image)
        if self.amount < 0.0 or self.sigma <= 0.0:
            raise ValueError("unsharp amount must be nonnegative and sigma positive")
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sigma)
        return cv2.addWeighted(image, 1.0 + self.amount, blurred, -self.amount, 0.0)

    def params(self) -> dict[str, Any]:
        return {"amount": self.amount, "sigma": self.sigma}


def make_enhancer(
    operation: str,
    *,
    gamma: float,
    clahe_clip_limit: float,
    clahe_tile_grid: tuple[int, int],
    unsharp_amount: float,
    unsharp_sigma: float,
) -> Enhancer:
    if operation == "identity":
        return IdentityEnhancer()
    if operation == "gamma":
        return GammaEnhancer(gamma=gamma)
    if operation == "clahe":
        return ClaheEnhancer(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid,
        )
    if operation == "unsharp":
        return UnsharpEnhancer(amount=unsharp_amount, sigma=unsharp_sigma)
    raise ValueError(f"unsupported core enhancement operation: {operation}")


def route_top_operations(
    degradation_components: Mapping[str, float],
    *,
    available: Sequence[str],
    max_ops: int,
) -> tuple[str, ...]:
    """Route highest normalized degradation components to unique operations."""
    if max_ops < 0:
        raise ValueError("max_ops must be nonnegative")
    required = {"luminance", "contrast", "blur", "haze", "entropy"}
    if set(degradation_components) != required:
        raise ValueError(f"degradation components must be exactly {sorted(required)}")
    component_to_operation = {
        "luminance": "gamma",
        "contrast": "clahe",
        "haze": "clahe",
        "blur": "unsharp",
    }
    priority = {"luminance": 0, "contrast": 1, "haze": 2, "blur": 3, "entropy": 4}
    ordered_components = sorted(
        degradation_components,
        key=lambda name: (-float(degradation_components[name]), priority[name]),
    )
    allowed = set(available)
    operations: list[str] = []
    for component in ordered_components:
        operation = component_to_operation.get(component)
        if operation is None or operation not in allowed or operation in operations:
            continue
        operations.append(operation)
        if len(operations) >= max_ops:
            break
    return tuple(operations)


def apply_enhancer_batch(
    enhancer: Enhancer,
    images: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    return tuple(enhancer.apply(image) for image in images)


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("enhancer expects one non-empty uint8 RGB HWC image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("enhancer expects one non-empty uint8 RGB HWC image")
