from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from buse_uav.schemas import DetectionBatch, ImageRecord


class DetectorError(RuntimeError):
    """Detector error safe to show at the command line."""


class DetectorAdapter(ABC):
    """Backend-independent detector interface using original-image coordinates."""

    @property
    @abstractmethod
    def class_names(self) -> Sequence[str]:
        """Return detector class names in class-id order."""

    @abstractmethod
    def predict(
        self,
        records: Sequence[ImageRecord],
        *,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        fp16: bool,
    ) -> tuple[DetectionBatch, ...]:
        """Predict one batch while preserving input order and image IDs."""

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Return a serializable model identity."""


def validate_detection_batches(
    batches: Sequence[DetectionBatch],
    records: Sequence[ImageRecord],
    *,
    num_classes: int,
) -> None:
    """Fail closed on missing images, invalid classes, or unreasonable coordinates."""
    expected = {record.image_id: record for record in records}
    seen: set[int | str] = set()
    for batch in batches:
        if batch.image_id not in expected:
            raise DetectorError(f"prediction references unknown image {batch.image_id}")
        if batch.image_id in seen:
            raise DetectorError(f"duplicate prediction batch for image {batch.image_id}")
        seen.add(batch.image_id)
        record = expected[batch.image_id]
        for index, box in enumerate(batch.boxes):
            x1, y1, x2, y2 = box.xyxy
            if not all(math.isfinite(value) for value in (*box.xyxy, box.score)):
                raise DetectorError(
                    f"image {batch.image_id} box {index} contains non-finite values"
                )
            if not (0.0 <= x1 < x2 <= record.width and 0.0 <= y1 < y2 <= record.height):
                raise DetectorError(
                    f"image {batch.image_id} box {index} is outside "
                    f"{record.width}x{record.height}: {box.xyxy}"
                )
            if not 0.0 <= box.score <= 1.0:
                raise DetectorError(
                    f"image {batch.image_id} box {index} has invalid score {box.score}"
                )
            if not 0 <= box.class_id < num_classes:
                raise DetectorError(
                    f"image {batch.image_id} box {index} has invalid class {box.class_id}"
                )
    missing = set(expected) - seen
    if missing:
        first = sorted(str(value) for value in missing)[:3]
        raise DetectorError(f"predictions are missing {len(missing)} images; first={first}")


def project_ultralytics_config_dir(project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).resolve()
    return root / "data" / ".ultralytics"
