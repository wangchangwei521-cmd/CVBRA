from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from buse_uav.detectors.base import (
    DetectorAdapter,
    DetectorError,
    project_ultralytics_config_dir,
    validate_detection_batches,
)
from buse_uav.evaluation.timing import SynchronizedTimer
from buse_uav.schemas import Box, DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file

_MAX_STREAM_RECORDS = 8


def configure_ultralytics_environment(project_root: Path | None = None) -> Path:
    """Keep Ultralytics settings inside the ignored project data directory."""
    config_dir = project_ultralytics_config_dir(project_root)
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    return config_dir


class UltralyticsDetector(DetectorAdapter):
    def __init__(
        self,
        model_path: Path,
        *,
        model_name: str = "yolo11n",
        device: str,
        expected_class_names: Sequence[str],
        project_root: Path | None = None,
        stream_chunk_records: int = _MAX_STREAM_RECORDS,
        release_cuda_cache_between_chunks: bool = True,
    ) -> None:
        configure_ultralytics_environment(project_root)
        try:
            from ultralytics import RTDETR, YOLO  # type: ignore[attr-defined]
        except (ImportError, OSError, PermissionError) as exc:
            raise DetectorError(f"cannot import Ultralytics: {exc}") from exc
        if not model_path.is_file():
            raise DetectorError(
                f"detector weights do not exist: {model_path}. Run the smoke training first."
            )
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        if stream_chunk_records <= 0:
            raise DetectorError("stream_chunk_records must be positive")
        self.stream_chunk_records = stream_chunk_records
        self.release_cuda_cache_between_chunks = release_cuda_cache_between_chunks
        normalized_name = model_name.casefold().replace("-", "_")
        model_class: Any
        if normalized_name.startswith("rtdetr_"):
            model_class = RTDETR
        elif normalized_name.startswith("yolo"):
            model_class = YOLO
        else:
            raise DetectorError(f"unsupported Ultralytics detector family: {model_name}")
        self._model = model_class(str(model_path))
        raw_names = self._model.names
        if isinstance(raw_names, dict):
            names = tuple(str(raw_names[index]) for index in sorted(raw_names))
        else:
            names = tuple(str(name) for name in raw_names)
        expected = tuple(expected_class_names)
        if names != expected:
            raise DetectorError(
                f"detector class mapping mismatch: weights={names}, dataset={expected}"
            )
        self._class_names = names

    @property
    def class_names(self) -> Sequence[str]:
        return self._class_names

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
        if not records:
            return ()
        batches: list[DetectionBatch] = []
        try:
            with SynchronizedTimer(self.device) as timer:
                for start in range(0, len(records), self.stream_chunk_records):
                    record_chunk = records[start : start + self.stream_chunk_records]
                    results = self._model.predict(
                        source=[_prediction_source(record) for record in record_chunk],
                        imgsz=imgsz,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        device=self.device,
                        half=fp16 and self.device.casefold() != "cpu",
                        verbose=False,
                        stream=True,
                    )
                    for record, result in zip(record_chunk, results, strict=True):
                        batches.append(_convert_result(record, result))
                    del results
                    if self.release_cuda_cache_between_chunks:
                        _release_cuda_cache(self.device)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DetectorError(f"Ultralytics inference failed: {exc}") from exc
        elapsed_ms = timer.elapsed_ms
        if len(batches) != len(records):
            raise DetectorError(
                f"Ultralytics returned {len(batches)} results for {len(records)} inputs"
            )
        fallback_latency = elapsed_ms / len(records)
        batches = [
            replace(
                batch,
                latency_ms=fallback_latency,
                meta={
                    **batch.meta,
                    "cuda_synchronized": timer.synchronized,
                    "streamed_to_cpu": True,
                    "stream_chunk_records": self.stream_chunk_records,
                    "in_memory_input": batch.meta.get("in_memory_input", False),
                    "released_cuda_cache": self.release_cuda_cache_between_chunks,
                },
            )
            for batch in batches
        ]
        validate_detection_batches(
            batches,
            records,
            num_classes=len(self.class_names),
        )
        return tuple(batches)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "backend": "ultralytics",
            "name": self.model_name,
            "path": str(self.model_path.resolve()),
            "bytes": self.model_path.stat().st_size,
            "sha256": sha256_file(self.model_path),
            "class_names": list(self.class_names),
        }


def _cpu_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return list(value.tolist())


def _convert_result(record: ImageRecord, result: Any) -> DetectionBatch:
    boxes: list[Box] = []
    result_boxes = result.boxes
    if result_boxes is not None:
        xyxy = _cpu_list(result_boxes.xyxy)
        scores = _cpu_list(result_boxes.conf)
        classes = _cpu_list(result_boxes.cls)
        for coordinates, score, class_id in zip(
            xyxy,
            scores,
            classes,
            strict=True,
        ):
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            x1 = min(max(x1, 0.0), float(record.width))
            y1 = min(max(y1, 0.0), float(record.height))
            x2 = min(max(x2, 0.0), float(record.width))
            y2 = min(max(y2, 0.0), float(record.height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(
                Box(
                    xyxy=(x1, y1, x2, y2),
                    score=float(score),
                    class_id=int(class_id),
                )
            )
    return DetectionBatch(
        image_id=record.image_id,
        boxes=tuple(boxes),
        latency_ms=0.0,
        meta={
            "path": record.path,
            "backend": "ultralytics",
            "in_memory_input": record.image_bgr is not None,
        },
    )


def _prediction_source(record: ImageRecord) -> str | Any:
    image = record.image_bgr
    if image is None:
        return record.path
    if image.ndim != 3 or image.shape[2] != 3:
        raise DetectorError(f"in-memory image {record.image_id} must be BGR HWC")
    if image.shape[:2] != (record.height, record.width):
        raise DetectorError(
            f"in-memory image {record.image_id} shape {image.shape[:2]} does not match "
            f"record {(record.height, record.width)}"
        )
    return image


def _release_cuda_cache(device: str) -> None:
    if not device.casefold().startswith("cuda"):
        return
    try:
        import torch
    except (ImportError, OSError):
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
