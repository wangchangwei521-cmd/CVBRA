from __future__ import annotations

import importlib
import importlib.metadata
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from buse_uav.detectors.base import DetectorAdapter, DetectorError, validate_detection_batches
from buse_uav.evaluation.timing import SynchronizedTimer
from buse_uav.schemas import Box, DetectionBatch, ImageRecord
from buse_uav.utils.hashing import sha256_file


class MMDetectionDetector(DetectorAdapter):
    """MMDetection adapter that exposes the backend-independent detector contract."""

    def __init__(
        self,
        model_path: Path,
        model_config: Path,
        *,
        device: str,
        expected_class_names: Sequence[str],
    ) -> None:
        try:
            from mmdet.apis import init_detector  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            raise DetectorError(
                "cannot import MMDetection; use the isolated Phase 9 environment"
            ) from exc
        if not model_path.is_file():
            raise DetectorError(f"MMDetection weights do not exist: {model_path}")
        if not model_config.is_file():
            raise DetectorError(f"MMDetection model config does not exist: {model_config}")
        self.model_path = model_path
        self.model_config = model_config
        self.device = device
        try:
            _register_official_hazydet_dataset(model_config)
            self._model = init_detector(str(model_config), str(model_path), device=device)
            self._model.eval()
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise DetectorError(f"MMDetection model initialization failed: {exc}") from exc
        metadata = getattr(self._model, "dataset_meta", {})
        raw_names = metadata.get("classes") if isinstance(metadata, Mapping) else None
        if not isinstance(raw_names, (list, tuple)):
            raise DetectorError("MMDetection checkpoint/config does not expose dataset classes")
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
        if imgsz <= 0 or max_det <= 0 or not 0.0 <= conf <= 1.0 or not 0.0 < iou <= 1.0:
            raise DetectorError("invalid MMDetection inference thresholds or sizes")
        try:
            import torch
            from mmdet.apis import inference_detector
        except (ImportError, OSError) as exc:
            raise DetectorError(f"cannot import MMDetection inference dependencies: {exc}") from exc
        _configure_model_test(self._model, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)
        batches: list[DetectionBatch] = []
        try:
            with torch.inference_mode():
                for record in records:
                    use_fp16 = fp16 and self.device.casefold().startswith("cuda")
                    with (
                        SynchronizedTimer(self.device) as timer,
                        torch.autocast(device_type="cuda", enabled=use_fp16),
                    ):
                        source = record.image_bgr if record.image_bgr is not None else record.path
                        result = inference_detector(self._model, source)
                    batch = _convert_result(record, result, conf=conf, max_det=max_det)
                    batches.append(
                        replace(
                            batch,
                            latency_ms=timer.elapsed_ms,
                            meta={
                                **batch.meta,
                                "cuda_synchronized": timer.synchronized,
                                "streamed_to_cpu": True,
                                "stream_chunk_records": 1,
                                "in_memory_input": record.image_bgr is not None,
                            },
                        )
                    )
        except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as exc:
            raise DetectorError(f"MMDetection inference failed: {exc}") from exc
        validate_detection_batches(batches, records, num_classes=len(self.class_names))
        return tuple(batches)

    def fingerprint(self) -> dict[str, Any]:
        parameters = sum(int(parameter.numel()) for parameter in self._model.parameters())
        return {
            "backend": "mmdet",
            "name": type(self._model).__name__,
            "path": str(self.model_path.resolve()),
            "bytes": self.model_path.stat().st_size,
            "sha256": sha256_file(self.model_path),
            "model_config": str(self.model_config.resolve()),
            "model_config_sha256": sha256_file(self.model_config),
            "class_names": list(self.class_names),
            "parameters": parameters,
            "mmdet_version": _distribution_version("mmdet"),
            "mmcv_version": _distribution_version("mmcv"),
            "mmengine_version": _distribution_version("mmengine"),
        }


def _configure_model_test(
    model: Any,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> None:
    config = getattr(model, "cfg", None)
    model_config = _get(config, "model")
    test_configs = [
        _get(model_config, "test_cfg"),
        getattr(model, "test_cfg", None),
        _get(getattr(model, "bbox_head", None), "test_cfg"),
    ]
    available = [value for value in test_configs if value is not None]
    if not available:
        raise DetectorError("MMDetection config lacks model.test_cfg")
    for test_config in available:
        _set(test_config, "score_thr", conf)
        _set(test_config, "max_per_img", max_det)
        nms = _get(test_config, "nms")
        if nms is None:
            raise DetectorError("MMDetection config lacks model.test_cfg.nms")
        _set(nms, "iou_threshold", iou)
    pipeline = _get(_get(_get(config, "test_dataloader"), "dataset"), "pipeline")
    resize_count = _set_resize_scale(pipeline, imgsz)
    if resize_count == 0:
        raise DetectorError("MMDetection test pipeline has no resize transform")


def _set_resize_scale(value: Any, imgsz: int) -> int:
    count = 0
    if isinstance(value, Mapping):
        transform_type = str(value.get("type", ""))
        if "Resize" in transform_type:
            _set(value, "scale", (imgsz, imgsz))
            if "scales" in value:
                _set(value, "scales", [(imgsz, imgsz)])
            count += 1
        for nested in value.values():
            count += _set_resize_scale(nested, imgsz)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            count += _set_resize_scale(nested, imgsz)
    return count


def _convert_result(
    record: ImageRecord,
    result: Any,
    *,
    conf: float,
    max_det: int,
) -> DetectionBatch:
    if isinstance(result, list):
        if len(result) != 1:
            raise DetectorError(f"MMDetection returned {len(result)} samples for one image")
        result = result[0]
    instances = getattr(result, "pred_instances", None)
    if instances is None:
        raise DetectorError("MMDetection result lacks pred_instances")
    coordinates = _cpu_list(getattr(instances, "bboxes", []))
    scores = _cpu_list(getattr(instances, "scores", []))
    labels = _cpu_list(getattr(instances, "labels", []))
    if not (len(coordinates) == len(scores) == len(labels)):
        raise DetectorError("MMDetection result arrays have inconsistent lengths")
    candidates: list[Box] = []
    for raw_coordinates, raw_score, raw_label in zip(  # noqa: B905 - Phase 9 uses Python 3.9
        coordinates, scores, labels
    ):
        score = float(raw_score)
        if score < conf:
            continue
        x1, y1, x2, y2 = (float(number) for number in raw_coordinates)
        x1 = min(max(x1, 0.0), float(record.width))
        y1 = min(max(y1, 0.0), float(record.height))
        x2 = min(max(x2, 0.0), float(record.width))
        y2 = min(max(y2, 0.0), float(record.height))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            Box(
                xyxy=(x1, y1, x2, y2),
                score=score,
                class_id=int(raw_label),
            )
        )
    candidates.sort(key=lambda box: (-box.score, box.class_id, box.xyxy))
    return DetectionBatch(
        image_id=record.image_id,
        boxes=tuple(candidates[:max_det]),
        latency_ms=0.0,
        meta={
            "path": record.path,
            "backend": "mmdet",
            "in_memory_input": record.image_bgr is not None,
        },
    )


def _cpu_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _set(value: Any, key: str, item: Any) -> None:
    if isinstance(value, dict):
        value[key] = item
    else:
        setattr(value, key, item)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _register_official_hazydet_dataset(model_config: Path) -> None:
    if model_config.name != "fcos_r50_1x_hazydet.py":
        return
    source_root = model_config.resolve().parents[2]
    dataset_module = source_root / "HazyDet" / "datasets" / "hazydet.py"
    if not dataset_module.is_file():
        raise DetectorError(f"official HazyDet dataset module does not exist: {dataset_module}")
    source = str(source_root)
    sys.path.insert(0, source)
    try:
        importlib.import_module("HazyDet.datasets.hazydet")
    except (ImportError, OSError, AssertionError) as exc:
        raise DetectorError(f"cannot register official HazyDet dataset: {exc}") from exc
    finally:
        sys.path.remove(source)
