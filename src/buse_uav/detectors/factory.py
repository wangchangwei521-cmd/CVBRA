from __future__ import annotations

from collections.abc import Sequence

from buse_uav.detectors.base import DetectorAdapter, DetectorError
from buse_uav.detectors.mmdet_adapter import MMDetectionDetector
from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector
from buse_uav.schemas import DetectorConfig, RuntimeConfig


def build_detector(
    config: DetectorConfig,
    *,
    device: str,
    expected_class_names: Sequence[str],
    runtime: RuntimeConfig | None = None,
) -> DetectorAdapter:
    """Build a backend adapter without leaking detector branches into the method."""
    if config.backend == "ultralytics":
        return UltralyticsDetector(
            config.model,
            model_name=config.name,
            device=device,
            expected_class_names=expected_class_names,
            stream_chunk_records=(runtime.stream_chunk_records if runtime else 8),
            release_cuda_cache_between_chunks=(
                runtime.release_cuda_cache_between_chunks if runtime else True
            ),
        )
    if config.backend == "mmdet":
        if config.definition is None:
            raise DetectorError("backend=mmdet requires detector.definition")
        return MMDetectionDetector(
            config.model,
            config.definition,
            device=device,
            expected_class_names=expected_class_names,
        )
    raise DetectorError(f"unsupported detector backend: {config.backend}")
