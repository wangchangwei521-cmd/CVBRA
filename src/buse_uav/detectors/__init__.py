"""Detector adapters."""

from buse_uav.detectors.base import DetectorAdapter, DetectorError
from buse_uav.detectors.factory import build_detector
from buse_uav.detectors.mmdet_adapter import MMDetectionDetector
from buse_uav.detectors.ultralytics_adapter import UltralyticsDetector

__all__ = [
    "DetectorAdapter",
    "DetectorError",
    "MMDetectionDetector",
    "UltralyticsDetector",
    "build_detector",
]
