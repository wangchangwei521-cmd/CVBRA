"""Detection fusion methods."""

from buse_uav.fusion.pipeline import concatenate_pre_fusion, fuse_detection_batches
from buse_uav.fusion.soft_nms import soft_nms
from buse_uav.fusion.wbf import weighted_box_fusion

__all__ = [
    "concatenate_pre_fusion",
    "fuse_detection_batches",
    "soft_nms",
    "weighted_box_fusion",
]
