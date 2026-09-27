"""Accuracy, robustness, and efficiency evaluation."""

from buse_uav.evaluation.coco import (
    EvaluationError,
    coco_prediction_rows,
    evaluate_coco,
    evaluate_coco_per_class,
    write_coco_predictions,
)

__all__ = [
    "EvaluationError",
    "coco_prediction_rows",
    "evaluate_coco",
    "evaluate_coco_per_class",
    "write_coco_predictions",
]
"""Evaluation, timing, bootstrap, and paper aggregation utilities."""
