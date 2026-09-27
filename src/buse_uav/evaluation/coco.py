from __future__ import annotations

import contextlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from buse_uav.schemas import DetectionBatch
from buse_uav.utils.io import atomic_write_json


class EvaluationError(RuntimeError):
    """Evaluation error safe to display at the command line."""


def coco_prediction_rows(
    batches: Sequence[DetectionBatch],
    *,
    category_id_by_class: Mapping[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        for box in batch.boxes:
            if box.class_id not in category_id_by_class:
                raise EvaluationError(f"no COCO category mapping for class {box.class_id}")
            x1, y1, x2, y2 = box.xyxy
            values = (*box.xyxy, box.score)
            if not all(math.isfinite(value) for value in values):
                raise EvaluationError(f"non-finite prediction for image {batch.image_id}")
            if x2 <= x1 or y2 <= y1:
                raise EvaluationError(f"non-positive prediction for image {batch.image_id}")
            rows.append(
                {
                    "image_id": batch.image_id,
                    "category_id": category_id_by_class[box.class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": box.score,
                }
            )
    return rows


def write_coco_predictions(
    path: Path,
    batches: Sequence[DetectionBatch],
    *,
    category_id_by_class: Mapping[int, int],
) -> list[dict[str, Any]]:
    rows = coco_prediction_rows(
        batches,
        category_id_by_class=category_id_by_class,
    )
    atomic_write_json(path, rows)
    return rows


def evaluate_coco(
    annotation_path: Path,
    prediction_path: Path,
    *,
    max_det: int = 500,
    image_ids: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO  # type: ignore[import-untyped]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EvaluationError("pycocotools is not installed") from exc
    if not annotation_path.is_file():
        raise EvaluationError(f"COCO annotation file does not exist: {annotation_path}")
    if not prediction_path.is_file():
        raise EvaluationError(f"COCO prediction file does not exist: {prediction_path}")
    try:
        raw_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(f"cannot parse COCO predictions: {exc}") from exc
    if not isinstance(raw_predictions, list):
        raise EvaluationError("COCO prediction document must be a JSON list")
    if not raw_predictions:
        raise EvaluationError("prediction file contains no detections; this run cannot produce AP")

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        ground_truth.dataset.setdefault("info", {})
        detections = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluated_image_ids = sorted(
            (
                set(image_ids)
                if image_ids is not None
                else {
                    row["image_id"]
                    for row in detections.dataset.get("annotations", [])
                    if "image_id" in row
                }
            ),
            key=str,
        )
        if not evaluated_image_ids:
            raise EvaluationError("no image IDs are available for COCO evaluation")
        evaluator.params.imgIds = evaluated_image_ids
        evaluator.params.maxDets = [1, min(100, max_det), max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        summary_buffer = io.StringIO()
        with contextlib.redirect_stdout(summary_buffer):
            evaluator.summarize()

    stats = evaluator.stats.tolist()
    metrics = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR_1": stats[6],
        "AR_100": stats[7],
        "AR_max_det": stats[8],
        "images_evaluated": len(evaluated_image_ids),
        "max_det": max_det,
        "summary": summary_buffer.getvalue(),
    }
    return metrics


def evaluate_coco_per_class(
    annotation_path: Path,
    prediction_path: Path,
    *,
    max_det: int = 500,
    image_ids: Sequence[int | str] | None = None,
) -> list[dict[str, Any]]:
    """Return class-wise COCO metrics from one evaluation pass.

    The metric definitions intentionally mirror :func:`evaluate_coco`: AP uses
    COCO's standard 100-detection cap, while AP50/AP75 and size-stratified AP
    use ``max_det``. Supplying the complete split image IDs is required for
    paper reporting so images without detections remain in the evaluation.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise EvaluationError("pycocotools is not installed") from exc
    if not annotation_path.is_file():
        raise EvaluationError(f"COCO annotation file does not exist: {annotation_path}")
    if not prediction_path.is_file():
        raise EvaluationError(f"COCO prediction file does not exist: {prediction_path}")
    try:
        raw_predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(f"cannot parse COCO predictions: {exc}") from exc
    if not isinstance(raw_predictions, list):
        raise EvaluationError("COCO prediction document must be a JSON list")
    if not raw_predictions:
        raise EvaluationError("prediction file contains no detections; this run cannot produce AP")

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        ground_truth.dataset.setdefault("info", {})
        detections = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluated_image_ids = sorted(
            (
                set(image_ids)
                if image_ids is not None
                else {
                    row["image_id"]
                    for row in detections.dataset.get("annotations", [])
                    if "image_id" in row
                }
            ),
            key=str,
        )
        if not evaluated_image_ids:
            raise EvaluationError("no image IDs are available for COCO evaluation")
        evaluator.params.imgIds = evaluated_image_ids
        evaluator.params.maxDets = [1, min(100, max_det), max_det]
        evaluator.evaluate()
        evaluator.accumulate()

    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in ground_truth.dataset.get("categories", [])
    }
    category_ids = [int(value) for value in evaluator.params.catIds]
    rows: list[dict[str, Any]] = []
    for category_index, category_id in enumerate(category_ids):
        rows.append(
            {
                "category_id": category_id,
                "category_name": categories.get(category_id, str(category_id)),
                "AP": _category_precision(evaluator, category_index, max_det=100),
                "AP50": _category_precision(
                    evaluator,
                    category_index,
                    iou_threshold=0.5,
                    max_det=max_det,
                ),
                "AP75": _category_precision(
                    evaluator,
                    category_index,
                    iou_threshold=0.75,
                    max_det=max_det,
                ),
                "AP_small": _category_precision(
                    evaluator,
                    category_index,
                    area="small",
                    max_det=max_det,
                ),
                "AP_medium": _category_precision(
                    evaluator,
                    category_index,
                    area="medium",
                    max_det=max_det,
                ),
                "AP_large": _category_precision(
                    evaluator,
                    category_index,
                    area="large",
                    max_det=max_det,
                ),
                "images_evaluated": len(evaluated_image_ids),
                "max_det": max_det,
            }
        )
    return rows


def _category_precision(
    evaluator: Any,
    category_index: int,
    *,
    iou_threshold: float | None = None,
    area: str = "all",
    max_det: int,
) -> float:
    precision = evaluator.eval["precision"]
    area_indices = [
        index for index, label in enumerate(evaluator.params.areaRngLbl) if label == area
    ]
    max_det_indices = [
        index for index, value in enumerate(evaluator.params.maxDets) if value == max_det
    ]
    if not area_indices or not max_det_indices:
        raise EvaluationError(f"COCO metric slice is unavailable: area={area}, max_det={max_det}")
    if iou_threshold is None:
        values = precision[:, :, category_index, area_indices[0], max_det_indices[0]]
    else:
        iou_indices = np.where(np.isclose(evaluator.params.iouThrs, iou_threshold))[0]
        if len(iou_indices) != 1:
            raise EvaluationError(f"COCO IoU threshold is unavailable: {iou_threshold}")
        values = precision[iou_indices[0], :, category_index, area_indices[0], max_det_indices[0]]
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0
