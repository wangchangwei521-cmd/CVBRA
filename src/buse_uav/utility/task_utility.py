from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from buse_uav.schemas import Box, UtilityResult, UtilityWeightsConfig
from buse_uav.utility.matching import BoxMatch, hungarian_match

EPSILON = 1e-12


@dataclass(frozen=True)
class CandidateEvaluation:
    operation: str
    predictions: tuple[Box, ...]
    utility: UtilityResult
    num_reference: int
    num_predictions: int
    num_matches: int
    early_stopped: bool = False


@dataclass(frozen=True)
class CandidateChoice:
    operation: str
    predictions: tuple[Box, ...]
    q: float


def identity_evaluation(
    predictions: Sequence[Box],
    *,
    num_reference: int,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        operation="identity",
        predictions=tuple(predictions),
        utility=UtilityResult(
            q=0.0,
            confidence_gain=0.0,
            stability=0.0,
            rescue=0.0,
            unsupported_fp=0.0,
            count_explosion=0.0,
            compute=0.0,
        ),
        num_reference=num_reference,
        num_predictions=len(predictions),
        num_matches=0,
    )


def evaluate_candidate(
    reference: Sequence[Box],
    predictions: Sequence[Box],
    *,
    publish_conf: float,
    match_iou: float,
    max_count_ratio: float,
    crop_imgsz: int,
    full_imgsz: int,
    weights: UtilityWeightsConfig,
) -> CandidateEvaluation:
    if not 0.0 < publish_conf <= 1.0:
        raise ValueError("publish_conf must be in (0, 1]")
    if max_count_ratio <= 0.0 or min(crop_imgsz, full_imgsz) <= 0:
        raise ValueError("utility scale values must be positive")
    matches = hungarian_match(reference, predictions, iou_threshold=match_iou)
    components = _utility_components(
        reference,
        predictions,
        matches,
        publish_conf=publish_conf,
        max_count_ratio=max_count_ratio,
        crop_imgsz=crop_imgsz,
        full_imgsz=full_imgsz,
    )
    q = (
        weights.confidence_gain * components["confidence_gain"]
        + weights.stability * components["stability"]
        + weights.rescue * components["rescue"]
        - weights.unsupported_fp * components["unsupported_fp"]
        - weights.count_explosion * components["count_explosion"]
        - weights.compute * components["compute"]
    )
    result = UtilityResult(q=float(q), **components)
    return CandidateEvaluation(
        operation="candidate",
        predictions=tuple(predictions),
        utility=result,
        num_reference=len(reference),
        num_predictions=len(predictions),
        num_matches=len(matches),
    )


def with_operation(
    evaluation: CandidateEvaluation,
    operation: str,
) -> CandidateEvaluation:
    return replace(evaluation, operation=operation)


def choose_best_candidate(
    identity: CandidateEvaluation,
    candidates: Sequence[CandidateEvaluation],
) -> CandidateChoice:
    if identity.operation != "identity" or identity.utility.q != 0.0:
        raise ValueError("identity candidate must have operation=identity and Q=0")
    best = identity
    for candidate in candidates:
        if candidate.utility.q > best.utility.q:
            best = candidate
    return CandidateChoice(
        operation=best.operation,
        predictions=best.predictions,
        q=best.utility.q,
    )


def should_early_stop(
    result: UtilityResult,
    *,
    q_threshold: float,
    min_stability: float,
    max_unsupported_fp: float = 0.0,
) -> bool:
    if not 0.0 <= max_unsupported_fp <= 1.0:
        raise ValueError("max_unsupported_fp must be in [0, 1]")
    return (
        result.q >= q_threshold
        and result.unsupported_fp <= max_unsupported_fp + EPSILON
        and result.stability >= min_stability
    )


def _utility_components(
    reference: Sequence[Box],
    predictions: Sequence[Box],
    matches: Sequence[BoxMatch],
    *,
    publish_conf: float,
    max_count_ratio: float,
    crop_imgsz: int,
    full_imgsz: int,
) -> dict[str, float]:
    if matches:
        confidence_gain = sum(
            predictions[match.prediction_index].score - reference[match.reference_index].score
            for match in matches
        ) / len(matches)
        mean_iou = sum(match.iou for match in matches) / len(matches)
    else:
        confidence_gain = 0.0
        mean_iou = 0.0
    stability = (len(matches) / (len(reference) + EPSILON)) * mean_iou
    rescue = sum(
        predictions[match.prediction_index].score
        for match in matches
        if reference[match.reference_index].score
        < publish_conf
        <= predictions[match.prediction_index].score
    ) / (len(reference) + EPSILON)
    matched_predictions = {match.prediction_index for match in matches}
    unsupported_fp = sum(
        max(0.0, prediction.score - publish_conf)
        for index, prediction in enumerate(predictions)
        if index not in matched_predictions
    ) / (len(predictions) + EPSILON)
    count_explosion = max(
        0.0,
        len(predictions) / (len(reference) + 1.0) - max_count_ratio,
    )
    compute = (crop_imgsz / full_imgsz) ** 2
    return {
        "confidence_gain": _clip01(confidence_gain),
        "stability": _clip01(stability),
        "rescue": _clip01(rescue),
        "unsupported_fp": _clip01(unsupported_fp),
        "count_explosion": _clip01(count_explosion),
        "compute": _clip01(compute),
    }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
