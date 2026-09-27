from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from buse_uav.schemas import Region, RegionScore
from buse_uav.scoring.degradation import DegradationScore
from buse_uav.scoring.uncertainty import UncertaintyScore


@dataclass(frozen=True)
class SelectionResult:
    regions: tuple[Region, ...]
    ranking: tuple[int, ...]
    actual_area_ratio: float
    scores_by_region: dict[int, float]


def combine_scores(
    regions: Sequence[Region],
    degradation: Sequence[DegradationScore],
    uncertainty: Sequence[UncertaintyScore],
    *,
    alpha: float,
    interaction_lambda: float,
    blank_suppression: float,
) -> tuple[RegionScore, ...]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if interaction_lambda < 0.0:
        raise ValueError("interaction_lambda must be nonnegative")
    if not 0.0 <= blank_suppression <= 1.0:
        raise ValueError("blank_suppression must be in [0, 1]")
    degradation_by_id = {int(row["region_id"]): row for row in degradation}
    uncertainty_by_id = {row.region_id: row for row in uncertainty}
    expected_ids = {region.id for region in regions}
    if set(degradation_by_id) != expected_ids or set(uncertainty_by_id) != expected_ids:
        raise ValueError("D/U scores must cover every region exactly once")

    blank_count = math.ceil(len(regions) * 0.25)
    lowest_edge_ids = {
        score.region_id
        for score in sorted(
            uncertainty,
            key=lambda value: (value.edge_density, value.region_id),
        )[:blank_count]
    }
    output: list[RegionScore] = []
    for region in regions:
        degradation_row = degradation_by_id[region.id]
        uncertainty_row = uncertainty_by_id[region.id]
        degradation_value = float(degradation_row["degradation"])
        uncertainty_value = uncertainty_row.uncertainty
        difficulty = (
            alpha * degradation_value
            + (1.0 - alpha) * uncertainty_value
            + interaction_lambda * degradation_value * uncertainty_value
        ) / (1.0 + interaction_lambda)
        blank_suppressed = uncertainty_row.candidate_count == 0 and region.id in lowest_edge_ids
        if blank_suppressed:
            difficulty *= blank_suppression
        degradation_components = degradation_row["components"]
        components = {
            **{f"d_{name}": float(value) for name, value in degradation_components.items()},
            **{f"u_{name}": float(value) for name, value in uncertainty_row.components.items()},
            "edge_density": uncertainty_row.edge_density,
            "candidate_count": float(uncertainty_row.candidate_count),
            "blank_suppressed": float(blank_suppressed),
        }
        output.append(
            RegionScore(
                region_id=region.id,
                degradation=degradation_value,
                uncertainty=uncertainty_value,
                difficulty=max(0.0, min(1.0, difficulty)),
                components=components,
            )
        )
    return tuple(output)


def select_regions(
    regions: Sequence[Region],
    scores: Sequence[RegionScore],
    *,
    selection: str,
    area_budget: float,
    max_regions: int,
    seed: int,
    image_id: int | str,
) -> SelectionResult:
    if selection not in {"random", "degradation", "uncertainty", "joint"}:
        raise ValueError(f"unsupported region selection mode: {selection}")
    if not 0.0 <= area_budget <= 1.0:
        raise ValueError("area_budget must be in [0, 1]")
    if max_regions < 0:
        raise ValueError("max_regions must be nonnegative")
    region_by_id = {region.id: region for region in regions}
    score_by_id = {score.region_id: score for score in scores}
    if len(region_by_id) != len(regions) or set(region_by_id) != set(score_by_id):
        raise ValueError("regions and scores must have unique matching IDs")

    values: dict[int, float] = {}
    for region_id, score in score_by_id.items():
        if selection == "random":
            values[region_id] = _deterministic_random(seed, image_id, region_id)
        elif selection == "degradation":
            values[region_id] = score.degradation
        elif selection == "uncertainty":
            values[region_id] = score.uncertainty
        else:
            values[region_id] = score.difficulty
    ranking = tuple(sorted(region_by_id, key=lambda region_id: (-values[region_id], region_id)))
    selected: list[Region] = []
    used_area = 0.0
    for region_id in ranking:
        if len(selected) >= max_regions:
            break
        candidate = region_by_id[region_id]
        if used_area + candidate.area_ratio <= area_budget + 1e-12:
            selected.append(candidate)
            used_area += candidate.area_ratio
    return SelectionResult(
        regions=tuple(selected),
        ranking=ranking,
        actual_area_ratio=used_area,
        scores_by_region=values,
    )


def _deterministic_random(seed: int, image_id: int | str, region_id: int) -> float:
    payload = f"{seed}|{image_id}|{region_id}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64 - 1)
