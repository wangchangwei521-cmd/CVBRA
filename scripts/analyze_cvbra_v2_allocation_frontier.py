from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
ALLOCATION = (
    ROOT
    / "reports/development/cvbra_v1_allocation_sensitivity_v1"
    / "allocation_sensitivity_report.json"
)
REPLICATION = (
    ROOT
    / "reports/development/cvbra_v2_reviewer_closure_v1"
    / "replication_report.json"
)
SOURCE_POINTS = (
    ROOT
    / "reports/development/cvbra_v1_aldi_direct_baseline_v1"
    / "direct_comparison_analysis/point_report.json"
)
OUTPUT = (
    ROOT
    / "reports/development/cvbra_v2_reviewer_closure_v1"
    / "allocation_frontier_report.json"
)

VIEWS = ("original", "fog_0p6", "fog_1p0")


class AllocationFrontierError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AllocationFrontierError(f"expected JSON object: {path}")
    return value


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    coordinates = (*VIEWS, "HazyDet")
    return all(left[key] >= right[key] for key in coordinates) and any(
        left[key] > right[key] for key in coordinates
    )


def _allocation_points(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    summaries = report["summaries"]
    cells = {
        "qS_0_L10": ("replay_allocation", "qS_0"),
        "qS_0p125_L10": ("replay_allocation", "qS_0p125"),
        "qS_0p25_L0": ("freeze_boundary", "first_trainable_0"),
        "qS_0p25_L5": ("freeze_boundary", "first_trainable_5"),
        "qS_0p25_L10": ("freeze_boundary", "first_trainable_10"),
        "qS_0p25_L15": ("freeze_boundary", "first_trainable_15"),
    }
    output: dict[str, dict[str, float]] = {}
    for label, (axis, cell) in cells.items():
        values = summaries[axis]["cells"][cell]["AP"]
        output[label] = {key: float(values[key]) for key in (*VIEWS, "HazyDet")}
    return output


def _nondominated(points: dict[str, dict[str, float]]) -> list[str]:
    return sorted(
        label
        for label, values in points.items()
        if not any(
            other != label and _dominates(other_values, values)
            for other, other_values in points.items()
        )
    )


def _selection_intervals(
    points: dict[str, dict[str, float]], candidates: list[str]
) -> list[dict[str, Any]]:
    retention_values = sorted({points[label]["HazyDet"] for label in candidates})
    intervals: list[dict[str, Any]] = []
    lower: float | None = None
    for upper in retention_values:
        eligible = [
            label for label in candidates if points[label]["HazyDet"] >= upper
        ]
        selected = max(
            eligible,
            key=lambda label: (
                min(points[label][view] for view in VIEWS),
                points[label]["HazyDet"],
                label,
            ),
        )
        row = {
            "retention_floor_lower_exclusive": lower,
            "retention_floor_upper_inclusive": upper,
            "selected": selected,
            "worst_view_target_AP": min(
                points[selected][view] for view in VIEWS
            ),
            "source_retention_AP": points[selected]["HazyDet"],
        }
        if intervals and intervals[-1]["selected"] == selected:
            intervals[-1]["retention_floor_upper_inclusive"] = upper
        else:
            intervals.append(row)
        lower = upper
    return intervals


def _source_point(report: dict[str, Any]) -> dict[str, float]:
    target = {
        str(row["view"]): float(row["AP"])
        for row in report["target_rows"]
        if row.get("model") == "source"
        and row.get("scope") == "decontaminated_primary"
    }
    hazy = next(
        float(row["AP"])
        for row in report["hazydet_rows"]
        if row.get("model") == "source"
    )
    if set(target) != set(VIEWS):
        raise AllocationFrontierError("source target point is incomplete")
    return {**target, "HazyDet": hazy}


def _trajectory_checks(replication: dict[str, Any]) -> dict[str, Any]:
    lookup = {
        (
            int(row["boundary"]),
            int(row["seed"]),
            str(row["dataset"]),
            str(row["view"]),
        ): float(row["AP"])
        for row in replication["rows"]
    }
    output: dict[str, Any] = {}
    for seed in replication["seeds"]:
        points: dict[int, dict[str, float]] = {}
        for boundary in (5, 10):
            points[boundary] = {
                view: lookup[
                    (boundary, int(seed), "UAV_OBB_validation", view)
                ]
                for view in VIEWS
            }
            points[boundary]["HazyDet"] = lookup[
                (boundary, int(seed), "HazyDet_validation", "hazy")
            ]
        output[str(seed)] = {
            "L5_dominates_L10": _dominates(points[5], points[10]),
            "L5_minus_L10": {
                key: points[5][key] - points[10][key]
                for key in (*VIEWS, "HazyDet")
            },
            "L5_worst_view_target_AP": min(points[5][view] for view in VIEWS),
            "L10_worst_view_target_AP": min(points[10][view] for view in VIEWS),
        }
    return output


def analyze() -> dict[str, Any]:
    allocation = _load(ALLOCATION)
    replication = _load(REPLICATION)
    source_report = _load(SOURCE_POINTS)
    points = _allocation_points(allocation)
    nondominated = _nondominated(points)
    intervals = _selection_intervals(points, nondominated)
    contextual = {**points, "source_only": _source_point(source_report)}
    contextual_nondominated = _nondominated(contextual)
    contextual_intervals = _selection_intervals(
        contextual, contextual_nondominated
    )
    trajectories = _trajectory_checks(replication)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V2_ALLOCATION_FRONTIER_ANALYSIS",
        "inputs": {
            "allocation_report_sha256": sha256_file(ALLOCATION),
            "replication_report_sha256": sha256_file(REPLICATION),
            "source_point_report_sha256": sha256_file(SOURCE_POINTS),
        },
        "coordinates": [*VIEWS, "HazyDet"],
        "dominance_rule": "all_coordinates_no_smaller_and_at_least_one_larger",
        "allocation_points": points,
        "nondominated_allocations": nondominated,
        "retention_constrained_maximin_intervals": intervals,
        "context_including_source_only": {
            "nondominated": contextual_nondominated,
            "retention_constrained_maximin_intervals": contextual_intervals,
        },
        "independent_trajectory_checks": trajectories,
        "all_three_trajectories_L5_dominate_L10": all(
            row["L5_dominates_L10"] for row in trajectories.values()
        ),
        "test_results_used": False,
    }
    atomic_write_json(OUTPUT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the CVBRA allocation frontier")
    parser.parse_args()
    result = analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
