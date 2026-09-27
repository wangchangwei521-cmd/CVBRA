from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "development" / "cvbra_v1_data_synthesis"
REPORT = OUTPUT / "data_synthesis_report.json"
TABLE = OUTPUT / "cross_domain_metrics.csv"
COMPLETE = OUTPUT / "COMPLETE.json"

SOURCE_REPORTS = {
    "target_yolo": ROOT
    / "reports/development/cvbra_v1_matched_baselines/uav_obb_official_validation/evaluation"
    / "matched_point_report.json",
    "target_yolo_statistics": ROOT
    / "reports/development/cvbra_v1_matched_baselines/uav_obb_official_validation/evaluation"
    / "matched_baseline_report.json",
    "target_yolo_source_statistics": ROOT
    / "reports/development/cvbra_v1/uav_obb_official_validation/evaluation"
    / "validation_report.json",
    "target_rtdetr": ROOT
    / "reports/development/cvbra_v1_rtdetr_l/uav_obb_official_validation/evaluation"
    / "point_report.json",
    "target_rtdetr_statistics": ROOT
    / "reports/development/cvbra_v1_rtdetr_l/uav_obb_official_validation/evaluation"
    / "validation_report.json",
    "hazydet": ROOT
    / "reports/development/cvbra_v1_matched_baselines/hazydet_source_retention_v4"
    / "source_retention_report.json",
    "auair": ROOT / "reports/external/cvbra_v1_auair/external_report.json",
    "dronevehicle": ROOT
    / "reports/external/cvbra_v1_dronevehicle_validation/validation_report.json",
    "latency": ROOT
    / "reports/development/cvbra_v1_interleaved_latency_audit/latency_report.json",
    "training_seeds": ROOT
    / "reports/development/cvbra_v1_training_seed_robustness/seed_robustness_report.json",
    "training_seed_amendment": ROOT
    / "reports/development/cvbra_v1_training_seed_robustness"
    / "INEFFECTIVE_SEED_PERTURBATION_AMENDMENT_1.json",
    "training_order_v4": ROOT
    / "reports/development/cvbra_v1_training_order_robustness_v4"
    / "effective_order_robustness_v4_report.json",
}

METRIC_KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


class SynthesisError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SynthesisError(f"required locked report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SynthesisError(f"expected a JSON object: {path}")
    return value


def _mapping_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SynthesisError("report rows are missing or malformed")
    return rows


def _find_row(rows: Sequence[Mapping[str, Any]], **criteria: object) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(field) == expected for field, expected in criteria.items())
    ]
    if len(matches) != 1:
        raise SynthesisError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def _find_stat(rows: object, comparison: str) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise SynthesisError("statistics rows are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("comparison") == comparison
    ]
    if len(matches) != 1:
        raise SynthesisError(f"expected one statistic for {comparison}, found {len(matches)}")
    return matches[0]


def _ap(row: Mapping[str, Any]) -> float:
    value = row.get("AP")
    if not isinstance(value, (int, float)):
        raise SynthesisError("metric row does not contain numeric AP")
    return float(value)


def _metric_row(
    row: Mapping[str, Any],
    *,
    domain: str,
    detector: str,
    scope: str,
    view: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "domain": domain,
        "detector": detector,
        "scope": scope,
        "view": view,
        "model": row.get("model"),
        "seed": row.get("seed"),
        "endpoint": row.get("endpoint"),
        "images_evaluated": row.get("images_evaluated"),
    }
    for key in METRIC_KEYS:
        result[key] = row.get(key)
    return result


def _write_table(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "domain",
        "detector",
        "scope",
        "view",
        "model",
        "seed",
        "endpoint",
        *METRIC_KEYS,
        "images_evaluated",
    )
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    atomic_write_text(TABLE, buffer.getvalue())


def _source_manifest() -> dict[str, dict[str, Any]]:
    return {
        label: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for label, path in SOURCE_REPORTS.items()
    }


def build_synthesis() -> dict[str, Any]:
    reports = {label: _load_json(path) for label, path in SOURCE_REPORTS.items()}
    yolo_rows = _mapping_rows(reports["target_yolo"])
    rtdetr_rows = _mapping_rows(reports["target_rtdetr"])
    hazy_rows = _mapping_rows(reports["hazydet"])
    auair_rows = _mapping_rows(reports["auair"])
    drone_rows = _mapping_rows(reports["dronevehicle"])
    seed_rows = _mapping_rows(reports["training_seeds"])
    order_rows = _mapping_rows(reports["training_order_v4"])

    metrics: list[dict[str, Any]] = []
    for model in (
        "source",
        "CVBRA_v1",
        "STF",
        "CVBRA_noCV",
        "CVBRA_noReplay",
        "CVBRA_noFreeze",
    ):
        for view in ("original", "fog_1p0"):
            row = _find_row(
                yolo_rows,
                model=model,
                method="Identity",
                view=view,
                scope="decontaminated_primary",
            )
            metrics.append(
                _metric_row(
                    row,
                    domain="UAV-OBB",
                    detector="YOLO11n",
                    scope="decontaminated official validation (post-freeze)",
                    view=view,
                )
            )
    for model in ("source", "CVBRA_v1_RTDETR_L"):
        for view in ("original", "fog_1p0"):
            row = _find_row(
                rtdetr_rows,
                model=model,
                method="Identity",
                view=view,
                scope="decontaminated_primary",
            )
            metrics.append(
                _metric_row(
                    row,
                    domain="UAV-OBB",
                    detector="RT-DETR-L",
                    scope="decontaminated official validation (post-freeze)",
                    view=view,
                )
            )
    for row in hazy_rows:
        metrics.append(
            _metric_row(
                row,
                domain="HazyDet",
                detector="YOLO11n",
                scope="spent 1000-image validation (source-retention description)",
                view="hazy",
            )
        )
    for row in auair_rows:
        metrics.append(
            _metric_row(
                row,
                domain="AU-AIR",
                detector="YOLO11n",
                scope="external descriptive evaluation",
                view="original",
            )
        )
    for row in drone_rows:
        metrics.append(
            _metric_row(
                row,
                domain="DroneVehicle",
                detector="YOLO11n",
                scope="validation only; test historically consumed",
                view="adaptive",
            )
        )
    for row in seed_rows:
        domain = str(row.get("domain"))
        metrics.append(
            _metric_row(
                row,
                domain=domain,
                detector="YOLO11n",
                scope="post-freeze training-seed variability audit",
                view=domain.removeprefix("target_") if domain.startswith("target_") else "hazy",
            )
        )
    for row in order_rows:
        domain = str(row.get("domain"))
        metrics.append(
            _metric_row(
                row,
                domain=domain,
                detector="YOLO11n",
                scope="post-freeze effective-order robustness audit",
                view=domain.removeprefix("target_") if domain.startswith("target_") else "hazy",
            )
        )
    metrics.sort(
        key=lambda row: (
            str(row["domain"]),
            str(row["detector"]),
            str(row["model"]),
            str(row["view"]),
            str(row["seed"]),
            str(row["endpoint"]),
        )
    )

    def yolo(model: str, view: str, method: str = "Identity") -> Mapping[str, Any]:
        return _find_row(
            yolo_rows,
            model=model,
            method=method,
            view=view,
            scope="decontaminated_primary",
        )

    def rtdetr(model: str, view: str) -> Mapping[str, Any]:
        return _find_row(
            rtdetr_rows,
            model=model,
            method="Identity",
            view=view,
            scope="decontaminated_primary",
        )

    def hazy(model: str) -> Mapping[str, Any]:
        return _find_row(hazy_rows, model=model)

    target_source_stats = reports["target_yolo_source_statistics"].get("primary_statistics")
    target_ablation_stats = reports["target_yolo_statistics"].get("primary_statistics")
    latency_summary = reports["latency"].get("summary")
    seed_summary = reports["training_seeds"].get("summary")
    order_metric_summary = reports["training_order_v4"].get("metric_summary")
    if (
        not isinstance(latency_summary, dict)
        or not isinstance(seed_summary, dict)
        or not isinstance(order_metric_summary, dict)
    ):
        raise SynthesisError("latency or training-variability summary is malformed")
    contrasts = latency_summary.get("contrasts")
    if not isinstance(contrasts, dict):
        raise SynthesisError("latency contrasts are missing")
    cvbra_latency = contrasts.get("CVBRA_v1_over_source")
    if not isinstance(cvbra_latency, dict):
        raise SynthesisError("CVBRA/source latency contrast is missing")
    median_latency_ratio = float(cvbra_latency["median_cycle_ratio"])

    comparisons = {
        "target_YOLO11n_CVBRA_minus_source": {
            "original_AP": _ap(yolo("CVBRA_v1", "original"))
            - _ap(yolo("source", "original")),
            "fog_1p0_AP": _ap(yolo("CVBRA_v1", "fog_1p0"))
            - _ap(yolo("source", "fog_1p0")),
            "statistics": {
                "original": _find_stat(target_source_stats, "Identity_original"),
                "fog_1p0": _find_stat(target_source_stats, "Identity_primary_fog"),
            },
        },
        "target_YOLO11n_CVBRA_identity_minus_source_flip": {
            "original_AP": _ap(yolo("CVBRA_v1", "original"))
            - _ap(yolo("source", "original", "Flip_HardNMS")),
            "fog_1p0_AP": _ap(yolo("CVBRA_v1", "fog_1p0"))
            - _ap(yolo("source", "fog_1p0", "Flip_HardNMS")),
            "interpretation": (
                "one-pass adapted model versus two-pass source Flip on identical images"
            ),
        },
        "target_RTDETR_L_CVBRA_minus_source": {
            "original_AP": _ap(rtdetr("CVBRA_v1_RTDETR_L", "original"))
            - _ap(rtdetr("source", "original")),
            "fog_1p0_AP": _ap(rtdetr("CVBRA_v1_RTDETR_L", "fog_1p0"))
            - _ap(rtdetr("source", "fog_1p0")),
            "registered_statistics": reports["target_rtdetr_statistics"].get(
                "primary_statistics"
            ),
        },
        "HazyDet_CVBRA_minus_source": _ap(hazy("CVBRA_v1")) - _ap(hazy("source")),
        "HazyDet_CVBRA_minus_STF": _ap(hazy("CVBRA_v1")) - _ap(hazy("STF")),
        "AU-AIR": {
            "AP_deltas": reports["auair"].get("AP_deltas"),
            "paired_statistics": reports["auair"].get("paired_statistics"),
        },
        "DroneVehicle_validation": {
            "AP_deltas": reports["dronevehicle"].get("AP_deltas"),
            "paired_statistics": reports["dronevehicle"].get("paired_statistics"),
        },
    }

    order_checks = order_metric_summary.get("registered_metric_checks")
    if not isinstance(order_checks, dict):
        raise SynthesisError("registered effective-order checks are missing")
    target_yolo_gain = comparisons["target_YOLO11n_CVBRA_minus_source"]
    target_rtdetr_gain = comparisons["target_RTDETR_L_CVBRA_minus_source"]
    assert isinstance(target_yolo_gain, dict)
    assert isinstance(target_rtdetr_gain, dict)
    direction_checks = {
        "registered_effective_order_checks_pass": bool(
            reports["training_order_v4"].get("all_registered_checks_pass")
        ),
        "YOLO11n_target_original_and_fog_gains_positive": (
            float(target_yolo_gain["original_AP"]) > 0
            and float(target_yolo_gain["fog_1p0_AP"]) > 0
        ),
        "RTDETR_L_target_original_and_fog_gains_positive": (
            float(target_rtdetr_gain["original_AP"]) > 0
            and float(target_rtdetr_gain["fog_1p0_AP"]) > 0
        ),
        "interleaved_median_latency_ratio_at_most_1p15": median_latency_ratio <= 1.15,
    }
    freeze_supported = all(direction_checks.values())

    _write_table(metrics)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_CROSS_DOMAIN_DATA_SYNTHESIS",
        "source_reports": _source_manifest(),
        "cross_domain_metrics": TABLE.relative_to(ROOT).as_posix(),
        "cross_domain_metrics_sha256": sha256_file(TABLE),
        "comparisons": comparisons,
        "mechanism_evidence": {
            "visibility_curriculum": {
                "target_original_CVBRA_minus_noCV": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noCV_Identity_original"
                ),
                "target_fog1_CVBRA_minus_noCV": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noCV_Identity_fog_1p0"
                ),
                "HazyDet_CVBRA_minus_noCV": _find_stat(
                    reports["hazydet"].get("paired_statistics"),
                    "CVBRA_v1_minus_CVBRA_noCV",
                ),
            },
            "source_replay": {
                "target_original_CVBRA_minus_noReplay": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noReplay_Identity_original"
                ),
                "target_fog1_CVBRA_minus_noReplay": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noReplay_Identity_fog_1p0"
                ),
                "HazyDet_CVBRA_minus_noReplay": _find_stat(
                    reports["hazydet"].get("paired_statistics"),
                    "CVBRA_v1_minus_CVBRA_noReplay",
                ),
            },
            "frozen_early_layers": {
                "target_original_CVBRA_minus_noFreeze": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noFreeze_Identity_original"
                ),
                "target_fog1_CVBRA_minus_noFreeze": _find_stat(
                    target_ablation_stats, "CVBRA_minus_CVBRA_noFreeze_Identity_fog_1p0"
                ),
                "HazyDet_CVBRA_minus_noFreeze": _find_stat(
                    reports["hazydet"].get("paired_statistics"),
                    "CVBRA_v1_minus_CVBRA_noFreeze",
                ),
            },
            "ordinary_fine_tuning": {
                "target_original_CVBRA_minus_STF": _find_stat(
                    target_ablation_stats, "CVBRA_minus_STF_Identity_original"
                ),
                "target_fog1_CVBRA_minus_STF": _find_stat(
                    target_ablation_stats, "CVBRA_minus_STF_Identity_fog_1p0"
                ),
                "HazyDet_CVBRA_minus_STF": _find_stat(
                    reports["hazydet"].get("paired_statistics"), "CVBRA_v1_minus_STF"
                ),
            },
        },
        "latency": {
            "interleaved_CVBRA_over_source_median_ratio": median_latency_ratio,
            "registered_feasibility_ceiling": 1.5,
            "all_cycles_within_equivalence_band": cvbra_latency.get(
                "all_cycles_within_equivalence_band"
            ),
            "cold_cycle_retained_in_report": True,
            "full_summary": latency_summary,
        },
        "ineffective_exposed_seed_audit": {
            "numerical_summary": seed_summary,
            "amendment": reports["training_seed_amendment"],
            "stochastic_seed_variability_claim_authorized": False,
            "bitwise_reproducibility_claim_authorized": True,
        },
        "effective_order_robustness": reports["training_order_v4"],
        "direction_checks": direction_checks,
        "decision": (
            "FREEZE_CVBRA_V1_AS_MAIN_DATA_DIRECTION"
            if freeze_supported
            else "HOLD_CVBRA_V1_DIRECTION_AND_REPORT_FAILED_CHECKS"
        ),
        "ranking_policy": {
            "arbitrary_composite_score_used": False,
            "basis": (
                "domain-wise AP, paired uncertainty, seed variability, and latency are "
                "retained separately"
            ),
        },
        "evidence_boundary": {
            "test_content_accessed_by_synthesis": False,
            "new_model_or_endpoint_selection": False,
            "post_freeze_validation_reuse": True,
            "training_variability_audits_are_independent_confirmation": False,
            "ineffective_seed_audit_not_used_as_variability_evidence": True,
            "DroneVehicle_test_historically_consumed_no_new_test_access": True,
            "paper_body_modified": False,
        },
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "data_synthesis_report_sha256": sha256_file(REPORT),
            "cross_domain_metrics_sha256": sha256_file(TABLE),
        },
    )
    return payload


def main() -> int:
    result = build_synthesis()
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "report": REPORT.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
