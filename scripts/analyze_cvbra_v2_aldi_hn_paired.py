from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scripts import analyze_cvbra_v1_uav_obb_official_validation as target_analysis
from scripts import run_cvbra_v1_hazydet_source_retention_v3 as hazydet_reference
from scripts import run_cvbra_v1_training_seed_robustness as evidence_base
from scripts import run_cvbra_v2_aldi_horizon_normalized_v1 as aldi_hn
from scripts import run_cvbra_v2_reviewer_closure_v1 as closure

from buse_uav.evaluation.bootstrap import (
    ClusterBootstrapScope,
    paired_coco_ap_cluster_bootstrap_scopes,
)
from buse_uav.evaluation.supplementary import bootstrap_sign_pvalue, holm_adjust
from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v2_aldi_hn_paired_analysis_v1.yaml"
OUTPUT = (
    ROOT
    / "reports/development/cvbra_v2_aldi_horizon_normalized_v1"
    / "paired_comparison"
)
REGISTRATION = OUTPUT / "REGISTRATION_LOCK.json"
REPORT = OUTPUT / "paired_statistics.json"
COMPLETE = OUTPUT / "COMPLETE.json"

TARGET_RESAMPLES = 10_000
HAZY_RESAMPLES = 2_000
SEED = 20260822
VIEWS = ("original", "fog_0p6", "fog_1p0")
BOUNDARIES = (10, 5)
MAX_DET = 500


class ALDIHNPairedError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ALDIHNPairedError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _target_annotation() -> Path:
    conversion = _load(evidence_base.CONVERSION_LOCK)
    path = ROOT / str(conversion["annotation"])
    if not path.is_file():
        raise ALDIHNPairedError(f"target annotation is missing: {path}")
    return path


def register() -> dict[str, Any]:
    inputs = (
        PROTOCOL,
        aldi_hn.REGISTRATION,
        closure.REGISTRATION,
        evidence_base.CONVERSION_LOCK,
        _target_annotation(),
        evidence_base.HAZY_ANNOTATION,
    )
    for path in inputs:
        if not path.is_file():
            raise ALDIHNPairedError(f"registration input is missing: {path}")
    payload = {
        "schema_version": 1,
        "status": "ALDI_HN_PAIRED_ANALYSIS_REGISTERED",
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "aldi_hn_registration_sha256": sha256_file(aldi_hn.REGISTRATION),
        "cvbra_replication_registration_sha256": sha256_file(closure.REGISTRATION),
        "target_annotation_sha256": sha256_file(_target_annotation()),
        "hazydet_annotation_sha256": sha256_file(evidence_base.HAZY_ANNOTATION),
        "target_resamples": TARGET_RESAMPLES,
        "hazydet_resamples": HAZY_RESAMPLES,
        "seed": SEED,
        "test_predictions_used": False,
        "method_or_hyperparameter_selection": False,
    }
    if REGISTRATION.exists():
        existing = _load(REGISTRATION)
        stable = tuple(key for key in payload if key not in {"schema_version", "status"})
        if any(existing.get(key) != payload.get(key) for key in stable):
            raise ALDIHNPairedError("paired-analysis registration changed")
        return existing
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REGISTRATION, payload)
    return payload


def _checkpoint_deltas(path: Path, expected: int) -> list[float]:
    document = _load(path)
    values = document.get("deltas")
    if not isinstance(values, list) or len(values) != expected:
        raise ALDIHNPairedError(f"bootstrap checkpoint is incomplete: {path}")
    deltas = [float(value) for value in values]
    if not all(math.isfinite(value) for value in deltas):
        raise ALDIHNPairedError("bootstrap checkpoint has nonfinite values")
    return deltas


def _row(
    *,
    name: str,
    annotation: Path,
    baseline: Path,
    method: Path,
    clusters: dict[str, tuple[int, ...]],
    resamples: int,
    checkpoint: Path,
) -> dict[str, Any]:
    result = paired_coco_ap_cluster_bootstrap_scopes(
        annotation,
        baseline,
        method,
        {
            "analysis": ClusterBootstrapScope(
                clusters=clusters,
                checkpoint_path=checkpoint,
                checkpoint_identity={
                    "study": "cvbra_v2_aldi_hn_paired_analysis_v1",
                    "comparison": name,
                    "registration_sha256": sha256_file(REGISTRATION),
                    "test_predictions_used": False,
                },
            )
        },
        resamples=resamples,
        seed=SEED,
        max_det=MAX_DET,
        workers=4,
        chunk_resamples=100,
        accelerate_ap_only=True,
    )["analysis"]
    deltas = _checkpoint_deltas(checkpoint, resamples)
    deviation = stdev(deltas)
    if deviation <= 0 or not math.isfinite(deviation):
        raise ALDIHNPairedError(f"bootstrap variance failed: {name}")
    return {
        "comparison": name,
        **result,
        "bootstrap_mean_delta": fmean(deltas),
        "bootstrap_standard_deviation": deviation,
        "standardized_effect": float(result["delta"]) / deviation,
        "p_two_sided": bootstrap_sign_pvalue(deltas),
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def analyze() -> dict[str, Any]:
    register()
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        if _load(COMPLETE).get("report_sha256") != sha256_file(REPORT):
            raise ALDIHNPairedError("paired-analysis report changed")
        return report
    for path in (aldi_hn.COMPLETE, closure.COMPLETE):
        if not path.is_file():
            raise ALDIHNPairedError(f"required completed evaluation is missing: {path}")
    target_annotation = _target_annotation()
    target_clusters = target_analysis._clusters(target_annotation)
    target_rows = []
    for boundary in BOUNDARIES:
        for view in VIEWS:
            target_rows.append(
                _row(
                    name=f"CVBRA_L{boundary}_minus_ALDI_HN_{view}",
                    annotation=target_annotation,
                    baseline=aldi_hn._prediction("UAV_OBB_validation", view),
                    method=closure._prediction(
                        boundary, 42, "UAV_OBB_validation", view
                    ),
                    clusters=target_clusters,
                    resamples=TARGET_RESAMPLES,
                    checkpoint=OUTPUT
                    / "target_bootstrap"
                    / f"L{boundary}_{view}.json",
                )
            )
    target_adjusted = holm_adjust(
        [float(row["p_two_sided"]) for row in target_rows]
    )
    for row, adjusted in zip(target_rows, target_adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted
        row["holm_significant_0p05"] = adjusted < 0.05

    hazy_ids = [
        int(record.image_id)
        for record in evidence_base._hazy_records(verify_hashes=False)
    ]
    hazy_clusters = hazydet_reference.singleton_image_clusters(hazy_ids)
    hazy_rows = []
    for boundary in BOUNDARIES:
        hazy_rows.append(
            _row(
                name=f"CVBRA_L{boundary}_minus_ALDI_HN_HazyDet",
                annotation=evidence_base.HAZY_ANNOTATION,
                baseline=aldi_hn._prediction("HazyDet_validation", "hazy"),
                method=closure._prediction(
                    boundary, 42, "HazyDet_validation", "hazy"
                ),
                clusters=hazy_clusters,
                resamples=HAZY_RESAMPLES,
                checkpoint=OUTPUT
                / "hazydet_bootstrap"
                / f"L{boundary}_hazy.json",
            )
        )
    hazy_adjusted = holm_adjust([float(row["p_two_sided"]) for row in hazy_rows])
    for row, adjusted in zip(hazy_rows, hazy_adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted
        row["holm_significant_0p05"] = adjusted < 0.05

    payload = {
        "schema_version": 1,
        "status": "COMPLETE_ALDI_HN_PAIRED_ANALYSIS",
        "registration_sha256": sha256_file(REGISTRATION),
        "target": {
            "images": 167,
            "groups": 141,
            "resamples": TARGET_RESAMPLES,
            "multiple_testing": "Holm across six CVBRA-minus-ALDI-HN contrasts",
            "rows": target_rows,
        },
        "hazydet": {
            "images": 1000,
            "resamples": HAZY_RESAMPLES,
            "multiple_testing": "Holm across two CVBRA-minus-ALDI-HN contrasts",
            "rows": hazy_rows,
        },
        "test_predictions_used": False,
        "negative_results_retained": True,
    }
    atomic_write_json(REPORT, payload)
    atomic_write_json(
        COMPLETE,
        {"status": payload["status"], "report_sha256": sha256_file(REPORT)},
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired CVBRA versus ALDI-HN analysis")
    parser.add_argument("--stage", choices=("register", "analyze"), default="analyze")
    args = parser.parse_args()
    result = register() if args.stage == "register" else analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
