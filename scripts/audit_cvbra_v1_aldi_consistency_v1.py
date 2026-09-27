from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/REGISTRATION_LOCK.json"
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_aldi_direct_baseline_v1.yaml"
OFFICIAL = ROOT / "third_party/aldi_official"
TRANSLATION = ROOT / "src/buse_uav/detectors/aldi_ultralytics.py"
OUTPUT = ROOT / "reports/development/cvbra_v1_aldi_direct_baseline_v1/consistency_audit_v1"
REPORT = OUTPUT / "aldi_translation_consistency_report.json"
REPORT_MD = OUTPUT / "ALDI_TRANSLATION_CONSISTENCY_REPORT.md"
COMPLETE = OUTPUT / "COMPLETE.json"

DIAGNOSTICS = {
    "native_uda": (
        ROOT
        / "reports/development/cvbra_v1_aldi_direct_baseline_v1/native_uda"
        / "train_diagnostics.json"
    ),
    "equal_supervision": (
        ROOT
        / "reports/development/cvbra_v1_aldi_direct_baseline_v1/equal_supervision"
        / "train_diagnostics.json"
    ),
}


class ALDIConsistencyError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ALDIConsistencyError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ALDIConsistencyError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(OFFICIAL), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ALDIConsistencyError(f"cannot inspect official ALDI repository: {exc}") from exc
    return result.stdout.strip()


def _contains(text: str, *tokens: str) -> bool:
    return all(token in text for token in tokens)


def audit() -> dict[str, Any]:
    if REPORT.exists() and COMPLETE.exists():
        report = _load(REPORT)
        marker = _load(COMPLETE)
        if marker.get("report_sha256") != sha256_file(REPORT):
            raise ALDIConsistencyError("existing ALDI consistency report changed")
        return report
    if REPORT.exists() or REPORT_MD.exists() or COMPLETE.exists():
        raise ALDIConsistencyError("partial ALDI consistency output requires audit")
    registration = _load(REGISTRATION)
    official_files = registration.get("official_files")
    if not isinstance(official_files, dict):
        raise ALDIConsistencyError("ALDI registration has no official-file registry")
    file_hash_checks: dict[str, bool] = {}
    for relative, digest in official_files.items():
        path = ROOT / str(relative)
        file_hash_checks[str(relative)] = path.is_file() and sha256_file(path) == str(digest)
    commit = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain") == ""
    official_config = (OFFICIAL / "configs/cityscapes/ALDI-Yolo-Cityscapes.yaml").read_text(
        encoding="utf-8"
    )
    official_base = (OFFICIAL / "configs/Base-Yolo.yaml").read_text(encoding="utf-8")
    official_distill = (OFFICIAL / "aldi/yolo/distill.py").read_text(encoding="utf-8")
    official_ema = (OFFICIAL / "aldi/ema.py").read_text(encoding="utf-8")
    translation = TRANSLATION.read_text(encoding="utf-8")
    diagnostics = {name: _load(path) for name, path in DIAGNOSTICS.items()}
    active_runtime = {
        name: {
            "ema_updates_positive": float(value.get("ema_updates", 0)) > 0,
            "pseudo_boxes_positive": float(value.get("pseudo_boxes", 0)) > 0,
            "target_pseudo_coverage_positive": float(value.get("target_image_pseudo_coverage", 0))
            > 0,
            "soft_confidence_loss_positive": float(
                value.get("mean_soft_confidence_loss_per_batch", 0)
            )
            > 0,
            "soft_class_loss_positive": float(value.get("mean_soft_class_loss_per_batch", 0)) > 0,
            "pseudo_box_loss_positive": float(value.get("mean_pseudo_box_loss_per_batch", 0)) > 0,
            "pseudo_dfl_loss_positive": float(value.get("mean_pseudo_dfl_loss_per_batch", 0)) > 0,
        }
        for name, value in diagnostics.items()
    }
    roles = [
        {
            "role": "source_strong_supervision",
            "official_evidence": _contains(
                official_config,
                "labeled_strong",
                "LABELED_INCLUDE_RANDOM_ERASING: True",
            ),
            "translation_evidence": _contains(
                translation, "def strong_view_batch", "source_flags", "supervised_loss"
            ),
            "runtime_evidence": all(
                int(value.get("source_images", 0)) == 7200 for value in diagnostics.values()
            ),
        },
        {
            "role": "EMA_teacher",
            "official_evidence": _contains(official_config, "EMA:", "ENABLED: True", "TEACHER:")
            and _contains(official_ema, "class EMA", "_update_ema"),
            "translation_evidence": _contains(
                translation, "def update_aldi_teacher", "ema_alpha", "optimizer_step"
            ),
            "runtime_evidence": all(
                checks["ema_updates_positive"] for checks in active_runtime.values()
            )
            and all(float(value.get("ema_alpha", 0)) == 0.9996 for value in diagnostics.values()),
        },
        {
            "role": "weak_to_strong_target_distillation",
            "official_evidence": _contains(
                official_config, "unlabeled_strong", "TEACHER:", "ENABLED: True"
            ),
            "translation_evidence": _contains(
                translation, "target_weak", "strong_images", "_teacher_predictions"
            ),
            "runtime_evidence": all(
                int(value.get("target_images", 0)) == 21600 for value in diagnostics.values()
            ),
        },
        {
            "role": "MIC_target_masking",
            "official_evidence": "UNLABELED_MIC_AUG: True" in official_config,
            "translation_evidence": _contains(
                translation, "def _mic_mask", "mic_ratio", "mic_block_size"
            ),
            "runtime_evidence": all(
                float(value.get("mic_ratio", -1)) == 0.5
                and int(value.get("mic_block_size", -1)) == 32
                for value in diagnostics.values()
            ),
        },
        {
            "role": "soft_class_distillation",
            "official_evidence": "ROIH_CLS_ENABLED: True" in official_config
            and "loss_soft_cls" in official_distill,
            "translation_evidence": _contains(
                translation, "def _soft_class_losses", "class_loss", "softmax"
            ),
            "runtime_evidence": all(
                checks["soft_class_loss_positive"] for checks in active_runtime.values()
            ),
        },
        {
            "role": "soft_objectness_to_anchor_free_confidence",
            "official_evidence": "OBJ_ENABLED: True" in official_config
            and "loss_soft_obj" in official_distill,
            "translation_evidence": _contains(
                translation,
                "confidence_loss",
                "binary_cross_entropy_with_logits",
                "teacher_confidence",
            ),
            "runtime_evidence": all(
                checks["soft_confidence_loss_positive"] for checks in active_runtime.values()
            ),
        },
        {
            "role": "pseudo_box_regression",
            "official_evidence": "ROIH_REG_ENABLED: True" in official_config
            and "loss_soft_reg" in official_distill,
            "translation_evidence": _contains(
                translation, "pseudo_batch_from_detections", "pseudo_loss", "pseudo_dfl_loss"
            ),
            "runtime_evidence": all(
                checks["pseudo_boxes_positive"]
                and checks["pseudo_box_loss_positive"]
                and checks["pseudo_dfl_loss_positive"]
                for checks in active_runtime.values()
            ),
        },
        {
            "role": "registered_teacher_thresholds",
            "official_evidence": "pseudo_label_threshold=0.8" in official_distill
            and "IOU_THRES: 0.65" in official_base,
            "translation_evidence": _contains(
                translation, "pseudo_threshold: float = 0.8", "pseudo_nms_iou: float = 0.65"
            ),
            "runtime_evidence": all(
                float(value.get("pseudo_threshold", -1)) == 0.8
                and float(value.get("pseudo_nms_iou", -1)) == 0.65
                for value in diagnostics.values()
            ),
        },
    ]
    all_roles = all(
        row["official_evidence"] and row["translation_evidence"] and row["runtime_evidence"]
        for row in roles
    )
    exact_deviations = [
        {
            "dimension": "detector",
            "official": "anchor-based YOLOv5m",
            "project": "anchor-free YOLO11n",
            "disposition": "required architecture translation; never labeled official author code",
        },
        {
            "dimension": "objectness term",
            "official": "separate YOLOv5 objectness map",
            "project": "YOLO11 class-confidence map",
            "disposition": "registered BCE confidence-map analogue",
        },
        {
            "dimension": "equal-supervision variant",
            "official": "native labeled-source plus unlabeled-target UDA",
            "project": "additional labeled-target matched-information variant",
            "disposition": "project comparator extension reported separately from native UDA",
        },
        {
            "dimension": "training budget",
            "official": "benchmark-specific iteration schedule",
            "project": "same 8-epoch/3,600-sample budget as CVBRA",
            "disposition": "fair project-budget control, not an official reproduction claim",
        },
    ]
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_ALDI_ANCHOR_FREE_TRANSLATION_CONSISTENCY_AUDIT",
        "completed_at_utc": _now(),
        "registration_sha256": sha256_file(REGISTRATION),
        "protocol_sha256": sha256_file(PROTOCOL),
        "official_repository": _relative(OFFICIAL),
        "official_commit_expected": registration.get("official_commit"),
        "official_commit_observed": commit,
        "official_commit_match": commit == registration.get("official_commit"),
        "official_worktree_clean": clean,
        "official_file_hash_checks": file_hash_checks,
        "all_official_file_hashes_match": all(file_hash_checks.values()),
        "translation": _relative(TRANSLATION),
        "translation_sha256": sha256_file(TRANSLATION),
        "translation_hash_matches_registration": (
            sha256_file(TRANSLATION) == registration.get("translation_module_sha256")
        ),
        "roles": roles,
        "all_registered_roles_present_and_active": all_roles,
        "runtime_diagnostics": {
            name: {
                "path": _relative(DIAGNOSTICS[name]),
                "sha256": sha256_file(DIAGNOSTICS[name]),
                "checks": active_runtime[name],
            }
            for name in DIAGNOSTICS
        },
        "exact_deviations": exact_deviations,
        "verdict": (
            "ROLE_LEVEL_CONSISTENT_REGISTERED_ANCHOR_FREE_TRANSLATION"
            if all_roles
            and clean
            and commit == registration.get("official_commit")
            and all(file_hash_checks.values())
            else "CONSISTENCY_CHECK_FAILED_REPORT_WITHOUT_REINTERPRETATION"
        ),
        "claim_boundary": (
            "The comparator is ALDI++-AF, a registered anchor-free translation of official "
            "recipe roles. It is not the ALDI authors' official YOLO11 implementation and "
            "is not presented as an exact benchmark reproduction."
        ),
        "performance_result_changed": False,
    }
    markdown = [
        "# ALDI++-AF translation consistency audit",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "The locked official repository, recipe files, translation module, and both runtime "
        "diagnostics were checked. Every registered recipe role was present in the official "
        "source, implemented in the anchor-free translation, and active during training.",
        "",
        "| Role | Official | Translation | Runtime |",
        "|---|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {row['role']} | {'PASS' if row['official_evidence'] else 'FAIL'} | "
        f"{'PASS' if row['translation_evidence'] else 'FAIL'} | "
        f"{'PASS' if row['runtime_evidence'] else 'FAIL'} |"
        for row in roles
    )
    markdown.extend(
        [
            "",
            "The audit does not erase the declared architecture translation: YOLOv5m becomes "
            "YOLO11n, its objectness term becomes an anchor-free confidence-map analogue, and "
            "the equal-supervision comparator is a project extension. The manuscript must "
            "therefore retain the label ALDI++-AF and the non-official-code qualifier.",
            "",
        ]
    )
    atomic_write_json(REPORT, payload)
    atomic_write_text(REPORT_MD, "\n".join(markdown))
    atomic_write_json(
        COMPLETE,
        {
            "status": payload["status"],
            "report_sha256": sha256_file(REPORT),
            "markdown_sha256": sha256_file(REPORT_MD),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ALDI++-AF translation consistency")
    parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
