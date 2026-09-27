from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import run_cvbra_v1_training_order_robustness_v2 as engine
from scripts import run_cvbra_v1_training_seed_robustness as seed_base

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiment/cvbra_v1_training_order_robustness_v4.yaml"
PROTOCOL_SHA256 = "ade56987b5dc0e0988a9e430a9ad44d675972b323a24ed1cd025319f982257b5"
DATA_MANIFEST = ROOT / "data/processed/cvbra_v1_order_robustness_v4/manifest.json"
DATA_MANIFEST_SHA256 = "970629ffa092f3b883e751dd5b51853e6fcbf9867c5fe0bd49597f6ac62b9176"
GENERATOR = ROOT / "scripts/prepare_cvbra_v1_order_robustness_v4.py"
GENERATOR_SHA256 = "f9036caed5fc0eacb4ce8f192398fa31e98795ee082c87e68410cfa5b7edc6f3"
ENGINE = ROOT / "scripts/run_cvbra_v1_training_order_robustness_v2.py"
ENGINE_SHA256 = "9c05ac9df70d5a0ee1f53eec86b46e3ba636ef7502ccc37a2e45b2cab011bd01"
SEED_AMENDMENT = (
    ROOT
    / "reports/development/cvbra_v1_training_seed_robustness"
    / "INEFFECTIVE_SEED_PERTURBATION_AMENDMENT_1.json"
)
SEED_AMENDMENT_SHA256 = "8561c63d404d5337e96b65d9a6f1c5605dbed7c345ba943b0dc035281beeb2da"
V2_AMENDMENT = (
    ROOT
    / "reports/development/cvbra_v1_training_order_robustness_v2"
    / "INEFFECTIVE_TEXT_ORDER_AMENDMENT_1.json"
)
V2_AMENDMENT_SHA256 = "cec935aca8a6ca73f7fe2c91d4372694b93417cb9fcbb78377e3a55c43205c65"

ORDER_CONFIGS = {
    "hash_a": {
        "yaml": ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_a/dataset.yaml",
        "yaml_sha256": "bf8b52fb2882edb566496735717c50565db360f22b3a86e3bca7dd838cad301c",
        "mapping": ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_a/mapping.json",
        "mapping_sha256": "b7b458e6504900bc017a3801f305b0d949d9368788511adb96dbe4e19c6eb1f9",
    },
    "hash_b": {
        "yaml": ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_b/dataset.yaml",
        "yaml_sha256": "bfad260bd8ab368a06ba278f3d4ca951b8602a6ec65d173b20a60e9dda7d94c2",
        "mapping": ROOT / "data/processed/cvbra_v1_order_robustness_v4/hash_b/mapping.json",
        "mapping_sha256": "ca00201dd7396504ae923261bb1e64a4fc821e3bcce5254a39c46fc7fa589769",
    },
}
ORDERS = tuple(ORDER_CONFIGS)

OUTPUT = ROOT / "reports/development/cvbra_v1_training_order_robustness_v4"
RUN_ROOT = ROOT / "runs/cvbra_v1_training_order_robustness_v4"
IMPLEMENTATION_LOCK = OUTPUT / "implementation_lock.json"
FINAL_REPORT = OUTPUT / "effective_order_robustness_v4_report.json"
FINAL_COMPLETE = OUTPUT / "V4_COMPLETE.json"


class EffectiveOrderV4Error(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EffectiveOrderV4Error(f"cannot parse mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EffectiveOrderV4Error(f"expected JSON object: {path}")
    return value


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise EffectiveOrderV4Error(f"locked {label} changed: {path}")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _configure_engine() -> None:
    engine.PROTOCOL = PROTOCOL
    engine.PROTOCOL_SHA256 = PROTOCOL_SHA256
    engine.ORDER_MANIFEST = DATA_MANIFEST
    engine.ORDER_MANIFEST_SHA256 = DATA_MANIFEST_SHA256
    engine.ORDER_CONFIGS = {
        order: {
            "yaml": config["yaml"],
            "yaml_sha256": config["yaml_sha256"],
            "list": config["mapping"],
            "list_sha256": config["mapping_sha256"],
        }
        for order, config in ORDER_CONFIGS.items()
    }
    engine.ORDERS = ORDERS
    engine.ENDPOINTS = ("primary_locked", *ORDERS)
    engine.OUTPUT = OUTPUT
    engine.RUN_ROOT = RUN_ROOT
    engine.IMPLEMENTATION_LOCK = IMPLEMENTATION_LOCK
    engine.PREDICTION_LOCK = OUTPUT / "joint_prediction_lock.json"
    engine.PREDICTION_MARKER = OUTPUT / "PREDICTIONS_LOCKED"
    engine.METRICS = OUTPUT / "order_metrics.csv"
    engine.REPORT = OUTPUT / "order_robustness_engine_report.json"
    engine.COMPLETE = OUTPUT / "ENGINE_COMPLETE.json"
    engine.preflight = preflight


def _validate_implementation_lock() -> dict[str, Any]:
    lock = _load_mapping(IMPLEMENTATION_LOCK)
    if (
        lock.get("protocol_sha256") != PROTOCOL_SHA256
        or lock.get("wrapper_sha256") != sha256_file(Path(__file__).resolve())
        or lock.get("execution_engine_sha256") != ENGINE_SHA256
        or lock.get("data_manifest_sha256") != DATA_MANIFEST_SHA256
    ):
        raise EffectiveOrderV4Error("v4 implementation lock changed")
    return lock


def preflight() -> dict[str, Any]:
    seed_base.preflight()
    for path, expected, label in (
        (PROTOCOL, PROTOCOL_SHA256, "v4 protocol"),
        (DATA_MANIFEST, DATA_MANIFEST_SHA256, "v4 data manifest"),
        (GENERATOR, GENERATOR_SHA256, "v4 generator"),
        (ENGINE, ENGINE_SHA256, "execution engine"),
        (SEED_AMENDMENT, SEED_AMENDMENT_SHA256, "seed amendment"),
        (V2_AMENDMENT, V2_AMENDMENT_SHA256, "v2 amendment"),
        (engine.PRIMARY_CHECKPOINT, engine.PRIMARY_CHECKPOINT_SHA256, "primary checkpoint"),
    ):
        _assert_hash(path, expected, label=label)
    manifest = _load_mapping(DATA_MANIFEST)
    if (
        manifest.get("images_per_order") != 3600
        or manifest.get("same_training_multiset") is not True
        or manifest.get("image_or_label_bytes_modified") is not False
        or manifest.get("backend_sorted_order_verified_before_training") is not True
    ):
        raise EffectiveOrderV4Error("v4 materialization contract is invalid")
    for order, config in ORDER_CONFIGS.items():
        yaml_path = config["yaml"]
        mapping_path = config["mapping"]
        assert isinstance(yaml_path, Path)
        assert isinstance(mapping_path, Path)
        _assert_hash(yaml_path, str(config["yaml_sha256"]), label=f"{order} YAML")
        _assert_hash(mapping_path, str(config["mapping_sha256"]), label=f"{order} mapping")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, list) or len(mapping) != 3600:
            raise EffectiveOrderV4Error(f"{order} mapping length changed")
    target, primary_ids = seed_base._target_records(verify_hashes=True)
    hazy = seed_base._hazy_records(verify_hashes=True)
    if len(target["original"]) != 218 or len(primary_ids) != 167 or len(hazy) != 1000:
        raise EffectiveOrderV4Error("locked evaluation records changed")
    if IMPLEMENTATION_LOCK.exists():
        return _validate_implementation_lock()
    if any(
        path.exists()
        for path in (
            RUN_ROOT,
            engine.PREDICTION_LOCK,
            engine.METRICS,
            engine.REPORT,
            FINAL_REPORT,
        )
    ):
        raise EffectiveOrderV4Error("v4 outputs appeared before implementation lock")
    payload = {
        "schema_version": 1,
        "status": "CVBRA_V1_EFFECTIVE_ORDER_ROBUSTNESS_V4_IMPLEMENTATION_LOCKED",
        "locked_at_utc": _utc_now(),
        "protocol": _relative(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "wrapper": _relative(Path(__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "execution_engine": _relative(ENGINE),
        "execution_engine_sha256": ENGINE_SHA256,
        "generator_sha256": GENERATOR_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "orders": list(ORDERS),
        "test_content_accessed": False,
        "paper_body_change_authorized": False,
    }
    atomic_write_json(IMPLEMENTATION_LOCK, payload)
    return payload


def registered_checks_pass(engine_report: dict[str, Any]) -> bool:
    metric_summary = engine_report.get("metric_summary")
    state_diversity = engine_report.get("state_diversity")
    if not isinstance(metric_summary, dict) or not isinstance(state_diversity, dict):
        raise EffectiveOrderV4Error("engine report is missing registered checks")
    return bool(metric_summary.get("all_registered_metric_checks_pass")) and bool(
        state_diversity.get("all_registered_effective_perturbation_checks_pass")
    )


def finalize(engine_report: dict[str, Any]) -> dict[str, Any]:
    if FINAL_REPORT.exists() and FINAL_COMPLETE.exists():
        report = _load_mapping(FINAL_REPORT)
        marker = _load_mapping(FINAL_COMPLETE)
        if marker.get("effective_order_v4_report_sha256") != sha256_file(FINAL_REPORT):
            raise EffectiveOrderV4Error("v4 final report changed")
        return report
    if any(path.exists() for path in (FINAL_REPORT, FINAL_COMPLETE)):
        raise EffectiveOrderV4Error("partial v4 final report requires audit")
    if not engine.REPORT.is_file():
        raise EffectiveOrderV4Error("execution-engine report is missing")
    checks_pass = registered_checks_pass(engine_report)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_CVBRA_V1_EFFECTIVE_ORDER_ROBUSTNESS_V4_AUDIT",
        "completed_at_utc": _utc_now(),
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "execution_engine_report": _relative(engine.REPORT),
        "execution_engine_report_sha256": sha256_file(engine.REPORT),
        "rows": engine_report.get("rows"),
        "baselines": engine_report.get("baselines"),
        "metric_summary": engine_report.get("metric_summary"),
        "state_diversity": engine_report.get("state_diversity"),
        "all_registered_checks_pass": checks_pass,
        "decision": (
            "EFFECTIVE_ORDER_ROBUSTNESS_SUPPORTED"
            if checks_pass
            else "REPORT_FAILED_CHECKS_WITHOUT_METHOD_OR_ENDPOINT_RESELECTION"
        ),
        "superseded_audits": {
            "exposed_seed": _relative(SEED_AMENDMENT),
            "text_list_order": _relative(V2_AMENDMENT),
        },
        "evidence_boundary": {
            "effective_training_order_perturbation_verified": bool(
                engine_report.get("state_diversity", {}).get(
                    "all_registered_effective_perturbation_checks_pass"
                )
            ),
            "same_training_multiset": True,
            "post_freeze_validation_reuse": True,
            "independent_confirmation_claim": False,
            "method_or_endpoint_reselection": False,
            "test_content_accessed": False,
            "paper_body_modified": False,
        },
    }
    atomic_write_json(FINAL_REPORT, payload)
    atomic_write_json(
        FINAL_COMPLETE,
        {
            "status": payload["status"],
            "effective_order_v4_report_sha256": sha256_file(FINAL_REPORT),
        },
    )
    return payload


def main() -> int:
    _configure_engine()
    parser = argparse.ArgumentParser(description="Run effective CVBRA-v1 order robustness v4")
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "infer", "score", "all"),
        default="all",
    )
    parser.add_argument("--order", choices=ORDERS)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "train":
        result = engine.train_order(args.order) if args.order else engine.train()
    elif args.stage == "infer":
        result = engine.infer()
    elif args.stage == "score":
        result = finalize(engine.score())
    else:
        engine.train()
        result = finalize(engine.score())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
