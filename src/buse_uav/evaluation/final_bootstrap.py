from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buse_uav.data.corruptions import REQUIRED_CORRUPTIONS
from buse_uav.evaluation.aggregate import AggregateError, validate_run
from buse_uav.evaluation.bootstrap import paired_coco_ap_bootstrap
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text

FINAL_CONFIGURATION_NAMES = (
    "baseline",
    "full",
    "a6_hard_nms",
    "global_gamma",
    "global_clahe",
    "global_unsharp",
    "flip_tta",
    "random_identity_crop",
    "identity_crop",
    "full_candidate_bank",
    "all_identity_grid",
)
BOOTSTRAP_BASELINE_CONFIGURATION = "baseline"
BOOTSTRAP_METHOD_CONFIGURATION = "full"
EXPECTED_FINAL_CELLS = len(FINAL_CONFIGURATION_NAMES) * 19
EXPECTED_BOOTSTRAP_COMPARISONS = 19
_GROUP_PREFIX = "main_visdrone_frozen_"


@dataclass(frozen=True)
class FinalBootstrapPair:
    corruption: str
    severity: int
    baseline_run_id: str
    baseline_run_path: Path
    method_run_id: str
    method_run_path: Path

    @property
    def key(self) -> str:
        return f"{self.corruption}:{self.severity}"

    @property
    def group(self) -> str:
        return f"{_GROUP_PREFIX}{self.corruption}_l{self.severity}"


def load_final_bootstrap_pairs(path: Path) -> list[FinalBootstrapPair]:
    """Require the exact 209-cell matrix and return the 19 B0/Full pairs."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot parse final VisDrone manifest {path}: {exc}") from exc
    runs = document.get("runs") if isinstance(document, dict) else None
    if not isinstance(runs, list):
        raise ValueError("final VisDrone manifest has no run list")
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in runs:
        if not isinstance(raw, dict):
            raise ValueError("final VisDrone manifest contains a non-object row")
        try:
            configuration = str(raw["configuration"])
            corruption = str(raw["corruption"])
            severity = int(raw["severity"])
            run_id = str(raw["run_id"])
            run_path = str(raw["run_path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("final VisDrone manifest contains an invalid row") from exc
        key = (configuration, corruption, severity)
        if key in by_key:
            raise ValueError(f"final VisDrone manifest contains a duplicate cell: {key}")
        by_key[key] = {**raw, "run_id": run_id, "run_path": run_path}

    expected = {(configuration, "clean", 0) for configuration in FINAL_CONFIGURATION_NAMES} | {
        (configuration, corruption, severity)
        for configuration in FINAL_CONFIGURATION_NAMES
        for corruption in REQUIRED_CORRUPTIONS
        for severity in (1, 2, 3)
    }
    present = set(by_key)
    if len(runs) != EXPECTED_FINAL_CELLS or present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            "final VisDrone bootstrap requires the exact 209-cell matrix: "
            f"rows={len(runs)}, missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:1]}"
        )

    cell_order = [("clean", 0)] + [
        (corruption, severity) for corruption in REQUIRED_CORRUPTIONS for severity in (1, 2, 3)
    ]
    pairs: list[FinalBootstrapPair] = []
    for corruption, severity in cell_order:
        baseline = by_key[(BOOTSTRAP_BASELINE_CONFIGURATION, corruption, severity)]
        method = by_key[(BOOTSTRAP_METHOD_CONFIGURATION, corruption, severity)]
        pairs.append(
            FinalBootstrapPair(
                corruption=corruption,
                severity=severity,
                baseline_run_id=str(baseline["run_id"]),
                baseline_run_path=Path(str(baseline["run_path"])),
                method_run_id=str(method["run_id"]),
                method_run_path=Path(str(method["run_path"])),
            )
        )
    return pairs


def run_visdrone_final_bootstrap(
    manifest_path: Path,
    *,
    state_path: Path,
    output_path: Path,
    combined_path: Path,
    resamples: int = 1000,
    seed: int = 42,
    workers: int = 4,
    progress_path: Path | None = None,
    chunk_resamples: int = 5,
) -> dict[str, Any]:
    """Run or resume the frozen Full-vs-B0 image-paired COCO AP bootstrap."""
    if resamples != 1000:
        raise ValueError("final paper bootstrap requires exactly 1,000 resamples")
    if workers <= 0:
        raise ValueError("bootstrap workers must be positive")
    if chunk_resamples <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    pairs = load_final_bootstrap_pairs(manifest_path)
    identity: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "visdrone_final_paired_coco_ap",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "baseline_configuration": BOOTSTRAP_BASELINE_CONFIGURATION,
        "method_configuration": BOOTSTRAP_METHOD_CONFIGURATION,
        "resamples": resamples,
        "seed": seed,
    }
    state = _load_or_initialize_state(state_path, identity)
    results = state["results"]
    if not isinstance(results, list):
        raise ValueError("VisDrone bootstrap state has no result list")
    completed = {str(row["key"]): row for row in results if isinstance(row, dict)}
    if progress_path is not None:
        _clear_checkpoint_for_completed_pair(progress_path, completed)
    for pair in pairs:
        if pair.key in completed:
            continue
        baseline = validate_run(pair.baseline_run_path)
        method = validate_run(pair.method_run_path)
        _validate_pair(pair, baseline, method)
        image_ids = [
            row["image_id"]
            for row in method.data_manifest["images"]
            if isinstance(row, dict) and "image_id" in row
        ]
        if len(image_ids) != int(method.metrics["images_evaluated"]):
            raise AggregateError(f"bootstrap image ID coverage mismatch for {pair.key}")
        max_det = int(method.config["detector"]["max_det"])
        result = paired_coco_ap_bootstrap(
            _bootstrap_annotation(method.path, method.data_manifest),
            baseline.path / "predictions" / "final.coco.json",
            method.path / "predictions" / "final.coco.json",
            resamples=resamples,
            seed=seed,
            max_det=max_det,
            image_ids=image_ids,
            workers=workers,
            checkpoint_path=progress_path,
            checkpoint_identity={
                "manifest_sha256": identity["manifest_sha256"],
                "comparison_key": pair.key,
                "baseline_run_id": pair.baseline_run_id,
                "method_run_id": pair.method_run_id,
            },
            chunk_resamples=chunk_resamples,
        )
        expected_delta = float(method.metrics["AP"]) - float(baseline.metrics["AP"])
        if not math.isclose(float(result["delta"]), expected_delta, abs_tol=1e-9):
            raise AggregateError(
                f"bootstrap observed AP delta disagrees with frozen metrics for {pair.key}"
            )
        row = {
            "key": pair.key,
            "group": pair.group,
            "dataset": "visdrone",
            "split": "test-dev",
            "corruption": pair.corruption,
            "severity": pair.severity,
            "baseline_configuration": BOOTSTRAP_BASELINE_CONFIGURATION,
            "configuration": BOOTSTRAP_METHOD_CONFIGURATION,
            "baseline_run_id": pair.baseline_run_id,
            "method_run_id": pair.method_run_id,
            "method": "buse",
            "metric": "AP",
            "observed_baseline_AP": float(baseline.metrics["AP"]),
            "observed_method_AP": float(method.metrics["AP"]),
            **result,
            "algorithm_version": 1,
            "max_det": max_det,
            "cache_hit": False,
        }
        results.append(row)
        completed[pair.key] = row
        atomic_write_json(state_path, state)
        _write_csv(output_path, _ordered_results(results, pairs))
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
    ordered = _ordered_results(results, pairs)
    if len(ordered) != EXPECTED_BOOTSTRAP_COMPARISONS:
        raise RuntimeError("VisDrone final bootstrap stopped before all comparisons completed")
    _write_csv(output_path, ordered)
    merge_bootstrap_tables(combined_path, ordered)
    return {
        "comparisons": len(ordered),
        "resamples": resamples,
        "seed": seed,
        "chunk_resamples": chunk_resamples,
        "output": str(output_path.resolve()),
        "combined": str(combined_path.resolve()),
    }


def merge_bootstrap_tables(path: Path, visdrone_rows: Sequence[Mapping[str, Any]]) -> None:
    """Preserve non-VisDrone CIs and atomically replace the frozen VisDrone block."""
    existing: list[dict[str, Any]] = []
    if path.is_file():
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                existing = [dict(row) for row in csv.DictReader(stream)]
        except (csv.Error, OSError) as exc:
            raise ValueError(f"cannot read existing bootstrap table {path}: {exc}") from exc
    retained = [row for row in existing if not str(row.get("group", "")).startswith(_GROUP_PREFIX)]
    merged = retained + [dict(row) for row in visdrone_rows]
    _write_csv(path, merged)


def _validate_pair(pair: FinalBootstrapPair, baseline: Any, method: Any) -> None:
    if baseline.run_id != pair.baseline_run_id or method.run_id != pair.method_run_id:
        raise AggregateError(f"bootstrap run ID provenance mismatch for {pair.key}")
    if baseline.config["method"]["name"] != "baseline" or method.config["method"]["name"] != "buse":
        raise AggregateError(f"bootstrap method identity mismatch for {pair.key}")
    if (
        baseline.config["dataset"]["split"] != "test-dev"
        or method.config["dataset"]["split"] != "test-dev"
    ):
        raise AggregateError(f"bootstrap split mismatch for {pair.key}")
    if bool(baseline.config["runtime"]["tune"]) or bool(method.config["runtime"]["tune"]):
        raise AggregateError(f"bootstrap pair contains forbidden tuning for {pair.key}")
    if baseline.model_fingerprint["sha256"] != method.model_fingerprint["sha256"]:
        raise AggregateError(f"bootstrap pair mixes detector weights for {pair.key}")
    if stable_hash(baseline.data_manifest, length=64) != stable_hash(
        method.data_manifest, length=64
    ):
        raise AggregateError(f"bootstrap pair mixes data manifests for {pair.key}")


def _bootstrap_annotation(run_path: Path, data_manifest: Mapping[str, Any]) -> Path:
    annotation = data_manifest.get("annotation")
    path = Path(str(annotation)) if annotation is not None else run_path / "ground_truth.coco.json"
    if not path.is_file():
        raise AggregateError(f"bootstrap ground truth is unavailable: {path}")
    return path


def _load_or_initialize_state(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        state = {**identity, "results": []}
        atomic_write_json(path, state)
        return state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot parse VisDrone bootstrap state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("VisDrone bootstrap state is not an object")
    for key, value in identity.items():
        if state.get(key) != value:
            raise ValueError(f"VisDrone bootstrap state identity mismatch for {key}")
    return state


def _clear_checkpoint_for_completed_pair(
    path: Path, completed: Mapping[str, Mapping[str, Any]]
) -> None:
    if not path.is_file():
        return
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot parse VisDrone bootstrap progress {path}: {exc}") from exc
    identity = checkpoint.get("identity") if isinstance(checkpoint, dict) else None
    scope = identity.get("scope") if isinstance(identity, dict) else None
    key = scope.get("comparison_key") if isinstance(scope, dict) else None
    if isinstance(key, str) and key in completed:
        path.unlink()


def _ordered_results(
    rows: Sequence[Mapping[str, Any]], pairs: Sequence[FinalBootstrapPair]
) -> list[dict[str, Any]]:
    by_key = {str(row["key"]): dict(row) for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("VisDrone bootstrap state contains duplicate comparison keys")
    return [by_key[pair.key] for pair in pairs if pair.key in by_key]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "group",
        "dataset",
        "split",
        "corruption",
        "severity",
        "baseline_configuration",
        "configuration",
        "baseline_run_id",
        "method_run_id",
        "method",
        "metric",
        "delta",
        "ci_low",
        "ci_high",
        "resamples",
        "seed",
        "images",
    ]
    fieldnames = [field for field in preferred if field in fields]
    fieldnames.extend(field for field in fields if field not in fieldnames)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    atomic_write_text(path, stream.getvalue())
