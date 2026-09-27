from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from buse_uav.evaluation.ablation import AblationRegistryError, build_labeled_ablation_rows
from buse_uav.evaluation.bootstrap import paired_coco_ap_bootstrap
from buse_uav.evaluation.timing import pareto_mask, read_timing_traces, timing_summary
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text


class AggregateError(RuntimeError):
    """Aggregation error safe to display at the command line."""


_PRESERVED_BOOTSTRAP_GROUP_PREFIXES = ("main_visdrone_frozen_",)


@dataclass(frozen=True)
class ValidatedRun:
    path: Path
    run_id: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    data_manifest: dict[str, Any]
    model_fingerprint: dict[str, Any]
    timing_rows: list[dict[str, Any]]


def aggregate_plan(plan_path: Path, *, output: Path = Path("reports")) -> dict[str, Any]:
    """Validate paper runs, aggregate deterministic tables, and mark Pareto rows."""
    plan = _load_yaml_mapping(plan_path)
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise AggregateError("aggregation plan requires a nonempty groups list")
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cached_bootstrap_rows = _read_csv_rows(tables / "bootstrap_ci.csv")
    all_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, str]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    accepted_runs: list[ValidatedRun] = []
    for raw_group in groups:
        if not isinstance(raw_group, dict):
            raise AggregateError("each aggregation group must be a mapping")
        group_name = str(raw_group.get("name", "")).strip()
        if not group_name:
            raise AggregateError("each aggregation group requires a name")
        run_paths, explicit = _resolve_group_runs(raw_group)
        group_runs: list[ValidatedRun] = []
        for run_path in run_paths:
            try:
                group_runs.append(validate_run(run_path))
            except (AggregateError, OSError, ValueError) as exc:
                if explicit:
                    raise AggregateError(f"explicit run {run_path} is invalid: {exc}") from exc
                rejected_rows.append(
                    {
                        "group": group_name,
                        "run_id": run_path.name,
                        "reason": str(exc),
                    }
                )
        if not group_runs:
            raise AggregateError(f"group {group_name!r} contains no valid runs")
        _validate_fairness(group_name, group_runs)
        group_rows = [
            _run_row(group_name, run, repeat=index + 1)
            for index, run in enumerate(sorted(group_runs, key=lambda item: item.run_id))
        ]
        frontier = pareto_mask(group_rows)
        for row, is_pareto in zip(group_rows, frontier, strict=True):
            row["pareto"] = is_pareto
        _add_repeat_statistics(group_rows)
        table_name = raw_group.get("table")
        if table_name is not None:
            safe_name = Path(str(table_name)).name
            if not safe_name.endswith(".csv"):
                raise AggregateError(f"group table must be a CSV filename: {table_name}")
            _write_csv(tables / safe_name, group_rows)
        all_rows.extend(group_rows)
        accepted_runs.extend(group_runs)
        bootstrap_rows.extend(
            _bootstrap_group(
                raw_group,
                group_name,
                group_runs,
                resamples=int(plan.get("bootstrap_resamples", 1000)),
                seed=int(plan.get("bootstrap_seed", 42)),
                workers=int(plan.get("bootstrap_workers", 1)),
                cached_rows=cached_bootstrap_rows,
            )
        )
    all_rows.sort(key=lambda row: (str(row["group"]), str(row["method"]), str(row["run_id"])))
    bootstrap_rows.extend(
        _preserved_bootstrap_rows(cached_bootstrap_rows, generated_rows=bootstrap_rows)
    )
    bootstrap_rows.sort(key=lambda row: (str(row["group"]), str(row["method_run_id"])))
    efficiency_rows = [
        row for row in all_rows if row.get("dataset") == "hazydet" and row.get("split") == "val"
    ]
    _write_csv(tables / "all_runs.csv", all_rows)
    _write_csv(tables / "efficiency.csv", efficiency_rows)
    _write_csv(tables / "bootstrap_ci.csv", bootstrap_rows, empty_fields=_BOOTSTRAP_FIELDS)
    _write_csv(
        tables / "rejected_runs.csv",
        rejected_rows,
        empty_fields=("group", "run_id", "reason"),
    )
    labeled_table_metadata = _write_labeled_tables(
        plan.get("labeled_tables"),
        accepted_rows=all_rows,
        tables=tables,
    )
    _write_latex(tables / "efficiency.tex", efficiency_rows)
    pareto_source_rows = _pareto_source_rows(all_rows)
    for suffix in ("png", "svg", "pdf"):
        _write_pareto_figure(figures / f"pareto.{suffix}", pareto_source_rows)
    _write_csv(figures / "pareto_source.csv", pareto_source_rows)
    table_artifacts = {
        path.name: _artifact(path)
        for path in sorted(tables.iterdir())
        if path.is_file() and path.suffix in {".csv", ".tex"}
    }
    figure_artifacts = _figure_artifacts(figures)
    catalog_artifacts = _declared_artifact_catalog(
        plan.get("artifact_catalog"),
        plan_path=plan_path,
    )
    manifest = {
        "schema_version": 1,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "accepted_run_ids": [run.run_id for run in accepted_runs],
        "rejected_runs": rejected_rows,
        "labeled_tables": labeled_table_metadata,
        "tables": table_artifacts,
        "figures": figure_artifacts,
        **catalog_artifacts,
    }
    atomic_write_json(output / "paper_manifest.json", manifest)
    return {
        "runs": len(all_rows),
        "groups": len(groups),
        "rejected": len(rejected_rows),
        "bootstrap_comparisons": len(bootstrap_rows),
        "labeled_tables": len(labeled_table_metadata),
        "catalog_artifacts": sum(len(section) for section in catalog_artifacts.values()),
        "output": str(output.resolve()),
    }


def _write_labeled_tables(
    raw_tables: Any,
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    tables: Path,
) -> list[dict[str, Any]]:
    if raw_tables is None:
        return []
    if not isinstance(raw_tables, list):
        raise AggregateError("labeled_tables must be a list")
    metadata: list[dict[str, Any]] = []
    for raw in raw_tables:
        if not isinstance(raw, dict):
            raise AggregateError("each labeled table must be a mapping")
        table_name = Path(str(raw.get("table", ""))).name
        registry = Path(str(raw.get("registry", "")))
        if not table_name.endswith(".csv") or not registry.is_file():
            raise AggregateError("labeled table requires a CSV name and existing registry")
        try:
            rows = build_labeled_ablation_rows(registry, accepted_rows)
        except AblationRegistryError as exc:
            raise AggregateError(f"cannot build labeled table {table_name}: {exc}") from exc
        _write_csv(tables / table_name, rows)
        metadata.append(
            {
                "table": table_name,
                "rows": len(rows),
                "registry": str(registry.resolve()),
                "registry_sha256": sha256_file(registry),
            }
        )
    return metadata


def validate_run(run_path: Path) -> ValidatedRun:
    path = run_path.resolve()
    if not path.is_dir():
        raise AggregateError(f"run directory does not exist: {path}")
    if not (path / "SUCCESS").is_file():
        raise AggregateError("run has no SUCCESS marker")
    required = (
        "config_resolved.yaml",
        "data_manifest.json",
        "model_fingerprint.json",
        "metrics.json",
        "metrics_provenance.json",
        "predictions/final.coco.json",
    )
    missing = [relative for relative in required if not (path / relative).is_file()]
    if missing:
        raise AggregateError(f"run is missing required artifacts: {missing}")
    config = _load_yaml_mapping(path / "config_resolved.yaml")
    for section in ("dataset", "detector", "method", "experiment", "runtime"):
        if not isinstance(config.get(section), dict):
            raise AggregateError(f"configuration lacks required {section} section")
    required_config_fields = {
        "dataset": ("name", "split"),
        "detector": ("name", "max_det"),
        "method": ("name",),
        "experiment": ("name",),
        "runtime": ("tune",),
    }
    for section, fields in required_config_fields.items():
        missing_fields = [field for field in fields if field not in config[section]]
        if missing_fields:
            raise AggregateError(
                f"configuration section {section} lacks critical fields: {missing_fields}"
            )
    split = str(config["dataset"].get("split", "")).casefold()
    test_like = split in {"test", "test-dev", "rddts"} or split.startswith("test-")
    if test_like and bool(config["runtime"].get("tune", False)):
        raise AggregateError("test/RDDTS run has forbidden runtime.tune=true")
    metrics_path = path / "metrics.json"
    metrics = _load_mapping(metrics_path)
    provenance = _load_mapping(path / "metrics_provenance.json")
    expected_hashes = {
        "metrics_sha256": sha256_file(metrics_path),
        "prediction_sha256": sha256_file(path / "predictions" / "final.coco.json"),
        "config_sha256": sha256_file(path / "config_resolved.yaml"),
    }
    for name, expected in expected_hashes.items():
        if provenance.get(name) != expected:
            raise AggregateError(f"metrics provenance mismatch for {name}")
    data_manifest = _load_mapping(path / "data_manifest.json")
    images = data_manifest.get("images")
    if not isinstance(images, list) or not images:
        raise AggregateError("data manifest has no image entries")
    _validate_annotation_provenance(path, data_manifest, provenance)
    model_fingerprint = _load_mapping(path / "model_fingerprint.json")
    model_sha = model_fingerprint.get("sha256")
    if not isinstance(model_sha, str) or len(model_sha) != 64:
        raise AggregateError("model fingerprint has no valid SHA256")
    timing_rows = read_timing_traces(path / "traces")
    if len(timing_rows) != len(images):
        raise AggregateError(
            f"timing coverage mismatch: rows={len(timing_rows)}, images={len(images)}"
        )
    if {str(row.get("run_id")) for row in timing_rows} != {path.name}:
        raise AggregateError("timing rows do not all identify their containing run")
    if int(metrics.get("images_evaluated", -1)) != len(images):
        raise AggregateError("metric image count does not match the data manifest")
    return ValidatedRun(
        path=path,
        run_id=path.name,
        config=config,
        metrics=metrics,
        data_manifest=data_manifest,
        model_fingerprint=model_fingerprint,
        timing_rows=timing_rows,
    )


def _validate_fairness(group_name: str, runs: Sequence[ValidatedRun]) -> None:
    model_hashes = {str(run.model_fingerprint["sha256"]) for run in runs}
    if len(model_hashes) != 1:
        raise AggregateError(f"fairness group {group_name!r} mixes different model weight hashes")
    manifest_hashes = {stable_hash(run.data_manifest, length=64) for run in runs}
    if len(manifest_hashes) != 1:
        raise AggregateError(f"fairness group {group_name!r} mixes different data manifests")


def _run_row(
    group_name: str,
    run: ValidatedRun,
    *,
    repeat: int,
) -> dict[str, Any]:
    summary = timing_summary(run.timing_rows)
    config = run.config
    regions = _selection_statistics(run)
    candidate = _candidate_statistics(run)
    metrics = {
        key: value
        for key, value in run.metrics.items()
        if key != "summary" and isinstance(value, (int, float))
    }
    latency = summary["latency_mean_ms"]
    ap = float(metrics.get("AP", 0.0))
    base_ap = _optional_number(run.metrics.get("baseline_AP"))
    delta_ap = ap - base_ap if base_ap is not None else 0.0
    baseline_latency = _optional_number(run.metrics.get("baseline_latency_ms"))
    extra_latency = latency - (baseline_latency if baseline_latency is not None else latency)
    return {
        "group": group_name,
        "run_id": run.run_id,
        "repeat": repeat,
        "dataset": str(config["dataset"].get("name", "")),
        "split": str(config["dataset"].get("split", "")),
        "corruption_name": str(config["dataset"].get("corruption_name") or "clean"),
        "corruption_severity": config["dataset"].get("corruption_severity") or 0,
        "method": str(config["method"].get("name", "")),
        "detector": str(config["detector"].get("name", "")),
        "config_fingerprint": stable_hash(config, length=64),
        "model_sha256": str(run.model_fingerprint["sha256"]),
        "data_manifest_sha256": stable_hash(run.data_manifest, length=64),
        **metrics,
        **summary,
        **regions,
        **candidate,
        "extra_ms_per_ap": (extra_latency / delta_ap if delta_ap > 0.0 else math.nan),
        "extra_eic_per_ap": (
            (summary["eic_mean"] - 1.0) / delta_ap if delta_ap > 0.0 else math.nan
        ),
    }


def _validate_annotation_provenance(
    run_path: Path,
    data_manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    annotation_value = data_manifest.get("annotation")
    if annotation_value is not None:
        annotation = Path(str(annotation_value))
        if not annotation.is_file():
            raise AggregateError(f"data annotation is unavailable: {annotation}")
        annotation_sha256 = sha256_file(annotation)
        if data_manifest.get("annotation_sha256") != annotation_sha256:
            raise AggregateError("data annotation hash no longer matches the run manifest")
        if provenance.get("annotation_sha256") != annotation_sha256:
            raise AggregateError("metrics provenance mismatch for annotation_sha256")
        return
    annotations = data_manifest.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise AggregateError("data manifest has no annotation provenance")
    for row in annotations:
        if not isinstance(row, dict):
            raise AggregateError("data annotation manifest row is not a mapping")
        annotation = Path(str(row.get("path", "")))
        if not annotation.is_file():
            raise AggregateError(f"data annotation is unavailable: {annotation}")
        if row.get("sha256") != sha256_file(annotation):
            raise AggregateError("data annotation hash no longer matches the run manifest")
    ground_truth = run_path / "ground_truth.coco.json"
    if not ground_truth.is_file():
        raise AggregateError(f"generated COCO ground truth is unavailable: {ground_truth}")
    if provenance.get("annotation_sha256") != sha256_file(ground_truth):
        raise AggregateError("metrics provenance mismatch for annotation_sha256")


def _selection_statistics(run: ValidatedRun) -> dict[str, float]:
    path = run.path / "traces" / "selection_summary.json"
    if not path.is_file():
        return {"area_ratio_mean": 0.0}
    raw = _load_mapping(path)
    images = raw.get("images")
    if not isinstance(images, list):
        return {"area_ratio_mean": 0.0}
    values = [
        float(item["actual_area_ratio"])
        for item in images
        if isinstance(item, dict) and "actual_area_ratio" in item
    ]
    return {"area_ratio_mean": sum(values) / len(values) if values else 0.0}


def _add_repeat_statistics(rows: Sequence[dict[str, Any]]) -> None:
    by_method: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["detector"]),
            str(row["method"]),
            str(row["config_fingerprint"]),
        )
        by_method.setdefault(key, []).append(row)
    for method_rows in by_method.values():
        latency = [float(row["latency_mean_ms"]) for row in method_rows]
        accuracy = [float(row["AP"]) for row in method_rows]
        eic = [float(row["eic_mean"]) for row in method_rows]
        for row in method_rows:
            row.update(
                {
                    "repeat_count": len(method_rows),
                    "repeat_latency_mean_ms": fmean(latency),
                    "repeat_latency_std_ms": pstdev(latency),
                    "repeat_AP_mean": fmean(accuracy),
                    "repeat_AP_std": pstdev(accuracy),
                    "repeat_eic_mean": fmean(eic),
                    "repeat_run_ids": ";".join(sorted(str(item["run_id"]) for item in method_rows)),
                }
            )


def _candidate_statistics(run: ValidatedRun) -> dict[str, float]:
    path = run.path / "traces" / "candidates.jsonl"
    if not path.is_file():
        return {"early_stop_rate": 0.0}
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    enhanced = [row for row in rows if row.get("operation") != "identity"]
    if not enhanced:
        return {"early_stop_rate": 0.0}
    stopped = sum(bool(row.get("early_stopped", False)) for row in enhanced)
    return {"early_stop_rate": stopped / len(enhanced)}


def _bootstrap_group(
    group: dict[str, Any],
    group_name: str,
    runs: Sequence[ValidatedRun],
    *,
    resamples: int,
    seed: int,
    workers: int,
    cached_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if group.get("bootstrap", True) is False:
        return []
    baseline_method = str(group.get("baseline_method", "baseline"))
    baseline_runs = sorted(
        (run for run in runs if run.config["method"].get("name") == baseline_method),
        key=lambda run: run.run_id,
    )
    if not baseline_runs:
        return []
    baseline = baseline_runs[0]
    output: list[dict[str, Any]] = []
    method_runs: dict[str, ValidatedRun] = {}
    for run in sorted(runs, key=lambda item: item.run_id):
        method = str(run.config["method"].get("name", ""))
        method_runs.setdefault(method, run)
    for method, run in method_runs.items():
        if method == baseline_method:
            continue
        image_ids = [
            image["image_id"]
            for image in run.data_manifest["images"]
            if isinstance(image, dict) and "image_id" in image
        ]
        max_det = int(run.config["detector"].get("max_det", 500))
        cached = _matching_bootstrap_row(
            cached_rows,
            group_name=group_name,
            baseline_run_id=baseline.run_id,
            method_run_id=run.run_id,
            method=method,
            resamples=resamples,
            seed=seed,
            images=len(image_ids),
            max_det=max_det,
        )
        result = (
            cached
            if cached is not None
            else paired_coco_ap_bootstrap(
                _bootstrap_annotation(run),
                baseline.path / "predictions" / "final.coco.json",
                run.path / "predictions" / "final.coco.json",
                resamples=resamples,
                seed=seed,
                max_det=max_det,
                image_ids=image_ids,
                workers=workers,
            )
        )
        output.append(
            {
                "group": group_name,
                "baseline_run_id": baseline.run_id,
                "method_run_id": run.run_id,
                "method": method,
                **result,
                "algorithm_version": 1,
                "max_det": max_det,
                "cache_hit": cached is not None,
            }
        )
    return output


def _matching_bootstrap_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_name: str,
    baseline_run_id: str,
    method_run_id: str,
    method: str,
    resamples: int,
    seed: int,
    images: int,
    max_det: int,
) -> dict[str, float | int] | None:
    for row in rows:
        try:
            matches = (
                row.get("group") == group_name
                and row.get("baseline_run_id") == baseline_run_id
                and row.get("method_run_id") == method_run_id
                and row.get("method") == method
                and int(row.get("resamples", -1)) == resamples
                and int(row.get("seed", -1)) == seed
                and int(row.get("images", -1)) == images
                and int(row.get("algorithm_version", 1)) == 1
                and int(row.get("max_det", max_det)) == max_det
            )
            if matches:
                return {
                    "delta": float(row["delta"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "resamples": resamples,
                    "seed": seed,
                    "images": images,
                }
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _resolve_group_runs(group: Mapping[str, Any]) -> tuple[list[Path], bool]:
    raw_runs = group.get("runs")
    if raw_runs is not None:
        if not isinstance(raw_runs, list) or not raw_runs:
            raise AggregateError("explicit group runs must be a nonempty list")
        return sorted((Path(str(value)) for value in raw_runs), key=str), True
    discover = group.get("discover")
    if not isinstance(discover, dict):
        raise AggregateError("group requires either runs or discover")
    root = Path(str(discover.get("root", "runs")))
    if not root.is_dir():
        raise AggregateError(f"discovery root does not exist: {root}")
    paths: list[Path] = []
    for path in sorted(
        (item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name
    ):
        config_path = path / "config_resolved.yaml"
        if not config_path.is_file():
            continue
        try:
            config = _load_yaml_mapping(config_path)
        except AggregateError:
            continue
        if _matches_discovery(config, discover):
            expected_model_sha = discover.get("model_sha256")
            if expected_model_sha is not None:
                fingerprint_path = path / "model_fingerprint.json"
                if not fingerprint_path.is_file():
                    continue
                try:
                    fingerprint = _load_mapping(fingerprint_path)
                except AggregateError:
                    continue
                if fingerprint.get("sha256") != expected_model_sha:
                    continue
            paths.append(path)
    if not paths:
        raise AggregateError(f"run discovery matched no directories under {root}")
    return paths, False


def _matches_discovery(config: Mapping[str, Any], discover: Mapping[str, Any]) -> bool:
    paths = {
        "experiment": ("experiment", "name"),
        "dataset": ("dataset", "name"),
        "split": ("dataset", "split"),
        "corruption_name": ("dataset", "corruption_name"),
        "corruption_severity": ("dataset", "corruption_severity"),
        "method": ("method", "name"),
        "detector": ("detector", "name"),
    }
    for filter_name, (section, key) in paths.items():
        if filter_name not in discover:
            continue
        section_value = config.get(section)
        if not isinstance(section_value, Mapping):
            return False
        expected = discover[filter_name]
        actual = section_value.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    if rows:
        fields = sorted({key for row in rows for key in row})
        preferred = [
            "group",
            "run_id",
            "repeat",
            "dataset",
            "split",
            "corruption_name",
            "corruption_severity",
            "method",
            "detector",
            "AP",
            "AP50",
            "latency_mean_ms",
            "latency_p95_ms",
            "fps",
            "eic_mean",
            "pareto",
        ]
        fieldnames = [field for field in preferred if field in fields]
        fieldnames.extend(field for field in fields if field not in fieldnames)
    else:
        fieldnames = list(empty_fields)
    lines: list[str] = []
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    lines.append(stream.getvalue())
    atomic_write_text(path, "".join(lines))


def _bootstrap_annotation(run: ValidatedRun) -> Path:
    annotation = run.data_manifest.get("annotation")
    if annotation is not None:
        return Path(str(annotation))
    ground_truth = run.path / "ground_truth.coco.json"
    if not ground_truth.is_file():
        raise AggregateError(f"bootstrap ground truth is unavailable: {ground_truth}")
    return ground_truth


def _write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}\n",
        "\\toprule\n",
        "run\\_id & method & AP & latency (ms) & EIC & Pareto \\\\\n",
        "\\midrule\n",
    ]
    for row in rows:
        values = [
            _latex_escape(str(row.get("run_id", ""))),
            _latex_escape(str(row.get("method", ""))),
            _format_number(row.get("AP")),
            _format_number(row.get("latency_mean_ms")),
            _format_number(row.get("eic_mean")),
            "yes" if row.get("pareto") else "no",
        ]
        lines.append(" & ".join(values) + " \\\\\n")
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n"])
    if rows and any("run_id" not in row for row in rows):
        raise AggregateError("every LaTeX table row must include run_id")
    atomic_write_text(path, "".join(lines))


def _pareto_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counters = {"validation": 0, "test": 0, "RDDTS": 0}
    prefixes = {"validation": "V", "test": "T", "RDDTS": "R"}
    output: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        group = str(row.get("group", "")).casefold()
        panel = "RDDTS" if "rddts" in group else "test" if "test" in group else "validation"
        counters[panel] += 1
        row["plot_panel"] = panel
        row["plot_label"] = f"{prefixes[panel]}{counters[panel]:02d}"
        output.append(row)
    return output


def _write_pareto_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.ticker import FormatStrFormatter
    except (ImportError, OSError) as exc:
        raise AggregateError(f"cannot render Pareto figure: {exc}") from exc
    panels = (
        ("validation", "HazyDet validation"),
        ("test", "HazyDet test"),
        ("RDDTS", "HazyDet RDDTS"),
    )
    # Render at the journal's final two-column width so point labels and axes
    # are not silently reduced below the 8-pt artwork minimum in LaTeX.
    figure, axes = plt.subplots(1, 3, figsize=(6.5, 3.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    eic_values = [float(row["eic_mean"]) for row in rows]
    normalization = Normalize(vmin=min(eic_values), vmax=max(eic_values) + 1e-12)
    color_map = plt.get_cmap("viridis")
    offset_cycle = ((4, 6), (4, -12), (-14, 6), (-14, -12), (4, 14), (-14, 14))
    for axis, (panel, title) in zip(axes, panels, strict=True):
        panel_rows = [row for row in rows if str(row.get("plot_panel")) == panel]
        if not panel_rows:
            axis.set_title(title)
            axis.text(0.5, 0.5, "No declared rows", ha="center", va="center")
            axis.axis("off")
            continue
        for is_pareto, marker in ((False, "x"), (True, "o")):
            subset = [row for row in panel_rows if bool(row.get("pareto")) is is_pareto]
            if not subset:
                continue
            axis.scatter(
                [float(row["latency_mean_ms"]) for row in subset],
                [float(row["AP"]) for row in subset],
                c=[float(row["eic_mean"]) for row in subset],
                cmap=color_map,
                norm=normalization,
                marker=marker,
                s=30 if is_pareto else 22,
                linewidths=0.75,
                edgecolors="#1F2937" if is_pareto else None,
                alpha=0.9,
            )
        annotated = (
            [row for row in panel_rows if bool(row.get("pareto"))]
            if panel == "validation"
            else panel_rows
        )
        x_values = [float(row["latency_mean_ms"]) for row in panel_rows]
        y_values = [float(row["AP"]) for row in panel_rows]
        for index, row in enumerate(annotated):
            if panel == "validation":
                offset = offset_cycle[index % len(offset_cycle)]
            else:
                offset = (
                    -78 if float(row["latency_mean_ms"]) == max(x_values) else 7,
                    -16 if float(row["AP"]) == max(y_values) else 8,
                )
            method_label = {
                "baseline": "B0",
                "buse": "D-U-Q-WBF",
            }.get(str(row["method"]), str(row["method"]).replace("_", " "))
            annotation = (
                str(row["plot_label"])
                if panel == "validation"
                else f"{row['plot_label']}\n{method_label}"
            )
            axis.annotate(
                annotation,
                (float(row["latency_mean_ms"]), float(row["AP"])),
                fontsize=8.0,
                xytext=offset,
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#475569"},
            )
        axis.set_xscale("log")
        axis.set_title(title, fontsize=9.0)
        axis.tick_params(axis="both", labelsize=8.0)
        axis.yaxis.set_major_formatter(
            FormatStrFormatter("%.3f" if panel == "validation" else "%.5f")
        )
        axis.margins(x=0.12, y=0.20)
        axis.set_facecolor("white")
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )
    color_bar.set_label("Equivalent inference calls (EIC)", fontsize=8.0)
    color_bar.ax.tick_params(labelsize=8.0)
    figure.supxlabel("Mean latency (ms/image, log scale)", fontsize=8.0)
    figure.supylabel("COCO AP", fontsize=8.0)
    figure.suptitle(
        "Accuracy-efficiency trade-off under frozen evaluation",
        fontsize=10.0,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    format_name = path.suffix.casefold().removeprefix(".")
    metadata = {
        "pdf": {"CreationDate": None, "ModDate": None},
        "svg": {"Date": None},
    }.get(format_name)
    try:
        with matplotlib.rc_context({"svg.hashsalt": "buse-uav-pareto-v1"}):
            figure.savefig(
                temporary,
                format=format_name,
                dpi=180,
                metadata=metadata,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _figure_artifacts(figures: Path) -> dict[str, dict[str, Any]]:
    supported = {".png", ".svg", ".pdf", ".csv", ".json"}
    output: dict[str, dict[str, Any]] = {}
    paths = [
        path
        for path in sorted(figures.iterdir())
        if path.is_file() and path.suffix.casefold() in supported
    ]
    for path in paths:
        suffix = path.suffix.casefold().removeprefix(".")
        key = (
            path.stem
            if suffix == "png" or path.stem.endswith("_source")
            else f"{path.stem}_{suffix}"
        )
        if key in output:
            raise AggregateError(f"figure artifact key collision: {key}")
        output[key] = _artifact(path)
    return output


def _declared_artifact_catalog(
    raw_catalog: Any,
    *,
    plan_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Hash declared final documents, data metadata, and model files."""
    if raw_catalog is None:
        return {}
    if not isinstance(raw_catalog, dict):
        raise AggregateError("artifact_catalog must be a mapping")
    raw_root = raw_catalog.get("root", ".")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise AggregateError("artifact_catalog.root must be a nonempty path")
    root = (plan_path.resolve().parent / raw_root).resolve()
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for section in ("documents", "data_metadata", "models"):
        raw_section = raw_catalog.get(section)
        if raw_section is None:
            continue
        if not isinstance(raw_section, dict):
            raise AggregateError(f"artifact_catalog.{section} must be a mapping")
        entries: dict[str, dict[str, Any]] = {}
        for raw_name, raw_entry in raw_section.items():
            name = str(raw_name).strip()
            if not name:
                raise AggregateError(f"artifact_catalog.{section} contains an empty key")
            metadata: dict[str, Any] = {}
            raw_path_value: Any
            if isinstance(raw_entry, str):
                raw_path_value = raw_entry
            elif isinstance(raw_entry, dict):
                raw_path_value = raw_entry.get("path")
                metadata = {key: value for key, value in raw_entry.items() if key != "path"}
            else:
                raise AggregateError(f"artifact_catalog.{section}.{name} is invalid")
            if not isinstance(raw_path_value, str) or not raw_path_value.strip():
                raise AggregateError(f"artifact_catalog.{section}.{name} lacks a path")
            raw_path = raw_path_value
            if {"path", "sha256", "bytes"}.intersection(metadata):
                raise AggregateError(
                    f"artifact_catalog.{section}.{name} metadata uses a reserved key"
                )
            allow_missing = metadata.pop("allow_missing", False)
            if not isinstance(allow_missing, bool):
                raise AggregateError(
                    f"artifact_catalog.{section}.{name}.allow_missing must be boolean"
                )
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise AggregateError(
                    f"artifact_catalog.{section}.{name} escapes root: {path}"
                ) from exc
            if not resolved.is_file():
                if allow_missing:
                    continue
                raise AggregateError(f"artifact_catalog.{section}.{name} is not a file: {resolved}")
            entries[name] = {**_artifact(resolved), **metadata}
        output[section] = entries
    return output


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AggregateError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateError(f"expected a JSON object: {path}")
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        raise AggregateError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateError(f"expected a YAML mapping: {path}")
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (csv.Error, OSError):
        return []


def _preserved_bootstrap_rows(
    cached_rows: Sequence[Mapping[str, Any]],
    *,
    generated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain validated externally generated frozen bootstrap blocks across aggregation."""
    generated_keys = {
        (
            str(row.get("group", "")),
            str(row.get("baseline_run_id", "")),
            str(row.get("method_run_id", "")),
        )
        for row in generated_rows
    }
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in cached_rows:
        group = str(raw.get("group", ""))
        if not group.startswith(_PRESERVED_BOOTSTRAP_GROUP_PREFIXES):
            continue
        baseline_run_id = str(raw.get("baseline_run_id", ""))
        method_run_id = str(raw.get("method_run_id", ""))
        key = (group, baseline_run_id, method_run_id)
        if not baseline_run_id or not method_run_id:
            raise AggregateError(f"preserved bootstrap row lacks run provenance: {group}")
        if key in seen:
            raise AggregateError(f"duplicate preserved bootstrap row: {key}")
        seen.add(key)
        if key in generated_keys:
            continue
        try:
            resamples = int(str(raw.get("resamples", "")))
            seed = int(str(raw.get("seed", "")))
            delta = float(str(raw.get("delta", "")))
            ci_low = float(str(raw.get("ci_low", "")))
            ci_high = float(str(raw.get("ci_high", "")))
        except (TypeError, ValueError) as exc:
            raise AggregateError(f"preserved bootstrap row is malformed: {group}") from exc
        if (
            resamples != 1000
            or seed != 42
            or not all(math.isfinite(value) for value in (delta, ci_low, ci_high))
            or ci_low > ci_high
        ):
            raise AggregateError(f"preserved bootstrap row violates frozen protocol: {group}")
        output.append(dict(raw))
    return output


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{number:.4f}" if math.isfinite(number) else "--"


def _optional_number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


_BOOTSTRAP_FIELDS = (
    "group",
    "baseline_run_id",
    "method_run_id",
    "method",
    "delta",
    "ci_low",
    "ci_high",
    "resamples",
    "seed",
    "images",
    "algorithm_version",
    "max_det",
    "cache_hit",
)
