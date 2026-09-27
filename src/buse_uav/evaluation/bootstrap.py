from __future__ import annotations

import contextlib
import copy
import io
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from buse_uav.evaluation.coco import EvaluationError
from buse_uav.utils.hashing import sha256_file, stable_hash
from buse_uav.utils.io import atomic_write_json


@dataclass(frozen=True)
class _PreparedGroundTruth:
    info: Any
    licenses: Any
    categories: Any
    images_by_id: dict[Any, dict[str, Any]]
    annotations_by_image: dict[Any, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class _PreparedPredictions:
    predictions_by_image: dict[Any, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class _CocoEvaluationCache:
    eval_imgs: tuple[Any, ...]
    params: Any
    image_position: dict[Any, int]


@dataclass(frozen=True)
class _AcceleratedCocoApOnlyCache:
    offsets: np.ndarray
    score_ranks: np.ndarray
    matches: np.ndarray
    ignores: np.ndarray
    gt_counts: np.ndarray
    rank_counts: np.ndarray
    recall_thresholds: np.ndarray
    image_position: dict[Any, int]


@dataclass(frozen=True)
class _ClusterPositionCache:
    offsets: np.ndarray
    image_positions: np.ndarray


@dataclass(frozen=True)
class ClusterBootstrapScope:
    """One registered cluster-bootstrap scope sharing detector-side caches."""

    clusters: Mapping[str, Sequence[Any]]
    checkpoint_path: Path | None = None
    checkpoint_identity: Mapping[str, Any] | None = None


_COCO_AREA_STAT_INDEX = {"all": 0, "small": 3, "medium": 4, "large": 5}

_BOOTSTRAP_WORKER_CACHED_CONTEXT: tuple[_CocoEvaluationCache, _CocoEvaluationCache, str] | None = (
    None
)
_BOOTSTRAP_WORKER_CLUSTER_CONTEXT: (
    tuple[
        _CocoEvaluationCache,
        _CocoEvaluationCache,
        tuple[tuple[Any, ...], ...],
    ]
    | None
) = None
_BOOTSTRAP_WORKER_PREPARED_CONTEXT: (
    tuple[
        _PreparedGroundTruth,
        _PreparedPredictions,
        _PreparedPredictions,
        int,
        str,
    ]
    | None
) = None
_BOOTSTRAP_WORKER_RAW_CONTEXT: (
    tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        int,
        str,
    ]
    | None
) = None


def paired_bootstrap_delta(
    baseline: Sequence[float],
    method: Sequence[float],
    *,
    resamples: int = 1000,
    seed: int = 42,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float | int]:
    """Paired image bootstrap for an arbitrary image-level statistic."""
    baseline_array = np.asarray(baseline, dtype=np.float64)
    method_array = np.asarray(method, dtype=np.float64)
    if baseline_array.ndim != 1 or method_array.ndim != 1:
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    if baseline_array.size == 0 or baseline_array.shape != method_array.shape:
        raise ValueError("paired bootstrap inputs must be nonempty and equal length")
    if not np.isfinite(baseline_array).all() or not np.isfinite(method_array).all():
        raise ValueError("paired bootstrap inputs must be finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    reducer = statistic or (lambda values: float(np.mean(values)))
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, baseline_array.size, size=baseline_array.size)
        deltas[index] = reducer(method_array[sampled]) - reducer(baseline_array[sampled])
    return {
        "delta": reducer(method_array) - reducer(baseline_array),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "images": int(baseline_array.size),
    }


def paired_coco_ap_bootstrap(
    annotation_path: Path,
    baseline_prediction_path: Path,
    method_prediction_path: Path,
    *,
    resamples: int = 1000,
    seed: int = 42,
    max_det: int = 500,
    image_ids: Sequence[Any] | None = None,
    workers: int = 1,
    reuse_indexes: bool = True,
    cache_evaluations: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    chunk_resamples: int | None = None,
    area_label: str = "all",
) -> dict[str, float | int]:
    """Recompute COCO AP for every paired image bootstrap resample."""
    ground_truth = _load_mapping(annotation_path)
    baseline = _load_rows(baseline_prediction_path)
    method = _load_rows(method_prediction_path)
    raw_images = ground_truth.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise EvaluationError("COCO ground truth contains no images")
    evaluated_image_ids = (
        list(image_ids)
        if image_ids is not None
        else [image["id"] for image in raw_images if isinstance(image, dict) and "id" in image]
    )
    if not evaluated_image_ids:
        raise EvaluationError("COCO ground truth contains no valid image IDs")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if checkpoint_identity is not None and checkpoint_path is None:
        raise ValueError("checkpoint identity requires a checkpoint path")
    if chunk_resamples is not None and chunk_resamples <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    if area_label not in _COCO_AREA_STAT_INDEX:
        raise ValueError(f"unsupported COCO area label: {area_label}")
    if cache_evaluations and len(set(evaluated_image_ids)) != len(evaluated_image_ids):
        raise ValueError("cached COCO bootstrap requires unique evaluated image IDs")

    prepared_ground_truth = _prepare_ground_truth(ground_truth)
    prepared_baseline = _prepare_predictions(baseline)
    prepared_method = _prepare_predictions(method)
    if reuse_indexes:
        observed_baseline = _coco_ap_for_prepared_sample(
            prepared_ground_truth,
            prepared_baseline,
            evaluated_image_ids,
            max_det=max_det,
            area_label=area_label,
        )
        observed_method = _coco_ap_for_prepared_sample(
            prepared_ground_truth,
            prepared_method,
            evaluated_image_ids,
            max_det=max_det,
            area_label=area_label,
        )
    else:
        observed_baseline = _coco_ap_for_sample(
            ground_truth,
            baseline,
            evaluated_image_ids,
            max_det=max_det,
            area_label=area_label,
        )
        observed_method = _coco_ap_for_sample(
            ground_truth,
            method,
            evaluated_image_ids,
            max_det=max_det,
            area_label=area_label,
        )
    rng = np.random.default_rng(seed)
    image_id_array = np.asarray(evaluated_image_ids, dtype=object)
    samples = [
        list(
            image_id_array[
                rng.integers(
                    0,
                    len(evaluated_image_ids),
                    size=len(evaluated_image_ids),
                )
            ]
        )
        for _ in range(resamples)
    ]
    checkpoint_state: dict[str, Any] | None = None
    delta_slots: list[float | None] = [None] * resamples
    if checkpoint_path is not None:
        identity = _bootstrap_checkpoint_identity(
            annotation_path=annotation_path,
            baseline_prediction_path=baseline_prediction_path,
            method_prediction_path=method_prediction_path,
            evaluated_image_ids=evaluated_image_ids,
            resamples=resamples,
            seed=seed,
            max_det=max_det,
            scope=checkpoint_identity,
            area_label=area_label,
        )
        checkpoint_state, delta_slots = _load_or_initialize_checkpoint(
            checkpoint_path,
            identity=identity,
            observed_baseline=observed_baseline,
            observed_method=observed_method,
            resamples=resamples,
        )

    task_size = chunk_resamples or max(1, (resamples + workers - 1) // workers)
    pending_chunks = _pending_chunks(samples, delta_slots, chunk_size=task_size)
    if workers == 1:
        cached_baseline: _CocoEvaluationCache | None = None
        cached_method: _CocoEvaluationCache | None = None
        if cache_evaluations and pending_chunks:
            cached_baseline = _build_evaluation_cache(
                prepared_ground_truth,
                prepared_baseline,
                evaluated_image_ids,
                max_det=max_det,
            )
            cached_method = _build_evaluation_cache(
                prepared_ground_truth,
                prepared_method,
                evaluated_image_ids,
                max_det=max_det,
            )
        for indices, sampled_chunk in pending_chunks:
            if cached_baseline is not None and cached_method is not None:
                chunk = _bootstrap_delta_chunk_cached(
                    cached_baseline,
                    cached_method,
                    sampled_chunk,
                    area_label=area_label,
                )
            elif reuse_indexes:
                chunk = _bootstrap_delta_chunk_prepared(
                    prepared_ground_truth,
                    prepared_baseline,
                    prepared_method,
                    sampled_chunk,
                    max_det=max_det,
                    area_label=area_label,
                )
            else:
                chunk = [
                    _coco_ap_for_sample(
                        ground_truth,
                        method,
                        sampled,
                        max_det=max_det,
                        area_label=area_label,
                    )
                    - _coco_ap_for_sample(
                        ground_truth,
                        baseline,
                        sampled,
                        max_det=max_det,
                        area_label=area_label,
                    )
                    for sampled in sampled_chunk
                ]
            _record_completed_chunk(
                indices,
                chunk,
                delta_slots=delta_slots,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
    elif pending_chunks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_bootstrap_worker,
            initargs=(
                str(annotation_path),
                str(baseline_prediction_path),
                str(method_prediction_path),
                max_det,
                reuse_indexes,
                cache_evaluations,
                evaluated_image_ids,
                area_label,
            ),
        ) as executor:
            futures = {
                executor.submit(_bootstrap_delta_chunk, sampled_chunk): indices
                for indices, sampled_chunk in pending_chunks
            }
            for future in as_completed(futures):
                indices = futures[future]
                _record_completed_chunk(
                    indices,
                    future.result(),
                    delta_slots=delta_slots,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                )
    if any(delta is None for delta in delta_slots):
        raise RuntimeError("COCO bootstrap stopped before every resample completed")
    deltas = np.asarray([delta for delta in delta_slots if delta is not None], dtype=np.float64)
    return {
        "delta": observed_method - observed_baseline,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "images": len(evaluated_image_ids),
    }


def paired_coco_ap_cluster_bootstrap(
    annotation_path: Path,
    baseline_prediction_path: Path,
    method_prediction_path: Path,
    clusters: Mapping[str, Sequence[Any]],
    *,
    resamples: int = 10000,
    seed: int = 42,
    max_det: int = 500,
    workers: int = 1,
    checkpoint_path: Path | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    chunk_resamples: int = 25,
) -> dict[str, float | int]:
    """Paired COCO AP bootstrap with complete clusters sampled as the unit."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if chunk_resamples <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    if checkpoint_identity is not None and checkpoint_path is None:
        raise ValueError("checkpoint identity requires a checkpoint path")
    cluster_names, cluster_rows, evaluated_image_ids = _validated_clusters(clusters)

    ground_truth = _load_mapping(annotation_path)
    baseline = _load_rows(baseline_prediction_path)
    method = _load_rows(method_prediction_path)
    raw_images = ground_truth.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise EvaluationError("COCO ground truth contains no images")
    available_ids = {
        image["id"] for image in raw_images if isinstance(image, dict) and "id" in image
    }
    if any(image_id not in available_ids for image_id in evaluated_image_ids):
        raise EvaluationError("cluster bootstrap includes an image absent from ground truth")

    prepared_ground_truth = _prepare_ground_truth(ground_truth)
    prepared_baseline = _prepare_predictions(baseline)
    prepared_method = _prepare_predictions(method)
    observed_baseline = _coco_ap_for_prepared_sample(
        prepared_ground_truth,
        prepared_baseline,
        evaluated_image_ids,
        max_det=max_det,
    )
    observed_method = _coco_ap_for_prepared_sample(
        prepared_ground_truth,
        prepared_method,
        evaluated_image_ids,
        max_det=max_det,
    )

    rng = np.random.default_rng(seed)
    cluster_count = len(cluster_rows)
    cluster_samples = [
        list(row)
        for row in rng.integers(
            0,
            cluster_count,
            size=(resamples, cluster_count),
        )
    ]
    checkpoint_state: dict[str, Any] | None = None
    delta_slots: list[float | None] = [None] * resamples
    if checkpoint_path is not None:
        if len(cluster_names) != len(cluster_rows):
            raise RuntimeError("cluster name and row counts differ")
        protected_scope = {
            "sampling_unit": "cluster",
            "cluster_names": cluster_names,
            "clusters_sha256": stable_hash(
                [
                    {"name": name, "image_ids": list(rows)}
                    for name, rows in zip(  # noqa: B905 -- lengths checked above
                        cluster_names, cluster_rows
                    )
                ],
                length=64,
            ),
        }
        scope = dict(checkpoint_identity or {})
        for key, value in protected_scope.items():
            if key in scope and scope[key] != value:
                raise ValueError(f"checkpoint identity conflicts with cluster lock: {key}")
            scope[key] = value
        identity = _bootstrap_checkpoint_identity(
            annotation_path=annotation_path,
            baseline_prediction_path=baseline_prediction_path,
            method_prediction_path=method_prediction_path,
            evaluated_image_ids=evaluated_image_ids,
            resamples=resamples,
            seed=seed,
            max_det=max_det,
            scope=scope,
        )
        checkpoint_state, delta_slots = _load_or_initialize_checkpoint(
            checkpoint_path,
            identity=identity,
            observed_baseline=observed_baseline,
            observed_method=observed_method,
            resamples=resamples,
        )

    pending_chunks = _pending_chunks(
        cluster_samples,
        delta_slots,
        chunk_size=chunk_resamples,
    )
    frozen_clusters = tuple(tuple(rows) for rows in cluster_rows)
    if workers == 1:
        baseline_cache: _CocoEvaluationCache | None = None
        method_cache: _CocoEvaluationCache | None = None
        if pending_chunks:
            baseline_cache = _build_evaluation_cache(
                prepared_ground_truth,
                prepared_baseline,
                evaluated_image_ids,
                max_det=max_det,
            )
            method_cache = _build_evaluation_cache(
                prepared_ground_truth,
                prepared_method,
                evaluated_image_ids,
                max_det=max_det,
            )
        for indices, sampled_chunk in pending_chunks:
            if baseline_cache is None or method_cache is None:
                raise RuntimeError("cluster bootstrap cache was not initialized")
            values = _bootstrap_cluster_delta_chunk_cached(
                baseline_cache,
                method_cache,
                frozen_clusters,
                sampled_chunk,
            )
            _record_completed_chunk(
                indices,
                values,
                delta_slots=delta_slots,
                checkpoint_path=checkpoint_path,
                checkpoint_state=checkpoint_state,
            )
    elif pending_chunks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_cluster_bootstrap_worker,
            initargs=(
                str(annotation_path),
                str(baseline_prediction_path),
                str(method_prediction_path),
                max_det,
                evaluated_image_ids,
                frozen_clusters,
            ),
        ) as executor:
            futures = {
                executor.submit(_bootstrap_cluster_delta_chunk, sampled_chunk): indices
                for indices, sampled_chunk in pending_chunks
            }
            for future in as_completed(futures):
                _record_completed_chunk(
                    futures[future],
                    future.result(),
                    delta_slots=delta_slots,
                    checkpoint_path=checkpoint_path,
                    checkpoint_state=checkpoint_state,
                )
    if any(delta is None for delta in delta_slots):
        raise RuntimeError("cluster bootstrap stopped before every resample completed")
    deltas = np.asarray([delta for delta in delta_slots if delta is not None], dtype=np.float64)
    return {
        "delta": observed_method - observed_baseline,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "images": len(evaluated_image_ids),
        "clusters": cluster_count,
    }


def paired_coco_ap_cluster_bootstrap_scopes(
    annotation_path: Path,
    baseline_prediction_path: Path,
    method_prediction_path: Path,
    scopes: Mapping[str, ClusterBootstrapScope],
    *,
    resamples: int = 10000,
    seed: int = 42,
    max_det: int = 500,
    workers: int = 1,
    chunk_resamples: int = 25,
    accelerate_ap_only: bool = False,
) -> dict[str, dict[str, float | int]]:
    """Run registered cluster scopes with one shared pair of COCO match caches."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _paired_coco_ap_cluster_bootstrap_scopes_impl(
            annotation_path,
            baseline_prediction_path,
            method_prediction_path,
            scopes,
            resamples=resamples,
            seed=seed,
            max_det=max_det,
            workers=workers,
            chunk_resamples=chunk_resamples,
            accelerate_ap_only=accelerate_ap_only,
        )


def _paired_coco_ap_cluster_bootstrap_scopes_impl(
    annotation_path: Path,
    baseline_prediction_path: Path,
    method_prediction_path: Path,
    scopes: Mapping[str, ClusterBootstrapScope],
    *,
    resamples: int = 10000,
    seed: int = 42,
    max_det: int = 500,
    workers: int = 1,
    chunk_resamples: int = 25,
    accelerate_ap_only: bool = False,
) -> dict[str, dict[str, float | int]]:
    if not scopes:
        raise ValueError("cluster bootstrap requires at least one scope")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if chunk_resamples <= 0:
        raise ValueError("bootstrap chunk size must be positive")

    ground_truth = _load_mapping(annotation_path)
    baseline = _load_rows(baseline_prediction_path)
    method = _load_rows(method_prediction_path)
    raw_images = ground_truth.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise EvaluationError("COCO ground truth contains no images")
    available_ids = {
        image["id"] for image in raw_images if isinstance(image, dict) and "id" in image
    }

    scope_rows: dict[str, dict[str, Any]] = {}
    cached_image_ids: list[Any] = []
    cached_image_id_set: set[Any] = set()
    for raw_name, specification in scopes.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("cluster bootstrap scope names must be nonempty")
        if specification.checkpoint_identity is not None and specification.checkpoint_path is None:
            raise ValueError("checkpoint identity requires a checkpoint path")
        cluster_names, cluster_rows, evaluated_image_ids = _validated_clusters(
            specification.clusters
        )
        if any(image_id not in available_ids for image_id in evaluated_image_ids):
            raise EvaluationError("cluster bootstrap includes an image absent from ground truth")
        for image_id in evaluated_image_ids:
            if image_id not in cached_image_id_set:
                cached_image_ids.append(image_id)
                cached_image_id_set.add(image_id)
        scope_rows[name] = {
            "specification": specification,
            "cluster_names": cluster_names,
            "clusters": tuple(tuple(rows) for rows in cluster_rows),
            "evaluated_image_ids": evaluated_image_ids,
        }

    prepared_ground_truth = _prepare_ground_truth(ground_truth)
    baseline_cache = _build_evaluation_cache(
        prepared_ground_truth,
        _prepare_predictions(baseline),
        cached_image_ids,
        max_det=max_det,
    )
    method_cache = _build_evaluation_cache(
        prepared_ground_truth,
        _prepare_predictions(method),
        cached_image_ids,
        max_det=max_det,
    )

    for row in scope_rows.values():
        specification = row["specification"]
        cluster_names = row["cluster_names"]
        clusters = row["clusters"]
        evaluated_image_ids = row["evaluated_image_ids"]
        observed_baseline = _coco_ap_from_evaluation_cache_ap_only(
            baseline_cache, evaluated_image_ids
        )
        observed_method = _coco_ap_from_evaluation_cache_ap_only(method_cache, evaluated_image_ids)
        cluster_count = len(clusters)
        checkpoint_state: dict[str, Any] | None = None
        delta_slots: list[float | None] = [None] * resamples
        checkpoint_path = specification.checkpoint_path
        if checkpoint_path is not None:
            if len(cluster_names) != len(clusters):
                raise RuntimeError("cluster name and row counts differ")
            protected_scope = {
                "sampling_unit": "cluster",
                "cluster_names": cluster_names,
                "clusters_sha256": stable_hash(
                    [
                        {"name": cluster_name, "image_ids": list(cluster)}
                        for cluster_name, cluster in zip(  # noqa: B905 -- lengths checked above
                            cluster_names, clusters
                        )
                    ],
                    length=64,
                ),
            }
            identity_scope = dict(specification.checkpoint_identity or {})
            for key, value in protected_scope.items():
                if key in identity_scope and identity_scope[key] != value:
                    raise ValueError(f"checkpoint identity conflicts with cluster lock: {key}")
                identity_scope[key] = value
            identity = _bootstrap_checkpoint_identity(
                annotation_path=annotation_path,
                baseline_prediction_path=baseline_prediction_path,
                method_prediction_path=method_prediction_path,
                evaluated_image_ids=evaluated_image_ids,
                resamples=resamples,
                seed=seed,
                max_det=max_det,
                scope=identity_scope,
            )
            checkpoint_state, delta_slots = _load_or_initialize_checkpoint(
                checkpoint_path,
                identity=identity,
                observed_baseline=observed_baseline,
                observed_method=observed_method,
                resamples=resamples,
            )
        row.update(
            {
                "checkpoint_state": checkpoint_state,
                "delta_slots": delta_slots,
                "observed_baseline": observed_baseline,
                "observed_method": observed_method,
                "cluster_count": cluster_count,
            }
        )

    accelerated_baseline: _AcceleratedCocoApOnlyCache | None = None
    accelerated_method: _AcceleratedCocoApOnlyCache | None = None
    if accelerate_ap_only:
        accelerated_baseline = _prepare_accelerated_coco_ap_only_cache(baseline_cache)
        accelerated_method = _prepare_accelerated_coco_ap_only_cache(method_cache)
        for row in scope_rows.values():
            image_ids = row["evaluated_image_ids"]
            if (
                _coco_ap_from_accelerated_cache(accelerated_baseline, image_ids)
                != row["observed_baseline"]
                or _coco_ap_from_accelerated_cache(accelerated_method, image_ids)
                != row["observed_method"]
            ):
                raise RuntimeError("accelerated COCO AP does not match the locked point estimate")

    for row in scope_rows.values():
        _run_streamed_cluster_scope(
            baseline_cache,
            method_cache,
            row,
            resamples=resamples,
            seed=seed,
            workers=workers,
            chunk_resamples=chunk_resamples,
            accelerated_baseline=accelerated_baseline,
            accelerated_method=accelerated_method,
        )

    results: dict[str, dict[str, float | int]] = {}
    for name, row in scope_rows.items():
        delta_slots = row["delta_slots"]
        if any(delta is None for delta in delta_slots):
            raise RuntimeError(f"cluster bootstrap scope stopped before completion: {name}")
        deltas = np.asarray([delta for delta in delta_slots if delta is not None], dtype=np.float64)
        results[name] = {
            "delta": row["observed_method"] - row["observed_baseline"],
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "resamples": resamples,
            "seed": seed,
            "images": len(row["evaluated_image_ids"]),
            "clusters": len(row["clusters"]),
        }
    return results


def _run_streamed_cluster_scope(
    baseline_cache: _CocoEvaluationCache,
    method_cache: _CocoEvaluationCache,
    row: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
    workers: int,
    chunk_resamples: int,
    accelerated_baseline: _AcceleratedCocoApOnlyCache | None,
    accelerated_method: _AcceleratedCocoApOnlyCache | None,
) -> None:
    """Generate registered samples in bounded chunks without changing RNG order."""
    cluster_count = int(row["cluster_count"])
    delta_slots = row["delta_slots"]
    if not isinstance(delta_slots, list) or len(delta_slots) != resamples:
        raise RuntimeError("cluster bootstrap delta slots are malformed")
    specification = row["specification"]
    rng = np.random.default_rng(seed)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    accelerated_clusters: _ClusterPositionCache | None = None
    if accelerated_baseline is not None or accelerated_method is not None:
        if accelerated_baseline is None or accelerated_method is None:
            raise RuntimeError("accelerated COCO AP pair is incomplete")
        accelerated_clusters = _prepare_cluster_position_cache(
            row["clusters"], accelerated_baseline.image_position
        )
        if accelerated_method.image_position != accelerated_baseline.image_position:
            raise RuntimeError("accelerated COCO AP pair has different image identities")
    try:
        for start in range(0, resamples, chunk_resamples):
            stop = min(start + chunk_resamples, resamples)
            raw_samples = rng.integers(
                0,
                cluster_count,
                size=(stop - start, cluster_count),
            )
            pending = [
                (start + offset, tuple(int(value) for value in raw_sample))
                for offset, raw_sample in enumerate(raw_samples)
                if delta_slots[start + offset] is None
            ]
            if not pending:
                continue
            grouped: dict[tuple[int, ...], list[int]] = {}
            for index, sample in pending:
                grouped.setdefault(sample, []).append(index)
            unique = list(grouped.items())
            groups = [indices for _, indices in unique]
            samples = [list(sample) for sample, _ in unique]
            if accelerated_clusters is not None:
                if accelerated_baseline is None or accelerated_method is None:
                    raise RuntimeError("accelerated COCO AP pair disappeared")
                if executor is None or len(samples) == 1:
                    values = _bootstrap_cluster_delta_chunk_ap_only_accelerated(
                        accelerated_baseline,
                        accelerated_method,
                        accelerated_clusters,
                        samples,
                    )
                else:
                    partitions = [samples[index::workers] for index in range(workers)]
                    futures = [
                        executor.submit(
                            _bootstrap_cluster_delta_chunk_ap_only_accelerated,
                            accelerated_baseline,
                            accelerated_method,
                            accelerated_clusters,
                            partition,
                        )
                        for partition in partitions
                        if partition
                    ]
                    partition_values = [future.result() for future in futures]
                    values = [
                        partition_values[index % workers][index // workers]
                        for index in range(len(samples))
                    ]
            elif executor is None or len(samples) == 1:
                values = _bootstrap_cluster_delta_chunk_ap_only_cached(
                    baseline_cache,
                    method_cache,
                    row["clusters"],
                    samples,
                )
            else:
                partitions = [samples[index::workers] for index in range(workers)]
                futures = [
                    executor.submit(
                        _bootstrap_cluster_delta_chunk_ap_only_cached,
                        baseline_cache,
                        method_cache,
                        row["clusters"],
                        partition,
                    )
                    for partition in partitions
                    if partition
                ]
                partition_values = [future.result() for future in futures]
                values = [
                    partition_values[index % workers][index // workers]
                    for index in range(len(samples))
                ]
            _record_completed_sample_groups(
                groups,
                values,
                delta_slots=delta_slots,
                checkpoint_path=specification.checkpoint_path,
                checkpoint_state=row["checkpoint_state"],
            )
    finally:
        if executor is not None:
            executor.shutdown()


def _validated_clusters(
    clusters: Mapping[str, Sequence[Any]],
) -> tuple[list[str], list[list[Any]], list[Any]]:
    if not clusters:
        raise ValueError("cluster bootstrap requires at least one cluster")
    names: list[str] = []
    rows: list[list[Any]] = []
    flattened: list[Any] = []
    seen: set[Any] = set()
    for raw_name, raw_rows in clusters.items():
        name = str(raw_name).strip()
        values = list(raw_rows)
        if not name or not values:
            raise ValueError("cluster names and memberships must be nonempty")
        if len(values) != len(set(values)) or any(value in seen for value in values):
            raise ValueError("cluster image memberships must be unique and disjoint")
        names.append(name)
        rows.append(values)
        flattened.extend(values)
        seen.update(values)
    return names, rows, flattened


def _initialize_cluster_bootstrap_worker(
    annotation_path: str,
    baseline_prediction_path: str,
    method_prediction_path: str,
    max_det: int,
    evaluated_image_ids: Sequence[Any],
    clusters: tuple[tuple[Any, ...], ...],
) -> None:
    global _BOOTSTRAP_WORKER_CLUSTER_CONTEXT
    ground_truth = _prepare_ground_truth(_load_mapping(Path(annotation_path)))
    baseline = _prepare_predictions(_load_rows(Path(baseline_prediction_path)))
    method = _prepare_predictions(_load_rows(Path(method_prediction_path)))
    _BOOTSTRAP_WORKER_CLUSTER_CONTEXT = (
        _build_evaluation_cache(
            ground_truth,
            baseline,
            evaluated_image_ids,
            max_det=max_det,
        ),
        _build_evaluation_cache(
            ground_truth,
            method,
            evaluated_image_ids,
            max_det=max_det,
        ),
        clusters,
    )


def _bootstrap_cluster_delta_chunk(samples: Sequence[Sequence[Any]]) -> list[float]:
    if _BOOTSTRAP_WORKER_CLUSTER_CONTEXT is None:
        raise RuntimeError("cluster bootstrap worker is not initialized")
    baseline, method, clusters = _BOOTSTRAP_WORKER_CLUSTER_CONTEXT
    return _bootstrap_cluster_delta_chunk_cached(baseline, method, clusters, samples)


def _bootstrap_cluster_delta_chunk_cached(
    baseline: _CocoEvaluationCache,
    method: _CocoEvaluationCache,
    clusters: Sequence[Sequence[Any]],
    samples: Sequence[Sequence[Any]],
) -> list[float]:
    output: list[float] = []
    for sampled_clusters in samples:
        sampled_image_ids = [
            image_id for raw_index in sampled_clusters for image_id in clusters[int(raw_index)]
        ]
        output.append(
            _coco_ap_from_evaluation_cache(method, sampled_image_ids)
            - _coco_ap_from_evaluation_cache(baseline, sampled_image_ids)
        )
    return output


def _bootstrap_cluster_delta_chunk_ap_only_cached(
    baseline: _CocoEvaluationCache,
    method: _CocoEvaluationCache,
    clusters: Sequence[Sequence[Any]],
    samples: Sequence[Sequence[Any]],
) -> list[float]:
    output: list[float] = []
    for sampled_clusters in samples:
        sampled_image_ids = [
            image_id for raw_index in sampled_clusters for image_id in clusters[int(raw_index)]
        ]
        output.append(
            _coco_ap_from_evaluation_cache_ap_only(method, sampled_image_ids)
            - _coco_ap_from_evaluation_cache_ap_only(baseline, sampled_image_ids)
        )
    return output


def _bootstrap_cluster_delta_chunk_ap_only_accelerated(
    baseline: _AcceleratedCocoApOnlyCache,
    method: _AcceleratedCocoApOnlyCache,
    clusters: _ClusterPositionCache,
    samples: Sequence[Sequence[Any]],
) -> list[float]:
    sample_array = np.asarray(samples, dtype=np.int64)
    if sample_array.ndim != 2:
        raise RuntimeError("accelerated cluster bootstrap samples must form a matrix")
    baseline_precision, method_precision = _accelerated_cluster_precision_batch(
        baseline.offsets,
        baseline.score_ranks,
        baseline.matches,
        baseline.ignores,
        baseline.gt_counts,
        baseline.rank_counts,
        baseline.recall_thresholds,
        method.offsets,
        method.score_ranks,
        method.matches,
        method.ignores,
        method.gt_counts,
        method.rank_counts,
        method.recall_thresholds,
        clusters.offsets,
        clusters.image_positions,
        sample_array,
    )
    return [
        _mean_valid_coco_precision(method_precision[index])
        - _mean_valid_coco_precision(baseline_precision[index])
        for index in range(len(samples))
    ]


@njit(cache=True, nogil=True)
def _accelerated_cluster_precision_batch(
    baseline_offsets: np.ndarray,
    baseline_score_ranks: np.ndarray,
    baseline_matches: np.ndarray,
    baseline_ignores: np.ndarray,
    baseline_gt_counts: np.ndarray,
    baseline_rank_counts: np.ndarray,
    baseline_recall_thresholds: np.ndarray,
    method_offsets: np.ndarray,
    method_score_ranks: np.ndarray,
    method_matches: np.ndarray,
    method_ignores: np.ndarray,
    method_gt_counts: np.ndarray,
    method_rank_counts: np.ndarray,
    method_recall_thresholds: np.ndarray,
    cluster_offsets: np.ndarray,
    cluster_image_positions: np.ndarray,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_precision = np.empty(
        (
            samples.shape[0],
            baseline_matches.shape[0],
            len(baseline_recall_thresholds),
            baseline_offsets.shape[0],
        ),
        dtype=np.float64,
    )
    method_precision = np.empty(
        (
            samples.shape[0],
            method_matches.shape[0],
            len(method_recall_thresholds),
            method_offsets.shape[0],
        ),
        dtype=np.float64,
    )
    for sample_index in range(samples.shape[0]):
        image_count = 0
        for sampled_cluster in samples[sample_index]:
            image_count += cluster_offsets[sampled_cluster + 1] - cluster_offsets[sampled_cluster]
        image_positions = np.empty(image_count, dtype=np.int64)
        cursor = 0
        for sampled_cluster in samples[sample_index]:
            start = cluster_offsets[sampled_cluster]
            stop = cluster_offsets[sampled_cluster + 1]
            for source_index in range(start, stop):
                image_positions[cursor] = cluster_image_positions[source_index]
                cursor += 1
        baseline_precision[sample_index] = _accelerated_coco_precision(
            baseline_offsets,
            baseline_score_ranks,
            baseline_matches,
            baseline_ignores,
            baseline_gt_counts,
            baseline_rank_counts,
            baseline_recall_thresholds,
            image_positions,
        )
        method_precision[sample_index] = _accelerated_coco_precision(
            method_offsets,
            method_score_ranks,
            method_matches,
            method_ignores,
            method_gt_counts,
            method_rank_counts,
            method_recall_thresholds,
            image_positions,
        )
    return baseline_precision, method_precision


@njit(cache=True, nogil=True)
def _accelerated_coco_precision(
    offsets: np.ndarray,
    score_ranks: np.ndarray,
    matches: np.ndarray,
    ignores: np.ndarray,
    gt_counts: np.ndarray,
    rank_counts: np.ndarray,
    recall_thresholds: np.ndarray,
    sample_positions: np.ndarray,
) -> np.ndarray:
    category_count = offsets.shape[0]
    threshold_count = matches.shape[0]
    recall_count = len(recall_thresholds)
    precision = np.full((threshold_count, recall_count, category_count), -1.0, dtype=np.float64)
    epsilon = np.spacing(1.0)
    for category in range(category_count):
        positive_ground_truth = 0
        event_count = 0
        for position in sample_positions:
            positive_ground_truth += gt_counts[category, position]
            event_count += offsets[category, position + 1] - offsets[category, position]
        if positive_ground_truth == 0:
            continue
        counts = np.zeros(rank_counts[category], dtype=np.int64)
        source_events = np.empty(event_count, dtype=np.int64)
        cursor = 0
        for position in sample_positions:
            for source in range(offsets[category, position], offsets[category, position + 1]):
                source_events[cursor] = source
                counts[score_ranks[source]] += 1
                cursor += 1
        starts = np.empty(len(counts), dtype=np.int64)
        running = 0
        for rank in range(len(counts)):
            starts[rank] = running
            running += int(counts[rank])
        next_slot = starts.copy()
        ordered = np.empty(event_count, dtype=np.int64)
        for raw_source in source_events:
            source_index = int(raw_source)
            rank = int(score_ranks[source_index])
            ordered[next_slot[rank]] = source_index
            next_slot[rank] += 1
        recall = np.empty(event_count, dtype=np.float64)
        category_precision = np.empty(event_count, dtype=np.float64)
        for threshold in range(threshold_count):
            true_positive = 0
            false_positive = 0
            for index in range(event_count):
                source = ordered[index]
                ignored = ignores[threshold, source] != 0
                matched = matches[threshold, source] != 0
                if not ignored:
                    if matched:
                        true_positive += 1
                    else:
                        false_positive += 1
                recall[index] = true_positive / positive_ground_truth
                category_precision[index] = true_positive / (
                    false_positive + true_positive + epsilon
                )
            for index in range(event_count - 1, 0, -1):
                if category_precision[index] > category_precision[index - 1]:
                    category_precision[index - 1] = category_precision[index]
            cursor = 0
            for recall_index in range(recall_count):
                target = recall_thresholds[recall_index]
                while cursor < event_count and recall[cursor] < target:
                    cursor += 1
                precision[threshold, recall_index, category] = (
                    category_precision[cursor] if cursor < event_count else 0.0
                )
    return precision


def _initialize_bootstrap_worker(
    annotation_path: str,
    baseline_prediction_path: str,
    method_prediction_path: str,
    max_det: int,
    reuse_indexes: bool,
    cache_evaluations: bool,
    evaluated_image_ids: Sequence[Any],
    area_label: str,
) -> None:
    global _BOOTSTRAP_WORKER_CACHED_CONTEXT
    global _BOOTSTRAP_WORKER_PREPARED_CONTEXT, _BOOTSTRAP_WORKER_RAW_CONTEXT
    ground_truth = _load_mapping(Path(annotation_path))
    baseline = _load_rows(Path(baseline_prediction_path))
    method = _load_rows(Path(method_prediction_path))
    if cache_evaluations:
        prepared_ground_truth = _prepare_ground_truth(ground_truth)
        _BOOTSTRAP_WORKER_CACHED_CONTEXT = (
            _build_evaluation_cache(
                prepared_ground_truth,
                _prepare_predictions(baseline),
                evaluated_image_ids,
                max_det=max_det,
            ),
            _build_evaluation_cache(
                prepared_ground_truth,
                _prepare_predictions(method),
                evaluated_image_ids,
                max_det=max_det,
            ),
            area_label,
        )
        _BOOTSTRAP_WORKER_PREPARED_CONTEXT = None
        _BOOTSTRAP_WORKER_RAW_CONTEXT = None
    elif reuse_indexes:
        _BOOTSTRAP_WORKER_CACHED_CONTEXT = None
        _BOOTSTRAP_WORKER_PREPARED_CONTEXT = (
            _prepare_ground_truth(ground_truth),
            _prepare_predictions(baseline),
            _prepare_predictions(method),
            max_det,
            area_label,
        )
        _BOOTSTRAP_WORKER_RAW_CONTEXT = None
    else:
        _BOOTSTRAP_WORKER_CACHED_CONTEXT = None
        _BOOTSTRAP_WORKER_PREPARED_CONTEXT = None
        _BOOTSTRAP_WORKER_RAW_CONTEXT = (
            ground_truth,
            baseline,
            method,
            max_det,
            area_label,
        )


def _bootstrap_delta_chunk(samples: Sequence[Sequence[Any]]) -> list[float]:
    if _BOOTSTRAP_WORKER_CACHED_CONTEXT is not None:
        cached_baseline, cached_method, area_label = _BOOTSTRAP_WORKER_CACHED_CONTEXT
        return _bootstrap_delta_chunk_cached(
            cached_baseline,
            cached_method,
            samples,
            area_label=area_label,
        )
    if _BOOTSTRAP_WORKER_PREPARED_CONTEXT is not None:
        ground_truth, prepared_baseline, prepared_method, max_det, area_label = (
            _BOOTSTRAP_WORKER_PREPARED_CONTEXT
        )
        return _bootstrap_delta_chunk_prepared(
            ground_truth,
            prepared_baseline,
            prepared_method,
            samples,
            max_det=max_det,
            area_label=area_label,
        )
    if _BOOTSTRAP_WORKER_RAW_CONTEXT is not None:
        raw_ground_truth, raw_baseline, raw_method, max_det, area_label = (
            _BOOTSTRAP_WORKER_RAW_CONTEXT
        )
        return [
            _coco_ap_for_sample(
                raw_ground_truth,
                raw_method,
                sampled,
                max_det=max_det,
                area_label=area_label,
            )
            - _coco_ap_for_sample(
                raw_ground_truth,
                raw_baseline,
                sampled,
                max_det=max_det,
                area_label=area_label,
            )
            for sampled in samples
        ]
    raise RuntimeError("bootstrap worker is not initialized")


def _bootstrap_delta_chunk_cached(
    baseline: _CocoEvaluationCache,
    method: _CocoEvaluationCache,
    samples: Sequence[Sequence[Any]],
    *,
    area_label: str = "all",
) -> list[float]:
    return [
        _coco_ap_from_evaluation_cache(method, sampled, area_label=area_label)
        - _coco_ap_from_evaluation_cache(baseline, sampled, area_label=area_label)
        for sampled in samples
    ]


def _bootstrap_delta_chunk_prepared(
    ground_truth: _PreparedGroundTruth,
    baseline: _PreparedPredictions,
    method: _PreparedPredictions,
    samples: Sequence[Sequence[Any]],
    *,
    max_det: int,
    area_label: str = "all",
) -> list[float]:
    return [
        _coco_ap_for_prepared_sample(
            ground_truth,
            method,
            sampled,
            max_det=max_det,
            area_label=area_label,
        )
        - _coco_ap_for_prepared_sample(
            ground_truth,
            baseline,
            sampled,
            max_det=max_det,
            area_label=area_label,
        )
        for sampled in samples
    ]


def _coco_ap_for_sample(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    sampled_image_ids: Sequence[Any],
    *,
    max_det: int,
    area_label: str = "all",
) -> float:
    return _coco_ap_for_prepared_sample(
        _prepare_ground_truth(ground_truth),
        _prepare_predictions(predictions),
        sampled_image_ids,
        max_det=max_det,
        area_label=area_label,
    )


def _prepare_ground_truth(ground_truth: Mapping[str, Any]) -> _PreparedGroundTruth:
    images_by_id = {
        row["id"]: row
        for row in ground_truth.get("images", [])
        if isinstance(row, dict) and "id" in row
    }
    raw_annotations: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in ground_truth.get("annotations", []):
        if isinstance(row, dict) and "image_id" in row:
            raw_annotations[row["image_id"]].append(row)
    return _PreparedGroundTruth(
        info=ground_truth.get("info", {}),
        licenses=ground_truth.get("licenses", []),
        categories=ground_truth.get("categories", []),
        images_by_id=images_by_id,
        annotations_by_image={key: tuple(rows) for key, rows in raw_annotations.items()},
    )


def _prepare_predictions(predictions: Sequence[dict[str, Any]]) -> _PreparedPredictions:
    raw_predictions: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        raw_predictions[row["image_id"]].append(row)
    return _PreparedPredictions(
        predictions_by_image={key: tuple(rows) for key, rows in raw_predictions.items()}
    )


def _coco_ap_for_prepared_sample(
    ground_truth: _PreparedGroundTruth,
    predictions: _PreparedPredictions,
    sampled_image_ids: Sequence[Any],
    *,
    max_det: int,
    area_label: str = "all",
) -> float:
    evaluator = _evaluate_prepared_sample(
        ground_truth,
        predictions,
        sampled_image_ids,
        max_det=max_det,
    )
    return 0.0 if evaluator is None else _accumulate_coco_ap(evaluator, area_label=area_label)


def _evaluate_prepared_sample(
    ground_truth: _PreparedGroundTruth,
    predictions: _PreparedPredictions,
    sampled_image_ids: Sequence[Any],
    *,
    max_det: int,
) -> Any | None:
    try:
        from pycocotools.coco import COCO  # type: ignore[import-untyped]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped]
    except ImportError as exc:
        raise EvaluationError("pycocotools is not installed") from exc

    boot_images: list[dict[str, Any]] = []
    boot_annotations: list[dict[str, Any]] = []
    boot_predictions: list[dict[str, Any]] = []
    next_annotation_id = 1
    for new_image_id, source_image_id in enumerate(sampled_image_ids, start=1):
        source_image = ground_truth.images_by_id.get(source_image_id)
        if source_image is None:
            raise EvaluationError(f"bootstrap image is absent from ground truth: {source_image_id}")
        boot_images.append({**source_image, "id": new_image_id})
        for annotation in ground_truth.annotations_by_image.get(source_image_id, ()):
            boot_annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": new_image_id,
                }
            )
            next_annotation_id += 1
        for prediction in predictions.predictions_by_image.get(source_image_id, ()):
            boot_predictions.append({**prediction, "image_id": new_image_id})
    if not boot_predictions:
        return 0.0
    document = {
        "info": ground_truth.info,
        "licenses": ground_truth.licenses,
        "images": boot_images,
        "annotations": boot_annotations,
        "categories": ground_truth.categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = document
        coco.createIndex()
        detections = coco.loadRes(boot_predictions)
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = list(range(1, len(sampled_image_ids) + 1))
        evaluator.params.maxDets = [1, min(100, max_det), max_det]
        evaluator.evaluate()
    return evaluator


def _accumulate_coco_ap(evaluator: Any, *, area_label: str = "all") -> float:
    try:
        stat_index = _COCO_AREA_STAT_INDEX[area_label]
    except KeyError as exc:
        raise ValueError(f"unsupported COCO area label: {area_label}") from exc
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[stat_index])


def _build_evaluation_cache(
    ground_truth: _PreparedGroundTruth,
    predictions: _PreparedPredictions,
    evaluated_image_ids: Sequence[Any],
    *,
    max_det: int,
) -> _CocoEvaluationCache:
    evaluator = _evaluate_prepared_sample(
        ground_truth,
        predictions,
        evaluated_image_ids,
        max_det=max_det,
    )
    positions = {image_id: index for index, image_id in enumerate(evaluated_image_ids)}
    if evaluator is None:
        return _CocoEvaluationCache(eval_imgs=(), params=None, image_position=positions)
    return _CocoEvaluationCache(
        eval_imgs=tuple(evaluator.evalImgs),
        params=copy.deepcopy(evaluator.params),
        image_position=positions,
    )


def _coco_ap_from_evaluation_cache(
    cache: _CocoEvaluationCache,
    sampled_image_ids: Sequence[Any],
    *,
    area_label: str = "all",
) -> float:
    if cache.params is None:
        return 0.0
    try:
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise EvaluationError("pycocotools is not installed") from exc
    sample_positions: list[int] = []
    for image_id in sampled_image_ids:
        position = cache.image_position.get(image_id)
        if position is None:
            raise EvaluationError(f"bootstrap image is absent from cache: {image_id}")
        sample_positions.append(position)
    params = copy.deepcopy(cache.params)
    params.imgIds = list(range(1, len(sample_positions) + 1))
    image_count = len(cache.image_position)
    area_count = len(params.areaRng)
    category_count = len(params.catIds) if params.useCats else 1
    eval_imgs = [
        cache.eval_imgs[(category * area_count + area) * image_count + position]
        for category in range(category_count)
        for area in range(area_count)
        for position in sample_positions
    ]
    evaluator = COCOeval(None, None, "bbox")
    evaluator.params = params
    evaluator._paramsEval = copy.deepcopy(params)
    evaluator.evalImgs = eval_imgs
    return _accumulate_coco_ap(evaluator, area_label=area_label)


def _coco_ap_from_evaluation_cache_ap_only(
    cache: _CocoEvaluationCache, sampled_image_ids: Sequence[Any]
) -> float:
    """Accumulate only the exact area=all/maxDet slice used by COCO AP."""
    if cache.params is None:
        return 0.0
    try:
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise EvaluationError("pycocotools is not installed") from exc
    sample_positions: list[int] = []
    for image_id in sampled_image_ids:
        position = cache.image_position.get(image_id)
        if position is None:
            raise EvaluationError(f"bootstrap image is absent from cache: {image_id}")
        sample_positions.append(position)
    full_params = cache.params
    area_labels = list(full_params.areaRngLbl)
    if "all" not in area_labels or 100 not in full_params.maxDets:
        raise EvaluationError("COCO bootstrap cache lacks the registered AP slice")
    all_area_index = area_labels.index("all")
    full_area_count = len(full_params.areaRng)
    category_count = len(full_params.catIds) if full_params.useCats else 1
    image_count = len(cache.image_position)
    eval_imgs = [
        cache.eval_imgs[(category * full_area_count + all_area_index) * image_count + position]
        for category in range(category_count)
        for position in sample_positions
    ]
    params = copy.deepcopy(full_params)
    params.imgIds = list(range(1, len(sample_positions) + 1))
    params.areaRng = [copy.deepcopy(full_params.areaRng[all_area_index])]
    params.areaRngLbl = [area_labels[all_area_index]]
    # pycocotools stats[0] uses its fixed AP default of maxDets=100.
    params.maxDets = [100]
    evaluator = COCOeval(None, None, "bbox")
    evaluator.params = params
    evaluator._paramsEval = copy.deepcopy(params)
    evaluator.evalImgs = eval_imgs
    evaluator.accumulate()
    precision = evaluator.eval.get("precision")
    if not isinstance(precision, np.ndarray):
        raise EvaluationError("COCO AP-only accumulation produced no precision tensor")
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def _prepare_accelerated_coco_ap_only_cache(
    cache: _CocoEvaluationCache,
) -> _AcceleratedCocoApOnlyCache:
    if cache.params is None:
        raise EvaluationError("accelerated COCO AP requires a nonempty evaluation cache")
    params = cache.params
    area_labels = list(params.areaRngLbl)
    if "all" not in area_labels or 100 not in params.maxDets:
        raise EvaluationError("COCO bootstrap cache lacks the registered AP slice")
    all_area_index = area_labels.index("all")
    area_count = len(params.areaRng)
    category_count = len(params.catIds) if params.useCats else 1
    image_count = len(cache.image_position)
    threshold_count = len(params.iouThrs)
    offsets = np.zeros((category_count, image_count + 1), dtype=np.int64)
    rows: list[list[Any | None]] = []
    total_detections = 0
    for category in range(category_count):
        offsets[category, 0] = total_detections
        category_rows: list[Any | None] = []
        for image in range(image_count):
            row = cache.eval_imgs[(category * area_count + all_area_index) * image_count + image]
            category_rows.append(row)
            total_detections += min(100, len(row["dtScores"])) if row is not None else 0
            offsets[category, image + 1] = total_detections
        rows.append(category_rows)
    score_ranks = np.zeros(total_detections, dtype=np.int32)
    matches = np.zeros((threshold_count, total_detections), dtype=np.uint8)
    ignores = np.zeros((threshold_count, total_detections), dtype=np.uint8)
    gt_counts = np.zeros((category_count, image_count), dtype=np.int64)
    rank_counts = np.zeros(category_count, dtype=np.int64)
    for category, category_rows in enumerate(rows):
        scores: list[float] = []
        for image, row in enumerate(category_rows):
            if row is None:
                continue
            start = int(offsets[category, image])
            stop = int(offsets[category, image + 1])
            detection_count = stop - start
            values = np.asarray(row["dtScores"][:detection_count], dtype=np.float64)
            scores.extend(values.tolist())
            matches[:, start:stop] = np.asarray(
                row["dtMatches"][:, :detection_count] != 0, dtype=np.uint8
            )
            ignores[:, start:stop] = np.asarray(
                row["dtIgnore"][:, :detection_count], dtype=np.uint8
            )
            gt_counts[category, image] = int(np.count_nonzero(np.asarray(row["gtIgnore"]) == 0))
        if scores:
            values = np.asarray(scores, dtype=np.float64)
            unique = np.unique(values)
            rank_counts[category] = len(unique)
            start = int(offsets[category, 0])
            stop = int(offsets[category, -1])
            score_ranks[start:stop] = (len(unique) - 1 - np.searchsorted(unique, values)).astype(
                np.int32
            )
    return _AcceleratedCocoApOnlyCache(
        offsets=offsets,
        score_ranks=score_ranks,
        matches=matches,
        ignores=ignores,
        gt_counts=gt_counts,
        rank_counts=rank_counts,
        recall_thresholds=np.asarray(params.recThrs, dtype=np.float64),
        image_position=dict(cache.image_position),
    )


def _prepare_cluster_position_cache(
    clusters: Sequence[Sequence[Any]], image_position: Mapping[Any, int]
) -> _ClusterPositionCache:
    offsets = np.zeros(len(clusters) + 1, dtype=np.int64)
    positions: list[int] = []
    for cluster_index, cluster in enumerate(clusters):
        for image_id in cluster:
            position = image_position.get(image_id)
            if position is None:
                raise EvaluationError(f"cluster image is absent from cache: {image_id}")
            positions.append(position)
        offsets[cluster_index + 1] = len(positions)
    return _ClusterPositionCache(
        offsets=offsets,
        image_positions=np.asarray(positions, dtype=np.int64),
    )


def _coco_ap_from_accelerated_cache(
    cache: _AcceleratedCocoApOnlyCache, sampled_image_ids: Sequence[Any]
) -> float:
    positions: list[int] = []
    for image_id in sampled_image_ids:
        position = cache.image_position.get(image_id)
        if position is None:
            raise EvaluationError(f"bootstrap image is absent from cache: {image_id}")
        positions.append(position)
    precision = _accelerated_coco_precision(
        cache.offsets,
        cache.score_ranks,
        cache.matches,
        cache.ignores,
        cache.gt_counts,
        cache.rank_counts,
        cache.recall_thresholds,
        np.asarray(positions, dtype=np.int64),
    )
    return _mean_valid_coco_precision(precision)


def _mean_valid_coco_precision(precision: np.ndarray) -> float:
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def _bootstrap_checkpoint_identity(
    *,
    annotation_path: Path,
    baseline_prediction_path: Path,
    method_prediction_path: Path,
    evaluated_image_ids: Sequence[Any],
    resamples: int,
    seed: int,
    max_det: int,
    scope: Mapping[str, Any] | None,
    area_label: str = "all",
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "protocol": "paired_coco_ap_bootstrap_chunks",
        "annotation": str(annotation_path.resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "baseline_prediction": str(baseline_prediction_path.resolve()),
        "baseline_prediction_sha256": sha256_file(baseline_prediction_path),
        "method_prediction": str(method_prediction_path.resolve()),
        "method_prediction_sha256": sha256_file(method_prediction_path),
        "image_ids_sha256": stable_hash(list(evaluated_image_ids), length=64),
        "images": len(evaluated_image_ids),
        "resamples": resamples,
        "seed": seed,
        "max_det": max_det,
        "scope": dict(scope or {}),
    }
    if area_label != "all":
        identity["area_label"] = area_label
    return identity


def _load_or_initialize_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    observed_baseline: float,
    observed_method: float,
    resamples: int,
) -> tuple[dict[str, Any], list[float | None]]:
    if not path.is_file():
        deltas: list[float | None] = [None] * resamples
        state = {
            "identity": dict(identity),
            "observed_baseline": observed_baseline,
            "observed_method": observed_method,
            "completed_resamples": 0,
            "deltas": deltas,
        }
        atomic_write_json(path, state)
        return state, deltas
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot parse COCO bootstrap checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("COCO bootstrap checkpoint is not an object")
    if state.get("identity") != dict(identity):
        raise ValueError("COCO bootstrap checkpoint identity mismatch")
    if state.get("observed_baseline") != observed_baseline:
        raise ValueError("COCO bootstrap checkpoint baseline AP mismatch")
    if state.get("observed_method") != observed_method:
        raise ValueError("COCO bootstrap checkpoint method AP mismatch")
    raw_deltas = state.get("deltas")
    if not isinstance(raw_deltas, list) or len(raw_deltas) != resamples:
        raise ValueError("COCO bootstrap checkpoint has invalid delta coverage")
    deltas = []
    for value in raw_deltas:
        if value is None:
            deltas.append(None)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            deltas.append(float(value))
        else:
            raise ValueError("COCO bootstrap checkpoint contains an invalid delta")
    completed = sum(delta is not None for delta in deltas)
    if state.get("completed_resamples") != completed:
        raise ValueError("COCO bootstrap checkpoint completed count mismatch")
    return state, deltas


def _pending_chunks(
    samples: Sequence[list[Any]],
    deltas: Sequence[float | None],
    *,
    chunk_size: int,
) -> list[tuple[list[int], list[list[Any]]]]:
    pending = [index for index, delta in enumerate(deltas) if delta is None]
    return [
        (indices, [samples[index] for index in indices])
        for start in range(0, len(pending), chunk_size)
        if (indices := pending[start : start + chunk_size])
    ]


def _pending_unique_sample_chunks(
    samples: Sequence[list[Any]],
    deltas: Sequence[float | None],
    *,
    chunk_size: int,
) -> list[tuple[list[list[int]], list[list[Any]]]]:
    if len(samples) != len(deltas):
        raise RuntimeError("COCO bootstrap sample and delta counts differ")
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, (sample, delta) in enumerate(
        zip(samples, deltas)  # noqa: B905 -- lengths checked above
    ):
        if delta is None:
            grouped.setdefault(tuple(sample), []).append(index)
    unique = list(grouped.items())
    return [
        (
            [indices for _, indices in chunk],
            [list(sample) for sample, _ in chunk],
        )
        for start in range(0, len(unique), chunk_size)
        if (chunk := unique[start : start + chunk_size])
    ]


def _record_completed_sample_groups(
    index_groups: Sequence[Sequence[int]],
    values: Sequence[float],
    *,
    delta_slots: list[float | None],
    checkpoint_path: Path | None,
    checkpoint_state: dict[str, Any] | None,
) -> None:
    if len(index_groups) != len(values):
        raise RuntimeError("COCO bootstrap worker returned the wrong unique-sample count")
    indices = [index for group in index_groups for index in group]
    expanded = [
        value
        for group, value in zip(  # noqa: B905 -- lengths checked above
            index_groups, values
        )
        for _ in group
    ]
    _record_completed_chunk(
        indices,
        expanded,
        delta_slots=delta_slots,
        checkpoint_path=checkpoint_path,
        checkpoint_state=checkpoint_state,
    )


def _record_completed_chunk(
    indices: Sequence[int],
    values: Sequence[float],
    *,
    delta_slots: list[float | None],
    checkpoint_path: Path | None,
    checkpoint_state: dict[str, Any] | None,
) -> None:
    if len(indices) != len(values):
        raise RuntimeError("COCO bootstrap worker returned the wrong chunk size")
    for index, value in zip(  # noqa: B905 -- lengths checked above
        indices, values
    ):
        if delta_slots[index] is not None:
            raise RuntimeError(f"COCO bootstrap resample {index} completed twice")
        if not np.isfinite(value):
            raise RuntimeError(f"COCO bootstrap resample {index} is not finite")
        delta_slots[index] = float(value)
    if checkpoint_path is not None and checkpoint_state is not None:
        checkpoint_state["deltas"] = delta_slots
        checkpoint_state["completed_resamples"] = sum(delta is not None for delta in delta_slots)
        atomic_write_json(checkpoint_path, checkpoint_state)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"expected a JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise EvaluationError(f"expected a JSON row list: {path}")
    return value
