from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from buse_uav.utils.io import atomic_write_text

TIMING_FIELDS = (
    "run_id",
    "image_id",
    "load_ms",
    "base_ms",
    "flip_ms",
    "scoring_ms",
    "crop_ms",
    "enhance_ms",
    "candidate_ms",
    "utility_ms",
    "fusion_ms",
    "total_ms",
    "num_regions",
    "num_candidates",
    "num_calls",
    "num_full_calls",
    "eic",
    "peak_vram_mb",
    "cpu_memory_mb",
)


def synchronize_cuda(device: str) -> bool:
    """Synchronize a requested CUDA device and report whether it was used."""
    if not device.casefold().startswith("cuda"):
        return False
    try:
        import torch
    except (ImportError, OSError):
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize(_cuda_index(device))
    return True


@dataclass
class SynchronizedTimer(AbstractContextManager["SynchronizedTimer"]):
    """Wall-clock timer with mandatory CUDA synchronization at both boundaries."""

    device: str = "cpu"
    elapsed_ms: float = 0.0
    synchronized: bool = False
    _started: float = 0.0

    def __enter__(self) -> SynchronizedTimer:
        self.synchronized = synchronize_cuda(self.device)
        self._started = perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.synchronized = synchronize_cuda(self.device) or self.synchronized
        self.elapsed_ms = (perf_counter() - self._started) * 1000.0


def equivalent_inference_cost(
    crop_sizes: Iterable[int | tuple[int, int]],
    *,
    full_size: int | tuple[int, int],
    full_calls: int = 1,
) -> float:
    """Compute the specification's detection-equivalent inference cost (EIC)."""
    if full_calls < 0:
        raise ValueError("full_calls must be nonnegative")
    full_area = _size_area(full_size)
    if full_area <= 0:
        raise ValueError("full_size must have positive dimensions")
    eic = float(full_calls)
    for size in crop_sizes:
        area = _size_area(size)
        if area <= 0:
            raise ValueError("crop sizes must have positive dimensions")
        eic += area / full_area
    return eic


def timing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate per-image timing traces into deterministic efficiency metrics."""
    if not rows:
        raise ValueError("at least one timing row is required")
    totals = _finite_values(rows, "total_ms")
    if not totals or any(value <= 0.0 for value in totals):
        raise ValueError("timing rows require finite, positive total_ms values")
    output = {
        "latency_mean_ms": float(np.mean(totals)),
        "latency_excluding_load_mean_ms": float(
            np.mean([float(row["total_ms"]) - float(row.get("load_ms", 0.0)) for row in rows])
        ),
        "latency_std_ms": float(np.std(totals, ddof=0)),
        "latency_p50_ms": float(np.percentile(totals, 50)),
        "latency_p95_ms": float(np.percentile(totals, 95)),
        "fps": 1000.0 / float(np.mean(totals)),
        "peak_vram_mb": max(_finite_values(rows, "peak_vram_mb") or [0.0]),
        "cpu_memory_peak_mb": max(_finite_values(rows, "cpu_memory_mb") or [0.0]),
        "num_regions_mean": _mean_or_zero(rows, "num_regions"),
        "num_candidates_mean": _mean_or_zero(rows, "num_candidates"),
        "num_calls_mean": _mean_or_zero(rows, "num_calls"),
        "eic_mean": _mean_or_zero(rows, "eic"),
    }
    for stage in (
        "load_ms",
        "base_ms",
        "flip_ms",
        "scoring_ms",
        "crop_ms",
        "enhance_ms",
        "candidate_ms",
        "utility_ms",
        "fusion_ms",
    ):
        output[f"{stage.removesuffix('_ms')}_mean_ms"] = _mean_or_zero(rows, stage)
    return output


def pareto_mask(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_key: str = "AP",
    latency_key: str = "latency_mean_ms",
    eic_key: str = "eic_mean",
) -> tuple[bool, ...]:
    """Return True for configurations not dominated on AP, latency, and EIC."""
    points = [
        (
            _finite_number(row, accuracy_key),
            _finite_number(row, latency_key),
            _finite_number(row, eic_key),
        )
        for row in rows
    ]
    result: list[bool] = []
    for index, (accuracy, latency, eic) in enumerate(points):
        dominated = False
        for other_index, (other_accuracy, other_latency, other_eic) in enumerate(points):
            if index == other_index:
                continue
            no_worse = other_accuracy >= accuracy and other_latency <= latency and other_eic <= eic
            strictly_better = (
                other_accuracy > accuracy or other_latency < latency or other_eic < eic
            )
            if no_worse and strictly_better:
                dominated = True
                break
        result.append(not dominated)
    return tuple(result)


def peak_vram_mb(device: str) -> float:
    if not device.casefold().startswith("cuda"):
        return 0.0
    try:
        import torch
    except (ImportError, OSError):
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(_cuda_index(device)) / (1024**2))


def cpu_memory_mb() -> float:
    try:
        import psutil
    except ImportError:
        return 0.0
    return float(psutil.Process().memory_info().rss / (1024**2))


def reset_peak_vram(device: str) -> None:
    if not device.casefold().startswith("cuda"):
        return
    try:
        import torch
    except (ImportError, OSError):
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(_cuda_index(device))
        except RuntimeError:
            # Some Windows/PyTorch driver combinations reject this API even
            # though synchronization and allocation queries remain available.
            return


def write_timing_traces(
    trace_directory: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write readable JSONL plus the specification-required Parquet trace."""
    trace_directory.mkdir(parents=True, exist_ok=True)
    normalized = [_normalized_timing_row(row) for row in rows]
    import json

    atomic_write_text(
        trace_directory / "timings.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in normalized
        ),
    )
    frame = pd.DataFrame(normalized, columns=TIMING_FIELDS)
    destination = trace_directory / "timings.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def read_timing_traces(trace_directory: Path) -> list[dict[str, Any]]:
    parquet = trace_directory / "timings.parquet"
    if parquet.is_file():
        raw_rows = pd.read_parquet(parquet).to_dict(orient="records")
        return [{str(key): value for key, value in row.items()} for row in raw_rows]
    jsonl = trace_directory / "timings.jsonl"
    if not jsonl.is_file():
        raise ValueError(f"timing trace is missing: {parquet}")
    import json

    rows: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"invalid timing row in {jsonl}")
        rows.append(parsed)
    return rows


def _normalized_timing_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in TIMING_FIELDS if field not in row]
    if missing:
        raise ValueError(f"timing row is missing fields: {missing}")
    output = {field: row[field] for field in TIMING_FIELDS}
    for field in TIMING_FIELDS[2:]:
        value = _finite_number(output, field)
        if value < 0.0:
            raise ValueError(f"timing field {field} must be nonnegative")
        output[field] = value
    return output


def _size_area(size: int | tuple[int, int]) -> float:
    if isinstance(size, int):
        return float(size * size)
    width, height = size
    return float(width * height)


def _cuda_index(device: str) -> int | None:
    _, separator, raw_index = device.partition(":")
    if not separator:
        return None
    try:
        return int(raw_index)
    except ValueError as exc:
        raise ValueError(f"invalid CUDA device: {device}") from exc


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key not in row:
            continue
        values.append(_finite_number(row, key))
    return values


def _mean_or_zero(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = _finite_values(rows, key)
    return float(np.mean(values)) if values else 0.0


def _finite_number(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
