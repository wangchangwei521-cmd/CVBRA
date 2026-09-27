from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

from buse_uav.detectors.base import DetectorAdapter
from buse_uav.evaluation.timing import cpu_memory_mb, peak_vram_mb
from buse_uav.pipeline.selective import SelectiveOutput, apply_identity_crop_selection
from buse_uav.pipeline.trace import RunDirectory
from buse_uav.regions.grid import make_grid
from buse_uav.schemas import AppConfig, DetectionBatch, ImageRecord
from buse_uav.scoring.uncertainty import max_region_uncertainty
from buse_uav.utils.io import atomic_write_json, atomic_write_text


def apply_fast_cawbf_gate(
    config: AppConfig,
    run: RunDirectory,
    detector: DetectorAdapter,
    records: Sequence[ImageRecord],
    base_probe: Sequence[DetectionBatch],
) -> SelectiveOutput:
    """Run the unchanged CAWBF path only for high-U images."""
    if config.method.name != "buse_cawbf_fast":
        raise ValueError("fast CAWBF gate requires method.name=buse_cawbf_fast")
    if config.fast_gate.score != "max_region_uncertainty":
        raise ValueError(f"unsupported fast gate score: {config.fast_gate.score}")

    base_by_id = {batch.image_id: batch for batch in base_probe}
    if set(base_by_id) != {record.image_id for record in records}:
        raise ValueError("fast gate base detections must cover every input image")

    decisions: list[dict[str, Any]] = []
    gate_ms_by_id: dict[int | str, float] = {}
    active_records: list[ImageRecord] = []
    active_base: list[DetectionBatch] = []
    for record in records:
        started = perf_counter()
        regions = make_grid(
            (record.height, record.width, 3),
            rows=config.regions.rows,
            cols=config.regions.cols,
            context_padding=config.regions.context_padding,
        )
        score = max_region_uncertainty(
            regions,
            base_by_id[record.image_id],
            publish_conf=config.detector.publish_conf,
            threshold_sigma=config.scoring.threshold_sigma,
            density_kappa=config.scoring.density_kappa,
            rank_mix=config.scoring.rank_mix,
            weights=config.scoring.uncertainty_weights,
        )
        gate_ms = (perf_counter() - started) * 1000.0
        activated = score >= config.fast_gate.threshold
        gate_ms_by_id[record.image_id] = gate_ms
        decisions.append(
            {
                "image_id": record.image_id,
                "score": score,
                "threshold": config.fast_gate.threshold,
                "activated": activated,
                "gate_ms": gate_ms,
            }
        )
        if activated:
            active_records.append(record)
            active_base.append(base_by_id[record.image_id])

    active_output = (
        apply_identity_crop_selection(
            config,
            run,
            detector,
            active_records,
            active_base,
        )
        if active_records
        else _empty_active_output(config, run)
    )
    activated_ids = {record.image_id for record in active_records}
    active_final = {batch.image_id: batch for batch in active_output.final}
    active_local = {batch.image_id: batch for batch in active_output.local_pre_fusion}
    active_pre_fusion = {batch.image_id: batch for batch in active_output.pre_fusion}
    active_timings = {row["image_id"]: row for row in active_output.timings}
    vram = peak_vram_mb(config.experiment.device)
    memory = cpu_memory_mb()

    final: list[DetectionBatch] = []
    local_pre_fusion: list[DetectionBatch] = []
    pre_fusion: list[DetectionBatch] = []
    timings: list[dict[str, Any]] = []
    for record in records:
        image_id = record.image_id
        gate_ms = gate_ms_by_id[image_id]
        if image_id in activated_ids:
            final.append(
                replace(
                    active_final[image_id],
                    meta={**active_final[image_id].meta, "fast_gate_activated": True},
                )
            )
            local_pre_fusion.append(active_local[image_id])
            pre_fusion.append(active_pre_fusion[image_id])
            timing = dict(active_timings[image_id])
            timing["scoring_ms"] = float(timing["scoring_ms"]) + gate_ms
            timing["total_ms"] = float(timing["total_ms"]) + gate_ms
            timings.append(timing)
            continue

        base = base_by_id[image_id]
        bypass_boxes = tuple(box for box in base.boxes if box.score >= config.detector.publish_conf)
        final.append(
            DetectionBatch(
                image_id=image_id,
                boxes=bypass_boxes,
                latency_ms=base.latency_ms,
                meta={
                    "method": "fast_gate_b0_bypass",
                    "fusion_ms": 0.0,
                    "fast_gate_activated": False,
                },
            )
        )
        local_pre_fusion.append(
            DetectionBatch(
                image_id=image_id,
                boxes=(),
                latency_ms=0.0,
                meta={
                    "method": "fast_gate_b0_bypass",
                    "selected_regions": 0,
                    "local_q_by_region": {},
                },
            )
        )
        pre_fusion.append(
            DetectionBatch(
                image_id=image_id,
                boxes=base.boxes,
                latency_ms=base.latency_ms,
                meta={"method": "fast_gate_b0_bypass_pre_fusion"},
            )
        )
        timings.append(
            {
                "run_id": run.run_id,
                "image_id": image_id,
                "load_ms": 0.0,
                "base_ms": base.latency_ms,
                "flip_ms": 0.0,
                "scoring_ms": gate_ms,
                "crop_ms": 0.0,
                "enhance_ms": 0.0,
                "candidate_ms": 0.0,
                "utility_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": base.latency_ms + gate_ms,
                "num_regions": 0,
                "num_candidates": 0,
                "num_calls": 1,
                "num_full_calls": 1,
                "eic": 1.0,
                "peak_vram_mb": vram,
                "cpu_memory_mb": memory,
            }
        )

    atomic_write_json(
        run.path / "traces" / "fast_gate_summary.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "method": config.method.name,
            "score": config.fast_gate.score,
            "threshold": config.fast_gate.threshold,
            "images": len(records),
            "activated_images": len(active_records),
            "activation_rate": len(active_records) / len(records) if records else 0.0,
            "decisions": decisions,
        },
    )
    return SelectiveOutput(
        final=tuple(final),
        local_pre_fusion=tuple(local_pre_fusion),
        pre_fusion=tuple(pre_fusion),
        timings=tuple(timings),
    )


def _empty_active_output(config: AppConfig, run: RunDirectory) -> SelectiveOutput:
    atomic_write_text(run.path / "traces" / "regions.jsonl", "")
    atomic_write_json(
        run.path / "traces" / "selection_summary.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "method": config.method.name,
            "selection": config.method.selection,
            "images": [],
        },
    )
    if config.runtime.save_trace:
        atomic_write_text(run.path / "traces" / "candidates.jsonl", "")
        atomic_write_json(
            run.path / "traces" / "candidate_summary.json",
            {"schema_version": 1, "run_id": run.run_id, "images": []},
        )
    return SelectiveOutput(final=(), local_pre_fusion=(), pre_fusion=(), timings=())
