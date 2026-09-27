from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class Box:
    """A detection box in absolute original-image coordinates."""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    source: str = "base"
    region_id: int | None = None
    operation: str | None = None


@dataclass(frozen=True)
class ImageRecord:
    image_id: int | str
    path: str
    width: int
    height: int
    image_bgr: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class DetectionBatch:
    image_id: int | str
    boxes: tuple[Box, ...]
    latency_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    id: int
    core_xyxy: tuple[int, int, int, int]
    crop_xyxy: tuple[int, int, int, int]
    area_ratio: float


@dataclass(frozen=True)
class RegionScore:
    region_id: int
    degradation: float
    uncertainty: float
    difficulty: float
    components: dict[str, float]


@dataclass(frozen=True)
class UtilityResult:
    q: float
    confidence_gain: float
    stability: float
    rescue: float
    unsupported_fp: float
    count_explosion: float
    compute: float


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str = "buse_uav"
    seed: int = 42
    deterministic: bool = True
    output_root: Path = Path("runs")


class DatasetConfig(StrictModel):
    name: str
    root: Path
    split: str
    images: Path
    annotations: Path
    class_names: list[str]
    image_variant: Literal["hazy", "clean"] | None = None
    images_root: Path | None = None
    corruption_name: str | None = None
    corruption_severity: int | None = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def validate_corruption_identity(self) -> DatasetConfig:
        fields = (self.corruption_name, self.corruption_severity)
        if (fields[0] is None) != (fields[1] is None):
            raise ValueError("dataset corruption_name and corruption_severity must be set together")
        if fields[0] is not None and self.name != "visdrone":
            raise ValueError("dataset corruptions are supported only for visdrone")
        return self


class DetectorConfig(StrictModel):
    backend: Literal["ultralytics", "mmdet"]
    name: str
    model: Path
    definition: Path | None = None
    device: str
    full_imgsz: int = Field(gt=0)
    crop_imgsz: int = Field(gt=0)
    probe_conf: float = Field(ge=0.0, le=1.0)
    publish_conf: float = Field(gt=0.0, le=1.0)
    nms_iou: float = Field(gt=0.0, le=1.0)
    max_det: int = Field(gt=0)
    fp16: bool

    @model_validator(mode="after")
    def validate_thresholds_and_sizes(self) -> DetectorConfig:
        if self.probe_conf >= self.publish_conf:
            raise ValueError("detector.probe_conf must be smaller than publish_conf")
        if self.crop_imgsz > self.full_imgsz:
            raise ValueError("detector.crop_imgsz must not exceed full_imgsz")
        if self.backend == "mmdet" and self.definition is None:
            raise ValueError("detector.definition is required for backend=mmdet")
        return self


class RegionsConfig(StrictModel):
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    context_padding: float = Field(ge=0.0, le=0.5)
    area_budget: float = Field(ge=0.0, le=1.0)
    max_regions: int = Field(ge=0)
    blank_suppression: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_region_count(self) -> RegionsConfig:
        if self.max_regions > self.rows * self.cols:
            raise ValueError("regions.max_regions must not exceed rows * cols")
        return self


class FlipConsistencyConfig(StrictModel):
    enabled: bool
    weight: float = Field(ge=0.0)


class ScoringConfig(StrictModel):
    calibration: Path
    rank_mix: float = Field(ge=0.0, le=1.0)
    threshold_sigma: float = Field(gt=0.0)
    density_kappa: float = Field(gt=0.0)
    degradation_weights: dict[str, float]
    uncertainty_weights: dict[str, float]
    flip_consistency: FlipConsistencyConfig

    @model_validator(mode="after")
    def validate_branch_weights(self) -> ScoringConfig:
        if set(self.degradation_weights) != {
            "luminance",
            "contrast",
            "blur",
            "haze",
            "entropy",
        }:
            raise ValueError("scoring.degradation_weights has unexpected component names")
        if set(self.uncertainty_weights) != {
            "confidence_entropy",
            "threshold_proximity",
            "class_conflict",
            "low_conf_density",
        }:
            raise ValueError("scoring.uncertainty_weights has unexpected component names")
        _validate_nonnegative_unit_sum(self.degradation_weights, "scoring.degradation_weights")
        _validate_nonnegative_unit_sum(self.uncertainty_weights, "scoring.uncertainty_weights")
        return self


class DifficultyConfig(StrictModel):
    alpha: float = Field(ge=0.0, le=1.0)
    interaction_lambda: float = Field(ge=0.0)


class FastGateConfig(StrictModel):
    score: Literal["max_region_uncertainty"] = "max_region_uncertainty"
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


class GuardConfig(StrictModel):
    score: Literal["max_region_uncertainty"] = "max_region_uncertainty"
    activation_rate: float = Field(default=0.30, gt=0.0, le=1.0)
    fusion_mode: Literal["hard_nms_union", "consensus_anchor"] = "hard_nms_union"
    consensus_iou: float = Field(default=0.50, gt=0.0, le=1.0)
    rescue_conf: float = Field(default=0.25, gt=0.0, le=1.0)


class PackedViewConfig(StrictModel):
    score: Literal["max_region_uncertainty"] = "max_region_uncertainty"
    canvas_imgsz: Literal[896, 1024, 1152] = 1024
    activation_rate: float = Field(default=0.5, gt=0.0, le=1.0)
    grid: tuple[int, int] = (4, 4)
    selected_regions: int = Field(default=4, gt=0)
    context_padding: float = Field(default=0.10, ge=0.0, le=0.5)
    layout: tuple[int, int] = (2, 2)
    horizontal_flip: bool = True
    auxiliary_calls_per_activated_image: int = Field(default=1, gt=0)
    content_overlap_min: float = Field(default=0.80, gt=0.0, le=1.0)
    fill_value: int = Field(default=114, ge=0, le=255)

    @model_validator(mode="after")
    def validate_registered_geometry(self) -> PackedViewConfig:
        if self.activation_rate not in {0.5, 0.75, 1.0}:
            raise ValueError("packed_view.activation_rate must be one of 0.50, 0.75, or 1.00")
        if self.grid != (4, 4):
            raise ValueError("packed_view.grid is frozen at [4, 4]")
        if self.selected_regions != 4:
            raise ValueError("packed_view.selected_regions is frozen at 4")
        if self.context_padding != 0.10:
            raise ValueError("packed_view.context_padding is frozen at 0.10")
        if self.layout != (2, 2):
            raise ValueError("packed_view.layout is frozen at [2, 2]")
        if not self.horizontal_flip:
            raise ValueError("packed_view.horizontal_flip must remain enabled")
        if self.auxiliary_calls_per_activated_image != 1:
            raise ValueError("packed_view.auxiliary_calls_per_activated_image is frozen at 1")
        if self.content_overlap_min != 0.80:
            raise ValueError("packed_view.content_overlap_min is frozen at 0.80")
        if self.fill_value != 114:
            raise ValueError("packed_view.fill_value is frozen at 114")
        return self


class EnhancementConfig(StrictModel):
    include_identity: bool
    candidates: list[Literal["gamma", "clahe", "unsharp", "dcp"]]
    max_ops_per_region: int = Field(ge=0)
    gamma: float = Field(gt=0.0)
    clahe_clip_limit: float = Field(gt=0.0)
    clahe_tile_grid: tuple[int, int]
    unsharp_amount: float = Field(ge=0.0)
    unsharp_sigma: float = Field(gt=0.0)
    early_stop_q: float
    early_stop_min_stability: float = Field(ge=0.0, le=1.0)
    early_stop_max_unsupported_fp: float = Field(default=0.0, ge=0.0, le=1.0)


class UtilityWeightsConfig(StrictModel):
    confidence_gain: float = Field(ge=0.0)
    stability: float = Field(ge=0.0)
    rescue: float = Field(ge=0.0)
    unsupported_fp: float = Field(ge=0.0)
    count_explosion: float = Field(ge=0.0)
    compute: float = Field(ge=0.0)


class UtilityConfig(StrictModel):
    match_iou: float = Field(gt=0.0, le=1.0)
    max_count_ratio: float = Field(gt=0.0)
    conservative_gate: bool = False
    weights: UtilityWeightsConfig


class FusionConfig(StrictModel):
    method: Literal["wbf", "anchored_wbf", "soft_nms", "hard_nms"]
    iou: float = Field(gt=0.0, le=1.0)
    soft_nms_sigma: float = Field(gt=0.0)
    base_weight: float = Field(gt=0.0)
    local_weight_min: float = Field(gt=0.0)
    local_weight_max: float = Field(gt=0.0)
    anchor_iou: float = Field(default=0.70, gt=0.0, le=1.0)
    reliability_q_min: float = 0.0
    rescue_conf: float = Field(default=0.35, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_local_weights(self) -> FusionConfig:
        if self.local_weight_min > self.local_weight_max:
            raise ValueError("fusion.local_weight_min must not exceed local_weight_max")
        if self.anchor_iou < self.iou:
            raise ValueError("fusion.anchor_iou must not be below fusion.iou")
        return self


class RuntimeConfig(StrictModel):
    batch_candidates: bool
    cross_shape_candidate_batching: bool = False
    num_workers: int = Field(ge=0)
    degradation_workers: int = Field(default=1, ge=0)
    save_trace: bool
    save_intermediates: bool
    profile: bool
    resume: bool
    tune: bool = False
    require_paths: bool = False
    in_memory_candidates: bool = False
    stream_chunk_records: int = Field(default=8, gt=0)
    release_cuda_cache_between_chunks: bool = True


class MethodConfig(StrictModel):
    name: str
    selection: Literal["random", "degradation", "uncertainty", "joint"]
    enhancement_enabled: bool
    utility_enabled: bool
    early_stop_enabled: bool
    fusion_enabled: bool


class TrainConfig(StrictModel):
    epochs: int = Field(gt=0)
    resource_cap_epochs: int | None = Field(default=None, gt=0)
    imgsz: int = Field(gt=0)
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    patience: int | None = Field(default=None, ge=0)
    batch: int | None = Field(default=None, gt=0)
    mosaic: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_resource_cap(self) -> TrainConfig:
        if self.resource_cap_epochs is not None and self.resource_cap_epochs > self.epochs:
            raise ValueError("train.resource_cap_epochs must not exceed train.epochs")
        return self


class ExperimentConfig(StrictModel):
    name: str
    max_images: int | None = Field(default=None, gt=0)
    device: str
    train: TrainConfig | None


class CorruptionConfig(StrictModel):
    name: str
    splits: list[str]
    corruptions: dict[str, list[float]]


class AppConfig(StrictModel):
    project: ProjectConfig
    dataset: DatasetConfig
    detector: DetectorConfig
    method: MethodConfig
    experiment: ExperimentConfig
    corruption: CorruptionConfig
    regions: RegionsConfig
    scoring: ScoringConfig
    difficulty: DifficultyConfig
    fast_gate: FastGateConfig = Field(default_factory=FastGateConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    packed_view: PackedViewConfig = Field(default_factory=PackedViewConfig)
    enhancement: EnhancementConfig
    utility: UtilityConfig
    fusion: FusionConfig
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def enforce_data_policy_and_paths(self) -> AppConfig:
        split = self.dataset.split.casefold()
        test_like = split in {"test", "test-dev", "rddts"} or split.startswith("test-")
        if test_like and self.runtime.tune:
            raise ValueError("runtime.tune=true is forbidden for test and RDDTS splits")
        if self.dataset.name == "uavdt":
            if self.runtime.tune:
                raise ValueError("runtime.tune=true is forbidden for the zero-shot UAVDT protocol")
            if self.experiment.train is not None:
                raise ValueError("training is forbidden for the zero-shot UAVDT protocol")
        if self.dataset.name == "auair":
            if self.runtime.tune:
                raise ValueError("runtime.tune=true is forbidden for the zero-shot AU-AIR protocol")
            if self.experiment.train is not None:
                raise ValueError("training is forbidden for the zero-shot AU-AIR protocol")
        if self.dataset.name == "dronevehicle":
            if self.runtime.tune:
                raise ValueError(
                    "runtime.tune=true is forbidden for the zero-shot DroneVehicle protocol"
                )
            if self.experiment.train is not None:
                raise ValueError("training is forbidden for the zero-shot DroneVehicle protocol")
        if self.method.name == "buse_cawbf_fast":
            if self.scoring.flip_consistency.enabled:
                raise ValueError(
                    "buse_cawbf_fast forbids flip consistency because its gate must use only "
                    "the single B0 pass"
                )
            if self.fusion.method != "anchored_wbf":
                raise ValueError("buse_cawbf_fast requires fusion.method=anchored_wbf")
        if self.method.name == "duq_guard":
            if self.scoring.flip_consistency.enabled:
                raise ValueError("duq_guard uses an explicit active-image Flip pass")
            if self.regions.max_regions != 1:
                raise ValueError("duq_guard requires exactly one selected region")
            if self.enhancement.max_ops_per_region != 1:
                raise ValueError("duq_guard requires exactly one routed enhancement")
            if self.fusion.method != "hard_nms":
                raise ValueError("duq_guard requires inner fusion.method=hard_nms")
        if self.method.name == "duq_pvf":
            if self.scoring.flip_consistency.enabled:
                raise ValueError("duq_pvf forbids the separate Flip-consistency pass")
            if self.method.selection != "joint":
                raise ValueError("duq_pvf requires frozen joint D-U selection")
            if self.method.enhancement_enabled or self.method.early_stop_enabled:
                raise ValueError("duq_pvf forbids the historical enhancement candidate bank")
            if not self.method.utility_enabled or not self.method.fusion_enabled:
                raise ValueError("duq_pvf requires frozen Q admission and WBF")
            if self.fusion.method != "wbf":
                raise ValueError("duq_pvf requires fusion.method=wbf")
            if (self.regions.rows, self.regions.cols) != self.packed_view.grid:
                raise ValueError("duq_pvf regions must match packed_view.grid=[4, 4]")
            if self.regions.context_padding != self.packed_view.context_padding:
                raise ValueError("duq_pvf requires exactly 10% region context")
            if self.regions.max_regions != self.packed_view.selected_regions:
                raise ValueError("duq_pvf requires exactly four selected regions")
            if not self.runtime.in_memory_candidates:
                raise ValueError("duq_pvf requires in-memory packed auxiliary views")
            if self.detector.name not in {"yolo11n", "rtdetr_l"}:
                raise ValueError("duq_pvf is registered only for YOLO11n and RT-DETR-L")
            expected_model = {
                "yolo11n": Path("weights/hazydet/yolo11n_best.pt"),
                "rtdetr_l": Path("weights/hazydet/rtdetr_l_best.pt"),
            }[self.detector.name]
            if self.detector.model != expected_model:
                raise ValueError(
                    f"duq_pvf requires frozen {self.detector.name} weight {expected_model}"
                )
            if self.experiment.train is not None:
                raise ValueError("training is forbidden for the duq_pvf study")
            if self.dataset.name == "hazydet":
                if split != "val":
                    raise ValueError("duq_pvf HazyDet development is validation-only")
                if self.experiment.max_images is None or self.experiment.max_images > 300:
                    raise ValueError(
                        "duq_pvf HazyDet access is capped at the registered 300 images"
                    )
                if not self.runtime.tune:
                    raise ValueError(
                        "duq_pvf HazyDet candidate-bank runs require runtime.tune=true"
                    )
            elif self.dataset.name in {"visdrone", "uavdt", "auair", "dronevehicle"}:
                raise ValueError(
                    f"duq_pvf is forbidden on frozen historical dataset {self.dataset.name}"
                )
            elif self.dataset.name not in {"eagle", "dawn"}:
                raise ValueError(f"duq_pvf is not registered for dataset {self.dataset.name}")
            elif self.runtime.tune:
                raise ValueError("duq_pvf external EAGLE/DAWN runs forbid runtime.tune=true")
        if self.runtime.cross_shape_candidate_batching and self.method.name not in {
            "buse_cawbf_fast",
            "duq_guard",
        }:
            raise ValueError(
                "runtime.cross_shape_candidate_batching is restricted to Fast or Guard"
            )
        if self.fusion.rescue_conf < self.detector.publish_conf:
            raise ValueError("fusion.rescue_conf must not be below detector.publish_conf")
        if self.guard.rescue_conf < self.detector.publish_conf:
            raise ValueError("guard.rescue_conf must not be below detector.publish_conf")
        if self.runtime.require_paths:
            self._validate_required_paths()
        return self

    def _validate_required_paths(self) -> None:
        images_root = self.dataset.images_root or self.dataset.root
        checks = {
            "dataset.root": self.dataset.root,
            "dataset.images": images_root / self.dataset.images,
            "dataset.annotations": self.dataset.root / self.dataset.annotations,
            "scoring.calibration": self.scoring.calibration,
        }
        if self.experiment.train is None:
            checks["detector.model"] = self.detector.model
            if self.detector.definition is not None:
                checks["detector.definition"] = self.detector.definition
        missing = [f"{name}={path}" for name, path in checks.items() if not path.exists()]
        if missing:
            joined = "; ".join(missing)
            raise ValueError(
                "required local paths are unavailable: "
                f"{joined}. Set BUSE_DATA_ROOT or pass Hydra path overrides."
            )


def _validate_nonnegative_unit_sum(weights: dict[str, float], name: str) -> None:
    if not weights:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"{name} values must be nonnegative")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1 within 1e-6")
