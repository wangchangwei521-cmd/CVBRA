from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as functional

from buse_uav.utils.io import atomic_write_json

ALDIVariant = Literal["native_uda", "equal_supervision"]
_VALID_VARIANTS = frozenset({"native_uda", "equal_supervision"})


@dataclass(frozen=True)
class ALDIRuntimeConfig:
    """Frozen runtime inputs for one ALDI++ anchor-free translation run."""

    variant: ALDIVariant
    source_image_paths: tuple[Path, ...]
    expected_source_images: int
    diagnostics_path: Path
    protocol_sha256: str
    registration_sha256: str
    implementation_lock_sha256: str
    source_checkpoint_sha256: str
    data_manifest_sha256: str
    seed: int = 42
    ema_alpha: float = 0.9996
    ema_initialize_from_first_student_update: bool = False
    pseudo_threshold: float = 0.8
    pseudo_nms_iou: float = 0.65
    max_det: int = 500
    mic_ratio: float = 0.5
    mic_block_size: int = 32
    confidence_weight: float = 0.7
    class_weight: float = 0.3


_RUNTIME: ALDIRuntimeConfig | None = None


def configure_aldi_runtime(config: ALDIRuntimeConfig) -> None:
    """Configure one fail-closed ALDI translation process before trainer creation."""
    global _RUNTIME
    if config.variant not in _VALID_VARIANTS:
        raise RuntimeError(f"unsupported ALDI translation variant: {config.variant}")
    if _RUNTIME is not None and config != _RUNTIME:
        raise RuntimeError("ALDI runtime was configured twice with different inputs")
    _RUNTIME = config


def _runtime() -> ALDIRuntimeConfig:
    if _RUNTIME is None:
        raise RuntimeError("ALDI runtime was not configured")
    return _RUNTIME


def normalize_image_path(value: str | Path) -> str:
    """Return a Windows-stable absolute path key without requiring file I/O."""
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def subset_detection_batch(batch: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    """Subset a YOLO detection batch and densely reindex its annotation image IDs."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise RuntimeError("batch subset indices must be a one-dimensional int64 tensor")
    images = batch["img"]
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise RuntimeError("batch images are missing or malformed")
    if indices.numel() == 0:
        device = images.device
        return {
            "img": images[:0],
            "batch_idx": torch.zeros((0,), device=device, dtype=torch.long),
            "cls": torch.zeros((0, 1), device=device, dtype=batch["cls"].dtype),
            "bboxes": torch.zeros((0, 4), device=device, dtype=batch["bboxes"].dtype),
        }
    if int(indices.min()) < 0 or int(indices.max()) >= int(images.shape[0]):
        raise RuntimeError("batch subset index is out of range")
    old_batch_idx = batch["batch_idx"].view(-1).long()
    annotation_mask = torch.zeros_like(old_batch_idx, dtype=torch.bool)
    remapped = torch.full_like(old_batch_idx, -1)
    for new_index, old_index in enumerate(indices.tolist()):
        selected = old_batch_idx == int(old_index)
        annotation_mask |= selected
        remapped[selected] = int(new_index)
    return {
        "img": images.index_select(0, indices),
        "batch_idx": remapped[annotation_mask],
        "cls": batch["cls"][annotation_mask],
        "bboxes": batch["bboxes"][annotation_mask],
    }


def slice_detection_predictions(preds: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    """Slice anchor-free YOLO raw predictions by batch index."""
    required = {"boxes", "scores", "feats"}
    if set(preds) < required:
        raise RuntimeError("YOLO raw prediction schema changed")
    feats = preds["feats"]
    if not isinstance(feats, list):
        raise RuntimeError("YOLO feature pyramid schema changed")
    return {
        "boxes": preds["boxes"].index_select(0, indices),
        "scores": preds["scores"].index_select(0, indices),
        "feats": [feature.index_select(0, indices) for feature in feats],
    }


def pseudo_batch_from_detections(
    detections: list[torch.Tensor], images: torch.Tensor
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Convert NMS detections into normalized YOLO training targets."""
    if images.ndim != 4 or len(detections) != int(images.shape[0]):
        raise RuntimeError("pseudo-label image coverage changed")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    batch_indices: list[torch.Tensor] = []
    classes: list[torch.Tensor] = []
    boxes: list[torch.Tensor] = []
    images_with_pseudo = 0
    for image_index, detection in enumerate(detections):
        if detection.ndim != 2 or detection.shape[1] < 6:
            raise RuntimeError("NMS detection schema changed")
        if detection.numel() == 0:
            continue
        xyxy = detection[:, :4]
        x1, y1, x2, y2 = xyxy.unbind(dim=1)
        xywh = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=1)
        scale = xywh.new_tensor([width, height, width, height])
        xywh = (xywh / scale).clamp_(0.0, 1.0)
        valid = (xywh[:, 2] > 0.0) & (xywh[:, 3] > 0.0)
        if not bool(valid.any()):
            continue
        xywh = xywh[valid]
        boxes.append(xywh)
        classes.append(detection[valid, 5:6].to(dtype=images.dtype))
        batch_indices.append(
            torch.full(
                (int(xywh.shape[0]),),
                image_index,
                device=images.device,
                dtype=torch.long,
            )
        )
        images_with_pseudo += 1
    if boxes:
        out_boxes = torch.cat(boxes, dim=0)
        out_classes = torch.cat(classes, dim=0)
        out_indices = torch.cat(batch_indices, dim=0)
    else:
        out_boxes = images.new_zeros((0, 4))
        out_classes = images.new_zeros((0, 1))
        out_indices = torch.zeros((0,), device=images.device, dtype=torch.long)
    return (
        {
            "img": images,
            "batch_idx": out_indices,
            "cls": out_classes,
            "bboxes": out_boxes,
        },
        int(out_boxes.shape[0]),
        images_with_pseudo,
    )


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
    value = (image * weights).sum(dim=0, keepdim=True)
    return value.expand_as(image)


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3.0 * sigma))
    kernel_size = radius * 2 + 1
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size)
    return functional.conv2d(image.unsqueeze(0), kernel, padding=radius, groups=3).squeeze(0)


def _random_erase(
    image: torch.Tensor,
    rng: random.Random,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
) -> None:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    area = height * width
    for _ in range(100):
        target_area = rng.uniform(*scale) * area
        aspect = rng.uniform(*ratio)
        erase_h = round(math.sqrt(target_area * aspect))
        erase_w = round(math.sqrt(target_area / aspect))
        if 1 < erase_h < height and 1 < erase_w < width:
            top = rng.randint(0, height - erase_h - 1)
            left = rng.randint(0, width - erase_w - 1)
            noise_seed = rng.randrange(0, 2**31 - 1)
            generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise = torch.rand((3, erase_h, erase_w), generator=generator)
            image[:, top : top + erase_h, left : left + erase_w] = noise.to(
                device=image.device, dtype=image.dtype
            )
            return


def _mic_mask(image: torch.Tensor, rng: random.Random, ratio: float, block_size: int) -> None:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    mask_h = max(1, round(height / block_size))
    mask_w = max(1, round(width / block_size))
    generator = torch.Generator(device="cpu").manual_seed(rng.randrange(0, 2**31 - 1))
    coarse = (torch.rand((1, 1, mask_h, mask_w), generator=generator) > ratio).float()
    mask = functional.interpolate(coarse, size=(height, width), mode="nearest").squeeze(0)
    image.mul_(mask.to(device=image.device, dtype=image.dtype))


def strong_view_batch(
    images: torch.Tensor,
    source_flags: list[bool],
    *,
    seed: int,
    step: int,
    mic_ratio: float,
    mic_block_size: int,
) -> torch.Tensor:
    """Apply the registered ALDI strong-view recipe without changing geometry."""
    if images.ndim != 4 or len(source_flags) != int(images.shape[0]):
        raise RuntimeError("strong-view batch coverage changed")
    output = images.clone()
    for image_index, is_source in enumerate(source_flags):
        rng = random.Random(seed + step * 1_000_003 + image_index * 9_973)
        image = output[image_index]
        if rng.random() < 0.8:
            contrast = rng.uniform(0.6, 1.4)
            brightness = rng.uniform(0.6, 1.4)
            saturation = rng.uniform(0.6, 1.4)
            mean = image.mean(dim=(1, 2), keepdim=True)
            image.copy_((image - mean) * contrast + mean)
            image.mul_(brightness)
            gray = _grayscale(image)
            image.copy_(gray + saturation * (image - gray))
        if rng.random() < 0.2:
            image.copy_(_grayscale(image))
        if rng.random() < 0.5:
            image.copy_(_gaussian_blur(image, rng.uniform(0.1, 2.0)))
        if is_source:
            for probability, scale, ratio in (
                (0.7, (0.05, 0.20), (0.3, 3.3)),
                (0.5, (0.02, 0.20), (0.1, 6.0)),
                (0.3, (0.02, 0.20), (0.05, 8.0)),
            ):
                if rng.random() < probability:
                    _random_erase(image, rng, scale=scale, ratio=ratio)
        else:
            _mic_mask(image, rng, mic_ratio, mic_block_size)
        image.clamp_(0.0, 1.0)
    return output


try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.torch_utils import unwrap_model
except (ImportError, OSError, PermissionError) as exc:  # pragma: no cover
    raise RuntimeError(f"Ultralytics ALDI translation support is unavailable: {exc}") from exc


class ALDITranslationDetectionModel(DetectionModel):
    """YOLO11 detection model with a training-only ALDI++ translation loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]
        object.__setattr__(self, "_aldi_teacher", None)
        self._aldi_source_paths: frozenset[str] = frozenset()
        self._aldi_variant: ALDIVariant | None = None
        self._aldi_step = 0
        self._aldi_updates = 0
        self._aldi_totals: dict[str, float] = {
            "batches": 0.0,
            "source_images": 0.0,
            "target_images": 0.0,
            "pseudo_boxes": 0.0,
            "target_images_with_pseudo": 0.0,
            "soft_confidence_loss": 0.0,
            "soft_class_loss": 0.0,
            "pseudo_box_loss": 0.0,
            "pseudo_dfl_loss": 0.0,
        }

    def configure_aldi(self, teacher: torch.nn.Module, config: ALDIRuntimeConfig) -> None:
        if self._aldi_teacher is not None:
            raise RuntimeError("ALDI teacher was configured twice")
        paths = frozenset(normalize_image_path(path) for path in config.source_image_paths)
        if len(paths) != config.expected_source_images:
            raise RuntimeError("ALDI source path coverage changed")
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        object.__setattr__(self, "_aldi_teacher", teacher)
        self._aldi_source_paths = paths
        self._aldi_variant = config.variant

    @property
    def aldi_teacher(self) -> torch.nn.Module:
        teacher = self._aldi_teacher
        if not isinstance(teacher, torch.nn.Module):
            raise RuntimeError("ALDI teacher is not configured")
        return teacher

    @torch.no_grad()
    def update_aldi_teacher(self) -> None:
        config = _runtime()
        teacher_state = self.aldi_teacher.state_dict()
        student_state = self.state_dict()
        if tuple(teacher_state) != tuple(student_state):
            raise RuntimeError("ALDI teacher/student state schema changed")
        alpha = config.ema_alpha
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach().to(device=teacher_value.device)
            if config.ema_initialize_from_first_student_update and self._aldi_updates == 0:
                teacher_value.copy_(student_value)
            elif teacher_value.is_floating_point():
                teacher_value.mul_(alpha).add_(student_value, alpha=1.0 - alpha)
            else:
                teacher_value.copy_(student_value)
        self._aldi_updates += 1

    def _source_flags(self, batch: dict[str, Any]) -> list[bool]:
        files = batch.get("im_file")
        if not isinstance(files, (list, tuple)) or len(files) != int(batch["img"].shape[0]):
            raise RuntimeError("ALDI batch image paths are missing")
        flags: list[bool] = []
        for value in files:
            key = normalize_image_path(str(value))
            is_source = key in self._aldi_source_paths
            if not is_source and "target_" not in Path(str(value)).name.casefold():
                raise RuntimeError(f"ALDI batch path is outside the locked multiset: {value}")
            flags.append(is_source)
        return flags

    def _teacher_predictions(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        teacher = self.aldi_teacher
        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            output = teacher(images)
        if was_training:
            teacher.train()
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], torch.Tensor)
            or not isinstance(output[1], dict)
        ):
            raise RuntimeError("ALDI teacher output schema changed")
        return output[0], output[1]

    def _soft_class_losses(
        self, student_scores: torch.Tensor, teacher_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = _runtime()
        if student_scores.shape != teacher_scores.shape:
            raise RuntimeError("ALDI teacher/student class-map shapes differ")
        teacher_logits = teacher_scores.detach()
        teacher_probs = teacher_logits.sigmoid()
        teacher_confidence = teacher_probs.max(dim=1).values
        student_confidence_logits = student_scores.max(dim=1).values
        confidence_loss = functional.binary_cross_entropy_with_logits(
            student_confidence_logits, teacher_confidence, reduction="mean"
        )
        foreground = teacher_confidence > config.pseudo_threshold
        if bool(foreground.any()):
            teacher_class = functional.softmax(teacher_logits, dim=1).permute(0, 2, 1)
            student_log_class = functional.log_softmax(student_scores, dim=1).permute(0, 2, 1)
            class_loss = (
                -(teacher_class[foreground] * student_log_class[foreground]).sum(dim=1).mean()
            )
        else:
            class_loss = student_scores.sum() * 0.0
        return confidence_loss, class_loss

    def loss(self, batch: dict[str, Any], preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        if preds is not None:
            raise RuntimeError("ALDI translation requires its registered weak/strong forward path")
        config = _runtime()
        if self._aldi_variant != config.variant:
            raise RuntimeError("ALDI model/runtime variant mismatch")
        source_flags = self._source_flags(batch)
        device = batch["img"].device
        source_indices = torch.tensor(
            [index for index, value in enumerate(source_flags) if value],
            device=device,
            dtype=torch.long,
        )
        target_indices = torch.tensor(
            [index for index, value in enumerate(source_flags) if not value],
            device=device,
            dtype=torch.long,
        )
        strong_images = strong_view_batch(
            batch["img"],
            source_flags,
            seed=config.seed,
            step=self._aldi_step,
            mic_ratio=config.mic_ratio,
            mic_block_size=config.mic_block_size,
        )
        strong_batch = dict(batch)
        strong_batch["img"] = strong_images
        student_preds = self.forward(strong_images)  # type: ignore[no-untyped-call]
        if not isinstance(student_preds, dict):
            raise RuntimeError("ALDI student output schema changed")

        zero = student_preds["scores"].new_zeros(3)
        if config.variant == "equal_supervision":
            supervised_loss, _ = super().loss(  # type: ignore[no-untyped-call]
                strong_batch, student_preds
            )
        elif source_indices.numel():
            supervised_batch = subset_detection_batch(strong_batch, source_indices)
            supervised_preds = slice_detection_predictions(student_preds, source_indices)
            supervised_loss, _ = super().loss(  # type: ignore[no-untyped-call]
                supervised_batch, supervised_preds
            )
        else:
            supervised_loss = zero

        distillation = zero.clone()
        pseudo_count = 0
        pseudo_images = 0
        confidence_loss = zero[0]
        class_loss = zero[0]
        pseudo_loss = zero
        if target_indices.numel():
            target_weak = batch["img"].index_select(0, target_indices)
            decoded, teacher_raw = self._teacher_predictions(target_weak)
            detections = non_max_suppression(
                decoded,
                conf_thres=config.pseudo_threshold,
                iou_thres=config.pseudo_nms_iou,
                max_det=config.max_det,
                nc=int(cast(int, self.nc)),
            )
            target_preds = slice_detection_predictions(student_preds, target_indices)
            target_images = strong_images.index_select(0, target_indices)
            pseudo_batch, pseudo_count, pseudo_images = pseudo_batch_from_detections(
                detections, target_images
            )
            pseudo_loss, _ = super().loss(  # type: ignore[no-untyped-call]
                pseudo_batch, target_preds
            )
            confidence_loss, class_loss = self._soft_class_losses(
                target_preds["scores"], teacher_raw["scores"]
            )
            if getattr(self, "criterion", None) is None:
                raise RuntimeError("ALDI detection criterion was not initialized")
            classification_gain = float(self.criterion.hyp.cls)
            target_count = int(target_indices.numel())
            soft_classification = (
                (config.confidence_weight * confidence_loss + config.class_weight * class_loss)
                * classification_gain
                * target_count
            )
            distillation[0] = pseudo_loss[0]
            distillation[1] = soft_classification
            distillation[2] = pseudo_loss[2]

        total_loss = supervised_loss + distillation
        batch_size = int(batch["img"].shape[0])
        self._aldi_totals["batches"] += 1.0
        self._aldi_totals["source_images"] += float(source_indices.numel())
        self._aldi_totals["target_images"] += float(target_indices.numel())
        self._aldi_totals["pseudo_boxes"] += float(pseudo_count)
        self._aldi_totals["target_images_with_pseudo"] += float(pseudo_images)
        self._aldi_totals["soft_confidence_loss"] += float(confidence_loss.detach())
        self._aldi_totals["soft_class_loss"] += float(class_loss.detach())
        self._aldi_totals["pseudo_box_loss"] += float(pseudo_loss[0].detach())
        self._aldi_totals["pseudo_dfl_loss"] += float(pseudo_loss[2].detach())
        self._aldi_step += 1
        return total_loss, total_loss.detach() / max(batch_size, 1)

    def aldi_diagnostics(self) -> dict[str, Any]:
        config = _runtime()
        totals = dict(self._aldi_totals)
        target_images = max(totals["target_images"], 1.0)
        batches = max(totals["batches"], 1.0)
        return {
            "schema_version": 1,
            "status": "ALDI_TRANSLATION_TRAIN_DIAGNOSTICS",
            "variant": config.variant,
            "batches": int(totals["batches"]),
            "source_images": int(totals["source_images"]),
            "target_images": int(totals["target_images"]),
            "pseudo_boxes": int(totals["pseudo_boxes"]),
            "target_images_with_pseudo": int(totals["target_images_with_pseudo"]),
            "mean_pseudo_boxes_per_target_image": totals["pseudo_boxes"] / target_images,
            "target_image_pseudo_coverage": totals["target_images_with_pseudo"] / target_images,
            "mean_soft_confidence_loss_per_batch": totals["soft_confidence_loss"] / batches,
            "mean_soft_class_loss_per_batch": totals["soft_class_loss"] / batches,
            "mean_pseudo_box_loss_per_batch": totals["pseudo_box_loss"] / batches,
            "mean_pseudo_dfl_loss_per_batch": totals["pseudo_dfl_loss"] / batches,
            "ema_updates": self._aldi_updates,
            "ema_alpha": config.ema_alpha,
            "ema_initialized_from_first_student_update": (
                config.ema_initialize_from_first_student_update
            ),
            "pseudo_threshold": config.pseudo_threshold,
            "pseudo_nms_iou": config.pseudo_nms_iou,
            "mic_ratio": config.mic_ratio,
            "mic_block_size": config.mic_block_size,
            "target_ground_truth_used_for_training": config.variant == "equal_supervision",
            "protocol_sha256": config.protocol_sha256,
            "registration_sha256": config.registration_sha256,
            "implementation_lock_sha256": config.implementation_lock_sha256,
            "source_checkpoint_sha256": config.source_checkpoint_sha256,
            "data_manifest_sha256": config.data_manifest_sha256,
        }


class ALDITranslationDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer for the locked anchor-free ALDI++ translation."""

    def get_model(
        self,
        cfg: str | dict[str, Any] | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> ALDITranslationDetectionModel:
        model = ALDITranslationDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)  # type: ignore[no-untyped-call]
        return model

    def _setup_train(self) -> None:
        super()._setup_train()  # type: ignore[no-untyped-call]
        model = unwrap_model(cast(torch.nn.Module, self.model))
        if not isinstance(model, ALDITranslationDetectionModel):
            raise RuntimeError("ALDI trainer did not receive the registered model class")
        teacher = copy.deepcopy(model).to(self.device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        model.configure_aldi(teacher, _runtime())

    def optimizer_step(self) -> None:
        super().optimizer_step()  # type: ignore[no-untyped-call]
        model = unwrap_model(cast(torch.nn.Module, self.model))
        if not isinstance(model, ALDITranslationDetectionModel):
            raise RuntimeError("ALDI model class changed during optimization")
        model.update_aldi_teacher()

    def validate(self) -> tuple[dict[str, float], float]:
        """Skip Ultralytics' forced final-epoch validation without reading labels."""
        return {}, 0.0

    def final_eval(self) -> None:
        """Keep the registered fixed last-epoch endpoint free of post-fit validation."""
        return None

    def save_model(self) -> bool:
        model = unwrap_model(cast(torch.nn.Module, self.model))
        if not isinstance(model, ALDITranslationDetectionModel):
            raise RuntimeError("ALDI model class changed before checkpointing")
        if self.ema is None:
            raise RuntimeError("Ultralytics EMA container is unavailable")
        original_ema = self.ema.ema
        self.ema.ema = copy.deepcopy(model.aldi_teacher)
        try:
            saved = bool(super().save_model())
        finally:
            self.ema.ema = original_ema
        if saved:
            atomic_write_json(_runtime().diagnostics_path, model.aldi_diagnostics())
        return saved
