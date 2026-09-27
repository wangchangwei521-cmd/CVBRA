from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json


@dataclass(frozen=True)
class EWCRuntimeConfig:
    """Locked runtime inputs for the matched EWC retention baseline."""

    source_image_list: Path
    expected_source_images: int
    fisher_tensor_path: Path
    fisher_report_path: Path
    fisher_marker_path: Path
    protocol_sha256: str
    registration_sha256: str
    implementation_lock_sha256: str
    source_checkpoint_sha256: str
    data_manifest_sha256: str
    reference_displacement_fraction: float = 0.01
    reference_loss_fraction: float = 0.05


_RUNTIME: EWCRuntimeConfig | None = None


def configure_ewc_runtime(config: EWCRuntimeConfig) -> None:
    """Configure the one registered EWC training run before trainer creation."""
    global _RUNTIME
    if _RUNTIME is not None and config != _RUNTIME:
        raise RuntimeError("EWC runtime was configured twice with different inputs")
    _RUNTIME = config


def _runtime() -> EWCRuntimeConfig:
    if _RUNTIME is None:
        raise RuntimeError("EWC runtime was not configured")
    return _RUNTIME


try:
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import unwrap_model
except (ImportError, OSError, PermissionError) as exc:  # pragma: no cover - environment gate
    raise RuntimeError(f"Ultralytics EWC support is unavailable: {exc}") from exc


class EWCDetectionModel(DetectionModel):  # type: ignore[misc]
    """YOLO detection model with a training-only diagonal EWC penalty."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ewc_reference: dict[str, torch.Tensor] | None = None
        self._ewc_fisher: dict[str, torch.Tensor] | None = None
        self._ewc_fisher_sum = 0.0
        self._ewc_coefficient = 0.0
        self.last_ewc_penalty = 0.0
        self.last_ewc_weighted_loss = 0.0

    def configure_ewc(
        self,
        *,
        reference: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        coefficient: float,
    ) -> None:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(reference) != names or set(fisher) != names:
            raise RuntimeError("EWC tensors do not match the trainable parameter schema")
        fisher_sum = sum(float(value.double().sum()) for value in fisher.values())
        if not math.isfinite(fisher_sum) or fisher_sum <= 0.0:
            raise RuntimeError("EWC Fisher mass is not positive and finite")
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise RuntimeError("EWC coefficient is not positive and finite")
        self._ewc_reference = {name: value.detach() for name, value in reference.items()}
        self._ewc_fisher = {name: value.detach() for name, value in fisher.items()}
        self._ewc_fisher_sum = fisher_sum
        self._ewc_coefficient = float(coefficient)

    def loss(self, batch: dict[str, Any], preds: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        detection_loss, loss_items = super().loss(batch, preds)
        if self._ewc_reference is None or self._ewc_fisher is None:
            return detection_loss, loss_items
        weighted = detection_loss.new_zeros(())
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            reference = self._ewc_reference[name]
            fisher = self._ewc_fisher[name]
            weighted = weighted + (fisher * (parameter - reference).square()).sum()
        penalty = weighted / self._ewc_fisher_sum
        batch_size = int(batch["img"].shape[0])
        weighted_loss = penalty * self._ewc_coefficient * batch_size
        self.last_ewc_penalty = float(penalty.detach())
        self.last_ewc_weighted_loss = float(weighted_loss.detach())
        return detection_loss + weighted_loss, loss_items


class EWCDetectionTrainer(DetectionTrainer):  # type: ignore[misc]
    """Detection trainer that estimates and locks Fisher before optimization."""

    def get_model(
        self,
        cfg: str | dict[str, Any] | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> EWCDetectionModel:
        model = EWCDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def _setup_train(self) -> None:
        super()._setup_train()
        self._estimate_and_lock_fisher()

    def _estimate_and_lock_fisher(self) -> None:
        config = _runtime()
        if any(
            path.exists()
            for path in (
                config.fisher_tensor_path,
                config.fisher_report_path,
                config.fisher_marker_path,
            )
        ):
            raise RuntimeError("EWC Fisher output existed before registered estimation")
        if not config.source_image_list.is_file():
            raise RuntimeError(f"EWC source list is missing: {config.source_image_list}")

        model = unwrap_model(self.model)
        if not isinstance(model, EWCDetectionModel):
            raise RuntimeError("EWC trainer did not receive the registered model class")
        source_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        if not trainable:
            raise RuntimeError("EWC trainer has no trainable floating parameters")
        fisher = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in trainable.items()
        }
        reference = {name: parameter.detach().clone() for name, parameter in trainable.items()}

        source_dataset = build_yolo_dataset(
            self.args,
            str(config.source_image_list.resolve()),
            1,
            self.data,
            mode="train",
            rect=False,
            stride=int(self.stride),
        )
        if len(source_dataset) != config.expected_source_images:
            raise RuntimeError(
                "EWC source coverage changed: "
                f"expected {config.expected_source_images}, observed {len(source_dataset)}"
            )
        source_loader = build_dataloader(
            source_dataset,
            batch=1,
            workers=0,
            shuffle=False,
            rank=-1,
            drop_last=False,
            pin_memory=False,
        )

        model.train()
        loss_sum = 0.0
        observed_images = 0
        for raw_batch in source_loader:
            batch = self.preprocess_batch(raw_batch)
            model.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            per_image_loss = loss.sum() / int(batch["img"].shape[0])
            per_image_loss.backward()
            observed_images += int(batch["img"].shape[0])
            loss_sum += float(per_image_loss.detach()) * int(batch["img"].shape[0])
            for name, parameter in trainable.items():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().float().square())
        if observed_images != config.expected_source_images:
            raise RuntimeError(
                "EWC Fisher image count changed: "
                f"expected {config.expected_source_images}, observed {observed_images}"
            )
        for value in fisher.values():
            value.div_(observed_images)

        incompatible = model.load_state_dict(source_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("EWC source-state restoration failed")
        model.zero_grad(set_to_none=True)

        fisher_sum = sum(float(value.double().sum()) for value in fisher.values())
        mean_source_loss = loss_sum / observed_images
        reference_weighted = 0.0
        for name in trainable:
            rms = float(reference[name].detach().float().square().mean().sqrt())
            displacement = config.reference_displacement_fraction * max(rms, 1e-8)
            reference_weighted += float(fisher[name].double().sum()) * displacement**2
        reference_penalty = reference_weighted / fisher_sum
        coefficient = config.reference_loss_fraction * mean_source_loss / reference_penalty
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (fisher_sum, mean_source_loss, reference_penalty, coefficient)
        ):
            raise RuntimeError("EWC Fisher calibration produced an invalid value")

        model.configure_ewc(reference=reference, fisher=fisher, coefficient=coefficient)
        config.fisher_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = config.fisher_tensor_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema_version": 1,
                "fisher": {name: value.detach().cpu() for name, value in fisher.items()},
            },
            temporary,
        )
        temporary.replace(config.fisher_tensor_path)
        positive = torch.cat(
            [
                value[value > 0].detach().float().cpu()
                for value in fisher.values()
                if bool((value > 0).any())
            ]
        )
        report = {
            "schema_version": 1,
            "status": "CVBRA_EWC_EMPIRICAL_FISHER_LOCKED_BEFORE_OPTIMIZATION",
            "protocol_sha256": config.protocol_sha256,
            "registration_sha256": config.registration_sha256,
            "implementation_lock_sha256": config.implementation_lock_sha256,
            "source_checkpoint_sha256": config.source_checkpoint_sha256,
            "data_manifest_sha256": config.data_manifest_sha256,
            "source_image_list": str(config.source_image_list.resolve()),
            "source_image_list_sha256": sha256_file(config.source_image_list),
            "source_images": observed_images,
            "trainable_parameter_tensors": len(trainable),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
            "fisher_sum": fisher_sum,
            "fisher_positive_entries": int(positive.numel()),
            "fisher_positive_minimum": float(positive.min()),
            "fisher_maximum": max(float(value.max()) for value in fisher.values()),
            "mean_source_detection_loss_per_image": mean_source_loss,
            "reference_displacement_fraction": config.reference_displacement_fraction,
            "reference_loss_fraction": config.reference_loss_fraction,
            "reference_penalty": reference_penalty,
            "ewc_coefficient": coefficient,
            "fisher_tensor": str(config.fisher_tensor_path.resolve()),
            "fisher_tensor_sha256": sha256_file(config.fisher_tensor_path),
            "validation_metric_used_for_Fisher_or_strength": False,
            "official_test_access": "prohibited",
        }
        atomic_write_json(config.fisher_report_path, report)
        atomic_write_json(
            config.fisher_marker_path,
            {
                "status": report["status"],
                "fisher_report_sha256": sha256_file(config.fisher_report_path),
            },
        )
