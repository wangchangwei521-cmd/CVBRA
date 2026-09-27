from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

import buse_uav.detectors.aldi_ultralytics as aldi_module
from buse_uav.detectors.aldi_ultralytics import (
    ALDIRuntimeConfig,
    ALDITranslationDetectionModel,
    ALDITranslationDetectionTrainer,
    pseudo_batch_from_detections,
    slice_detection_predictions,
    strong_view_batch,
    subset_detection_batch,
)

ROOT = Path(__file__).resolve().parents[2]


def test_subset_detection_batch_reindexes_annotations() -> None:
    batch = {
        "img": torch.zeros((3, 3, 16, 16)),
        "batch_idx": torch.tensor([0, 0, 2], dtype=torch.long),
        "cls": torch.tensor([[0.0], [1.0], [2.0]]),
        "bboxes": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ]
        ),
    }
    subset = subset_detection_batch(batch, torch.tensor([2, 0], dtype=torch.long))
    assert subset["img"].shape == (2, 3, 16, 16)
    assert subset["batch_idx"].tolist() == [1, 1, 0]
    assert subset["cls"].view(-1).tolist() == [0.0, 1.0, 2.0]


def test_slice_detection_predictions_preserves_pyramid_schema() -> None:
    predictions: dict[str, Any] = {
        "boxes": torch.arange(4 * 64 * 10).reshape(4, 64, 10),
        "scores": torch.arange(4 * 3 * 10).reshape(4, 3, 10),
        "feats": [
            torch.zeros((4, 8, 8, 8)),
            torch.zeros((4, 16, 4, 4)),
            torch.zeros((4, 32, 2, 2)),
        ],
    }
    sliced = slice_detection_predictions(
        predictions, torch.tensor([3, 1], dtype=torch.long)
    )
    assert sliced["boxes"].shape == (2, 64, 10)
    assert sliced["scores"].shape == (2, 3, 10)
    assert [value.shape[0] for value in sliced["feats"]] == [2, 2, 2]
    assert torch.equal(sliced["scores"][0], predictions["scores"][3])


def test_pseudo_batch_normalizes_boxes_and_counts_coverage() -> None:
    images = torch.zeros((2, 3, 100, 200))
    detections = [
        torch.tensor(
            [
                [20.0, 10.0, 60.0, 50.0, 0.95, 2.0],
                [40.0, 20.0, 100.0, 80.0, 0.90, 1.0],
            ]
        ),
        torch.zeros((0, 6)),
    ]
    batch, boxes, covered = pseudo_batch_from_detections(detections, images)
    assert boxes == 2
    assert covered == 1
    assert batch["batch_idx"].tolist() == [0, 0]
    assert batch["cls"].view(-1).tolist() == [2.0, 1.0]
    assert torch.allclose(
        batch["bboxes"],
        torch.tensor([[0.20, 0.30, 0.20, 0.40], [0.35, 0.50, 0.30, 0.60]]),
    )


def test_strong_view_is_deterministic_and_applies_target_mic() -> None:
    images = torch.ones((2, 3, 64, 64)) * 0.5
    first = strong_view_batch(
        images,
        [True, False],
        seed=42,
        step=7,
        mic_ratio=0.5,
        mic_block_size=16,
    )
    second = strong_view_batch(
        images,
        [True, False],
        seed=42,
        step=7,
        mic_ratio=0.5,
        mic_block_size=16,
    )
    different = strong_view_batch(
        images,
        [True, False],
        seed=42,
        step=8,
        mic_ratio=0.5,
        mic_block_size=16,
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert bool((first[1] == 0.0).any())
    assert bool(torch.isfinite(first).all())
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_protocol_has_native_and_equal_supervision_variants() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/settings/cvbra_v1_aldi_direct_baseline_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    variants = protocol["variants"]
    assert set(variants) == {
        "ALDIpp_AF_Y11_native_UDA",
        "ALDIpp_AF_Y11_equal_supervision",
    }
    assert variants["ALDIpp_AF_Y11_native_UDA"]["target_ground_truth_used_for_training"] is False
    assert (
        variants["ALDIpp_AF_Y11_equal_supervision"][
            "target_ground_truth_used_for_training"
        ]
        is True
    )


def test_trainer_skips_unregistered_final_validation() -> None:
    trainer = object.__new__(ALDITranslationDetectionTrainer)
    assert trainer.validate() == ({}, 0.0)
    assert trainer.final_eval() is None


def test_teacher_first_update_copies_student_before_ema(monkeypatch: Any) -> None:
    config = ALDIRuntimeConfig(
        variant="equal_supervision",
        source_image_paths=(ROOT / "source.jpg",),
        expected_source_images=1,
        diagnostics_path=ROOT / "diagnostics.json",
        protocol_sha256="protocol",
        registration_sha256="registration",
        implementation_lock_sha256="implementation",
        source_checkpoint_sha256="checkpoint",
        data_manifest_sha256="manifest",
        ema_alpha=0.9,
        ema_initialize_from_first_student_update=True,
    )
    monkeypatch.setattr(aldi_module, "_RUNTIME", config)

    student = object.__new__(ALDITranslationDetectionModel)
    torch.nn.Module.__init__(student)
    student.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    )
    teacher = torch.nn.Module()
    teacher.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32))
    )
    object.__setattr__(student, "_aldi_teacher", teacher)
    student._aldi_updates = 0

    student.update_aldi_teacher()
    assert torch.equal(teacher.weight, torch.tensor([2.0]))

    with torch.no_grad():
        student.weight.fill_(4.0)
    student.update_aldi_teacher()
    assert torch.allclose(teacher.weight, torch.tensor([2.2]))
