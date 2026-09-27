from __future__ import annotations

from typing import Any

import torch
from scripts.run_cvbra_v1_training import (
    SOURCE_REPLAY_IMAGES,
    _combined_state,
    _source_replay_rows,
    _target_labels,
)


def test_combined_state_restores_frozen_layers_and_keeps_adapted_layers() -> None:
    source = {
        "model.0.weight": torch.tensor([1.0]),
        "model.9.buffer": torch.tensor([2]),
        "model.10.weight": torch.tensor([3.0]),
        "model.23.weight": torch.tensor([4.0]),
    }
    trained = {
        "model.0.weight": torch.tensor([10.0]),
        "model.9.buffer": torch.tensor([20]),
        "model.10.weight": torch.tensor([30.0]),
        "model.23.weight": torch.tensor([40.0]),
    }
    combined = _combined_state(source, trained)
    assert torch.equal(combined["model.0.weight"], source["model.0.weight"])
    assert torch.equal(combined["model.9.buffer"], source["model.9.buffer"])
    assert torch.equal(combined["model.10.weight"], trained["model.10.weight"])
    assert torch.equal(combined["model.23.weight"], trained["model.23.weight"])


def test_source_replay_selection_is_fixed_and_exact_size() -> None:
    annotation: dict[str, Any] = {
        "images": [
            {"id": image_id, "file_name": f"{image_id:05d}.jpg"} for image_id in range(1, 8001)
        ]
    }
    first = _source_replay_rows(annotation)
    second = _source_replay_rows(annotation)
    assert len(first) == SOURCE_REPLAY_IMAGES
    assert first == second
    assert len({int(row["id"]) for row in first}) == SOURCE_REPLAY_IMAGES


def test_target_training_projection_clips_without_changing_evaluation_payload() -> None:
    images = [{"id": image_id, "width": 1920, "height": 1080} for image_id in range(1, 901)]
    annotation: dict[str, Any] = {
        "images": images,
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [-192.0, 108.0, 576.0, 108.0],
            }
        ],
    }
    labels, report = _target_labels(annotation)
    assert labels[1] == "0 0.10000000 0.15000000 0.20000000 0.10000000\n"
    assert labels[2] == ""
    assert report["outside_image_extent"] == 1
    assert report["clipped_for_training"] == 1
    assert report["degenerate_excluded_for_training"] == 0
    assert annotation["annotations"][0]["bbox"] == [-192.0, 108.0, 576.0, 108.0]
