from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


def _pattern_image(path: Path, *, size: tuple[int, int], offset: int) -> None:
    image = Image.new("RGB", size, (30 + offset, 50 + offset, 80 + offset))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 4, size[0] // 2, size[1] // 2), fill=(180, 120, 60))
    draw.line((0, size[1] - 1, size[0] - 1, 0), fill=(240, 240, 240), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@pytest.fixture()
def synthetic_hazydet(tmp_path: Path) -> Path:
    root = tmp_path / "HazyDet"
    split = root / "val"
    for variant, offset in (("clean_images", 0), ("hazy_images", 20)):
        _pattern_image(split / variant / "0001.png", size=(64, 48), offset=offset)
        _pattern_image(split / variant / "0002.png", size=(80, 60), offset=offset + 5)
    document = {
        "images": [
            {"id": 10, "file_name": "0001.png", "width": 64, "height": 48},
            {"id": 11, "file_name": "0002.png", "width": 80, "height": 60},
        ],
        "categories": [
            {"id": 3, "name": "car"},
            {"id": 7, "name": "truck"},
            {"id": 9, "name": "bus"},
        ],
        "annotations": [
            {
                "id": 100,
                "image_id": 10,
                "category_id": 3,
                "bbox": [8, 6, 16, 12],
                "area": 192,
                "iscrowd": 0,
            },
            {
                "id": 101,
                "image_id": 11,
                "category_id": 9,
                "bbox": [20, 15, 30, 20],
                "area": 600,
                "iscrowd": 0,
            },
        ],
    }
    (split / "val_coco.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def synthetic_visdrone(tmp_path: Path) -> Path:
    root = tmp_path / "VisDrone"
    split = root / "VisDrone2019-DET-val"
    for image_id, offset in (("000001", 0), ("000002", 15)):
        _pattern_image(
            split / "images" / f"{image_id}.jpg",
            size=(96, 72),
            offset=offset,
        )
        annotation = "\n".join(
            [
                "10,8,20,16,1,4,0,1",
                "45,30,18,14,1,1,1,2",
                "0,0,12,12,0,0,0,0",
            ]
        )
        annotation_path = split / "annotations" / f"{image_id}.txt"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(annotation + "\n", encoding="utf-8")
    return root
