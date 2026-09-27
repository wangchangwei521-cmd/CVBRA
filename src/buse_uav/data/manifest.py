from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from buse_uav.utils.hashing import sha256_file
from buse_uav.utils.io import atomic_write_json


def file_manifest_entry(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def build_file_manifest(paths: Iterable[Path], *, root: Path) -> list[dict[str, Any]]:
    unique = sorted({path.resolve() for path in paths}, key=lambda path: path.as_posix())
    return [file_manifest_entry(path, root=root) for path in unique]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)
