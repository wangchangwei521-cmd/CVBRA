from __future__ import annotations

import shutil
from pathlib import Path

from buse_uav.pipeline.trace import RunDirectory
from buse_uav.schemas import AppConfig


def intermediate_shard_path(
    config: AppConfig,
    run: RunDirectory,
    *parts: str,
) -> Path:
    """Return a persistent or transient path for detector image inputs."""
    root = (
        run.path / "shards"
        if config.runtime.save_intermediates
        else run.path / ".scratch" / "shards"
    )
    return root.joinpath(*parts)


def score_map_path(
    config: AppConfig,
    run: RunDirectory,
    image_id: int | str,
) -> Path | None:
    """Return the diagnostic score-map path only when persistence is enabled."""
    if not config.runtime.save_intermediates:
        return None
    return run.path / "traces" / "score_maps" / f"{image_id}.jpg"


def cleanup_transient_intermediates(config: AppConfig, run: RunDirectory) -> None:
    """Remove transient images while never touching structured run artifacts."""
    if config.runtime.save_intermediates:
        return
    for target in (run.path / ".scratch", run.path / "shards"):
        _remove_run_subtree(run.path, target)


def _remove_run_subtree(run_path: Path, target: Path) -> None:
    if not target.exists() and not target.is_symlink():
        return
    resolved_run = run_path.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_run or resolved_run not in resolved_target.parents:
        raise ValueError(f"refusing to clean path outside run directory: {resolved_target}")
    if target.is_symlink() or target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)
