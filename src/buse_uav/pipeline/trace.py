from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from buse_uav.utils.hashing import stable_hash
from buse_uav.utils.io import atomic_write_json, atomic_write_text
from buse_uav.utils.logging import StructuredLogger


@dataclass(frozen=True)
class RunDirectory:
    """The immutable identity and filesystem location of one experiment run."""

    run_id: str
    path: Path
    logger: StructuredLogger

    @classmethod
    def create(
        cls,
        *,
        output_root: Path,
        config: Mapping[str, Any],
        command: Sequence[str],
        run_id: str | None = None,
    ) -> RunDirectory:
        actual_run_id = run_id or build_run_id(config)
        run_path = output_root / actual_run_id
        try:
            run_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(
                f"run directory already exists: {run_path}; choose a new run_id or use resume"
            ) from exc

        for relative in ("predictions", "traces", "shards"):
            (run_path / relative).mkdir()

        atomic_write_text(
            run_path / "config_resolved.yaml",
            yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        )
        atomic_write_text(run_path / "command.txt", " ".join(command) + "\n")
        atomic_write_json(run_path / "env.json", environment_snapshot())
        atomic_write_json(run_path / "git_state.json", git_snapshot())
        atomic_write_json(run_path / "data_manifest.json", {"status": "not_collected"})
        atomic_write_json(run_path / "model_fingerprint.json", {"status": "not_collected"})
        atomic_write_text(run_path / "stdout.log", "")

        logger = StructuredLogger(run_path / "events.jsonl", run_id=actual_run_id)
        logger.info("run_created", path=str(run_path))
        return cls(actual_run_id, run_path, logger)

    @classmethod
    def resume(
        cls,
        path: Path,
        *,
        config: Mapping[str, Any],
        command: Sequence[str],
    ) -> RunDirectory:
        """Open an interrupted run after verifying its immutable configuration."""
        run_path = path.resolve()
        config_path = run_path / "config_resolved.yaml"
        if not config_path.is_file():
            raise ValueError(f"cannot resume run without configuration: {config_path}")
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(saved, Mapping) or stable_hash(saved, length=64) != stable_hash(
            config, length=64
        ):
            raise ValueError("resume configuration does not match the existing run")
        for relative in ("predictions", "traces", "shards"):
            (run_path / relative).mkdir(exist_ok=True)
        logger = StructuredLogger(run_path / "events.jsonl", run_id=run_path.name)
        logger.info("run_resumed", command=list(command))
        with (run_path / "resume_commands.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(" ".join(command) + "\n")
        return cls(run_path.name, run_path, logger)

    @property
    def successful(self) -> bool:
        return (self.path / "SUCCESS").is_file()

    def mark_success(self) -> None:
        self.logger.info("run_completed")
        atomic_write_text(self.path / "SUCCESS", "")


def build_run_id(config: Mapping[str, Any], *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    dataset = _nested_name(config, "dataset")
    method = _nested_name(config, "method")
    detector = _nested_name(config, "detector")
    return f"{timestamp}_{dataset}_{method}_{detector}_{stable_hash(config)}"


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def git_snapshot() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                encoding="utf-8",
                errors="replace",
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain"),
    }


def _nested_name(config: Mapping[str, Any], key: str) -> str:
    section = config.get(key, {})
    if isinstance(section, Mapping):
        value = section.get("name", key)
        return str(value).replace(" ", "-")
    return key
