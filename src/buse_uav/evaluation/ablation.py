from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class AblationRegistryError(RuntimeError):
    """Raised when a paper ablation registry cannot be traced to accepted runs."""


def build_labeled_ablation_rows(
    registry_path: Path,
    accepted_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand a one-factor registry into a traceable long-form paper table."""
    try:
        document = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AblationRegistryError(
            f"cannot parse ablation registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AblationRegistryError("ablation registry requires schema_version 1")
    axes = document.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise AblationRegistryError("ablation registry requires a nonempty axes mapping")

    run_index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in accepted_rows:
        key = (str(row.get("group", "")), str(row.get("run_id", "")))
        run_index.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    seen_values: set[tuple[str, str]] = set()
    for axis_order, (axis, raw_axis) in enumerate(axes.items()):
        if not isinstance(raw_axis, dict):
            raise AblationRegistryError(f"ablation axis {axis!r} must be a mapping")
        description = str(raw_axis.get("description", "")).strip()
        entries = raw_axis.get("entries")
        if not description or not isinstance(entries, list) or not entries:
            raise AblationRegistryError(f"ablation axis {axis!r} is incomplete")
        for value_order, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                raise AblationRegistryError(f"ablation axis {axis!r} has a non-object entry")
            group = str(raw_entry.get("group", "")).strip()
            run_id = str(raw_entry.get("run_id", "")).strip()
            value = str(raw_entry.get("value", "")).strip()
            label = str(raw_entry.get("label", "")).strip()
            if not group or not run_id or not value or not label:
                raise AblationRegistryError(f"ablation axis {axis!r} has an incomplete entry")
            value_key = (str(axis), value)
            if value_key in seen_values:
                raise AblationRegistryError(f"duplicate ablation value: {value_key}")
            seen_values.add(value_key)
            matches = run_index.get((group, run_id), [])
            if len(matches) != 1:
                raise AblationRegistryError(
                    f"ablation source must match exactly one accepted row: {(group, run_id)}; "
                    f"matches={len(matches)}"
                )
            source = dict(matches[0])
            output.append(
                {
                    "ablation_axis": str(axis),
                    "ablation_value": value,
                    "ablation_label": label,
                    "axis_description": description,
                    "axis_order": axis_order,
                    "value_order": value_order,
                    "source_group": group,
                    "registry": str(registry_path.resolve()),
                    **source,
                }
            )
    return output
