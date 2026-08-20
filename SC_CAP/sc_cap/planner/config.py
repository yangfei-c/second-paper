from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .core import PlannerConfig


def planner_config_from_mapping(value: Mapping[str, Any]) -> PlannerConfig:
    """Build the one canonical planner configuration from a mapping."""
    unknown = sorted(set(value) - set(PlannerConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown canonical planner parameters: {unknown}")
    return PlannerConfig(**dict(value))


def load_planner_config(path: str | Path) -> PlannerConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    planner = raw.get("planner", raw)
    if not isinstance(planner, dict):
        raise ValueError("planner configuration must be a mapping")
    return planner_config_from_mapping(planner)
