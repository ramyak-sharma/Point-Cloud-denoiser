"""YAML configuration loading shared by every pipeline stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a pipeline config, resolving relative data paths against the repo root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def resolve(cfg: dict[str, Any], relative_path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
