"""Minimal YAML configuration loading.

``AGENTS.md`` §14 requires experimental parameters to live in configuration
files rather than source code, and ``architecture.md`` §14 fixes the expected
layout under ``configs/``.

This module intentionally does NOT validate scientific content. The config files
currently hold structural placeholders with unresolved ``TODO`` values, and
coercing them into a strict schema now would mean inventing the very values this
milestone is supposed to leave open. Typed validation belongs in the milestone
that first consumes a given config section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

#: Configuration files expected by ``architecture.md`` §14.
#:
#: ``features`` is not in the architecture document's original four-file list.
#: It was added when Milestone 3 began consuming feature parameters, because
#: ``AGENTS.md`` §14 requires them to live in configuration rather than in
#: feature functions, and folding them into ``segmentation.yaml`` would put the
#: feature contract inside the file that owns model and threshold choices.
#:
#: ``impact`` holds the infrastructure-exposure and road-connectivity
#: assumptions (OSM filters, hubs, buffer, disruption thresholds), which
#: AGENTS.md §9 requires to be explicit and variable for sensitivity analysis.
EXPECTED_CONFIGS: tuple[str, ...] = (
    "data",
    "preprocessing",
    "features",
    "segmentation",
    "evaluation",
    "impact",
)


def project_root() -> Path:
    """Absolute path to the repository root.

    Resolved relative to this file (``src/floodmap/utils/config.py``) rather
    than the process working directory, so loading behaves identically from a
    notebook, a test and a script (``AGENTS.md`` §14: avoid hidden state).
    """
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    """Absolute path to the ``configs/`` directory."""
    return project_root() / "configs"


def config_path(name: str) -> Path:
    """Path to a named config, with or without the ``.yaml`` suffix."""
    stem = name[:-5] if name.endswith(".yaml") else name
    return configs_dir() / f"{stem}.yaml"


def load_config(name: str) -> Dict[str, Any]:
    """Load one configuration file as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the config does not exist. Surfaced explicitly rather than returning
        an empty dict, because a silently empty configuration would let a
        pipeline run with undocumented defaults.
    TypeError
        If the file does not parse to a mapping.
    """
    path = config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)

    if parsed is None:
        raise TypeError(f"configuration file is empty: {path}")
    if not isinstance(parsed, dict):
        raise TypeError(
            f"configuration must parse to a mapping, got {type(parsed).__name__}: {path}"
        )
    return parsed


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Load every expected configuration file, keyed by name."""
    return {name: load_config(name) for name in EXPECTED_CONFIGS}


def missing_configs() -> List[str]:
    """Names of expected configuration files that are absent."""
    return [name for name in EXPECTED_CONFIGS if not config_path(name).is_file()]


__all__ = [
    "EXPECTED_CONFIGS",
    "config_path",
    "configs_dir",
    "load_all_configs",
    "load_config",
    "missing_configs",
    "project_root",
]
