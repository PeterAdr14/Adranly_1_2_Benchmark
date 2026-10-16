"""Configuration: YAML loading with ${VAR} expansion, dotted overrides, entry filters and validation.

Paths in the YAML refer to environment variables (``${OCSR_WORK_DIR}``) so the same file works on a laptop and on
the cluster. ``${VAR:-default}`` supplies a fallback; an unset variable without a default is an error that names it.
Any value can be overridden from the command line, e.g. ``--set run.max_samples=20`` (list entries are addressed by
their ``name``: ``--set datasets.clef.params.max_samples=100``).
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration."""


@dataclass
class Spec:
    """A dataset or tool entry from the configuration."""
    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


def get_specs(cfg: Dict[str, Any], section: str) -> List[Spec]:
    """Enabled entries of `datasets` / `tools` (entries default to enabled)."""
    return [Spec(e["name"], e["kind"], dict(e.get("params") or {}))
            for e in cfg[section] if e.get("enabled", True)]


def validate_config(cfg: Dict[str, Any]) -> None:
    """Fail early, with a clear message, on a bad configuration."""
    for key in ("run", "metrics", "datasets", "tools"):
        if key not in cfg:
            raise ConfigError(f"Configuration is missing the '{key}' section.")
    run = cfg["run"]
    if run["batch_size"] < 1:
        raise ConfigError("run.batch_size must be >= 1")
    if run["max_samples"] is not None and run["max_samples"] < 1:
        raise ConfigError("run.max_samples must be >= 1 or null")
    if run["device"] not in {"auto", "cuda", "cpu"}:
        raise ConfigError("run.device must be one of: auto, cuda, cpu")
    if not run["name"] or "/" in run["name"] or "\\" in run["name"]:
        raise ConfigError("run.name must be a simple folder name")
    for section in ("datasets", "tools"):
        entries = cfg[section]
        names = [e.get("name") for e in entries]
        if not entries or len(set(names)) != len(names):
            raise ConfigError(f"'{section}' needs at least one entry and unique names: {names}")
        if any("kind" not in e for e in entries):
            raise ConfigError(f"Every '{section}' entry needs a 'kind'.")
        if not any(e.get("enabled", True) for e in entries):
            raise ConfigError(f"All '{section}' entries are disabled; enable at least one.")


# ---- ${VAR} expansion ------------------------------------------------------------------------------
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in strings."""
    if isinstance(value, str):
        def replace(match: "re.Match[str]") -> str:
            name, default = match.group(1), match.group(2)
            if os.environ.get(name):
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name} is not set (needed by the configuration value {value!r}). "
                              "On the cluster, `source slurm/env.sh` sets it.")
        return _VAR.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# ---- overrides and filters --------------------------------------------------------------------------
def _step(node: Any, part: str) -> Any:
    """Descend one level of a dotted path through dicts and name/index-addressed lists."""
    if isinstance(node, list):
        if part.isdigit():
            return node[int(part)]
        for item in node:
            if isinstance(item, dict) and item.get("name") == part:
                return item
        raise ConfigError(f"No list entry named '{part}' in the override path.")
    if isinstance(node, dict):
        return node.setdefault(part, {})
    raise ConfigError(f"Cannot descend into '{part}': the parent is not a mapping or list.")


def apply_override(raw: Dict[str, Any], expression: str) -> None:
    """Apply one ``dotted.path=value`` override in place (the value is parsed as YAML)."""
    key, separator, value = expression.partition("=")
    if not separator or not key.strip():
        raise ConfigError(f"An override must look like 'section.key=value', got: {expression!r}")
    parts = key.strip().split(".")
    node: Any = raw
    for part in parts[:-1]:
        node = _step(node, part)
    if not isinstance(node, dict):
        raise ConfigError(f"Cannot set '{key}': the parent is not a mapping.")
    node[parts[-1]] = yaml.safe_load(value)


def _restrict(entries: List[Dict[str, Any]], wanted: Optional[Sequence[str]], what: str) -> None:
    if not wanted:
        return
    names = [e["name"] for e in entries]
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ConfigError(f"Unknown {what} {unknown}; available: {names}")
    for entry in entries:
        entry["enabled"] = entry["name"] in set(wanted)


def load_config(path: Any, overrides: Iterable[str] = (), only_datasets: Optional[Sequence[str]] = None,
                only_tools: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML configuration, apply overrides and filters, expand ``${VAR}`` and validate."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    raw = copy.deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    for expression in overrides:
        apply_override(raw, expression)
    cfg = expand_env(raw)
    validate_config(cfg)
    _restrict(cfg["datasets"], only_datasets, "dataset")
    _restrict(cfg["tools"], only_tools, "tool")
    validate_config(cfg)
    return cfg
