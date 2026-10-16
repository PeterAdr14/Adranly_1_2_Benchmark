"""Small shared helpers: seeding, device choice, paths, hashing and environment capture."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("ocsr_bench")

ROOT = Path(os.environ.get("OCSR_BENCH_ROOT") or Path.cwd()).resolve()
TRACKED_PACKAGES = ("numpy", "pandas", "rdkit", "torch", "datasets", "huggingface_hub", "gdown",
                    "openbabel-wheel", "MolScribe")




def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(preference: str = "auto") -> str:
    """'cuda' or 'cpu' for a preference of auto / cuda / cpu."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if preference == "cuda" and not cuda_ok:
        raise RuntimeError("device='cuda' requested but no CUDA device is available.")
    return "cuda" if cuda_ok else "cpu"


def resolve_dir(path: Any) -> Path:
    """Resolve a relative folder against the working directory (ROOT)."""
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


# Settings that say *where* things live (or whether to resume), not *what* is computed. They are left out of
# the hash so a run can be resumed after moving data or environments (e.g. to another scratch directory).
_LOCATION_RUN_KEYS = ("data_dir", "output_dir", "resume")
_LOCATION_TOOL_KEYS = ("checkpoint_path", "python_executable", "venv_dir", "repo_path", "biominer_repo_path")


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the effective configuration, ignoring location-only settings."""
    reduced = copy.deepcopy(cfg)
    for key in _LOCATION_RUN_KEYS:
        reduced.get("run", {}).pop(key, None)
    for tool in reduced.get("tools", []):
        for key in _LOCATION_TOOL_KEYS:
            (tool.get("params") or {}).pop(key, None)
    return hashlib.sha256(json.dumps(reduced, sort_keys=True, default=str).encode()).hexdigest()[:16]


def collect_environment() -> Dict[str, Any]:
    """Interpreter, OS, key package versions and GPU, stored with every run."""
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions, "gpu": gpu}


def pip_freeze() -> str:
    """`pip freeze` of the active interpreter (empty string on failure)."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""
