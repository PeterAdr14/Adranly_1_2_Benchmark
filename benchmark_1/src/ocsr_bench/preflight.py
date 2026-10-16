"""Environment preflight: check every prerequisite (packages, GPU, remote files, tool code) before a long run."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chem import openbabel_available
from .config import get_specs
from .tools.common import HF_TOKEN_ENV, git_head
from .tools.molglyph import MOLGLYPH_DEFAULTS, MolGlyphTool
from .tools.molnextr import MOLNEXTR_DEFAULTS, MolNexTRTool
from .utils import resolve_dir

def _importable(module: str) -> bool:
    """True if `module` can be imported (tolerates modules without a spec)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):
        return module in sys.modules


def _remote_file_check(checks: List[Tuple[str, str, str]], label: str, repo: str, filename: str,
                       repo_type: str, token: Optional[str]) -> None:
    if os.environ.get("OCSR_SKIP_REMOTE") == "1":  # set by the GPU job scripts: the prefetch job already cached it
        checks.append((label, "OK", "skipped (OCSR_SKIP_REMOTE=1; fetched by the prefetch step)"))
        return
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        meta = get_hf_file_metadata(hf_hub_url(repo, filename, repo_type=repo_type), token=token)
        size = f"{meta.size / 2**20:.0f} MiB" if meta.size else "size unknown"
        checks.append((label, "OK", f"{repo}/{filename} ({size})"))
    except Exception as exc:
        checks.append((label, "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))


def run_preflight(cfg: Dict[str, Any]) -> int:
    """Print an OK / WARN / FAIL report and return the number of failures. Downloads no model weights."""
    token = os.environ.get(HF_TOKEN_ENV) or None
    tools = get_specs(cfg, "tools")
    kinds = {t.kind for t in tools}
    checks: List[Tuple[str, str, str]] = [("python", "OK", sys.version.split()[0])]
    modules = ["numpy", "pandas", "PIL", "rdkit", "gdown", "datasets", "huggingface_hub", "torch"]
    if "molglyph" in kinds:
        modules += ["cv2", "timm", "albumentations", "SmilesPE"]  # needed by BioMiner's MolGlyph code
    for module in modules:
        found = _importable(module)
        hint = {"cv2": "opencv-python-headless", "PIL": "pillow"}.get(module, module)
        checks.append((f"import {module}", "OK" if found else "FAIL", "" if found else f"pip install {hint}  (cluster: see slurm/setup_envs.sbatch)"))
    if "molglyph" in kinds:  # really import onmt: installed is not enough, its own imports must work too
        try:
            importlib.import_module("onmt")
            checks.append(("import onmt (OpenNMT-py)", "OK", ""))
        except ModuleNotFoundError as exc:
            missing = exc.name or "?"
            install = "OpenNMT-py==2.2.0" if missing == "onmt" else missing
            checks.append(("import onmt (OpenNMT-py)", "FAIL", f"missing module '{missing}': pip install {install}"))
        except Exception as exc:  # e.g. an incompatibility between onmt/torchtext and the installed torch
            checks.append(("import onmt (OpenNMT-py)", "FAIL", f"{type(exc).__name__}: {exc}"))
        try:
            albu = metadata.version("albumentations")
            checks.append(("albumentations version", "OK" if albu == "2.0.5" else "WARN", albu + (
                "" if albu == "2.0.5" else " (BioMiner pins 2.0.5: pip install albumentations==2.0.5 in the main environment)")))
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        checks.append(("CUDA GPU", "OK" if torch.cuda.is_available() else "WARN",
                       torch.cuda.get_device_name(0) if torch.cuda.is_available()
                       else "No GPU visible: inference will be very slow. Submit with --gres=gpu:1 on the gpu partition."))
    except ImportError:
        checks.append(("CUDA GPU", "WARN", "torch not installed"))
    if cfg["metrics"]["openbabel_cross_check"] and not openbabel_available():
        checks.append(("Open Babel", "WARN", "optional cross-check will be skipped (pip install openbabel-wheel)"))
    checks.append(("git", "OK" if shutil.which("git") else "FAIL", "" if shutil.which("git") else "git is needed to fetch tool code"))

    seen = set()
    for spec in get_specs(cfg, "datasets"):
        p = spec.params
        if spec.kind == "hf_archive" and (p.get("hf_repo"), p.get("hf_filename")) not in seen:
            seen.add((p.get("hf_repo"), p.get("hf_filename")))
            _remote_file_check(checks, f"dataset archive {p['hf_filename']}", p["hf_repo"], p["hf_filename"],
                               p.get("repo_type", "dataset"), token)
        elif spec.kind == "gdrive_zip":
            checks.append((f"dataset {spec.name}", "OK", "Google Drive archive is downloaded on first use "
                           "(if Drive's quota blocks it, switch the dataset to kind 'local_dir')"))

    for spec in tools:
        run, data_dir = cfg["run"], resolve_dir(cfg["run"]["data_dir"])
        if spec.kind == "molglyph":
            p = {**MOLGLYPH_DEFAULTS, **spec.params}
            if p["checkpoint_path"]:
                ok = Path(str(p["checkpoint_path"])).expanduser().is_file()
                checks.append(("MolGlyph checkpoint (local)", "OK" if ok else "FAIL", str(p["checkpoint_path"])))
            else:
                checks.append((f"{HF_TOKEN_ENV} available", "OK" if token else "FAIL",
                               "" if token else "MolGlyph is gated: accept its terms on Hugging Face and add the secret"))
                _remote_file_check(checks, "MolGlyph checkpoint reachable", p["checkpoint_repo"],
                                   p["checkpoint_filename"], "model", token)
            try:
                tool = MolGlyphTool(spec, run, data_dir)
                tool.load_code()
                checks.append(("BioMiner MolGlyph code imports", "OK", f"commit {git_head(tool._repo)}"))
            except Exception as exc:
                checks.append(("BioMiner MolGlyph code imports", "FAIL", f"{type(exc).__name__}: {exc}"))
        elif spec.kind == "molnextr":
            p = {**MOLNEXTR_DEFAULTS, **spec.params}
            if p["checkpoint_path"]:
                ok = Path(str(p["checkpoint_path"])).expanduser().is_file()
                checks.append(("MolNexTR checkpoint (local)", "OK" if ok else "FAIL", str(p["checkpoint_path"])))
            else:
                _remote_file_check(checks, "MolNexTR checkpoint reachable", p["checkpoint_repo"],
                                   p["checkpoint_filename"], p["checkpoint_repo_type"], token)
            try:
                tool = MolNexTRTool(spec, run, data_dir)
                repo = tool._ensure_repo()
                checks.append(("MolNexTR repository", "OK", f"commit {git_head(repo)}"))
                if p["python_executable"]:
                    checks.append(("MolNexTR interpreter", "OK" if Path(str(p["python_executable"])).exists() else "FAIL",
                                   str(p["python_executable"])))
                else:
                    built = (tool._work_dir / "molnextr" / ".env_ok").exists()
                    checks.append(("MolNexTR isolated environment", "OK" if built else "WARN",
                                   "already built" if built else f"will be built on first use (Python {p['python_version']}, "
                                   "several minutes)"))
                    uv_ok = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True).returncode == 0
                    checks.append(("uv (environment builder)", "OK" if uv_ok else "WARN",
                                   "" if uv_ok else "will be pip-installed automatically"))
            except Exception as exc:
                checks.append(("MolNexTR repository", "FAIL", f"{type(exc).__name__}: {exc}"))
        elif spec.kind == "osra":
            checks.append(("osra program", "OK" if shutil.which("osra") else "WARN",
                           "" if shutil.which("osra") else "not on PATH; load() will try `apt-get install osra`"))

    width = max(len(name) for name, _, _ in checks)
    for name, status, detail in checks:
        print(f"[{status:4}] {name.ljust(width)}  {detail}")
    failures = sum(status == "FAIL" for _, status, _ in checks)
    print(f"\n{failures} failure(s), {sum(s == 'WARN' for _, s, _ in checks)} warning(s).")
    return failures
