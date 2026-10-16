"""Helpers shared by tool adapters."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .base import ToolSetupError

HF_TOKEN_ENV = "HF_TOKEN"


def git_head(repo: Path) -> Optional[str]:
    """Commit SHA of a checkout (None if unavailable)."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def clone_repo(url: str, dest: Path, ref: Optional[str] = None) -> Path:
    """Clone `url` into `dest` (reusing an existing clone); check out `ref` if given (needs a full clone)."""
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", url, str(dest)] if ref else ["git", "clone", "--depth", "1", url, str(dest)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ToolSetupError(f"git clone of {url} failed: {getattr(exc, 'stderr', '') or exc}") from exc
    if ref:
        try:
            subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise ToolSetupError(f"git checkout {ref} failed: {exc.stderr}") from exc
    return dest
