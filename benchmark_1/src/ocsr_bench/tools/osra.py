"""OSRA, the proposal's classical baseline (optional)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence

from .base import OCSRTool, RawPrediction, ToolSetupError, register_tool

@register_tool("osra")
class OSRATool(OCSRTool):
    """OSRA, the proposal's classical baseline, called as a command-line program (optional, UNVERIFIED).

    Assumes `osra -f can <image>` prints one canonical SMILES per recognised structure, one per line; the
    first line is used. Confidence is not reported.
    """

    def load(self) -> None:
        if shutil.which("osra") is None:
            try:
                subprocess.run(["apt-get", "install", "-y", "-qq", "osra"], check=True, capture_output=True, text=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ToolSetupError("The 'osra' program is not installed and `apt-get install osra` failed "
                                     f"({getattr(exc, 'stderr', '') or exc}). Install OSRA manually or disable this tool.") from exc

    def predict_batch(self, image_paths: Sequence[Path]) -> List[RawPrediction]:
        timeout = float(self.spec.params.get("timeout_s", 120))
        results = []
        for path in image_paths:
            proc = subprocess.run(["osra", "-f", "can", str(path)], capture_output=True, text=True, timeout=timeout)
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            results.append(RawPrediction(
                smiles=lines[0].split()[0] if lines else "",
                extra={"n_structures_reported": len(lines), "returncode": proc.returncode,
                       "stderr_tail": proc.stderr[-200:] if proc.returncode else ""}))
        return results
