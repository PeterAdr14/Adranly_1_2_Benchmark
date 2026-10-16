"""Tool interface and registry: every OCSR model is an `OCSRTool` registered under a `kind`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..config import Spec

logger = logging.getLogger("ocsr_bench")

class ToolSetupError(RuntimeError):
    """Raised when a tool cannot be installed, authenticated or loaded."""


@dataclass
class RawPrediction:
    """What a tool returns for one image."""
    smiles: Optional[str]
    confidence: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class OCSRTool:
    """An image -> SMILES model. Subclasses implement `load` and `predict_batch`."""

    def __init__(self, spec: Spec, run: Dict[str, Any], data_dir: Path) -> None:
        self.spec, self.run, self.data_dir = spec, run, data_dir

    @property
    def name(self) -> str:
        return self.spec.name

    def load(self) -> None:
        """Acquire weights and prepare the model for inference."""
        raise NotImplementedError

    def predict_batch(self, image_paths: Sequence[Path]) -> List[RawPrediction]:
        """Return exactly one RawPrediction per input image, in order."""
        raise NotImplementedError

    def prefetch(self) -> None:
        """Fetch everything the tool needs (weights, code, environment) without running inference.

        Run on a CPU node before any GPU job so GPU time is spent only on inference.
        """

    def resource_summary(self) -> Dict[str, Any]:
        """Optional resource usage / provenance (e.g. peak GPU memory) recorded in the run metadata."""
        return {}

    def close(self) -> None:
        """Release GPU memory. Called once after all datasets are processed."""


TOOL_REGISTRY: Dict[str, type] = {}


def register_tool(kind: str):
    """Class decorator that registers an adapter under `kind`."""
    def decorator(cls):
        TOOL_REGISTRY[kind] = cls
        return cls
    return decorator


def build_tool(spec: Spec, run: Dict[str, Any], data_dir: Path) -> OCSRTool:
    if spec.kind not in TOOL_REGISTRY:
        raise ToolSetupError(f"Unknown tool kind '{spec.kind}'. Registered: {sorted(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[spec.kind](spec, run, data_dir)
