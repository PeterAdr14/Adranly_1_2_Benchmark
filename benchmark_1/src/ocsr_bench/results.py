"""Load a finished run back from its output folder (for re-reporting without recomputing)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pipeline import RunResult


def load_result(out_dir: Path) -> RunResult:
    """Rebuild a `RunResult` from the files a run wrote."""
    out_dir = Path(out_dir)
    if not (out_dir / "predictions.csv").is_file():
        raise FileNotFoundError(f"No predictions.csv in {out_dir}; run the benchmark first.")
    predictions = pd.read_csv(out_dir / "predictions.csv")
    for column in ("gt_valid", "pred_valid"):
        predictions[column] = predictions[column].astype(bool)
    for column in ("pred_smiles", "error", "failure_type"):
        if column in predictions:
            predictions[column] = predictions[column].astype(object).where(predictions[column].notna(), None)

    def optional(name: str) -> pd.DataFrame:
        path = out_dir / name
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return RunResult(out_dir, predictions, metrics, optional("comparison.csv"), optional("pairwise_comparison.csv"),
                     optional("failure_summary.csv"))
