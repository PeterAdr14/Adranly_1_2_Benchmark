"""Turn scored predictions and aggregate metrics into files on disk."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---- output files -------------------------------------------------------------------------------
COMPARISON_COLUMNS = [
    "tool", "dataset", "category", "n_scored", "n_gt_unparseable", "n_tool_errors",
    "exact_match", "exact_match_ci_low", "exact_match_ci_high", "exact_match_no_stereo",
    "inchikey_match", "inchikey_connectivity_match", "exact_match_openbabel",
    "mean_tanimoto", "median_tanimoto", "pred_validity_rate",
    "mean_confidence", "mean_confidence_correct", "mean_confidence_incorrect",
    "runtime_mean_s", "runtime_p50_s", "runtime_p95_s", "throughput_img_per_s",
]
FLAG_COLUMNS = ["gt_valid", "pred_valid", "exact_match", "exact_match_no_stereo", "inchikey_match",
                "inchikey_connectivity_match", "exact_match_openbabel"]


def build_comparison(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Flatten nested metrics into one row per tool / dataset / category (the side-by-side table)."""
    rows = [{"tool": tool, "dataset": dataset, "category": category, **values}
            for tool, per_dataset in metrics.items() for dataset, entry in per_dataset.items()
            for category, values in entry["categories"].items()]
    frame = pd.DataFrame(rows)
    return frame[[c for c in COMPARISON_COLUMNS if c in frame.columns]]


def _clean(value: Any) -> Any:
    """Make nested data JSON-safe: numpy -> python, NaN / inf -> None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def build_summary_markdown(comparison: pd.DataFrame, metadata_: Dict[str, Any],
                           failures: Optional[pd.DataFrame] = None, pairwise: Optional[pd.DataFrame] = None) -> str:
    rows = []
    for _, r in comparison[comparison["category"] == "all"].iterrows():
        ci = f"[{_fmt(r.get('exact_match_ci_low'))}, {_fmt(r.get('exact_match_ci_high'))}]"
        rows.append([r["tool"], r["dataset"], str(int(r["n_scored"])), f"{_fmt(r.get('exact_match'))} {ci}",
                     _fmt(r.get("exact_match_no_stereo")), _fmt(r.get("inchikey_connectivity_match")),
                     _fmt(r.get("mean_tanimoto")), _fmt(r.get("pred_validity_rate")),
                     _fmt(r.get("throughput_img_per_s"), 1)])
    overall = _md_table(["tool", "dataset", "n", "exact match [95% CI]", "no-stereo", "InChIKey conn.",
                         "mean Tanimoto", "valid SMILES", "img/s"], rows)
    by_category = []
    for (tool, dataset), part in comparison.groupby(["tool", "dataset"], sort=False):
        cells = {r["category"]: r for _, r in part.iterrows()}

        def cell(name: str) -> str:
            r = cells.get(name)
            return "-" if r is None else f"{_fmt(r.get('exact_match'))} (n={int(r['n_scored'])})"

        by_category.append([tool, dataset, cell("plain"), cell("chiral"), cell("markush")])
    env = metadata_.get("environment", {})
    lines = [f"# OCSR benchmark: {metadata_.get('run_name')}", "",
             f"- Finished: {metadata_.get('finished_at')} | config hash: {metadata_.get('config_hash')} | GPU: {env.get('gpu')}",
             "", "## Overall (all scorable structures)", "", overall, "",
             "## Exact match by ground-truth category", "",
             _md_table(["tool", "dataset", "plain", "chiral", "markush"], by_category), ""]
    if pairwise is not None and len(pairwise):
        prow = [[r["dataset"], r["tool_a"], r["tool_b"], str(int(r["n_paired"])), _fmt(r["difference_a_minus_b"]),
                 f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]", "yes" if r["significant"] else "no"]
                for _, r in pairwise.iterrows()]
        lines += ["## Paired tool comparison (exact match, A minus B)", "",
                  _md_table(["dataset", "A", "B", "paired n", "difference", "95% CI", "CI excludes 0"], prow), ""]
    if failures is not None and len(failures):
        lines += ["## Failure types (counts)", "",
                  _md_table([str(c) for c in failures.columns],
                            [[str(v) for v in row] for row in failures.itertuples(index=False)]), ""]
    lines += ["Exact match = identical canonical isomeric SMILES. Unparseable predictions count as wrong. "
              "Categories: markush = wildcard/R-group atoms; chiral = has assigned stereo; plain = neither.", ""]
    return "\n".join(lines)


def write_outputs(out_dir: Path, predictions: pd.DataFrame, metrics: Dict[str, Any], comparison: pd.DataFrame,
                  metadata_: Dict[str, Any], freeze: str = "", pairwise: Optional[pd.DataFrame] = None,
                  failures: Optional[pd.DataFrame] = None) -> None:
    """Write predictions, comparison, pairwise, failure summary, metrics, metadata, summary (+ pip freeze)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = predictions.copy()
    for column in FLAG_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column].astype(float)).astype("Int64")
    frame.round(5).to_csv(out_dir / "predictions.csv", index=False)
    comparison.round(5).to_csv(out_dir / "comparison.csv", index=False)
    if pairwise is not None:
        pairwise.round(5).to_csv(out_dir / "pairwise_comparison.csv", index=False)
    if failures is not None:
        failures.to_csv(out_dir / "failure_summary.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(json.dumps(_clean(metadata_), indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(build_summary_markdown(comparison, metadata_, failures, pairwise), encoding="utf-8")
    if freeze:
        (out_dir / "requirements.lock.txt").write_text(freeze, encoding="utf-8")
