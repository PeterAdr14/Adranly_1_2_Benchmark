"""Per-sample scoring, aggregate metrics, failure taxonomy and paired tool comparison."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .chem import (RDKitBackend, is_markush, openbabel_available, openbabel_canonical,
                   strip_extended_smiles)

logger = logging.getLogger("ocsr_bench")

NAN = float("nan")
CATEGORIES = ("plain", "chiral", "markush")


class Scorer:
    """Scores predicted SMILES against ground truth using a chemistry backend."""

    def __init__(self, backend: Any, openbabel_fn: Optional[Callable] = None) -> None:
        self._backend, self._openbabel = backend, openbabel_fn

    @classmethod
    def from_config(cls, mcfg: Dict[str, Any]) -> "Scorer":
        backend = RDKitBackend(mcfg["fingerprint_radius"], mcfg["fingerprint_bits"])
        openbabel_fn = None
        if mcfg["openbabel_cross_check"]:
            if openbabel_available():
                openbabel_fn = openbabel_canonical
            else:
                logger.warning("openbabel_cross_check is on but Open Babel is not installed; skipping "
                               "(pip install openbabel-wheel).")
        return cls(backend, openbabel_fn)

    def score(self, gt_smiles: str, pred_smiles: Optional[str]) -> Dict[str, Any]:
        """Per-sample score columns for one (ground truth, prediction) pair."""
        gt = self._backend.analyze(gt_smiles)
        pred_clean = strip_extended_smiles(pred_smiles)
        pred = self._backend.analyze(pred_clean)
        row: Dict[str, Any] = {
            "gt_valid": gt is not None, "pred_valid": pred is not None, "gt_category": "unparseable",
            "gt_canonical": gt.canonical if gt else "", "pred_canonical": pred.canonical if pred else "",
            "exact_match": NAN, "exact_match_no_stereo": NAN, "inchikey_match": NAN,
            "inchikey_connectivity_match": NAN, "tanimoto": NAN, "exact_match_openbabel": NAN,
        }
        if gt is None:
            return row
        if is_markush(gt_smiles):
            row["gt_category"] = "markush"
        else:
            row["gt_category"] = "chiral" if gt.has_stereo else "plain"

        if pred is None:
            row.update(exact_match=0.0, exact_match_no_stereo=0.0, inchikey_match=0.0,
                       inchikey_connectivity_match=0.0, tanimoto=0.0)
        else:
            exact = gt.canonical == pred.canonical
            flat = gt.canonical_flat == pred.canonical_flat
            row["exact_match"], row["exact_match_no_stereo"] = float(exact), float(flat)
            if gt.inchikey and pred.inchikey:
                row["inchikey_match"] = float(gt.inchikey == pred.inchikey)
                row["inchikey_connectivity_match"] = float(gt.inchikey_connectivity == pred.inchikey_connectivity)
            else:
                row["inchikey_match"], row["inchikey_connectivity_match"] = float(exact), float(flat)
            row["tanimoto"] = self._backend.similarity(gt, pred)

        if self._openbabel is not None:
            gt_ob, pred_ob = self._openbabel(gt_smiles), self._openbabel(pred_clean)
            if gt_ob is not None:
                row["exact_match_openbabel"] = float(pred_ob is not None and gt_ob == pred_ob)
        return row


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0 or n_boot <= 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, int(2e7 // n))  # cap the memory of the index matrix
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        means[start:start + size] = values[rng.integers(0, n, size=(size, n))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def calibration_table(confidence: np.ndarray, correct: np.ndarray, n_bins: int):
    """Accuracy per confidence bin and the expected calibration error (ECE)."""
    mask = ~np.isnan(confidence)
    conf, hit = np.clip(confidence[mask], 0.0, 1.0), correct[mask].astype(float)
    if len(conf) == 0:
        return [], NAN
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        sel = bins == b
        count = int(sel.sum())
        if count == 0:
            continue
        mean_conf, acc = float(conf[sel].mean()), float(hit[sel].mean())
        ece += count / len(conf) * abs(acc - mean_conf)
        table.append({"bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "n": count,
                      "mean_confidence": mean_conf, "accuracy": acc})
    return table, float(ece)


def _nanmean(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.mean()) if len(values) else NAN


def summarize_group(df: pd.DataFrame, mcfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Aggregate metrics for one slice of scored rows (a tool / dataset / category)."""
    scored = df[df["gt_valid"]]
    n = len(scored)
    out: Dict[str, Any] = {
        "n_total": int(len(df)), "n_scored": int(n), "n_gt_unparseable": int((~df["gt_valid"]).sum()),
        "n_tool_errors": int(df["error"].fillna("").astype(str).str.len().gt(0).sum()),
    }
    if n == 0:
        return out
    out["pred_validity_rate"] = float(scored["pred_valid"].mean())
    for column in ("exact_match", "exact_match_no_stereo", "inchikey_match",
                   "inchikey_connectivity_match", "exact_match_openbabel"):
        out[column] = _nanmean(scored[column])
    out["exact_match_ci_low"], out["exact_match_ci_high"] = bootstrap_ci(
        scored["exact_match"].to_numpy(), mcfg["bootstrap_samples"], seed)
    out["mean_tanimoto"], out["median_tanimoto"] = _nanmean(scored["tanimoto"]), float(scored["tanimoto"].median())
    conf, hit = scored["confidence"].astype(float), scored["exact_match"].astype(float)
    out["mean_confidence"] = _nanmean(conf)
    out["mean_confidence_correct"] = _nanmean(conf[hit == 1.0])
    out["mean_confidence_incorrect"] = _nanmean(conf[hit == 0.0])
    runtime = scored["runtime_s"].astype(float).dropna()
    if len(runtime):
        total = float(runtime.sum())
        out.update(runtime_mean_s=float(runtime.mean()), runtime_p50_s=float(runtime.quantile(0.5)),
                   runtime_p95_s=float(runtime.quantile(0.95)),
                   throughput_img_per_s=float(len(runtime) / total) if total > 0 else NAN)
    return out


def summarize(df: pd.DataFrame, mcfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Metrics overall and per ground-truth category for one tool on one dataset."""
    result: Dict[str, Any] = {"categories": {"all": summarize_group(df, mcfg, seed)}}
    for category in CATEGORIES:
        part = df[df["gt_category"] == category]
        if len(part):
            result["categories"][category] = summarize_group(part, mcfg, seed)
    scored = df[df["gt_valid"]]
    table, ece = calibration_table(scored["confidence"].astype(float).to_numpy(),
                                   scored["exact_match"].astype(float).to_numpy(), mcfg["confidence_bins"])
    result["calibration"] = {"ece": ece, "bins": table}
    return result


# ---- failure taxonomy and paired tool comparison (proposal: error taxonomy, pairwise 95% bootstrap CIs) ----
FAILURE_ORDER = ["correct", "stereo_only", "near_miss", "wrong_structure", "markush_mismatch",
                 "invalid_output", "tool_error", "gt_unparseable"]


def classify_failure(row: pd.Series) -> str:
    """Automatic first-pass failure type for one scored row (a subset of the proposal's taxonomy)."""
    if not row["gt_valid"]:
        return "gt_unparseable"
    if row["exact_match"] == 1.0:
        return "correct"
    error = row.get("error")
    if isinstance(error, str) and error:
        return "tool_error"
    if not row["pred_valid"]:
        return "invalid_output"
    if row["gt_category"] == "markush":
        return "markush_mismatch"
    if row["exact_match_no_stereo"] == 1.0:
        return "stereo_only"
    if row["tanimoto"] >= 0.85:
        return "near_miss"
    return "wrong_structure"


def failure_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Counts of each failure type per tool and dataset (one row per tool / dataset)."""
    counts = predictions.groupby(["tool", "dataset", "failure_type"]).size().unstack(fill_value=0)
    ordered = [c for c in FAILURE_ORDER if c in counts.columns] + [c for c in counts.columns if c not in FAILURE_ORDER]
    return counts[ordered].reset_index()


def pairwise_comparison(predictions: pd.DataFrame, mcfg: Dict[str, Any], seed: int) -> pd.DataFrame:
    """Paired exact-match difference between every two tools on the same images, with a bootstrap 95% CI.

    A positive difference means tool A is more accurate. `significant` is True when the CI excludes zero.
    """
    rows: List[Dict[str, Any]] = []
    scored = predictions[predictions["gt_valid"]]
    for dataset, part in scored.groupby("dataset", sort=False):
        wide = part.pivot_table(index="sample_id", columns="tool", values="exact_match", aggfunc="mean")
        for tool_a, tool_b in itertools.combinations(list(wide.columns), 2):
            both = wide[[tool_a, tool_b]].dropna()
            if both.empty:
                continue
            diff = (both[tool_a] - both[tool_b]).to_numpy()
            low, high = bootstrap_ci(diff, mcfg["bootstrap_samples"], seed)
            rows.append({"dataset": dataset, "tool_a": tool_a, "tool_b": tool_b, "n_paired": int(len(both)),
                         "exact_match_a": float(both[tool_a].mean()), "exact_match_b": float(both[tool_b].mean()),
                         "difference_a_minus_b": float(diff.mean()), "ci_low": low, "ci_high": high,
                         "significant": bool(low > 0 or high < 0)})
    return pd.DataFrame(rows)
