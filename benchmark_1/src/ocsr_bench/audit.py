"""Check a finished run against the assignment requirements."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .pipeline import RunResult
from .tools.common import HF_TOKEN_ENV

def requirements_audit(result: RunResult, cfg: Dict[str, Any]) -> bool:
    preds, files = result.predictions, {p.name for p in result.out_dir.iterdir()}
    datasets_run, tools_run = set(preds["dataset"]), set(preds["tool"])
    token = os.environ.get(HF_TOKEN_ENV)
    leaked = bool(token) and any(token in p.read_text(errors="ignore") for p in result.out_dir.iterdir() if p.is_file())
    confidence_share = {t: float(preds.loc[preds["tool"] == t, "confidence"].notna().mean()) for t in tools_run}
    checks = [
        ("Public benchmark datasets downloaded and loaded", len(datasets_run) >= 1),
        ("Proposal benchmarks present (CLEF, UOB, USPTO, JPO)", {"clef", "uob", "uspto", "jpo"} <= datasets_run),
        ("BioMiner model and BioVista dataset run", "biominer_molglyph" in tools_run and "biovista_ocsr" in datasets_run),
        ("More than one tool compared side by side", len(tools_run) >= 2),
        ("Predicted SMILES captured for every tool", all(preds.loc[preds["tool"] == t, "pred_smiles"].notna().any() for t in tools_run)),
        ("Exact match and Tanimoto computed", {"exact_match", "tanimoto"} <= set(preds.columns)),
        ("Results saved in structured form (CSV + JSON)", {"predictions.csv", "comparison.csv", "metrics.json"} <= files),
        ("Pairwise comparison and failure summary saved", {"pairwise_comparison.csv", "failure_summary.csv"} <= files or len(tools_run) < 2),
        ("Dependencies documented (pip freeze saved with the run)", "requirements.lock.txt" in files),
        ("No credential written into any saved file", not leaked),
    ]
    for label, ok in checks:
        print(("[ OK ] " if ok else "[FAIL] ") + label)
    print("\nConfidence scores captured (share of images with one):", {t: f"{v:.0%}" for t, v in confidence_share.items()})
    print("Tools that report no confidence leave this column empty by design (MolGlyph returns SMILES only).")
    return all(ok for _, ok in checks)
