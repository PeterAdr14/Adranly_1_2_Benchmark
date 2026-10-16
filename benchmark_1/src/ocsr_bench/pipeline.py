"""End-to-end orchestration: data -> inference (resumable) -> scoring -> output files."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import pandas as pd

from .config import get_specs, validate_config
from .data import LoadedDataset, Sample, load_dataset_samples
from .metrics import (Scorer, classify_failure, failure_summary, pairwise_comparison, summarize)
from .report import build_comparison, write_outputs
from .tools import OCSRTool, RawPrediction, build_tool
from .utils import collect_environment, config_hash, pip_freeze, resolve_dir, set_seed

logger = logging.getLogger("ocsr_bench")

class PipelineError(RuntimeError):
    """Raised for run-level problems such as resuming with a changed config."""


@dataclass
class RunResult:
    out_dir: Path
    predictions: pd.DataFrame
    metrics: Dict[str, Any]
    comparison: pd.DataFrame
    pairwise: pd.DataFrame
    failures: pd.DataFrame


# ---- inference with incremental, resumable persistence -----------------------------------------
def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _read_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["sample_id"]] = record
    return records


def _record(sample: Sample, pred: RawPrediction, runtime_s: float, error: Optional[str]) -> Dict[str, Any]:
    return {"sample_id": sample.sample_id, "smiles": pred.smiles, "confidence": pred.confidence,
            "error": error, "runtime_s": runtime_s, "extra": pred.extra}


def _run_batch(tool: OCSRTool, batch: Sequence[Sample]) -> List[Dict[str, Any]]:
    start = time.perf_counter()
    try:
        predictions = tool.predict_batch([s.image_path for s in batch])
        if len(predictions) != len(batch):
            raise ValueError(f"tool returned {len(predictions)} predictions for {len(batch)} images")
        per_image = (time.perf_counter() - start) / len(batch)
        return [_record(s, p, per_image, None) for s, p in zip(batch, predictions)]
    except Exception as exc:  # isolate failures; retried per image below
        logger.warning("Batch failed (%s: %s); retrying images one by one.", type(exc).__name__, exc)
    records = []
    for sample in batch:
        start = time.perf_counter()
        try:
            prediction, error = tool.predict_batch([sample.image_path])[0], None
        except Exception as exc:
            prediction, error = RawPrediction(smiles=None), f"{type(exc).__name__}: {exc}"
        records.append(_record(sample, prediction, time.perf_counter() - start, error))
    return records


def predict_samples(tool: OCSRTool, samples: List[Sample], raw_path: Path, batch_size: int,
                    resume: bool) -> List[Dict[str, Any]]:
    """Run `tool` over `samples`, persisting each batch; returns one record per sample, in order."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and raw_path.exists():
        raw_path.unlink()
    done = _read_jsonl(raw_path)
    todo = [s for s in samples if s.sample_id not in done]
    logger.info("%s: %d samples to run (%d already done)", raw_path.stem, len(todo), len(done))
    started = time.perf_counter()
    with raw_path.open("a", encoding="utf-8") as sink:
        for number, batch in enumerate(_chunks(todo, batch_size), start=1):
            for record in _run_batch(tool, batch):
                sink.write(json.dumps(record) + "\n")
                done[record["sample_id"]] = record
            sink.flush()
            if number % 10 == 0 or number * batch_size >= len(todo):
                finished = min(number * batch_size, len(todo))
                rate = finished / max(time.perf_counter() - started, 1e-9)
                logger.info("%s: %d/%d images (%.1f img/s)", raw_path.stem, finished, len(todo), rate)
    return [done[s.sample_id] for s in samples]


# ---- assembly and scoring -----------------------------------------------------------------------
def _relative(path: Path, base: Path) -> str:
    """Path relative to `base` when possible (keeps outputs portable), else the absolute path."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:  # e.g. a local_dir dataset stored outside data_dir
        return path.as_posix()


def assemble_frame(tool_name: str, dataset: LoadedDataset, records: List[Dict[str, Any]],
                   data_dir: Path) -> pd.DataFrame:
    rows = [{"tool": tool_name, "dataset": dataset.name, "sample_id": s.sample_id,
             "image": _relative(s.image_path, data_dir), "gt_smiles": s.gt_smiles,
             "pred_smiles": r["smiles"], "confidence": r["confidence"], "error": r["error"],
             "runtime_s": r["runtime_s"]} for s, r in zip(dataset.samples, records)]
    return pd.DataFrame(rows)


def score_predictions(frame: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Append the per-sample score columns to a predictions frame."""
    scores = pd.DataFrame([scorer.score(gt, pred) for gt, pred in zip(frame["gt_smiles"], frame["pred_smiles"])])
    scored = pd.concat([frame.reset_index(drop=True), scores], axis=1)
    scored["failure_type"] = scored.apply(classify_failure, axis=1)
    return scored




# ---- orchestration ------------------------------------------------------------------------------
def _check_resume(out_dir: Path, cfg: Dict[str, Any], current_hash: str) -> None:
    meta_path = out_dir / "run_metadata.json"
    if cfg["run"]["resume"] and meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8")).get("config_hash")
        if previous != current_hash:
            raise PipelineError(
                f"'{out_dir}' holds results from a different configuration (hash {previous} vs {current_hash}). "
                "Use a new run name, delete the folder, or set resume to False.")


def run_benchmark(cfg: Dict[str, Any], scorer: Optional[Scorer] = None) -> RunResult:
    """Run every configured tool on every configured dataset and write all outputs."""
    validate_config(cfg)
    run = cfg["run"]
    started_at = datetime.now(timezone.utc)
    set_seed(run["seed"])
    data_dir, out_dir = resolve_dir(run["data_dir"]), resolve_dir(run["output_dir"]) / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    current_hash = config_hash(cfg)
    _check_resume(out_dir, cfg, current_hash)
    logger.info("Run '%s' -> %s (config %s)", run["name"], out_dir, current_hash)

    datasets = [load_dataset_samples(spec, data_dir, run["max_samples"], run["seed"])
                for spec in get_specs(cfg, "datasets")]
    scorer = scorer or Scorer.from_config(cfg["metrics"])

    frames: List[pd.DataFrame] = []
    resources: Dict[str, Any] = {}
    for tool_spec in get_specs(cfg, "tools"):
        tool = build_tool(tool_spec, run, data_dir)
        logger.info("Loading tool '%s' (kind=%s)", tool.name, tool_spec.kind)
        tool.load()
        try:
            for dataset in datasets:
                raw_path = out_dir / "raw" / f"{tool.name}__{dataset.name}.jsonl"
                records = predict_samples(tool, dataset.samples, raw_path, run["batch_size"], run["resume"])
                frames.append(assemble_frame(tool.name, dataset, records, data_dir))
        finally:
            resources[tool.name] = tool.resource_summary()
            tool.close()

    predictions = score_predictions(pd.concat(frames, ignore_index=True), scorer)
    metrics: Dict[str, Any] = {}
    dataset_info = {d.name: d.info for d in datasets}
    for (tool_name, dataset_name), group in predictions.groupby(["tool", "dataset"], sort=False):
        entry = summarize(group, cfg["metrics"], run["seed"])
        entry["dataset_info"] = dataset_info[dataset_name]
        entry["failure_counts"] = {str(k): int(v) for k, v in group["failure_type"].value_counts().items()}
        metrics.setdefault(tool_name, {})[dataset_name] = entry

    metadata_ = {"run_name": run["name"], "config_hash": current_hash, "config": cfg,
                 "started_at": started_at.isoformat(timespec="seconds"),
                 "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "environment": collect_environment(), "tool_resources": resources, "datasets": dataset_info}
    comparison = build_comparison(metrics)
    pairwise = pairwise_comparison(predictions, cfg["metrics"], run["seed"])
    failures = failure_summary(predictions)
    write_outputs(out_dir, predictions, metrics, comparison, metadata_, pip_freeze(), pairwise, failures)
    logger.info("Done. Results written to %s", out_dir)
    return RunResult(out_dir, predictions, metrics, comparison, pairwise, failures)
