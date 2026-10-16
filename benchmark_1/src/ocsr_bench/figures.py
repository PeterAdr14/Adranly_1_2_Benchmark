"""Headless figures (PNG) for a finished run: accuracy per dataset, failure types and an error gallery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import get_specs
from .data import load_dataset_samples
from .pipeline import RunResult
from .utils import resolve_dir

logger = logging.getLogger("ocsr_bench")


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")  # no display on a compute node
    import matplotlib.pyplot as plt
    return plt


def plot_exact_match(result: RunResult, path: Path) -> Path:
    """Exact match per dataset and tool with the bootstrap 95% CI as error bars."""
    plt = _pyplot()
    overall = result.comparison[result.comparison["category"] == "all"]
    datasets = list(dict.fromkeys(overall["dataset"]))
    tools = list(dict.fromkeys(overall["tool"]))
    width = 0.8 / max(len(tools), 1)
    fig, ax = plt.subplots(figsize=(1.8 * len(datasets) + 3, 4))
    for k, tool in enumerate(tools):
        part = overall[overall["tool"] == tool].set_index("dataset").reindex(datasets)
        x = [i + k * width for i in range(len(datasets))]
        err = [(part["exact_match"] - part["exact_match_ci_low"]).fillna(0),
               (part["exact_match_ci_high"] - part["exact_match"]).fillna(0)]
        ax.bar(x, part["exact_match"].fillna(0), width * 0.92, yerr=err, capsize=3, label=tool)
    ax.set_xticks([i + width * (len(tools) - 1) / 2 for i in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("exact match")
    ax.set_title("Exact match by dataset and tool (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_failure_types(result: RunResult, path: Path) -> Path:
    """Share of each failure type per tool and dataset."""
    plt = _pyplot()
    share = result.failures.set_index(["tool", "dataset"])
    share = share.div(share.sum(axis=1), axis=0)
    ax = share.plot(kind="bar", stacked=True, figsize=(max(8, 0.9 * len(share)), 4.5), colormap="tab20")
    ax.set_ylabel("share of images")
    ax.set_title("Failure types by tool and dataset")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_gallery(result: RunResult, data_dir: Path, path: Path, n: int = 6, seed: int = 0) -> Path:
    """A random sample of wrong predictions next to their images (skipped if the images are gone)."""
    plt = _pyplot()
    from PIL import Image
    preds = result.predictions
    wrong = preds[preds["gt_valid"] & (preds["exact_match"] == 0)]
    wrong = wrong[[(data_dir / img).is_file() for img in wrong["image"]]]
    if wrong.empty:
        raise FileNotFoundError("no wrong predictions with available images")
    sample = wrong.sample(min(n, len(wrong)), random_state=seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        ax.imshow(Image.open(data_dir / row["image"]))
        confidence = "n/a" if pd.isna(row["confidence"]) else f"{row['confidence']:.2f}"
        ax.set_title(f"[{row['tool']} | {row['dataset']}] {row['failure_type']} conf={confidence}\n"
                     f"GT:   {row['gt_smiles'][:48]}\nPred: {str(row['pred_smiles'])[:48]}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_dataset_samples(cfg: Dict[str, Any], path: Path, per_dataset: int = 3) -> Path:
    """Three images with ground-truth SMILES per enabled dataset (a first look)."""
    plt = _pyplot()
    from PIL import Image
    specs = get_specs(cfg, "datasets")
    run = cfg["run"]
    fig, axes = plt.subplots(len(specs), per_dataset, figsize=(4 * per_dataset, 3.2 * len(specs)), squeeze=False)
    for row_axes, spec in zip(axes, specs):
        loaded = load_dataset_samples(spec, resolve_dir(run["data_dir"]), run["max_samples"], run["seed"])
        for ax in row_axes:
            ax.axis("off")
        for ax, sample in zip(row_axes, loaded.samples[:per_dataset]):
            ax.imshow(Image.open(sample.image_path))
            ax.set_title(f"{spec.name}\n{sample.gt_smiles[:38]}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_figures(result: RunResult, data_dir: Path) -> List[Path]:
    """Write all run figures under `<run folder>/figures/`; individual failures are logged, not fatal."""
    folder = result.out_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    for label, make in (("exact_match_by_dataset.png", lambda p: plot_exact_match(result, p)),
                        ("failure_types.png", lambda p: plot_failure_types(result, p)),
                        ("error_gallery.png", lambda p: plot_error_gallery(result, data_dir, p))):
        try:
            made.append(make(folder / label))
        except Exception as exc:  # a missing image or a one-tool run must not fail the whole job
            logger.warning("Skipped figure %s (%s: %s)", label, type(exc).__name__, exc)
    return made
