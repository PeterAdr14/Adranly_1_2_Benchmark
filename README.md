# CV_XN_Benchmark_1

# Phase 1 Benchmark

This folder contains the "phase1_benchmark" dataset and accompanying materials used for the Phase 1 benchmark in this repository. The benchmark package contains the raw data, metadata/summary, and example scripts to load and evaluate models on the dataset.

> Note: The archive `phase1_benchmark.zip` includes a `phase1_benchmark/summary.md` file describing the dataset. Refer to that file for any dataset-specific notes preserved from the original provider.

## Goals
- Provide a reproducible benchmark dataset for Phase 1 experiments.
- Make it straightforward to inspect, run, and evaluate models against this benchmark.
- Keep dataset structure and evaluation procedures clear and scriptable.

## Contents
A typical unpacked `phase1_benchmark/` directory should include:
- `images/` — image files used in the benchmark (e.g., JPEG/PNG).
- `annotations/` — ground-truth labels in a machine-readable format (JSON, CSV, or similar).
- `splits/` — train/val/test split files or lists (optional).
- `scripts/` — example scripts for loading data and running baseline evaluations.
- `summary.md` — human-readable summary of the dataset contents, collection protocol, and caveats.
- `LICENSE` — license governing use of the dataset (if provided).
- `README.md` (this file)

If your ZIP has additional subfolders (e.g., `__MACOSX/` or hidden system files), you can ignore/remove those after unpacking.

## Quick start

1. Unzip the dataset into the repository root (or a data directory of your choice):

```bash
# unzip into repository root (safe: extracts 'phase1_benchmark/' folder)
unzip phase1_benchmark.zip
# or to a data directory
unzip phase1_benchmark.zip -d data/
