#Benchmark_1

Short description
-----------------
This repository contains code and experiments for cross-network computer vision benchmarking (CV_XN_Benchmark_1). It provides training and evaluation scripts, experiment configurations, and analysis notebooks to reproduce and compare benchmark results across models/datasets used in the experiments.

Quick summary
-------------
- Purpose: Run reproducible CV benchmark experiments, reproduce reported metrics, and inspect analysis notebooks.
- Users: researchers and engineers who want to reproduce or extend the benchmark experiments.

Table of contents
-----------------
- Installation
- Repository layout
- Reproducing results (full pipeline)
- Brief results breakdown
- Notes on hardware / randomness
- Contact / citation

Installation
------------
1. Clone the repo:
   git clone https://github.com/PeterAdr14/CV_XN_Benchmark_1.git
   cd CV_XN_Benchmark_1

2. Create a Python virtual environment and install dependencies:
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3. (Optional) If Docker is used:
   docker build -t cv_xn_benchmark:latest .
   docker run -it --gpus all -v $(pwd):/workspace cv_xn_benchmark:latest

Repository layout
-----------------
(Adjust these entries to match the actual top-level folders in this repository)
- data/               Dataset download, preprocessing scripts, data README
- src/                Core training, evaluation, and utility code (entry points: src/train.py, src/evaluate.py)
- configs/            Experiment configuration files (YAML/JSON)
- experiments/        Experiment run scripts, logs, and checkpoint conventions
- models/             Saved model checkpoints
- results/            Aggregated metrics, CSVs, plots
- notebooks/          Jupyter notebooks for analysis and visualization
- scripts/            Small shell/python helper scripts (download, preprocess, run)
- requirements.txt    Python dependencies
- README.md           This file

How it fits together
--------------------
- Data is downloaded / prepared by scripts in data/ (see data/README.md).
- Training is performed by src/train.py using configs in configs/; checkpoints are written to experiments/<exp-name>/checkpoints.
- Evaluation uses src/evaluate.py and writes metrics to experiments/<exp-name>/metrics/*. Results are aggregated into results/ for plotting / comparison.
- Notebooks in notebooks/ read results/*.csv to produce the figures included in the brief results section.

Reproducing the results (short path)
-----------------------------------
1. Prepare the data
   - Edit data/download_config.yaml (or run scripts/download_data.sh) to point to dataset sources and output location.
   - Run:
     bash scripts/download_data.sh
     python scripts/preprocess_data.py --input data/raw --output data/processed

2. Training (example)
   - Pick a config in configs/, e.g. configs/exp_resnet50.yaml
   - Run:
     python src/train.py --config configs/exp_resnet50.yaml --output experiments/resnet50_run1

   - Typical flags:
     --config <path-to-config>
     --output <experiment-output-dir>
     --seed <seed>
     --device cuda:0

3. Evaluation
   - After training finishes and checkpoints are in experiments/resnet50_run1/checkpoints:
     python src/evaluate.py --config configs/exp_resnet50.yaml --checkpoint experiments/resnet50_run1/checkpoints/best.ckpt --out experiments/resnet50_run1/metrics

4. Aggregation & plots
   - Aggregate metric CSVs:
     python scripts/aggregate_results.py --input experiments/*/metrics --out results/summary.csv
   - Generate figures (notebook or script):
     jupyter nbconvert --execute notebooks/analysis.ipynb --to html --output results/analysis.html

Recreating exactly the reported numbers
--------------------------------------
To get the same numbers reported in the brief results breakdown below:
- Use the same code commit (record commit hash in experiment metadata).
- Use the same config file (recorded in experiments/<exp>/config.yaml).
- Use the same random seed (include --seed in training and evaluation).
- Use the same preprocessing (use data/manifest or data/README).
- Run for the same number of epochs / checkpoint policy.

Brief results breakdown
-----------------------
Below is an example layout for the brief results summary. Replace placeholders with the actual numbers from results/summary.csv.

Summary (example)
- Model: ResNet-50
  - Dataset: DatasetName
  - Metric(s): Top-1 Acc = 78.4%, Top-5 Acc = 94.1%
  - Checkpoint: experiments/resnet50_run1/checkpoints/best.ckpt
- Model: ModelB
  - Dataset: DatasetName
  - Metric(s): Top-1 Acc = 80.2%, Top-5 Acc = 95.0%
  - Checkpoint: experiments/modelb_run1/checkpoints/best.ckpt

Example table (replace with your numbers)
| Model      | Dataset     | Top-1 (%) | Top-5 (%) | Notes |
|------------|-------------|-----------|-----------|-------|
| ResNet-50  | DatasetName | 78.4      | 94.1      | baseline run |
| ModelB     | DatasetName | 80.2      | 95.0      | +data-aug |

Where to find the exact result files
- Aggregated CSV: results/summary.csv
- Per-experiment metrics: experiments/<exp-name>/metrics/*.csv
- Plots/figures: results/plots/*.png or results/analysis.html

Recommended minimal commands to reproduce from scratch
-----------------------------------------------------
# after cloning and installing deps
bash scripts/download_data.sh
python scripts/preprocess_data.py --input data/raw --output data/processed
python src/train.py --config configs/exp_resnet50.yaml --output experiments/resnet50_run1 --seed 42
python src/evaluate.py --config configs/exp_resnet50.yaml --checkpoint experiments/resnet50_run1/checkpoints/best.ckpt --out experiments/resnet50_run1/metrics
python scripts/aggregate_results.py --input experiments/*/metrics --out results/summary.csv

Notes about hardware and randomness
---------------------------------
- The reported run used (example) 1 x NVIDIA A100, CUDA 11.x, PyTorch 2.x.
- CPU: Intel Xeon or equivalent.
- Randomness: seeds set via --seed. Non-determinism can arise from cuDNN / FP16 / distributed settings — see configs/README for deterministic flags.

What to update in this README once you run experiments
-----------------------------------------------------
- Replace the placeholder numbers in "Brief results breakdown" with the actual entries from results/summary.csv.
- Ensure experiments/README (or each experiment folder) contains: git commit hash, full config used, command line, timestamp, hardware used, and seed.
- If any folder names differ from the layout above, update the "Repository layout" section to match.

Contact / citation
------------------
For questions, contact: PeterAdr14 (GitHub) or add preferred email/contact here.

License
-------
Add your LICENSE file or replace this section with the license in the repository.
