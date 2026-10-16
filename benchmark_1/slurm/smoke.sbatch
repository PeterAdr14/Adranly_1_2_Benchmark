#!/bin/bash
#SBATCH --job-name=ocsr_smoke
#SBATCH --partition=courses-gpu   # research accounts: --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G                  # courses-gpu RAM limit; research accounts can ask for more
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Small GPU run (20 images per dataset) to confirm that everything works and to read the real throughput.
# Look at `throughput_img_per_s` in the output: full-run time per tool is roughly images / throughput.
# Submit from the repository root:  sbatch slurm/smoke.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
CONFIG="${CONFIG:-configs/default.yaml}"
nvidia-smi || true
# Data and weights were fetched by the prefetch job; do not depend on internet access from the GPU node.
export OCSR_SKIP_REMOTE=1
# shellcheck disable=SC2086
"${OCSR_PY}" -m ocsr_bench preflight --config "${CONFIG}" ${RUN_ARGS:-}
"${OCSR_PY}" -m ocsr_bench run --config "${CONFIG}" --set run.name=smoke_test --set run.max_samples=20 ${RUN_ARGS:-}
