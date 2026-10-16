#!/bin/bash
#SBATCH --job-name=ocsr_prefetch
#SBATCH --partition=courses         # research accounts: --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Downloads datasets, model checkpoints and tool code on a CPU node, so the GPU job does inference only
# (idle GPUs are monitored on the cluster). Needs HF_TOKEN (see README). Submit from the repository root.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
CONFIG="${CONFIG:-configs/default.yaml}"
# shellcheck disable=SC2086  # RUN_ARGS holds extra CLI flags and is split on purpose
"${OCSR_PY}" -m ocsr_bench prefetch --config "${CONFIG}" ${RUN_ARGS:-}
"${OCSR_PY}" -m ocsr_bench overview --config "${CONFIG}" ${RUN_ARGS:-}
