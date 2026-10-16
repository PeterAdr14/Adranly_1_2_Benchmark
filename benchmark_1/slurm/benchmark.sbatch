#!/bin/bash
#SBATCH --job-name=ocsr_bench
#SBATCH --partition=courses-gpu   # research accounts: --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G                  # courses-gpu RAM limit; research accounts can ask for more
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Full benchmark on one GPU. Submit from the repository root:  sbatch slurm/benchmark.sbatch
#
# Time limit: the run saves its progress after every batch, so hitting the partition's limit loses nothing.
# Slurm sends USR1 to this script 15 minutes before the limit; the handler stops the benchmark and submits this
# same script again (up to MAX_RESUBMITS times), which resumes where it left off. Check the real limit with
#   scontrol show partition gpu
# and lower --time here if it is shorter than 8 hours. On `courses-gpu` only ONE job may run at a time: the
# resubmitted job simply waits until this one has ended.
#
# Optional variables (pass with sbatch --export=ALL,NAME=value):  CONFIG, RUN_ARGS (extra CLI flags, e.g.
# "--set run.name=my_run"), MAX_RESUBMITS, RESUBMIT_COUNT (set automatically).
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh
CONFIG="${CONFIG:-configs/default.yaml}"
nvidia-smi || true

# Data and weights were fetched by the prefetch job; do not depend on internet access from the GPU node.
export OCSR_SKIP_REMOTE=1
# shellcheck disable=SC2086
"${OCSR_PY}" -m ocsr_bench preflight --config "${CONFIG}" ${RUN_ARGS:-}

# shellcheck disable=SC2086
"${OCSR_PY}" -m ocsr_bench run --config "${CONFIG}" ${RUN_ARGS:-} &
BENCH_PID=$!

requeue() {
    echo "[$(date '+%F %T')] Time limit approaching: stopping the benchmark (progress is saved) and resubmitting."
    kill -TERM "${BENCH_PID}" 2>/dev/null || true
    wait "${BENCH_PID}" 2>/dev/null || true
    local attempt="${RESUBMIT_COUNT:-0}"
    if [ "${attempt}" -lt "${MAX_RESUBMITS:-10}" ]; then
        # Resubmit with the SAME partition and memory as this job (they may have been overridden at submit time).
        sbatch --partition="${SLURM_JOB_PARTITION}" --mem="${SLURM_MEM_PER_NODE:-12288}M" \
               --export=ALL,RESUBMIT_COUNT=$((attempt + 1)) "${SLURM_SUBMIT_DIR:-$PWD}/slurm/benchmark.sbatch"
    else
        echo "Resubmit limit (${MAX_RESUBMITS:-10}) reached; run 'sbatch slurm/benchmark.sbatch' to continue."
    fi
    exit 0
}
trap requeue USR1

STATUS=0
wait "${BENCH_PID}" || STATUS=$?
echo "[$(date '+%F %T')] benchmark finished with status ${STATUS}"
exit "${STATUS}"
