#!/bin/bash
# Submit ONE job with the partition, memory and time limit from slurm/settings.sh (your OCSR_* variables).
# A bare `sbatch slurm/<job>.sbatch` would use the partition written inside the script (the course default), which
# an account without course access cannot use. This wrapper passes your settings explicitly.
#   bash slurm/submit.sh benchmark                          # full benchmark
#   bash slurm/submit.sh benchmark --exclude=d4054          # extra sbatch options go after the job name
#   bash slurm/submit.sh {setup_envs|prefetch|smoke|benchmark}
set -eo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. slurm/settings.sh
mkdir -p logs

name="${1:?usage: bash slurm/submit.sh {setup_envs|prefetch|smoke|benchmark} [sbatch options]}"
shift
case "${name}" in
    setup_envs|prefetch) options=(--partition="${OCSR_CPU_PARTITION}" --mem="${OCSR_CPU_MEM}") ;;
    smoke)               options=(--partition="${OCSR_GPU_PARTITION}" --mem="${OCSR_GPU_MEM}") ;;
    benchmark)           options=(--partition="${OCSR_GPU_PARTITION}" --mem="${OCSR_GPU_MEM}" --time="${OCSR_GPU_TIME}") ;;
    *) echo "Unknown job: ${name} (choose setup_envs, prefetch, smoke or benchmark)"; exit 2 ;;
esac
echo "submitting ${name} on partition ${options[0]#--partition=}"
sbatch "${options[@]}" "$@" "slurm/${name}.sbatch"
