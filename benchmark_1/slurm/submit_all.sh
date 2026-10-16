#!/bin/bash
# Submits the whole workflow as a dependency chain: setup -> prefetch -> smoke test -> full benchmark.
# Each step starts only if the previous one succeeded (--dependency=afterok). Partitions, memory and the time limit
# come from slurm/settings.sh (course-account defaults; export OCSR_* variables to change them).
#   bash slurm/submit_all.sh                # everything
#   bash slurm/submit_all.sh --skip-setup   # environments already built
#   bash slurm/submit_all.sh --skip-smoke   # go straight to the full benchmark
set -eo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. slurm/settings.sh
mkdir -p logs   # Slurm does not create the log directory, and a missing one makes the job fail silently

SKIP_SETUP=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "${arg}" in
        --skip-setup) SKIP_SETUP=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        *) echo "Unknown option: ${arg}"; exit 2 ;;
    esac
done

previous=""
submit() {  # submit <script> [sbatch options...]: chains onto the previous job and records its id
    local script="$1"; shift
    local dependency=()
    [ -n "${previous}" ] && dependency=(--dependency="afterok:${previous}")
    previous="$(sbatch --parsable "${dependency[@]}" "$@" "${script}")"
    echo "submitted ${script} as job ${previous}"
}

echo "CPU partition: ${OCSR_CPU_PARTITION} | GPU partition: ${OCSR_GPU_PARTITION} (${OCSR_GPU_MEM}, ${OCSR_GPU_TIME}) | work dir: ${OCSR_WORK_DIR}"
[ "${SKIP_SETUP}" = 1 ] || submit slurm/setup_envs.sbatch --partition="${OCSR_CPU_PARTITION}" --mem="${OCSR_CPU_MEM}"
submit slurm/prefetch.sbatch --partition="${OCSR_CPU_PARTITION}" --mem="${OCSR_CPU_MEM}"
[ "${SKIP_SMOKE}" = 1 ] || submit slurm/smoke.sbatch --partition="${OCSR_GPU_PARTITION}" --mem="${OCSR_GPU_MEM}"
submit slurm/benchmark.sbatch --partition="${OCSR_GPU_PARTITION}" --mem="${OCSR_GPU_MEM}" --time="${OCSR_GPU_TIME}"
echo "Follow progress with: squeue -u \$USER   and   tail -f logs/*.out"
