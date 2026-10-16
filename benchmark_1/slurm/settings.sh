#!/bin/bash
# Resource and location defaults. Sourced by slurm/submit_all.sh (at submit time, so it can pass partitions to sbatch)
# and by slurm/env.sh (inside every job). Override any of them by exporting it first.
#
# Defaults are for COURSE accounts (access granted through the class): partitions `courses` (CPU, 24 h max) and
# `courses-gpu` (8 h max, ONE running job, 12 GB RAM). For a research account use the regular partitions instead:
#   export OCSR_CPU_PARTITION=short OCSR_GPU_PARTITION=gpu OCSR_GPU_MEM=32G

: "${OCSR_CPU_PARTITION:=courses}"
: "${OCSR_GPU_PARTITION:=courses-gpu}"
: "${OCSR_CPU_MEM:=16G}"
: "${OCSR_GPU_MEM:=12G}"            # courses-gpu allows at most 12 GB of RAM per job
: "${OCSR_GPU_TIME:=08:00:00}"      # courses-gpu maximum; the benchmark job resubmits itself before the limit

# Where the datasets, caches, tool code and Python environments go (all re-creatable).
#  * Course account: set OCSR_COURSE_DIR to your folder, e.g. /courses/AAI6630.<term>/students/$USER
#    (course folders are deleted after the course ends: copy your results out before then).
#  * Otherwise: /scratch/$USER (large, but purged monthly).
if [ -z "${OCSR_WORK_DIR:-}" ]; then
    if [ -n "${OCSR_COURSE_DIR:-}" ]; then
        OCSR_WORK_DIR="${OCSR_COURSE_DIR}/ocsr_bench"
    else
        OCSR_WORK_DIR="/scratch/${USER}/ocsr_bench"
    fi
fi
export OCSR_CPU_PARTITION OCSR_GPU_PARTITION OCSR_CPU_MEM OCSR_GPU_MEM OCSR_GPU_TIME OCSR_WORK_DIR
