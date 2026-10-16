#!/bin/bash
# Shared settings for every job script. Use it with:  source slurm/env.sh
# Override anything by exporting it before you submit (e.g. export OCSR_WORK_DIR=/scratch/$USER/other).

# ---- where things live ---------------------------------------------------------------------------------
# OCSR_WORK_DIR (datasets, caches, cloned code, Python environments) is chosen in slurm/settings.sh:
# $OCSR_COURSE_DIR/ocsr_bench if you set it, else /scratch/$USER/ocsr_bench (temporary: purged monthly).
# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
# Results are small. They go in $HOME (75 GB limit) so a scratch purge cannot delete them; copy them to your
# group's /projects folder for long-term storage. Conda environments belong in /projects if you have one:
# export OCSR_ENV_MAIN=/projects/<group>/ocsr/envs/ocsr-main (and the same for OCSR_ENV_MOLNEXTR).
: "${OCSR_RESULTS_DIR:=${HOME}/ocsr_results}"
: "${OCSR_ENV_MAIN:=${OCSR_WORK_DIR}/envs/ocsr-main}"            # conda prefix, Python 3.10 (MolGlyph + framework)
: "${OCSR_ENV_MOLNEXTR:=${OCSR_WORK_DIR}/envs/ocsr-molnextr}"    # conda prefix, Python 3.8 (MolNexTR)
: "${OCSR_HF_TOKEN_FILE:=${HOME}/.hf_token}"                      # one line: your Hugging Face READ token (chmod 600)

# ---- modules (names/versions from the Explorer docs; check with `module avail`) ----------------------------
: "${OCSR_MODULES:=explorer anaconda3/2024.06 cuda/12.1.1}"

# ---- PyTorch wheels installed by setup_envs.sbatch -----------------------------------------------------------
# The reported results come from torch 2.6.0 + CUDA 12.4 wheels (cu124): they ran on a T4 node and an H200 node.
# A V100 node seen earlier had driver 545 (CUDA 12.3), where the CUDA 11.8 build is the conservative choice:
#   export OCSR_TORCH_INDEX_MAIN=https://download.pytorch.org/whl/cu118     (not tested)
: "${OCSR_TORCH_MAIN:=torch==2.6.0 torchvision==0.21.0}"                   # the versions BioMiner pins
: "${OCSR_TORCH_INDEX_MAIN:=https://download.pytorch.org/whl/cu124}"
: "${OCSR_TORCH_MOLNEXTR:=torch==2.4.1}"                           # newest torch that supports Python 3.8
: "${OCSR_TORCH_INDEX_MOLNEXTR:=https://download.pytorch.org/whl/cu121}"

export OCSR_WORK_DIR OCSR_RESULTS_DIR OCSR_ENV_MAIN OCSR_ENV_MOLNEXTR
export OCSR_MOLNEXTR_PYTHON="${OCSR_ENV_MOLNEXTR}/bin/python"      # read by configs/default.yaml

# Batch shells do not always define `module`; load its initialisation if it is missing.
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        # shellcheck disable=SC1090
        [ -r "${init}" ] && . "${init}" && break
    done
fi
if type module >/dev/null 2>&1; then
    module purge
    # shellcheck disable=SC2086  # word splitting is intended: one module per word
    module load ${OCSR_MODULES}
else
    echo "[env] WARNING: the 'module' command is unavailable; continuing without loading ${OCSR_MODULES}"
fi

# Keep every cache off $HOME (fixed, small quota).
export HF_HOME="${OCSR_WORK_DIR}/cache/huggingface"
export XDG_CACHE_HOME="${OCSR_WORK_DIR}/cache/xdg"
export PIP_CACHE_DIR="${OCSR_WORK_DIR}/cache/pip"
export UV_CACHE_DIR="${OCSR_WORK_DIR}/cache/uv"
export TORCH_HOME="${OCSR_WORK_DIR}/cache/torch"
export CONDA_PKGS_DIRS="${OCSR_WORK_DIR}/cache/conda-pkgs"
export PYSTOW_HOME="${OCSR_WORK_DIR}/cache/pystow"   # MolNexTR's code uses pystow, which would otherwise write to ~/.data
mkdir -p "${OCSR_WORK_DIR}/cache" "${OCSR_WORK_DIR}/envs" "${OCSR_WORK_DIR}/data" "${OCSR_RESULTS_DIR}"

# The token is read from a file outside the repository; it is never written to any output.
if [ -z "${HF_TOKEN:-}" ] && [ -r "${OCSR_HF_TOKEN_FILE}" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${OCSR_HF_TOKEN_FILE}")"
    export HF_TOKEN
fi

OCSR_PY="${OCSR_ENV_MAIN}/bin/python"   # interpreter of the main environment (used instead of `activate`)
export OCSR_PY
echo "[env] work=${OCSR_WORK_DIR} results=${OCSR_RESULTS_DIR} token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo MISSING)"
