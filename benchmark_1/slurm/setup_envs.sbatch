#!/bin/bash
#SBATCH --job-name=ocsr_setup
#SBATCH --partition=courses         # research accounts: --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Builds the two Python environments (CPU job: installing packages is CPU-heavy, so it does not belong on a
# login node, and it needs no GPU). Submit from the repository root:  sbatch slurm/setup_envs.sbatch
# Re-running skips environments that exist; FORCE=1 rebuilds them:  sbatch --export=ALL,FORCE=1 slurm/setup_envs.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

REPO_DIR="${OCSR_WORK_DIR}/data/_third_party/MolNexTR"      # the location the MolNexTR adapter reuses
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --depth 1 https://github.com/CYF2000127/MolNexTR.git "${REPO_DIR}"
fi

# ---- main environment: Python 3.10, MolGlyph, this framework -------------------------------------------
if [ ! -x "${OCSR_ENV_MAIN}/bin/python" ] || [ "${FORCE:-0}" = "1" ]; then
    rm -rf "${OCSR_ENV_MAIN}"
    conda create -y --prefix "${OCSR_ENV_MAIN}" python=3.10
fi
# shellcheck disable=SC2086
"${OCSR_ENV_MAIN}/bin/python" -m pip install ${OCSR_TORCH_MAIN} --index-url "${OCSR_TORCH_INDEX_MAIN}"
"${OCSR_ENV_MAIN}/bin/python" -m pip install -r requirements.txt
"${OCSR_ENV_MAIN}/bin/python" -m pip install -e .
"${OCSR_ENV_MAIN}/bin/python" -m ocsr_bench selfcheck

# ---- MolNexTR environment: Python 3.8 (its README) + the pins in its requirements.txt ------------------------
if [ ! -x "${OCSR_ENV_MOLNEXTR}/bin/python" ] || [ "${FORCE:-0}" = "1" ]; then
    rm -rf "${OCSR_ENV_MOLNEXTR}"
    conda create -y --prefix "${OCSR_ENV_MOLNEXTR}" python=3.8
fi
# shellcheck disable=SC2086
"${OCSR_ENV_MOLNEXTR}/bin/python" -m pip install ${OCSR_TORCH_MOLNEXTR} --index-url "${OCSR_TORCH_INDEX_MOLNEXTR}"
"${OCSR_ENV_MOLNEXTR}/bin/python" -m pip install -r "${REPO_DIR}/requirements.txt"
# MolNexTR's code imports packages that its requirements.txt does not list (pystow was the first one found).
"${OCSR_ENV_MOLNEXTR}/bin/python" -m pip install pystow
# Hard check with automatic repair: the package must import (a failure here is much cheaper to see now than in
# the GPU job). If the import fails because ONE module is missing, install it and try again.
PY38="${OCSR_ENV_MOLNEXTR}/bin/python"
for attempt in 1 2 3 4 5 6 7 8; do
    if output="$(PYTHONPATH="${REPO_DIR}" "${PY38}" -c "import MolNexTR; print('MolNexTR import OK')" 2>&1)"; then
        echo "${output}"
        break
    fi
    missing="$(echo "${output}" | sed -n "s/.*No module named '\([A-Za-z0-9_]*\).*/\1/p" | tail -n 1)"
    case "${missing}" in   # import names that differ from the pip package names
        cv2) package=opencv-python-headless ;;
        PIL) package=pillow ;;
        sklearn) package=scikit-learn ;;
        yaml) package=pyyaml ;;
        *) package="${missing}" ;;
    esac
    if [ -z "${missing}" ] || [ "${attempt}" = 8 ]; then
        echo "MolNexTR import failed:"
        echo "${output}"
        exit 1
    fi
    echo "[setup] MolNexTR needs the module '${missing}': installing ${package}"
    "${PY38}" -m pip install "${package}"
done

echo "Environments ready:"
echo "  main     : ${OCSR_ENV_MAIN}"
echo "  molnextr : ${OCSR_ENV_MOLNEXTR}"
