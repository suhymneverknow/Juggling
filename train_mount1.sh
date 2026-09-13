#!/usr/bin/env bash
#SBATCH --job-name=juggling-mount1-jax
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:NVIDIAL40:1
#SBATCH --time=2-00:00:00
#SBATCH --output=train/runs/slurm-%x-%j.out
#SBATCH --error=train/runs/slurm-%x-%j.err
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then PROJECT_DIR="${SLURM_SUBMIT_DIR:?}"; else PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; fi
cd "$PROJECT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PROJECT_DIR/train/.jax_cache}"
mkdir -p train/runs "$JAX_COMPILATION_CACHE_DIR"
ENV_NAME="${JUGGLING_CONDA_ENV:-loco_mujoco}"
if [[ -n "${JUGGLING_PYTHON:-}" ]]; then exec "$JUGGLING_PYTHON" train/train_mount1_hydra.py "$@"; fi
CONDA_EXE="${JUGGLING_CONDA_EXE:-}"
if [[ -z "$CONDA_EXE" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE="$(command -v conda)"
    elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
        CONDA_EXE="$HOME/miniforge3/bin/conda"
    elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        CONDA_EXE="$HOME/miniconda3/bin/conda"
    else
        echo "error: conda not found; set JUGGLING_CONDA_EXE or JUGGLING_PYTHON" >&2
        exit 1
    fi
fi
exec "$CONDA_EXE" run --no-capture-output -n "$ENV_NAME" python train/train_mount1_hydra.py "$@"
