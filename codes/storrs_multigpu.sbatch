#!/bin/bash
#SBATCH --job-name=tile-chol
#SBATCH --partition=general-gpu
#SBATCH --qos=general-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=tile-chol-%j.out
# Multi-GPU tile Cholesky pilot runs (2026-10-20).
# Assumes PYTHON points at a Python with a CUDA build of PyTorch (see the README).
set -euo pipefail
PYTHON=${PYTHON:-$HOME/miniconda3/envs/tlr/bin/python}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"   # Slurm runs a copy of this script from its spool directory
nvidia-smi --query-gpu=name,memory.total --format=csv
DEV=$($PYTHON -c "import torch; print(','.join(f'cuda:{i}' for i in range(torch.cuda.device_count())))")   # only the GPUs Slurm allocated
for N in 20000 40000 60000; do
  $PYTHON 13_multigpu_tile_cholesky.py --n $N --b 1000 --devices $DEV --check_single --out ../results
done
