#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --qos=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=../results/logs/%x-%j.out
# One-GPU pilot runs (2026-10-21); usage: sbatch --job-name=NAME storrs_gpu1.sbatch "<python args>"
# The scripts checkpoint and resume, so a run that hits the time limit can simply be resubmitted.
set -euo pipefail
PYTHON=${PYTHON:-$HOME/miniconda3/envs/tlr/bin/python}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) start=$(date)"
eval $PYTHON "$@"
echo "end=$(date)"
