#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../results/logs/%x-%j.out
# CPU pilot runs moved from the laptop to Storrs (2026-10-20); usage: sbatch --job-name=NAME storrs_cpu.sbatch "<python args>"
set -euo pipefail
PYTHON=${PYTHON:-$HOME/miniconda3/envs/tlr/bin/python}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
eval $PYTHON "$@"
echo "end=$(date)"
