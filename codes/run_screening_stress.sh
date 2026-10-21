#!/usr/bin/env bash
# Reproduce the Vecchia head-to-head behind panel (c) of the results figure and the README (2026-10-21).
#
# Runs, for each of the nine fields at N = 10,000 (Matern nu, range, relative nugget):
#   24 sim     simulate the field (fixed seed 20261020), exact and adaptive-TLR log-likelihood at the truth, exact fit
#   25         Vecchia (GpGp) log-likelihood at the truth for m = 5..120, predictions, Vecchia fits m = 30, 60
#   24 eval    Vecchia fits scored on the exact likelihood
#   26         Vecchia at the storage of the adaptive TLR factor (and m = 250)
#   27 fit     verified exact MLE, adaptive-TLR MLE, parameter grid (Sobol seed 20261021)
#   28         Vecchia (grouped and ungrouped, m = 30, 60) on the grid
#   27 score   fit gaps below the exact MLE and shape errors
#   30         Vecchia fits under 8 seeded orderings (GpGp's max-min ordering is randomized)
#   27 fitorders  each of those fits scored on the exact likelihood
#   31         KL divergence of the adaptive TLR approximation from the exact model at the true parameters
#   32         KL divergence of Vecchia (m = 10, 30, 60, 120, 8 orderings; then m = 250 and the TLR-matched m, 2 orderings)
# Panel (c) of the results figure plots the KL divergences (31, 32).
# then prints both report tables and redraws the results figure.  About 45 min per field on one 8-core laptop (3 fields at a time).
#
# Usage (from codes/):
#   ./run_screening_stress.sh [OUT_DIR] [PARALLEL]
#     OUT_DIR   default ../results/screening_stress_rerun (the committed run is ../results/screening_stress)
#     PARALLEL  fields run at once, default 3 (each needs about 4 GB for the dense N = 10,000 reference)
#   python3 29_check_reproduction.py OUT_DIR      compare the rerun with the committed numbers
# Environment of the committed run: see ENVIRONMENT_screening_stress.txt.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-../results/screening_stress_rerun}
PAR=${2:-3}
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-3} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-3}

FIELDS="nu15_r03_e005 1.5 0.3 0.05
nu15_r1_e005 1.5 1.0 0.05
nu15_r3_e005 1.5 3.0 0.05
nu25_r03_e005 2.5 0.3 0.05
nu25_r1_e005 2.5 1.0 0.05
nu25_r3_e005 2.5 3.0 0.05
nu15_r03_e05 1.5 0.3 0.5
nu25_r1_e05 2.5 1.0 0.5
nu25_r1_e1 2.5 1.0 1.0"

one() {
  local tag=$1 nu=$2 r=$3 e=$4 d="$OUT/$1" L="$OUT/logs/$1.log"
  {
    python3 24_screening_stress.py sim --n 10000 --nu "$nu" --range "$r" --eta "$e" --tag "$tag" --out "$OUT"
    Rscript 25_vecchia_stress.R "$d"
    python3 24_screening_stress.py eval --tag "$tag" --out "$OUT"
    Rscript 26_vecchia_matched_storage.R "$d"
    python3 27_inference_headtohead.py fit --tag "$tag" --out "$OUT"
    Rscript 28_vecchia_grid.R "$d"
    python3 27_inference_headtohead.py score --tag "$tag" --out "$OUT"
    Rscript 30_vecchia_fit_orderings.R "$d" 8
    python3 27_inference_headtohead.py fitorders --tag "$tag" --out "$OUT"
    python3 31_kl_tlr.py --tag "$tag" --out "$OUT"
    Rscript 32_kl_vecchia.R "$d" 8
    Rscript 32_kl_vecchia.R "$d" 2 matched
    echo "DONE $tag"
  } > "$L" 2>&1 || { echo "FAILED $tag (see $L)"; return 1; }
  echo "DONE $tag"
}
export -f one; export OUT
echo "$FIELDS" | xargs -P "$PAR" -L 1 bash -c 'one "$@"' _
python3 24_screening_stress.py report --out "$OUT" | tee "$OUT/report_likelihood_at_truth.md"
python3 27_inference_headtohead.py report --out "$OUT" | tee "$OUT/report_inference.md"
FIG_OUT="$OUT" SS_DIR="$OUT" python3 20_pilot_results_figure.py
