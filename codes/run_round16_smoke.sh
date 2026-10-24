#!/bin/bash
# Objective 3 pilot round smoke test (2026-10-24): every new script runs end to end at small sizes into an empty folder.
#   usage: cd codes && ./run_round16_smoke.sh <empty output dir>     (needs data/windgust_train_data.csv for 39 and 41)
set -euo pipefail
OUT=${1:?output dir}; mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=${PYTHON:-python3}
echo "== 18 with tolerance, uniform and block rows (N = 2000)"
$PY 18_joint_rank_precision_pilot.py --n 2000 --mp_eps 1e-2,1e-4 --blr_eps 1e-3 --extra_rows --out "$OUT/joint_tol" | grep -E "^(rank|joint|mp_tol|blr|uniform|block)" | cut -c1-140
echo "== 39 wind-gust covariance with the same rows (N = 2000)"
$PY 39_joint_windgust.py --n 2000 --b 250 --mp_eps 1e-2 --blr_eps 1e-3 --extra_rows --out "$OUT/gust_tol" | grep -E "^(rank|joint|uniform|block|fixed)" | cut -c1-120
echo "== 40 maximum likelihood with rounded factors (N = 1000, 30 iterations)"
$PY 40_joint_fit_pilot.py --n 1000 --maxiter 30 --out "$OUT/joint_fit" | grep -E "^FIT|loss" | cut -c1-160
echo "== 41 wind-gust fit with rounded factors (N = 1500, 25 iterations, dense and joint)"
$PY 41_windgust_fit.py --n 1500 --maxiter 25 --rules dense,joint --out "$OUT/gust_fit" | grep -E "^FIT|loss" | cut -c1-160
echo "== 38 chains driven by the rounded factors (N = 1000, 30 iterations, joint and rtp)"
for rule in joint rtp; do $PY 38_tlr_cholesky_mcmc.py --n 1000 --rule $rule:0.9 --iters 30 --burn 10 --chain_seed 101 --start_mult 1.3,0.8,1.5 --out "$OUT/mcmc/${rule}_0.9/c1" | grep "^done" | cut -c1-120; done
echo "== 17 rounded TLR Cholesky with recompression (N = 2000)"
$PY 17_tlr_cholesky_pilot.py --n 2000 --rules adaptive:0.9,mixed:0.9 --mle "" --out "$OUT/tlrchol_mixed" | grep -E "^(adaptive|mixed)" | cut -c1-200
echo "SMOKE DONE: $OUT"
