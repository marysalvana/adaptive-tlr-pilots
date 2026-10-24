#!/bin/bash
# Round-10 smoke test (2026-10-23): runs every new pilot script end to end at small sizes into an empty folder and
# prints the key numbers. Takes a few minutes on a laptop. The numbers are for the small sizes here, not the committed
# ones; the committed runs are reproduced with the sbatch lines in codes/ENVIRONMENT_round10.txt and README sections 12-14.
#   usage: cd codes && ./run_round10_smoke.sh <empty output dir>
set -euo pipefail
OUT=${1:?output dir}; mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PY=${PYTHON:-python3}
echo "== 36: error = first-order term + certified remainder (N = 2000)"
$PY 36_likelihood_error_terms.py --n 2000 --rules adaptive:0.9,adaptive:0.5,fixed:1e-4 --out "$OUT/bound_terms" | grep -E "^(adaptive|fixed)" | cut -c1-200
echo "== 37: oscillation of the TLR likelihood error over exact-posterior draws (N = 1000, 5 draws, chain of the N = 5000 pilot)"
$PY 37_posterior_oscillation.py --n 1000 --draws 5 --out "$OUT/posterior_oscillation" | grep -E "^(adaptive|fixed)" | cut -c1-220
echo "== 38: Metropolis-Hastings driven by the TLR Cholesky, self-tuned proposal (N = 1000, 120 iterations, 2 chains per rule)"
for rule in dense adaptive:0.9; do tag=${rule//:/_}; for k in 1 2; do
  $PY 38_tlr_cholesky_mcmc.py --n 1000 --rule $rule --iters 120 --burn 60 --chain_seed 10$k --start_mult $([ $k = 1 ] && echo 1.3,0.8,1.5 || echo 0.6,1.4,0.5) --out "$OUT/mcmc_tlrchol/$tag/c$k" | grep "^done" | cut -c1-160
done; done
$PY 38_tlr_cholesky_mcmc.py --n 1000 --burn 60 --pool --out "$OUT/mcmc_tlrchol" | grep -E "^(dense|adaptive)" | cut -c1-200
echo "== 18 with --nu 2.5 and 39 on the wind-gust covariance (N = 2000)"
$PY 18_joint_rank_precision_pilot.py --n 2000 --nu 2.5 --range 1.0 --out "$OUT/joint_nu25" | grep "joint /"
$PY 39_joint_windgust.py --n 2000 --b 250 --out "$OUT/joint_windgust" | grep -E "joint /|^rank_only|^fixed" | cut -c1-120
echo "SMOKE DONE: $OUT"
