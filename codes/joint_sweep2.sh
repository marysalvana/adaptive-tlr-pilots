#!/bin/bash
# Objective 3 regime sweep 2 (2026-10-23, Mary: "long range and smooth, strong dependence"): N = 10,000,
# Matern nu in 1.5/2.5, range in 0.3/1.0/3.0, nugget 0.05/0.2 (realistic), tile size 250/500.
set -u
PY=$HOME/miniconda3/envs/tlr/bin/python
for b in 250 500; do for nu in 1.5 2.5; do for eta in 0.05 0.2; do for r in 0.3 1.0 3.0; do
  out=../results/joint_sweep/nu${nu}_b${b}_eta${eta}_r${r}
  [ -f $out/joint_rank_precision_n10000.json ] && continue
  echo "=== nu=$nu b=$b eta=$eta range=$r $(date +%H:%M)"
  $PY 18_joint_rank_precision_pilot.py --n 10000 --nu $nu --b $b --eta $eta --range $r --out $out 2>&1 | grep -E "joint /|bytes=" | cut -c1-110
done; done; done; done
echo "SWEEP2 DONE"
