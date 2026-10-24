#!/bin/bash
# Objective 3 regime sweep (2026-10-23): where does the joint rank+precision choice beat rank-then-precision? N = 10,000.
set -u
PY=$HOME/miniconda3/envs/tlr/bin/python
for b in 250 500; do for eta in 0.05 0.2 0.5 1.0; do for r in 0.05 0.1 0.3; do
  out=../results/joint_sweep/b${b}_eta${eta}_r${r}
  [ -f $out/joint_rank_precision_n10000.json ] && continue
  echo "=== b=$b eta=$eta range=$r $(date +%H:%M)"
  $PY 18_joint_rank_precision_pilot.py --n 10000 --b $b --eta $eta --range $r --out $out 2>&1 | grep -E "joint /|bytes=" | cut -c1-110
done; done; done
echo "SWEEP DONE"
