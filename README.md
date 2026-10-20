# Adaptive tile low-rank (TLR) Gaussian process inference: pilot experiments

Small, reproducible experiments on choosing the rank of every covariance tile from a
condition-dependent error budget rather than a fixed tolerance, with a Vecchia baseline, a
multi-GPU tile Cholesky, standard errors, and a Metropolis-Hastings run with the adaptive-TLR
likelihood.  Every script fixes its random seed and writes a JSON file with its settings and
outputs into `results/`; the logs of the reported runs are in `results/logs/`.

## 1. Rank rules: fixed tolerance vs. the adaptive budget (`codes/11_tlr_rank_rules.py`)

Simulated Matern (nu = 3/2) field with nugget on N random sites in the unit square
(sigma2 = 1, range = 0.3, nugget eta = 0.05, strong dependence), Morton-ordered, tiles
of b = 250.  For every off-diagonal tile the rank is chosen by

* `fixed eps`: smallest rank with relative Frobenius error <= eps (ExaGeoStat/HiCMA rule);
* `adaptive tau`: smallest rank with absolute Frobenius error <= tau * b * m(theta) / N,
  m(theta) = sigma2 * eta, which guarantees ||Delta Sigma||_F <= tau * lambda_min(Sigma)
  and therefore positive definiteness of Sigma^TLR (Weyl's inequality).

Sigma^TLR is assembled densely and factorized exactly, so the only approximation is the
tile truncation.  Reported per rule: stored values relative to dense, positive
definiteness (Cholesky success), log-likelihood error at the true parameters,
maximum-likelihood estimates of (sigma2, range, eta) with ranks re-chosen at every
parameter value, and RMSE / CRPS / 90% coverage at 300 held-out sites.

    cd codes && python3 11_tlr_rank_rules.py --n 10000 --out ../results
    cd codes && python3 11_tlr_rank_rules.py --n 20000 --mle "dense,adaptive:0.9" --out ../results

Outputs: `tlr_rank_rules_n{N}.json`, `sim_n{N}_obs.csv`, `sim_n{N}_holdout.csv`, `results/logs/`.

## 2. Vecchia baseline (`codes/12_vecchia_baseline.R`, GpGp)

Fits the same simulated data by Vecchia's approximation with m = 30 nearest neighbors
(GpGp `matern15_isotropic`, max-min ordering) and predicts the same held-out sites.

    cd codes && Rscript 12_vecchia_baseline.R 10000 30 ../results

Output: `vecchia_n{N}_m{m}.json`.

## 3. Multi-GPU tile Cholesky (`codes/13_multigpu_tile_cholesky.py`)

Factorizes the same covariance, built tile by tile, with the block rows distributed
cyclically over the GPUs visible to PyTorch (dense FP64 tiles; the rank and precision
rules act on the tiles and do not change the communication pattern).  Reports wall
time, peak memory per GPU, and the log-determinant against a single-GPU run.

    python3 13_multigpu_tile_cholesky.py --n 4000 --b 500 --devices cpu,cpu --check_single   # logic test
    sbatch codes/storrs_multigpu.sbatch                                                    # UConn Storrs HPC

Output: `multigpu_tile_cholesky_n{N}_b{b}_{k}x{device}.json`.

## Environment

Local runs: macOS, Python 3 with NumPy 2.4 / SciPy 1.17 / PyTorch 2.10 (CPU), R 4.x with
GpGp and jsonlite.  Storrs HPC runs: see `codes/storrs_multigpu.sbatch` (partition
general-gpu, NVIDIA A30 24 GB or L40 46 GB GPUs depending on the node, PyTorch 2.5.1+cu121 in a conda environment under the
home directory, which compute nodes mount with execution permitted).  Two Slurm details the
script handles: it changes into `$SLURM_SUBMIT_DIR` because Slurm runs a copy of the batch
script from its spool directory, and it takes the GPU list from `torch.cuda.device_count()`
because `nvidia-smi -L` on a shared node lists GPUs that were not allocated to the job.  The
tile update solves `L_kk X = M^T` on contiguous operands; the transposed-view right-hand
form `solve_triangular(L_kk^T, M, left=False)` gave wrong factors on CUDA in PyTorch 2.5.

## Results obtained on 2026-10-20

* Rank rules, N = 10,000 (`tlr_rank_rules_n10000.json`, `results/logs/tlr_n10000.log`): at the true
  parameters, fixed eps = 1e-1, 1e-2, 1e-3 gave an indefinite Sigma^TLR (Cholesky failed);
  eps = 1e-4 was definite at 4.5% of dense storage.  The fixed-eps = 1e-2 MLE failed on all
  39 evaluations; the eps = 1e-3 MLE failed on 20 of 164 evaluations and returned
  (1.304, 0.101, 0.039) against the dense (1.394, 0.324, 0.036), with RMSE 0.219 against 0.206.
  Adaptive tau = 0.5, 0.9, 0.99 were definite at 5.5%, 5.3%, 5.2% of dense storage (mean rank
  7 to 8 of b = 250); the tau = 0.9 MLE completed all 80 evaluations and returned
  (1.396, 0.326, 0.036) with RMSE 0.2056, CRPS 0.1158, and 90% coverage 0.937, the dense values.
* Vecchia, N = 10,000, m = 30 (`vecchia_n10000_m30.json`): (1.398, 0.324, 0.036), RMSE 0.2055,
  CRPS 0.1160, coverage 0.937, fit in 3 s; the reference point for likelihood-changing methods such as Vecchia's approximation.
* Multi-GPU tile Cholesky, UConn Storrs (`multigpu_tile_cholesky_n*_b1000_{2,3}xcuda.json`,
  `results/logs/tile-chol-*.out`): on two NVIDIA A30 GPUs (24 GB, node gpu48) and three NVIDIA L40
  GPUs (46 GB, node gpu50) at N = 20,000, 40,000, 60,000 the log-determinant equals the
  single-GPU value to all printed digits.  At N = 60,000 the factor took 14.7 GB on one GPU,
  7.7 to 8.0 GB per GPU on the two A30s, and 5.7 to 6.0 GB per GPU on the three L40s, in
  9.4 s and 40.7 s against 11.5 s and 67.9 s on one GPU of the same node (block rows are
  staged through host memory, see Environment; the pilot measures memory per GPU, not speed).
* N = 20,000 rank-rule run (`results/logs/tlr_n20000.log`): in progress at the time of writing.
