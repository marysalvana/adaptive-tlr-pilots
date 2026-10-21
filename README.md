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
* Standard errors, N = 10,000 (`pilot_standard_errors_n10000.json`, `results/logs/pilot_se_n10000.log`):
  observed-information standard errors of the dense fit are (0.61, 0.051, 0.016) for
  (sigma2, range, eta); the adaptive tau = 0.9 estimates differ from the dense ones by at most
  0.05 standard errors, the Vecchia estimates by at most 0.01, and the fixed eps = 1e-3 range
  by 4.4 standard errors.
* Timing caveat for experiment 1: Sigma^TLR is assembled densely and factorized exactly, so
  the wall times of the rank rules (ranks re-chosen at every evaluation, SVD of every tile)
  are not comparable with a TLR Cholesky and should not be read as the method's cost.
* Metropolis-Hastings pilot: see Section 7 (GPU chains with effective sample sizes); the CPU chain of
  `codes/16_tlr_mcmc_pilot.py` (`results/logs/tlr_mcmc_n5000.log`) mixes too slowly to be compared.
* N = 20,000 rank-rule run (`tlr_rank_rules_n20000.json`, `results/logs/tlr-n20000-31949243.out`, Sigma^TLR assembled densely
  as in Section 1): dense MLE (1.3603, 0.3364, 0.0361) in 89 evaluations; adaptive tau = 0.9 MLE (1.3747, 0.3365, 0.0357)
  in 92 evaluations, no failures, within 0.02 dense standard errors, same RMSE/CRPS/coverage.

## Results added on 2026-10-21

### 4. TLR Cholesky with recompression (`codes/17_tlr_cholesky_pilot.py`)

The rank rules inside an actual tile low-rank factorization: off-diagonal tiles are stored as low-rank
pairs, the factorization proceeds tile by tile (dense POTRF on diagonal tiles, triangular solves on the
low-rank tiles, Schur-complement updates recompressed under the same rule), the log-likelihood comes from
the factor, and predictions use TLR triangular solves. Nothing is assembled densely. A fixed tolerance can
now fail in the way the software fails: an indefinite Schur complement stops the factorization.

    sbatch --job-name=tlrchol-n10000 storrs_cpu.sbatch "17_tlr_cholesky_pilot.py --n 10000 --out ../results/tlr_cholesky"

Outputs: `tlr_cholesky/tlr_cholesky_n{N}.json`, `results/logs/tlrchol-n{N}-*.out`. Memory is reported as stored
values of the factor relative to a dense factor (N(N+1)/2 values).

* N = 10,000, b = 250, true parameters: eps = 1e-2 and 1e-3 fail to factorize; eps = 1e-4 completes at
  11.3% of the dense factor, 2.6 log-likelihood units off; eps = 1e-5 at 13.2%, 0.42 off; adaptive
  tau = 0.9 at 9.3% (mean rank 5.6), 0.26 off. Maximum likelihood with the TLR likelihood (ranks
  re-chosen at every evaluation): dense (1.394, 0.3239, 0.036); eps = 1e-4 (1.3878, 0.3326, 0.0362) with one
  failed factorization during the search and the factor at 11.4%; adaptive (1.3932, 0.3223, 0.0359), no
  failures, factor at 9.5%, within 0.03 dense standard errors of the dense estimates (dense SEs 0.61, 0.051,
  0.016 from `pilot_standard_errors_n10000.json`); identical RMSE/CRPS/coverage (0.2055 / 0.1158 / 0.937).
* N = 20,000, true parameters: eps = 1e-4 completes at 8.0%, 10.9 units off; eps = 1e-5 at 9.6%, 0.33 off;
  adaptive at 6.0% (mean rank 4.4), 0.11 off. Maximum likelihood: dense (1.3603, 0.3364, 0.0361); eps = 1e-4
  (0.5884, 0.2479, 0.0836) with 4 failed factorizations in 210 evaluations and the factor at 7.9%; adaptive
  (1.3583, 0.3367, 0.0361), no failures, factor at 6.1%, within 0.01 dense standard errors (SEs 0.87, 0.076,
  0.023 from `pilot_standard_errors_n20000.json`); the same RMSE/CRPS/coverage for all three (0.229 / 0.130 / 0.89).

* Past the dense limit (dense references computed on 514 GB nodes; `tlrchol-n40000-*`, `tlrchol-n60000-*`):
  N = 40,000: eps = 1e-4 fails to factorize; adaptive completes at 4.1% of the dense factor (mean rank 3.6), 0.66 units
  off, predictions equal to dense. N = 60,000: eps = 1e-4 fails; adaptive completes at 3.5% (mean rank 3.3), 0.56 off,
  predictions equal to dense. N = 100,000: a dense factor is not possible with NumPy's LAPACK (matrix beyond 2^31
  entries), so the field is simulated from a tau = 0.02 TLR factor (`--sim tlr --dense 0`, 4.3% of a dense factor) and
  only completion and storage are reported: eps = 1e-4 fails to factorize; adaptive completes at 2.9% of a dense factor
  (mean rank 2.9), predictions RMSE 0.2204 (`tlrchol-n100000-31998652.out`).
* Figure: `codes/20_pilot_results_figure.py` draws factor storage, log-likelihood error, and the fitted nugget of the
  dense, adaptive-TLR, and Vecchia (30 and 60 neighbors) fits against N, with the failed tolerances marked
  (`results/pilot_results.pdf`).

### 5. Small nugget and the Vecchia baseline priced (`codes/11_tlr_rank_rules.py`, `codes/12_vecchia_baseline.R`)

Same experiment as Section 1 with eta = 0.005 (`small_nugget/`, N = 20,000, true (1, 0.3, 0.005)): fixed
eps = 1e-3 and 1e-4 give an indefinite Sigma^TLR; eps = 1e-5 is definite at 3.5% but 3.9 units off;
adaptive tau = 0.9 is definite at 4.4% (mean rank 8.0), 0.38 off. Maximum likelihood: dense (1.0366, 0.3058, 0.0047) in
164 evaluations, RMSE 0.0746; eps = 1e-4 (1.5033, 0.1694, 0.0123) with 27 failed factorizations, RMSE 0.1395
(wrong estimates and double the prediction error); eps = 1e-5 (1.1501, 0.3224, 0.0043), one failure, RMSE 0.0747;
adaptive tau = 0.9 (1.3614, 0.3352, 0.0036), no failures, RMSE 0.0746. The dense and adaptive estimates lie on the
same ridge of the likelihood: the identifiable ratio sigma2 / range^3 (Zhang 2004) is 36.2 for dense, 36.1 for
adaptive, 34.3 for eps = 1e-5, and 309 for eps = 1e-4.

Vecchia (GpGp, `12_vecchia_baseline.R`, data regenerated from the seed), stored values N(m+1):

| setting | m | theta_hat | fit time | RMSE | CRPS | cover90 | stored values |
|---|---|---|---|---|---|---|---|
| standard, N = 10,000 (dense MLE 1.394, 0.324, 0.036) | 30 | 1.398, 0.324, 0.036 | 3 s | 0.2055 | 0.1160 | 0.937 | 310,000 |
| standard, N = 10,000 | 60 | 1.540, 0.335, 0.033 | 6 s | 0.2064 | 0.1164 | 0.943 | 610,000 |
| standard, N = 20,000 (dense MLE 1.360, 0.336, 0.036) | 30 | 1.540, 0.364, 0.032 | 4 s | 0.2319 | 0.1311 | 0.907 | 620,000 |
| standard, N = 20,000 | 60 | 0.862, 0.288, 0.057 | 17 s | 0.2293 | 0.1298 | 0.897 | 1,220,000 |
| small nugget, N = 20,000 (dense MLE 1.037, 0.306, 0.0047) | 30 | 1.069, 0.307, 0.0046 | 5 s | 0.0750 | 0.0425 | 0.893 | 620,000 |
| small nugget, N = 20,000 | 60 | 1.066, 0.309, 0.0046 | 11 s | 0.0750 | 0.0424 | 0.897 | 1,220,000 |
| standard, N = 40,000 (truth 1, 0.3, 0.05; dense MLE pending) | 30 | 0.515, 0.239, 0.097 | 9 s | 0.2250 | 0.1267 | 0.913 | 1,240,000 |
| standard, N = 40,000 | 60 | 0.487, 0.231, 0.102 | 25 s | 0.2261 | 0.1272 | 0.907 | 2,440,000 |
| standard, N = 60,000 (dense MLE pending) | 30 | 0.567, 0.231, 0.089 | 14 s | 0.2130 | 0.1206 | 0.937 | 1,860,000 |
| standard, N = 60,000 | 60 | 0.832, 0.273, 0.060 | 37 s | 0.2132 | 0.1208 | 0.943 | 3,660,000 |
| standard, N = 100,000 (TLR-simulated field; no dense) | 30 | 0.954, 0.284, 0.052 | 24 s | 0.2274 | 0.1283 | 0.903 | 3,100,000 |
| standard, N = 100,000 | 60 | 0.860, 0.272, 0.058 | 95 s | 0.2243 | 0.1264 | 0.900 | 6,100,000 |

Vecchia is fast and predicts well everywhere, and at N = 100,000 with 30 neighbors its estimates are close to the
truth; at N = 40,000 and 60,000 both neighbor counts put the nugget at about twice its true value and the variance at
about half (the adaptive rule's predictions at those sizes are equally good, RMSE 0.2260 and 0.2111). Head-to-head
likelihood evaluations at these points (`codes/23_vecchia_headtohead.py`, `vecchia_headtohead_n{N}.json`) were
pending when this file was written. Its parameter estimates move with the neighbor count
(sigma2 1.54 and 0.86 at N = 20,000 against the dense 1.36, i.e. 0.2 to 0.9 dense standard errors with
`pilot_standard_errors_n20000.json`: SEs 0.87, 0.076, 0.023 by finite differences at the dense MLE, whose
gradient there is not quite zero, so these SEs are approximate), and nothing short of a dense fit says which
neighbor count to trust. The 60-neighbor value at N = 10,000 may be a local optimum of GpGp's optimizer.

### 6. One budget on rank and storage precision (`codes/18_joint_rank_precision_pilot.py`, Objective 3)

For every tile, the cheapest (rank, precision) pair, precision in FP8 E4M3 / FP16 / FP32 / FP64 (rounding
simulated in NumPy after per-factor scaling), whose stored-tile error meets the adaptive budget e_max, against
rank-only compression in FP64 at the same budget and a fixed eps = 1e-4 in FP64. Sigma^TLR is assembled
densely from the rounded factors to check positive definiteness and the log-likelihood.

| N | rule | off-diagonal bytes / dense FP64 | precision mix (tiles) | budget met | PD | log-lik error |
|---|---|---|---|---|---|---|
| 10,000 | fixed 1e-4, FP64 | 4.5% | all FP64 | no | yes | -0.89 |
| 10,000 | rank only, FP64 | 5.7% | all FP64 | yes | yes | -0.25 |
| 10,000 | rank then precision | 2.8% | 779 FP32, 1 FP16 | yes | yes | -0.25 |
| 10,000 | joint | 2.8% | 777 FP32, 3 FP16 | yes | yes | -0.26 |
| 20,000 | rank only, FP64 | 4.7% | all FP64 | yes | yes | -0.06 |
| 20,000 | joint | 2.4% | 3,157 FP32, 3 FP16 | yes | yes | -0.06 |

The joint rule halves the bytes of rank-only compression at both sizes with the budget met and the matrix
definite; at this budget choosing rank first and then the lowest precision that fits gives the same bytes,
and no tile qualifies for FP8. Outputs: `joint/joint_rank_precision_n{N}.json`.

### 7. Metropolis-Hastings on a GPU (`codes/19_tlr_mcmc_gpu.py`, Objective 2)

Random-walk Metropolis-Hastings for (sigma2, range, eta) on the N = 5,000 field of Section 1's design
(same seed as `16_tlr_mcmc_pilot.py`), written after the companion mixed-precision Bayesian GP code: PyTorch on
one GPU, the tile compression batched (`--svd rand`: randomized range finder with power iterations and
the tile error verified in FP64, same ranks and log-likelihood as a full SVD), a failed factorization
rejects the proposal, checkpoints every 100 iterations, effective sample sizes by the initial positive
sequence estimator. Short isotropic-proposal chains (3,000 iterations, `mcmc_gpu_v4/`) have effective
sample sizes below 10 because the posterior is a ridge (correlation 0.86 between log sigma2 and log range),
so the reported comparison uses the proposal shaped by the exact chain's posterior covariance
(`prop_logcov_n5000.npy`, scale 2.38/sqrt(3) tuned in burn-in) and 20,000 iterations with 2,000 burn-in,
one chain per GPU job (`mcmc_long/{dense,adaptive_0.9,fixed_1e-4}/`, `results/logs/mcmc-long-*.out`):

| chain | posterior mean (sigma2, range, eta) | posterior SD | ESS | mean diff. in exact SD | SD ratio | failed factorizations |
|---|---|---|---|---|---|---|
| exact | 1.373, 0.337, 0.0444 | 0.754, 0.058, 0.018 | 600, 650, 601 | - | - | 0 |
| adaptive TLR, tau = 0.9 | 1.351, 0.336, 0.0444 | 0.643, 0.055, 0.018 | 642, 709, 641 | 0.03, 0.03, 0.00 | 0.85, 0.94, 0.99 | 0 |
| fixed eps = 1e-4 | 1.142, 0.322, 0.0469 | 0.294, 0.030, 0.012 | 989, 1281, 992 | 0.31, 0.26, 0.14 | 0.39, 0.51, 0.68 | 47 |

Monte Carlo standard errors of the posterior means are 0.03, 0.002, 0.0007 (exact chain). The adaptive-TLR
posterior matches the exact one within Monte Carlo error; the fixed tolerance shifts the posterior means by
0.1 to 0.3 SD, shrinks the posterior to 0.4 to 0.7 of its width, and fails 47 times in 20,000 proposals. A
fixed eps = 1e-3 cannot start a chain at the dispersed starting value (indefinite). Held-out RMSE, CRPS, and
coverage at the posterior means are the same for all three chains (0.234 / 0.132 / 0.92). The adaptive
chain runs at 0.24 s per iteration on an A30 or A100 (exact chain 0.02 s); the CPU version of the same
chain ran at 11 to 14 s per iteration.
