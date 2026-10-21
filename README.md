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

    cd codes && python3 11_tlr_rank_rules.py --n 10000 --out ../results/preliminary
    cd codes && python3 11_tlr_rank_rules.py --n 20000 --mle "dense,adaptive:0.9" --out ../results/preliminary

Outputs: `tlr_rank_rules_n{N}.json`, `sim_n{N}_obs.csv`, `sim_n{N}_holdout.csv`, `results/logs/`.

## 2. Vecchia baseline (`codes/12_vecchia_baseline.R`, GpGp)

Fits the same simulated data by Vecchia's approximation with m = 30 nearest neighbors
(GpGp `matern15_isotropic`, max-min ordering) and predicts the same held-out sites.

    cd codes && Rscript 12_vecchia_baseline.R 10000 30 ../results/preliminary

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
* Figure: `codes/20_pilot_results_figure.py` draws factor storage and the log-likelihood error at the true parameters
  against N, with the failed tolerances marked, and in panel (c) the KL divergence of adaptive TLR and of Vecchia from the
  exact model on the nine fields of Section 8 (`pilot_results.pdf`).

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

Vecchia is fast and predicts well everywhere. Its raw estimates move with N and with the neighbor count (at N = 40,000
and 60,000 both neighbor counts give a relative nugget of about 0.1 and a variance of about 0.5), but the quantities
the data identify are recovered by every method at every size: the noise variance sigma2 * eta is 0.049 to 0.050 for
dense, adaptive, and Vecchia fits against a true 0.05, and the microergodic ratio sigma2 / range^3 is 32 to 46 around a
true 37 for all of them (dense 41.0 and 35.7 at N = 10,000 and 20,000). The variance and range move together along a
ridge of the likelihood, so a halved variance with a doubled relative nugget is not an error. Head-to-head likelihood
evaluations at the fitted points (`codes/23_vecchia_headtohead.py`, `vecchia_headtohead_n{N}.json`): at N = 10,000 and
20,000 every fit, Vecchia's included, lies within about one log-likelihood unit of the best point. Its raw parameter
estimates move with the neighbor count
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

### 8. Vecchia head-to-head on nine fields: KL divergence and fits (codes 24-32, panel (c) of the results figure)

Where Vecchia's screening assumption weakens, does the adaptive rule keep the exact model's accuracy? Nine Matern fields at
N = 10,000 on the unit square (smoothness nu = 3/2 or 5/2, range 0.3, 1 or 3, relative nugget 0.05, 0.5 or 1; the first is
the field of Section 4, same seed and values), each with a dense exact reference. Folders `screening_stress/<tag>/`,
tag = nu{15,25}_r{03,1,3}_e{005,05,1}.

Accuracy of the approximation, the measure of Guinness (2018) and Katzfuss and Guinness (2021): the KL divergence
KL(N(0, Sigma) || N(0, Sigma_tilde)) at the true parameters, exact at this N. Adaptive TLR (tau = 0.9, 17's tile Cholesky
with recompression): `31_kl_tlr.py` -> `kl_tlr.json`. Response Vecchia (GpGp 0.5.1, max-min ordering, sparse inverse
Cholesky from `vecchia_Linv`): `32_kl_vecchia.R` -> `kl_vecchia.json` (m = 10, 30, 60, 120) and `kl_vecchia_matched.json`
(m = 250 and the m whose N(m+1) stored values equal the adaptive factor's). Both formulas were checked on exact cases:
Vecchia with every point as a neighbor and TLR with tau = 1e-8 give KL = 0 to rounding.

GpGp's `order_maxmin` is randomized, so the Vecchia likelihood and the Vecchia fits depend on the ordering drawn. Every
Vecchia number here is the mean over orderings drawn with `set.seed(1..8)` (KL at m = 250 and the matched m: 2 orderings);
the KL varies by under 1% across orderings, the fits much more (table below).

| field (nu, range, nugget) | adaptive TLR: storage, max rank, KL | Vecchia KL m = 10 / 30 / 60 / 120 | Vecchia KL at m = 250 / at TLR storage (m) |
|---|---|---|---|
| 3/2, 0.3, 0.05 (`nu15_r03_e005`) | 9.3%, 19, 0.078 | 250.6 / 43.5 / 22.2 / 6.7 | 0.37 / 0.065 (466) |
| 3/2, 1, 0.05 (`nu15_r1_e005`) | 8.1%, 9, 0.087 | 386.1 / 96.1 / 33.3 / 11.9 | 5.74 / 2.543 (405) |
| 3/2, 3, 0.05 (`nu15_r3_e005`) | 7.2%, 5, 0.065 | 440.4 / 135.3 / 59.1 / 23.5 | 7.73 / 4.461 (358) |
| 3/2, 0.3, 0.5 (`nu15_r03_e05`) | 8.0%, 11, 0.071 | 323.1 / 66.2 / 18.3 / 7.1 | 2.94 / 0.743 (399) |
| 5/2, 0.3, 0.05 (`nu25_r03_e005`) | 9.0%, 13, 0.093 | 361.4 / 89.0 / 43.0 / 24.5 | 6.12 / 1.147 (449) |
| 5/2, 1, 0.05 (`nu25_r1_e005`) | 7.7%, 6, 0.091 | 436.5 / 134.6 / 60.1 / 25.8 | 11.82 / 8.336 (385) |
| 5/2, 3, 0.05 (`nu25_r3_e005`) | 6.7%, 4, 0.147 | 458.0 / 150.9 / 72.6 / 34.4 | 15.01 / 10.515 (336) |
| 5/2, 1, 0.5 (`nu25_r1_e05`) | 7.1%, 4, 0.080 | 427.9 / 138.6 / 63.1 / 26.9 | 10.16 / 6.318 (356) |
| 5/2, 1, 1 (`nu25_r1_e1`) | 6.9%, 4, 0.106 | 411.9 / 138.6 / 64.1 / 27.5 | 10.33 / 6.566 (345) |

The adaptive KL divergence is 0.07 to 0.15 on every field. Vecchia's is 18 to 151 with 30 or 60 neighbors. At the storage
of the adaptive factor (m = 336 to 466, where Vecchia's O(N m^3) cost exceeds a dense Cholesky's at this N) Vecchia matches
the adaptive rule on the pilot field (0.065 vs 0.078) and is 10 to 90 times larger on the other eight, the gap growing with
the range and the smoothness. Response Vecchia is the variant tested; latent or grouped variants handle nuggets differently
and were not run.

Also in each folder: `stress.json` (24: field, exact and adaptive-TLR log-likelihood at the truth, Vecchia scored at the
truth), `vecchia.json` (25: Vecchia log-likelihood at the truth for m = 5..120, predictions, fits at m = 30, 60),
`vecchia_matched.json` (26), `grid.csv`, `vecchia_grid.csv` (27/28: exact, TLR and Vecchia log-likelihoods on 33 parameter
values around the exact MLE), `vecchia_fit_orderings.json` (30), `inference.json` (27: verified exact MLE, adaptive-TLR
MLE, fits scored on the exact likelihood).

Fits scored on the exact likelihood (`inference.json`, `27_inference_headtohead.py report`): the exact MLE is verified by
Nelder-Mead restarted until it stops improving, from the best of the truth, the earlier fits and every Vecchia fit; the
adaptive-TLR MLE and the Vecchia fits (fit_model, 8 orderings, `30_vecchia_fit_orderings.R`) are scored by the exact
log-likelihood they lose. The adaptive-TLR MLE lost at most 0.36 units (under 0.1 on eight fields); the Vecchia fits lost
up to 8.2 units on average over orderings and up to 20.7 for a single ordering, worst at long range and high smoothness.

| field (nu, range, nugget) | verified exact MLE (sigma2, range, nugget) | adaptive-TLR fit: exact log-lik lost | Vecchia m = 30 fits: mean (min to max) over 8 orderings | Vecchia m = 60 fits | grouped Vecchia KL m = 30 / 60 |
|---|---|---|---|---|---|
| 3/2, 0.3, 0.05 | (1.711, 0.348, 0.0293) | 0.02 | 0.47 (0.05 to 1.00) | 0.07 (0.01 to 0.22) | 28.4 / 7.6 |
| 3/2, 1, 0.05 | (1.626, 1.136, 0.0310) | 0.07 | 0.85 (0.34 to 1.63) | 0.55 (0.02 to 2.07) | 56.8 / 15.5 |
| 3/2, 3, 0.05 | (1.713, 3.594, 0.0294) | 0.07 | 4.76 (2.56 to 10.03) | 2.18 (0.38 to 5.92) | 82.2 / 28.9 |
| 3/2, 0.3, 0.5 | (1.211, 0.299, 0.4151) | 0.01 | 0.39 (0.03 to 0.76) | 0.26 (0.05 to 0.50) | 35.5 / 8.7 |
| 5/2, 0.3, 0.05 | (1.529, 0.316, 0.0329) | 0.00 | 0.43 (0.08 to 1.32) | 0.34 (0.04 to 0.67) | 59.6 / 23.5 |
| 5/2, 1, 0.05 | (1.669, 1.169, 0.0302) | 0.05 | 1.60 (0.03 to 9.97) | 1.34 (0.12 to 2.88) | 88.0 / 31.8 |
| 5/2, 3, 0.05 | (1.141, 2.741, 0.0442) | 0.36 | 8.20 (4.23 to 20.68) | 3.60 (1.19 to 9.03) | 95.3 / 39.6 |
| 5/2, 1, 0.5 | (0.947, 0.941, 0.5321) | 0.03 | 1.74 (0.09 to 7.23) | 4.49 (0.00 to 12.30) | 85.5 / 32.0 |
| 5/2, 1, 1 | (0.938, 0.912, 1.0746) | 0.06 | 3.52 (0.23 to 8.38) | 3.25 (0.06 to 6.32) | 85.4 / 31.8 |

Grouped Vecchia (`fit_model`'s default, `group = TRUE`), the variant the fits use, has no exposed inverse factor, so its KL
divergence is estimated by Monte Carlo (`33_kl_vecchia_grouped_mc.R`: KL = E[log p - log q] over 40 draws from the exact
model, 4 orderings, `kl_vecchia_grouped_mc.json`); on the same draws the ungrouped Monte Carlo estimate matches the exact
ungrouped KL of 32 within 1-2% on every field. Grouping roughly halves the KL (8 to 95 with 30 or 60 neighbors) and panel (c) of the results figure plots the grouped values; the equal-storage point stays ungrouped (exact).

Rerun everything (about 45 min per field on an 8-core laptop, 3 fields at a time; environment, seeds and versions in
`codes/ENVIRONMENT_screening_stress.txt`):

    cd codes && ./run_screening_stress.sh ../results/screening_stress_rerun 3
    python3 29_check_reproduction.py ../results/screening_stress_rerun

Smoke test (about 3 minutes; every stage end to end at N = 1,500 on two fields):

    cd codes && N=1500 FIELDS_ONLY="nu15_r03_e005 nu25_r3_e005" ./run_screening_stress.sh /tmp/smoke 2

Verified on 2026-10-21: the smoke test runs clean from the committed code, and rerunning the panel (c) of the results figure stages for the
pilot field reproduced every committed number exactly. The checker compares every quoted number with the committed one: quantities at fixed parameters (fields, log-likelihoods
and KL at the truth) should agree to rounding with the same BLAS; optimizer results (MLEs, fit losses) can move by a few
hundredths with another BLAS or thread count.
