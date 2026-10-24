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
* Metropolis-Hastings pilot: see Section 7 (GPU chains with effective sample sizes) and Section 11 (four chains per likelihood); the CPU chain of
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
adaptive tau = 0.9 (1.3614, 0.3352, 0.0036), no failures, RMSE 0.0746; eps = 1e-6 (added 2026-10-23, `small_nugget_eps6/`,
`codes/11` with the gesvd fallback after a non-converging tile SVD stopped the first run): (1.3541, 0.3333, 0.0036), no failures,
RMSE 0.0746, i.e. the same fit as the adaptive rule at the same storage (4.3% against 4.4%), 105 evaluations in 2.0 h. The dense and adaptive estimates lie on the
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

| field (nu, range, nugget) | verified exact MLE (sigma2, range, nugget) | adaptive-TLR fit: exact log-lik lost | Vecchia m = 30 fits WITH INTERCEPT (fit_model default): mean (min to max) over 8 orderings | Vecchia m = 60 fits with intercept | grouped Vecchia KL m = 30 / 60 |
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
ungrouped KL of 32 within 1-2% on every field. Grouping roughly halves the KL (8 to 95 with 30 or 60 neighbors) and Figure
2(c) plots the grouped values; the equal-storage point stays ungrouped (exact).

### 8a. Zero-mean Vecchia fits (added 2026-10-23; `codes/34_vecchia_fit_meanzero.R`)

A later check found a model mismatch in the fit table above: GpGp's `fit_model` profiles out an intercept (the design
matrix `X = 1` in `12_vecchia_baseline.R`, `25_vecchia_stress.R` and `30_vecchia_fit_orderings.R`), while the exact and
adaptive-TLR fits of `24`/`27` are zero-mean, and the Vecchia covariance estimates were scored on the zero-mean exact
likelihood. The fitted intercepts were not negligible on the long-range fields (0.87 and 0.91 at range 3), so part of the
loss in the table above is the fitted mean, not the approximation. Script 34 fits the same zero-mean model by Vecchia's
approximation with GpGp's own ingredients (`order_maxmin` under `set.seed(1..8)`, `find_ordered_nn`, `group_obs`, and
`vecchia_grouped_meanzero_loglik` maximized over log(sigma2, range, nugget) by restarted Nelder-Mead with a BFGS polish,
started from `get_start_parms`); `27_inference_headtohead.py fitorders --fo_file vecchia_fit_orderings_meanzero.json`
scores each fit on the exact likelihood (key `vecchia_fit_gap_over_orderings_meanzero` in `inference.json`). The
fixed-parameter results (KL divergences, likelihood ladders, matched-storage points, panel (c) of the results figure) are zero-mean on both
sides and are unaffected.

Exact log-likelihood lost below the verified exact MLE, mean (min to max) over the 8 orderings:

| field (nu, range, nugget) | adaptive-TLR fit | zero-mean Vecchia m = 30 | zero-mean Vecchia m = 60 | intercept fits, for comparison (m = 30 / 60 means) |
|---|---|---|---|---|
| 3/2, 0.3, 0.05 | 0.02 | 0.16 (0.01 to 0.34) | 0.07 (0.00 to 0.17) | 0.47 / 0.07 |
| 3/2, 1, 0.05 | 0.07 | 0.77 (0.01 to 2.45) | 0.31 (0.01 to 0.66) | 0.85 / 0.55 |
| 3/2, 3, 0.05 | 0.07 | 3.27 (0.04 to 8.49) | 0.32 (0.01 to 1.88) | 4.76 / 2.18 |
| 3/2, 0.3, 0.5 | 0.01 | 0.36 (0.04 to 1.42) | 0.10 (0.00 to 0.39) | 0.39 / 0.26 |
| 5/2, 0.3, 0.05 | 0.00 | 0.17 (0.01 to 0.70) | 0.22 (0.04 to 0.56) | 0.43 / 0.34 |
| 5/2, 1, 0.05 | 0.05 | 1.27 (0.26 to 3.29) | 0.60 (0.07 to 1.55) | 1.60 / 1.34 |
| 5/2, 3, 0.05 | 0.36 | 3.33 (0.22 to 7.90) | 0.47 (0.12 to 1.30) | 8.20 / 3.60 |
| 5/2, 1, 0.5 | 0.03 | 5.40 (0.08 to 16.45) | 2.64 (0.12 to 8.16) | 1.74 / 4.49 |
| 5/2, 1, 1 | 0.06 | 5.65 (0.04 to 18.32) | 1.73 (0.45 to 2.96) | 3.52 / 3.25 |

On the pilot field the zero-mean Vecchia fits lose 0.16 (m = 30) and 0.07 (m = 60) units (the intercept fits gave 0.47
and 0.07). Across the nine fields the adaptive-TLR fit loses at
most 0.36 units and the zero-mean Vecchia fits with 30 neighbors lose 0.16 to 5.7 on average, most on the long-range
(range 3) and noisy (relative nugget 0.5 and 1) fields; for the noisy fields Katzfuss and Guinness (2021) recommend the
sparse general Vecchia form rather than the standard (response) form run here, and the single-ordering maxima (up to 18
units) include optimizer stops on the flat sigma2-range ridge, so the means over orderings are the numbers to read. The
Vecchia fits again predict as well as the dense fit on every field.

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

### 9. Tolerances 1e-5 and 1e-6 past the dense limit (added 2026-10-23; `codes/17_tlr_cholesky_pilot.py`, `tlr_cholesky_eps/`)

Section 4 ran only eps = 1e-4 against the adaptive rule at N = 40,000 and beyond. This section asks what the tighter
tolerances do there. The fields of Section 4 are reproduced from their seeds (`--dense 0`, no dense factor), the rules
fixed 1e-5, fixed 1e-6 and adaptive tau = 0.9 are run at the true parameters, and the log-likelihood error is taken against
the dense value stored with the Section 4 run (`dense_loglik_true`), which `codes/20_pilot_results_figure.py` merges into
panels (a) and (b) of the results figure. At N = 100,000 the field is the tau = 0.02 TLR simulation of Section 4 and only completion, storage
and predictions are reported. `codes/17` now falls back to `gesvd` when the default tile SVD does not converge (the first
100,000-point run crashed in the 1e-6 pass).

    sbatch --job-name=tlrchol-eps-n40000 storrs_cpu_big.sbatch "17_tlr_cholesky_pilot.py --n 40000 --dense 0 --rules fixed:1e-5,fixed:1e-6,adaptive:0.9 --mle '' --out ../results/tlr_cholesky_eps"

| N | rule | completed | factor / dense factor | mean rank (max) | log-lik error | RMSE / CRPS / 90% cover | seconds |
|---|---|---|---|---|---|---|---|
| 40,000 | fixed 1e-5 | yes | 7.4% | 7.7 (20) | -1.56 | 0.2260 / 0.1272 / 0.907 | 275 |
| 40,000 | fixed 1e-6 | yes | 8.8% | 9.5 (28) | +0.18 | 0.2259 / 0.1272 / 0.907 | 317 |
| 40,000 | adaptive 0.9 | yes | 4.1% | 3.6 (18) | -0.66 | 0.2260 / 0.1272 / 0.907 | 231 |
| 60,000 | fixed 1e-5 | yes | 6.6% | 7.2 (19) | -2.63 | 0.2111 / 0.1190 / 0.93 | 15,050 |
| 60,000 | fixed 1e-6 | yes | 8.0% | 9.0 (26) | -0.35 | 0.2112 / 0.1190 / 0.93 | 937 |
| 60,000 | adaptive 0.9 | yes | 3.5% | 3.3 (17) | -0.56 | 0.2111 / 0.1190 / 0.93 | 892 |
| 100,000 | fixed 1e-5 | yes | 5.7% | - | no dense reference | 0.2204 | - |
| 100,000 | fixed 1e-6 | yes | 7.0% | 8.1 (23) | no dense reference | 0.2204 / 0.1242 / 0.91 | 19419 |
| 100,000 | adaptive 0.9 (Section 4) | yes | 2.9% | 2.9 | no dense reference | 0.2204 | - |

Both tolerances complete where 1e-4 fails, with 1.8 to 2.0 (1e-5) and 2.1 to 2.4 (1e-6) times the storage of the
adaptive factor and the same predictions; at N = 100,000 the 1e-6 factorization took 5.4 hours on 32 cores against
45 minutes for the adaptive factor. The 1e-5 factor's log-likelihood error grows with N (-1.56, -2.63), the
adaptive factor's does not (-0.66, -0.56).
The 1e-6 run at N = 100,000 ran separately after the gesvd fallback (`tlr_cholesky_eps6/`, job `tlrchol-eps6-n100000`)
and its entry is merged into `tlr_cholesky_eps/tlr_cholesky_n100000.json`. Outputs:
`tlr_cholesky_eps/tlr_cholesky_n{N}.json`, `results/logs/tlrchol-eps-n{N}-*.out`.

### 10. One budget on rank and storage precision at ranges 0.1 and 1.0 (added 2026-10-23; `codes/18_joint_rank_precision_pilot.py`, `joint_range0.1/`, `joint_range1.0/`)

Section 6 (range 0.3) found at most three tiles that qualified for FP16, none for FP8, and no difference between the joint
choice and rank first, then precision. The same pilot at range 0.1 (rougher field, higher ranks) and range 1.0 (smoother field), N = 10,000
and 20,000, sigma2 = 1, eta = 0.05, tau = 0.9, assembled matrix, is reported here.

    sbatch --job-name=joint_range0.1 storrs_cpu.sbatch "18_joint_rank_precision_pilot.py --n 10000 --range 0.1 --out ../results/joint_range0.1"

| range | N | rule | off-diagonal bytes / dense FP64 | bytes / rank only | precision mix (tiles FP8 / FP16 / FP32 / FP64) | budget met | PD | log-lik error |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 10,000 | fixed 1e-4, FP64 | 6.1% | 1.07 | 0 / 0 / 0 / 780 | no | yes | -0.69 |
| 0.1 | 10,000 | rank only, FP64 | 5.7% | 1.00 | 0 / 0 / 0 / 780 | yes | yes | +0.17 |
| 0.1 | 10,000 | rank then precision | 2.5% | 0.441 | 88 / 348 / 326 / 18 | yes | yes | +0.17 |
| 0.1 | 10,000 | joint | 2.5% | 0.434 | 89 / 379 / 294 / 18 | yes | yes | +0.10 |
| 0.1 | 20,000 | fixed 1e-4, FP64 | 4.5% | 1.03 | 0 / 0 / 0 / 3,160 | no | yes | +0.47 |
| 0.1 | 20,000 | rank only, FP64 | 4.4% | 1.00 | 0 / 0 / 0 / 3,160 | yes | yes | -0.03 |
| 0.1 | 20,000 | rank then precision | 1.9% | 0.441 | 396 / 1,257 / 1,419 / 88 | yes | yes | -0.02 |
| 0.1 | 20,000 | joint | 1.9% | 0.439 | 396 / 1,324 / 1,352 / 88 | yes | yes | -0.03 |
| 1.0 | 10,000 | fixed 1e-4, FP64 | 2.8% | 0.69 | 0 / 0 / 0 / 780 | no | no | - |
| 1.0 | 10,000 | rank only, FP64 | 4.0% | 1.00 | 0 / 0 / 0 / 780 | yes | yes | +0.32 |
| 1.0 | 10,000 | rank then precision | 2.0% | 0.500 | 0 / 0 / 780 / 0 | yes | yes | +0.32 |
| 1.0 | 10,000 | joint | 2.0% | 0.500 | 0 / 0 / 780 / 0 | yes | yes | +0.32 |
| 1.0 | 20,000 | fixed 1e-4, FP64 | 2.5% | 0.70 | 0 / 0 / 0 / 3,160 | no | no | - |
| 1.0 | 20,000 | rank only, FP64 | 3.6% | 1.00 | 0 / 0 / 0 / 3,160 | yes | yes | -0.29 |
| 1.0 | 20,000 | rank then precision | 1.8% | 0.500 | 0 / 0 / 3,160 / 0 | yes | yes | -0.29 |
| 1.0 | 20,000 | joint | 1.8% | 0.500 | 0 / 0 / 3,160 / 0 | yes | yes | -0.29 |

At range 0.1, 11 to 13% of the tiles qualify for FP8 and 42 to 49% for FP16, and the joint choice stores 0.434 and 0.439 of the
rank-only bytes against 0.441 for rank first, then precision, with the same log-likelihood and predictions (RMSE 0.2192
and 0.2397 for every rule that met the budget). At range 1.0 every tile meets the budget in FP32, both orders give exactly
half the rank-only bytes, and the fixed tolerance 1e-4 leaves the assembled matrix indefinite at both sizes. Outputs:
`joint_range{0.1,1.0}/joint_rank_precision_n{N}.json`, `results/logs/joint-r{0.1,1.0}-n{N}-*.out`.

### 11. Four chains per likelihood (added 2026-10-23; `codes/19_tlr_mcmc_gpu.py --chain_seed/--start_mult`, `codes/35_mcmc_multichain_diagnostics.py`)

Section 7 reported one chain per likelihood; this section adds three more. Chain 1 is the Section 7 chain (`mcmc_long/`, chain seed 20261021, start
1.3, 0.8, 1.5 times theta_true); chains 2-4 (`mcmc_multi/<rule>/c{2,3,4}/`, chain seeds 102-104) start from
(0.6, 1.4, 0.5), (2.0, 0.5, 2.0) and (0.8, 1.2, 0.8) times theta_true, with the same data, flat priors on log theta over
the box [0.1, 10] x [0.02, 2] x [0.001, 1], proposal covariance `prop_logcov_n5000.npy` scaled by 2.38/sqrt(3) and
tuned in burn-in, 20,000 iterations with 2,000 burn-in, one A30 or A100 per chain (adaptive chains 0.23 s per iteration).
Script 35 pools the four chains per likelihood on the natural scale and reports rank-normalized split R-hat and bulk
effective sample size (Vehtari et al. 2021), the Monte Carlo standard error of each mean, and the differences from the
pooled exact chains in exact posterior standard deviations (`mcmc_multi/multichain_diagnostics_n5000.json`).

| likelihood | posterior mean (sigma2, range, eta) | posterior SD | max split R-hat | bulk ESS | mean diff. in exact SD | SD ratio | failed factorizations per chain |
|---|---|---|---|---|---|---|---|
| exact | 1.445, 0.343, 0.0430 | 0.846, 0.062, 0.0181 | 1.004 | 2317, 2470, 2320 | - | - | 0, 0, 0, 0 |
| adaptive TLR (tau = 0.9) | 1.433, 0.341, 0.0432 | 0.838, 0.061, 0.0179 | 1.003 | 2311, 2479, 2312 | -0.01, -0.03, +0.01 | 0.99, 0.99, 0.99 | 0, 0, 0, 0 |
| fixed tolerance 1e-4 | 1.142, 0.322, 0.0472 | 0.303, 0.031, 0.0130 | 1.001 | 3594, 4376, 3591 | -0.36, -0.33, +0.23 | 0.36, 0.50, 0.72 | 47, 33, 50, 76 |

The pooled exact posterior has a heavier sigma2 tail than chain 1 alone (SD 0.85 against 0.75; chain SDs 0.75, 0.82,
0.80, 0.99), which is why a single chain could not resolve the adaptive chain's 0.85 SD ratio of Section 7: with four
chains the adaptive-TLR posterior matches the exact one to 0.03 SD in the means (within one Monte Carlo standard error)
and to 1% in the standard deviations. The fixed 1e-4 posterior is shifted by 0.23 to 0.36 SD (10 to 17 Monte Carlo
standard errors), narrowed to 0.36 to 0.72 of the exact SD, and truncated by 206 rejected factorizations in 80,000
proposals (1e-3 cannot start). Held-out RMSE and CRPS at the posterior means are 0.234 and 0.132 for every chain; the 90%
coverage is 0.937 for the exact and adaptive chains and 0.917 to 0.920 for the fixed-tolerance chains.

### 12. The log-likelihood error as a first-order term plus a certified remainder (added 2026-10-23; `codes/36_likelihood_error_terms.py`, `bound_terms/`)

The Objective 1 bound in hand grows with N (the eta route gives epsilon_ell = O(N tau)). This section measures what the
error is made of. With Delta = Sigma^TLR - Sigma (assembled, true parameters, same fields as section 4), alpha =
Sigma^{-1} y, z = L^{-1} y and E = L^{-1} Delta L^{-T}, the error ell^TLR - ell is split into the first-order term
T1 = -tr(Sigma^{-1} Delta)/2 + alpha^T Delta alpha / 2 and a remainder. The remainder is bounded without any factor N:
the log-determinant part by ||E||_F^2 / (4 (1 - ||E||_2)) and the quadratic form by ||E z||^2 / (2 (1 - ||E||_2)) with
||E z||^2 = (Delta alpha)^T Sigma^{-1} (Delta alpha). T1, ||E||_F, ||E||_2 and ||E z|| are all computable from the TLR
factor and the truncation residuals (Hutchinson estimates with TLR solves), so the pair (T1, remainder bound) is an a
posteriori certificate of the likelihood error at TLR cost. The eta-route column is the uniform bound
[N (-log(1 - eta)) + eta ||z||^2 / (1 - eta)] / 2 with eta = ||E||_2 (actual) and with eta_hat = sqrt(2 sum e_ij^2) /
m(theta) (what the rule guarantees a priori; "-" where eta_hat >= 1 and the bound is void); the "T1 tile bound" column is
the tile-by-tile Cauchy-Schwarz bound sum_{i>j} (||(Sigma^{-1})_{ij}||_F + ||alpha_i|| ||alpha_j||) e_ij on the
first-order term.

    sbatch --job-name=bound-terms-n40000 storrs_cpu_big.sbatch "36_likelihood_error_terms.py --n 40000 --rules adaptive:0.9,adaptive:0.5,fixed:1e-4,fixed:1e-5 --out ../results/bound_terms"

| N | rule | stored | error | first-order T1 | remainder | remainder bound | spectral norm of E | T1 tile bound | eta route (actual / a priori) |
|---|---|---|---|---|---|---|---|---|---|
| 10,000 | adaptive tau = 0.9 | 5.3% | -0.252 | -0.216 | -0.036 | 0.118 | 0.049 | 2,449 | 515 / 10,826 |
| 10,000 | adaptive tau = 0.5 | 5.5% | -0.064 | -0.053 | -0.011 | 0.036 | 0.027 | 1,341 | 275 / 4,108 |
| 10,000 | fixed eps = 0.0001 | 4.5% | -0.891 | +1.789 | -2.680 | 13.793 | 0.398 | 16,910 | 5,879 / - |
| 10,000 | fixed eps = 1e-05 | 5.4% | -0.436 | -0.391 | -0.045 | 0.142 | 0.038 | 1,867 | 397 / 10,012 |
| 20,000 | adaptive tau = 0.9 | 3.6% | -0.077 | -0.039 | -0.038 | 0.115 | 0.054 | 4,189 | 1,109 / 17,242 |
| 20,000 | adaptive tau = 0.5 | 3.8% | -0.255 | -0.245 | -0.011 | 0.035 | 0.029 | 2,358 | 581 / 7,242 |
| 20,000 | fixed eps = 0.0001 | 2.9% | -26.146 | -5.533 | -20.613 | 212.297 | 0.857 | 55,171 | 78,144 / - |
| 20,000 | fixed eps = 1e-05 | 3.5% | +0.568 | +0.743 | -0.175 | 0.548 | 0.072 | 6,586 | 1,519 / - |
| 40,000 | adaptive tau = 0.9 | 2.6% | +0.274 | +0.304 | -0.030 | 0.101 | 0.054 | 7,228 | 2,233 / 28,476 |
| 40,000 | adaptive tau = 0.5 | 2.8% | +0.175 | +0.185 | -0.010 | 0.033 | 0.031 | 4,198 | 1,280 / 12,949 |
| 40,000 | fixed eps = 0.0001 | 2.0% | not positive definite | | | | | | |
| 40,000 | fixed eps = 1e-05 | 2.4% | -0.506 | +0.021 | -0.527 | 2.059 | 0.169 | 22,226 | 7,749 / - |

For the adaptive rule the remainder bound is 0.10 to 0.12 at every N for tau = 0.9 and 0.02 to 0.04 for tau = 0.5, while
the eta-route bound grows from 515 to 2,234 (actual eta) and from 10,826 to 28,476 (a priori). The tile-by-tile
Cauchy-Schwarz bound on the first-order term is useless (thousands): the cancellation inside tr(Sigma^{-1} Delta) is what
makes the error small, so Task 2 of Objective 1 is to compute the first-order term from the truncation residuals and to
bound the remainder, not to bound the first-order term tile by tile. The fixed tolerances do not: the tolerance 1e-4
lets the spectral norm of E reach 0.40 and 0.86, so its remainder is 2.7 and 20.6 units with certificates of 13.8 and 212
(the assembled-matrix errors of -0.9 and -26.1 differ from the TLR Cholesky with recompression of section 4, -2.6 and
10.9 at the same N, which is a different object), and at N = 40,000 it does not give a positive definite assembled matrix;
1e-5 keeps the certificate at 0.14 to 2.1. The adaptive budget is what keeps ||E||_2 near 0.05 and the certificate near
0.1 at every N. Outputs:
`bound_terms/bound_terms_n{N}.json`, `results/logs/bound-terms-n{N}-*.out`.

### 13. When the joint rank-and-precision choice pays: regime sweep, larger N, and the wind-gust covariance (added 2026-10-23; `codes/18` with `--nu`, `codes/joint_sweep.sh`, `codes/joint_sweep2.sh`, `codes/39_joint_windgust.py`; `joint_sweep/`, `joint_range0.3/`, `joint_range0.1/`, `joint_windgust/`)

Sections 6 and 10 found the joint choice of rank and storage precision only 0 to 1.6% below "rank first, then the lowest
precision that fits" on Matern fields at nugget 0.05. This section asks where the joint choice pays. The per-tile budget is
e_max = tau b m(theta) / N with m(theta) = sigma2 eta; a tile can trade one extra rank column (a factor (r+1)/r in bytes)
for one precision level (a factor 1/2) only when its norm sits just above a rounding threshold at its minimal rank.

**Regime sweep at N = 10,000** (`joint_sweep/<tag>/joint_rank_precision_n10000.json`; "margin" = 1 - joint bytes / rank-then-precision bytes; the FP8/FP16 shares are the joint rule's; "fixed PD" = whether the fixed tolerance 1e-4 gave a positive definite assembled matrix). The nugget 0.5 and 1.0 rows are mechanism checks only; residual fields of scientific interest have relative nuggets of 0.05 to 0.2.

| nu | tile b | nugget | range | joint / rank-only | rank-then-precision / rank-only | margin | FP8 / FP16 share | mean rank | fixed PD |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 250 | 0.05 | 0.05 | 0.434 | 0.441 | 1.4% | 18% / 22% | 5.7 | yes |
| 1.5 | 250 | 0.05 | 0.1 | 0.434 | 0.441 | 1.5% | 11% / 49% | 7.1 | yes |
| 1.5 | 250 | 0.05 | 0.3 | 0.500 | 0.500 | 0.0% | 0% / 0% | 7.1 | yes |
| 1.5 | 250 | 0.05 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 5.0 | no |
| 1.5 | 250 | 0.05 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.5 | no |
| 1.5 | 250 | 0.2 | 0.05 | 0.398 | 0.418 | 4.8% | 19% / 24% | 4.5 | yes |
| 1.5 | 250 | 0.2 | 0.1 | 0.405 | 0.414 | 2.0% | 18% / 50% | 5.6 | yes |
| 1.5 | 250 | 0.2 | 0.3 | 0.484 | 0.485 | 0.2% | 0% / 11% | 5.8 | yes |
| 1.5 | 250 | 0.2 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 4.2 | yes |
| 1.5 | 250 | 0.2 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.1 | yes |
| 1.5 | 250 | 0.5 | 0.05 | 0.315 | 0.377 | 16.4% | 17% / 30% | 3.8 | yes |
| 1.5 | 250 | 0.5 | 0.1 | 0.356 | 0.378 | 5.8% | 23% / 54% | 4.7 | yes |
| 1.5 | 250 | 0.5 | 0.3 | 0.437 | 0.440 | 0.7% | 0% / 37% | 5.0 | yes |
| 1.5 | 250 | 1.0 | 0.05 | 0.247 | 0.327 | 24.5% | 12% / 32% | 3.1 | yes |
| 1.5 | 250 | 1.0 | 0.1 | 0.293 | 0.333 | 11.9% | 25% / 59% | 4.1 | yes |
| 1.5 | 250 | 1.0 | 0.3 | 0.379 | 0.389 | 2.7% | 0% / 64% | 4.5 | yes |
| 1.5 | 500 | 0.05 | 0.05 | 0.459 | 0.462 | 0.8% | 22% / 21% | 10.2 | yes |
| 1.5 | 500 | 0.05 | 0.1 | 0.456 | 0.460 | 0.9% | 9% / 43% | 11.1 | yes |
| 1.5 | 500 | 0.05 | 0.3 | 0.499 | 0.499 | 0.0% | 0% / 1% | 9.7 | yes |
| 1.5 | 500 | 0.05 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 6.6 | no |
| 1.5 | 500 | 0.05 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 4.1 | no |
| 1.5 | 500 | 0.2 | 0.05 | 0.392 | 0.444 | 11.7% | 19% / 34% | 8.3 | yes |
| 1.5 | 500 | 0.2 | 0.1 | 0.424 | 0.429 | 1.1% | 14% / 52% | 8.6 | yes |
| 1.5 | 500 | 0.2 | 0.3 | 0.490 | 0.491 | 0.1% | 0% / 8% | 7.7 | yes |
| 1.5 | 500 | 0.2 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 5.1 | yes |
| 1.5 | 500 | 0.2 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.5 | yes |
| 1.5 | 500 | 0.5 | 0.05 | 0.254 | 0.394 | 35.5% | 18% / 40% | 6.7 | yes |
| 1.5 | 500 | 0.5 | 0.1 | 0.368 | 0.402 | 8.5% | 21% / 58% | 7.2 | yes |
| 1.5 | 500 | 0.5 | 0.3 | 0.448 | 0.456 | 1.8% | 0% / 35% | 6.6 | yes |
| 1.5 | 500 | 1.0 | 0.05 | 0.245 | 0.294 | 16.6% | 16% / 34% | 5.5 | yes |
| 1.5 | 500 | 1.0 | 0.1 | 0.258 | 0.354 | 27.0% | 22% / 69% | 6.3 | yes |
| 1.5 | 500 | 1.0 | 0.3 | 0.391 | 0.410 | 4.6% | 0% / 63% | 5.8 | yes |
| 2.5 | 250 | 0.05 | 0.3 | 0.499 | 0.499 | 0.0% | 0% / 1% | 6.8 | no |
| 2.5 | 250 | 0.05 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 4.8 | no |
| 2.5 | 250 | 0.05 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.1 | yes |
| 2.5 | 250 | 0.2 | 0.3 | 0.480 | 0.482 | 0.4% | 0% / 13% | 5.6 | yes |
| 2.5 | 250 | 0.2 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.9 | no |
| 2.5 | 250 | 0.2 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.0 | yes |
| 2.5 | 500 | 0.05 | 0.3 | 0.499 | 0.499 | 0.0% | 0% / 1% | 8.8 | yes |
| 2.5 | 500 | 0.05 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 5.9 | no |
| 2.5 | 500 | 0.05 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.5 | no |
| 2.5 | 500 | 0.2 | 0.3 | 0.487 | 0.490 | 0.6% | 0% / 11% | 7.3 | yes |
| 2.5 | 500 | 0.2 | 1.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 4.8 | no |
| 2.5 | 500 | 0.2 | 3.0 | 0.500 | 0.500 | 0.0% | 0% / 0% | 3.1 | yes |

Strong dependence (range 1 and 3, or nu = 2.5): the far-field tiles are low-rank (mean rank 3 to 5) but their norms stay of
order sigma2 b, so no format below FP32 meets the budget, the joint rule reduces to the sequential one (margin 0) and the
fixed tolerance 1e-4 is not positive definite at nugget 0.05. Weak dependence (range 0.05 to 0.1): tiles are small in norm,
FP8 and FP16 qualify, and the margin reaches 4.8% (tile 250) and 11.7% (tile 500) at nugget 0.2, range 0.05.

**Larger N on the Matern field** (range 0.3 and 0.1, nugget 0.05, tile 250; sections 6 and 10 hold N = 10,000 and 20,000):

| N | range | joint / rank-only | rank-then-precision / rank-only | margin | FP8 / FP16 share | fixed PD |
|---|---|---|---|---|---|---|
| 40,000 | 0.3 | 0.500 | 0.500 | 0.0% | 0% / 0% | no |
| 40,000 | 0.1 | 0.437 | 0.439 | 0.3% | 12% / 40% | yes |
| 60,000 | 0.3 | 0.500 | 0.500 | 0.0% | 0% / 0% | no |
| 60,000 | 0.1 | 0.439 | 0.441 | 0.4% | 11% / 39% | yes |

The margin does not grow with N here: the budget tightens as 1/N and offsets the smaller far-field norms.

**The wind-gust space-time covariance** (`codes/39_joint_windgust.py`; the station-hours and their WRF predictors are in `data/windgust_train_data.csv`,
a copy of the SC26 application's training file, Jahan et al. 2025 wind gusts over the northeastern United States): random subsets of the 131,889 station-hours of the
SC26 application (2017-2019 storms, 266 stations), Morton order by spatial location, Gneiting-Matern(3/2) covariance with
beta = 0 at the parameters fitted by mixed precision in the SC26 paper (spatial range 19.72 km, temporal range 3.61 h,
relative nugget 0.928, sill 2.07), tau = 0.9. Only the covariance is needed: the tile choices and the positive definiteness
of the assembled Sigma^TLR do not involve the residuals. Most tile pairs are hours to years apart (the median off-diagonal
tile norm is 0), so most tiles get rank 0; the remaining tiles spread across all four precision thresholds.

| N | tile b | fixed 1e-4 | rank only | rank then precision | joint | joint / rank-only | r-t-p / rank-only | margin | joint tiles FP8 / FP16 / FP32 / FP64 (rank 0 counted as FP64) | mean rank fixed / adaptive |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 250 | 28.5% | 2.12% | 0.50% | 0.48% | 0.225 | 0.236 | 4.8% | 100 / 69 / 0 / 611 | 35.6 / 2.6 |
| 10,000 | 500 | 19.9% | 3.19% | 0.76% | 0.68% | 0.214 | 0.238 | 9.8% | 54 / 22 / 0 / 114 | 49.7 / 8.0 |
| 20,000 | 250 | 25.5% | 1.60% | 0.40% | 0.39% | 0.241 | 0.249 | 3.3% | 143 / 231 / 0 / 2786 | 31.8 / 2.0 |
| 20,000 | 500 | 18.9% | 2.31% | 0.57% | 0.56% | 0.242 | 0.245 | 1.0% | 69 / 108 / 0 / 603 | 47.2 / 5.8 |
| 40,000 | 250 | 22.4% | 1.47% | 0.44% | 0.35% | 0.239 | 0.303 | 21.1% | 415 / 668 / 0 / 11637 | 28.0 / 1.8 |
| 40,000 | 500 | 15.4% | 1.56% | 0.44% | 0.38% | 0.246 | 0.280 | 12.2% | 115 / 295 / 0 / 2750 | 38.4 / 3.9 |

(Off-diagonal bytes as a percentage of the dense FP64 off-diagonal bytes.) On this covariance the budget-governed precision
choice stores 0.21 to 0.25 of the rank-only bytes (a factor of four to five, against two on the Matern fields), the joint
choice is 1 to 21% below rank-then-precision with the largest margins at N = 40,000 (21% at tile 250, 12% at tile 500),
where the tighter budget leaves the sequential choice one precision level higher than the joint one, and the fixed
tolerance 1e-4 keeps a mean rank of 36 to 50 on tiles the budget drops to rank 0 to 3, storing 10 to 15 times the rank
rule's bytes. The fitted nugget of 0.93 is a property of these regression residuals, not a chosen setting. Outputs:
`joint_windgust/joint_windgust_n{N}_b{b}.json`, `results/logs/gust-joint-*.out`, `results/logs/joint-sweep*.out`.

### 14. Metropolis-Hastings driven by the TLR Cholesky factorization with a self-tuned proposal (added 2026-10-23, final 2026-10-24; `codes/38_tlr_cholesky_mcmc.py`, `mcmc_tlrchol/`)

Sections 7 and 11 assembled Sigma^TLR densely inside the chain and shaped the proposal from the exact chain's posterior
covariance. Here the likelihood is the TLR Cholesky factorization with recompression of `codes/17` (nothing dense is formed;
a failed factorization rejects the proposal), and the proposal is a random walk on log theta whose scale is adapted during
burn-in, re-shaped once at half burn-in by the covariance of the chain's own draws, and held fixed afterwards, so no exact
chain is used anywhere. Same data as section 7 (seed 20261020, N = 5,000, b = 250), four dispersed starts per likelihood
(chain seeds 101-104, start multipliers 1.3/0.8/1.5, 0.6/1.4/0.5, 2.0/0.5/2.0, 0.8/1.2/0.8), 10,000 iterations with 2,000
discarded, flat priors on log theta over [0.1, 10] x [0.02, 2] x [0.001, 1], one 4-core CPU job per chain, checkpoint every
100 iterations. `python3 38_tlr_cholesky_mcmc.py --pool --out ../results/mcmc_tlrchol` pools the chains
(rank-normalized split R-hat and bulk ESS of section 11's `codes/35`, differences from the pooled exact chains in exact SD).

    sbatch --requeue --job-name=mcmc-tc-adaptive_0.9-c1 storrs_cpu4.sbatch "38_tlr_cholesky_mcmc.py --n 5000 --rule adaptive:0.9 --iters 10000 --burn 2000 --chain_seed 101 --start_mult 1.3,0.8,1.5 --out ../results/mcmc_tlrchol/adaptive_0.9/c1"

| likelihood | chains | posterior mean (sigma2, range, eta) | posterior SD | max split R-hat | bulk ESS | mean diff. in exact SD | SD ratio | failed factorizations per chain | 90% coverage at the posterior mean | s / iteration |
|---|---|---|---|---|---|---|---|---|---|---|
| exact | 4 | 1.485, 0.346, 0.0420 | 0.866, 0.063, 0.0178 | 1.003 | 711, 735, 714 |  |  | 0, 0, 0, 0 | 0.937-0.937 | 0.8-2.0 |
| adaptive TLR Cholesky (tau = 0.9) | 4 | 1.466, 0.344, 0.0424 | 0.846, 0.062, 0.0176 | 1.004 | 752, 785, 750 | -0.02, -0.03, +0.02 | 0.98, 0.99, 0.99 | 0, 0, 0, 0 | 0.937-0.937 | 2.1-4.5 |
| fixed tolerance 1e-4, TLR Cholesky | 4 | 1.013, 0.306, 0.0521 | 0.229, 0.025, 0.0122 | 1.004 | 1394, 1514, 1380 | -0.55, -0.63, +0.57 | 0.26, 0.39, 0.69 | 0, 0, 0, 2 | 0.917-0.917 | 2.0-2.3 |

With the factorization the software runs and a proposal tuned from the chain itself, the adaptive rule reproduces the exact
posterior means to 0.03 exact SD and the standard deviations to 2% (four chains, split R-hat <= 1.004), with 0 failed
factorizations and the exact chains' coverage; the fixed tolerance 1e-4 is shifted by 0.55 to 0.63 SD and narrowed to 0.26 to
0.69 of the exact SD, the same bias and overconfidence as in the assembled-matrix pilot of section 11 (where it also failed
206 times; the TLR Cholesky with recompression at 1e-4 failed 2 times in 40,000 proposals at N = 5,000). The self-tuned
proposal mixes more slowly than the exact-covariance-shaped one of section 11 (bulk ESS about 750 against 2,300 for
comparable draws), which costs Monte Carlo precision, not accuracy. At the posterior mean the adaptive factor stores 0.153 of
a dense factor and the 1e-4 factor 0.169. Outputs: `mcmc_tlrchol/<rule>/c<k>/tlr_mcmc_n5000.json`, `chain_n5000.npy`, `mcmc_tlrchol/pooled_n5000.json`,
`results/logs/mcmc-tc-*.out`.

### 15. Oscillation of the TLR log-likelihood difference over the exact posterior: a numerical total variation bound (added 2026-10-24; `codes/37_posterior_oscillation.py`, `posterior_oscillation_200/`)

Objective 2's bound ||pi^TLR - pi||_TV <= tanh(epsilon/2) needs sup |ell^TLR - ell| over the region where the posterior
concentrates, and only half the oscillation of d(theta) = ell^TLR(theta) - ell(theta) matters there, since a constant shift
leaves pi^TLR unchanged. Script 37 evaluates d(theta) at 200 draws of the exact chain of section 7 (same data: seed 20261020,
N = 5,000, b = 250), thinned evenly after burn-in, for the assembled Sigma^TLR (exact tile SVDs) and for the TLR Cholesky
factorization with recompression of `codes/17`, for the adaptive rule (tau = 0.9) and the fixed tolerance 1e-4. The script
writes a partial JSON every 10 draws (`posterior_oscillation_n5000_partial.json`) and the full one at the end; a 300-draw
copy (`posterior_oscillation_300/`) and a 500-draw copy were also launched, the latter without partial saves and expected to
time out.

    sbatch --requeue --job-name=posterior-osc200-n5000 storrs_cpu.sbatch "37_posterior_oscillation.py --n 5000 --draws 200 --rules adaptive:0.9,fixed:1e-4 --out ../results/posterior_oscillation_200"

| rule | variant | draws evaluated | failed factorizations | range of d | half-oscillation delta | TV bound tanh(delta/2) |
|---|---|---|---|---|---|---|
| adaptive 0.9 | assembled | 200 | 0 | -0.248 to +0.299 | 0.274 | 0.136 |
| adaptive 0.9 | TLR Cholesky | 200 | 0 | -0.867 to -0.338 | 0.265 | 0.132 |
| fixed 1e-4 | assembled | 191 | 9 | -28.944 to +1.304 | 15.124 | 1.000 |
| fixed 1e-4 | TLR Cholesky | 194 | 6 | -29.516 to +1.815 | 15.665 | 1.000 |

For the adaptive rule the TLR Cholesky difference sits in a band of width 0.53 around -0.60 (at the true parameters -0.55,
at the posterior mean -0.35), so the numerical total variation bound is 0.13 (0.14 for the assembled matrix), against an
observed difference of 0.03 exact posterior standard deviations in the means (section 14); the a priori bound of section
12 is vacuous here. For the fixed tolerance 1e-4 the difference swings over 31 units and the factorization fails at 6 of
the 200 draws (9 of 200 for the assembled matrix), posterior tail draws with a large sigma2 and a small nugget, so its
bound is 1 and the chain's target is restricted to the region where the factorization completes. Outputs:
`posterior_oscillation_200/posterior_oscillation_n5000.json`, `logs/posterior-osc200-*.out`.

**300-draw confirmation** (finished 2026-10-24, 5.5 h; `posterior_oscillation_300/posterior_oscillation_n5000.json`,
`logs/posterior-osc300-*.out`): the same quantities over 300 evenly thinned draws.

| rule | variant | draws evaluated | failed factorizations | range of d | half-oscillation delta | TV bound tanh(delta/2) |
|---|---|---|---|---|---|---|
| adaptive 0.9 | assembled | 300 | 0 | -0.240 to +0.328 | 0.284 | 0.141 |
| adaptive 0.9 | TLR Cholesky | 300 | 0 | -0.845 to -0.265 | 0.290 | 0.144 |
| fixed 1e-4 | assembled | 290 | 10 | -68.631 to +1.298 | 34.965 | 1.000 |
| fixed 1e-4 | TLR Cholesky | 293 | 7 | -589.205 to +1.755 | 295.480 | 1.000 |

The adaptive band widens by 0.02 with 100 more draws (0.13 to 0.14), the fixed tolerance's swing grows from 31 to 591
units with one more failed draw per 100; the proposal quotes the 200-draw study.

### 16. Is the budget-chosen precision needed, and does inference with the rounded factors match dense inference? (added 2026-10-24; `codes/18` `--mp_eps/--blr_eps/--extra_rows`, `codes/39` same flags, `codes/40_joint_fit_pilot.py`, `codes/41_windgust_fit.py`, `codes/38 --rule joint:tau|rtp:tau`; `joint_tol/`, `joint_windgust_tol/`, `joint_fit*/`, `windgust_fit/`, `mcmc_tlrchol/joint_0.9/`, `mcmc_tlrchol/rtp_0.9/`)

Reviewers of the 2026-10-24 draft asked (a) whether the per-tile precision choice is needed at all, or whether one fixed precision or a rounding tolerance would do, (b) how the rule compares with block mixed precision (full-rank tiles in the lowest admissible precision, the construction of the SC26 paper [27]) under the same budget, and (c) whether inference with the rounded factors, not only the log-likelihood at the true parameters, matches dense inference. This section answers all three.

**16a. Precision rules compared.** Same fields and rules as sections 6 and 13 at N = 10,000, tile 250, tau = 0.9, plus: `uniform_16` / `uniform_32` (the rank rule's factors all stored in FP16 / FP32); `mp_tol_<eps>` (the rank rule's rank, then the lowest precision whose rounding error is within eps ||T||_F); `blr_<eps>` (rank and precision both from one relative tolerance eps, as in mixed-precision BLR [53]); `block_mp` (rank b, lowest precision meeting the budget e_max). Log-likelihood difference and held-out scores at the true parameters as in section 6. Off-diagonal bytes as a fraction of dense FP64 off-diagonal bytes; the diagonal tiles stay dense in FP64 for every rule. Nuggets 0.005 to 0.2 are the regimes of scientific interest.


Matern 3/2, nugget 0.005, range 0.1, N = 10,000 (e_max = 1.12e-04):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 8.05% | 1.000 | 0/0/0/780 | yes | +0.15 | 0.0798 / 0.0443 / 0.923 |
| rank then lowest admissible precision (budget) | 3.83% | 0.476 | 29/226/522/3 | yes | +0.13 | 0.0798 / 0.0443 / 0.923 |
| joint rank and precision (budget) | 3.81% | 0.473 | 30/249/498/3 | yes | +0.11 | 0.0798 / 0.0443 / 0.923 |
| rank rule, every factor in FP16 | 2.01% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, every factor in FP32 | 4.02% | 0.500 | 0/0/780/0 | yes | +0.14 | 0.0798 / 0.0443 / 0.923 |
| rank rule, precision by relative tolerance 1e-2 | 2.03% | 0.252 | 0/752/0/28 | no | fails |  |
| rank rule, precision by relative tolerance 1e-3 | 2.29% | 0.285 | 0/610/0/170 | no | fails |  |
| rank rule, precision by relative tolerance 1e-4 | 4.45% | 0.553 | 0/0/503/277 | yes | +0.14 | 0.0798 / 0.0443 / 0.923 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.61% | 0.075 | 0/779/1/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 1.09% | 0.135 | 0/751/29/0 | no | fails |  |
| one relative tolerance 1e-4 (BLR) | 3.04% | 0.377 | 0/0/780/0 | yes | -18.83 | 0.0805 / 0.0446 / 0.923 |
| one relative tolerance 1e-5 (BLR) | 4.24% | 0.527 | 0/0/780/0 | yes | +0.34 | 0.0799 / 0.0443 / 0.923 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 39.95% | 4.965 | 39/255/486/0 | yes | +0.03 | 0.0798 / 0.0443 / 0.923 |
| tolerance 1e-4 rank, FP64 (current practice) | 6.07% | 0.754 | 0/0/0/780 | yes | -18.83 | 0.0805 / 0.0446 / 0.923 |

Matern 3/2, nugget 0.005, range 0.3, N = 10,000 (e_max = 1.12e-04):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 7.93% | 1.000 | 0/0/0/780 | yes | +0.08 | 0.0677 / 0.0378 / 0.923 |
| rank then lowest admissible precision (budget) | 3.96% | 0.500 | 0/0/780/0 | yes | +0.08 | 0.0677 / 0.0378 / 0.923 |
| joint rank and precision (budget) | 3.96% | 0.500 | 0/0/780/0 | yes | +0.08 | 0.0677 / 0.0378 / 0.923 |
| rank rule, every factor in FP16 | 1.98% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, every factor in FP32 | 3.96% | 0.500 | 0/0/780/0 | yes | +0.08 | 0.0677 / 0.0378 / 0.923 |
| rank rule, precision by relative tolerance 1e-2 | 1.98% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, precision by relative tolerance 1e-3 | 1.98% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, precision by relative tolerance 1e-4 | 3.96% | 0.500 | 0/0/780/0 | yes | +0.08 | 0.0677 / 0.0378 / 0.923 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.45% | 0.056 | 0/780/0/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 0.77% | 0.097 | 0/765/15/0 | no | fails |  |
| one relative tolerance 1e-4 (BLR) | 2.08% | 0.263 | 0/0/780/0 | no | fails |  |
| one relative tolerance 1e-5 (BLR) | 2.97% | 0.375 | 0/0/780/0 | yes | -5.04 | 0.0678 / 0.0379 / 0.923 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 50.00% | 6.306 | 0/0/780/0 | yes | -0.01 | 0.0676 / 0.0378 / 0.923 |
| tolerance 1e-4 rank, FP64 (current practice) | 4.16% | 0.525 | 0/0/0/780 | no | fails |  |

Matern 3/2, nugget 0.05, range 0.1, N = 10,000 (e_max = 1.12e-03):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 5.67% | 1.000 | 0/0/0/780 | yes | +0.17 | 0.2192 / 0.1222 / 0.930 |
| rank then lowest admissible precision (budget) | 2.50% | 0.441 | 88/348/326/18 | yes | +0.17 | 0.2192 / 0.1222 / 0.930 |
| joint rank and precision (budget) | 2.46% | 0.434 | 89/379/294/18 | yes | +0.10 | 0.2192 / 0.1222 / 0.930 |
| rank rule, every factor in FP16 | 1.42% | 0.250 | 0/780/0/0 | yes | +0.82 | 0.2193 / 0.1223 / 0.927 |
| rank rule, every factor in FP32 | 2.84% | 0.500 | 0/0/780/0 | yes | +0.17 | 0.2192 / 0.1222 / 0.930 |
| rank rule, precision by relative tolerance 1e-2 | 1.51% | 0.267 | 0/655/0/125 | yes | +0.82 | 0.2193 / 0.1223 / 0.927 |
| rank rule, precision by relative tolerance 1e-3 | 1.84% | 0.325 | 0/482/4/294 | yes | +0.81 | 0.2193 / 0.1223 / 0.927 |
| rank rule, precision by relative tolerance 1e-4 | 3.61% | 0.637 | 0/0/286/494 | yes | +0.17 | 0.2192 / 0.1222 / 0.930 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.61% | 0.107 | 0/779/1/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 1.09% | 0.191 | 0/751/29/0 | yes | -22.66 | 0.2216 / 0.1237 / 0.933 |
| one relative tolerance 1e-4 (BLR) | 3.04% | 0.535 | 0/0/780/0 | yes | -0.69 | 0.2192 / 0.1222 / 0.930 |
| one relative tolerance 1e-5 (BLR) | 4.24% | 0.748 | 0/0/780/0 | yes | +0.14 | 0.2192 / 0.1222 / 0.930 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 31.86% | 5.618 | 142/353/285/0 | yes | -0.10 | 0.2192 / 0.1222 / 0.930 |
| tolerance 1e-4 rank, FP64 (current practice) | 6.07% | 1.071 | 0/0/0/780 | yes | -0.69 | 0.2192 / 0.1222 / 0.930 |

Matern 3/2, nugget 0.05, range 0.3, N = 10,000 (e_max = 1.12e-03):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 5.66% | 1.000 | 0/0/0/780 | yes | -0.25 | 0.2058 / 0.1159 / 0.933 |
| rank then lowest admissible precision (budget) | 2.83% | 0.500 | 0/1/779/0 | yes | -0.25 | 0.2058 / 0.1159 / 0.933 |
| joint rank and precision (budget) | 2.83% | 0.500 | 0/3/777/0 | yes | -0.26 | 0.2058 / 0.1159 / 0.933 |
| rank rule, every factor in FP16 | 1.42% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, every factor in FP32 | 2.83% | 0.500 | 0/0/780/0 | yes | -0.25 | 0.2058 / 0.1159 / 0.933 |
| rank rule, precision by relative tolerance 1e-2 | 1.42% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, precision by relative tolerance 1e-3 | 1.42% | 0.250 | 0/780/0/0 | no | fails |  |
| rank rule, precision by relative tolerance 1e-4 | 2.84% | 0.502 | 0/0/774/6 | yes | -0.25 | 0.2058 / 0.1159 / 0.933 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.45% | 0.079 | 0/780/0/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 0.77% | 0.135 | 0/765/15/0 | no | fails |  |
| one relative tolerance 1e-4 (BLR) | 2.08% | 0.368 | 0/0/780/0 | yes | -0.89 | 0.2058 / 0.1160 / 0.937 |
| one relative tolerance 1e-5 (BLR) | 2.97% | 0.525 | 0/0/780/0 | yes | -0.44 | 0.2057 / 0.1159 / 0.937 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 49.78% | 8.790 | 0/7/773/0 | yes | +0.02 | 0.2057 / 0.1159 / 0.937 |
| tolerance 1e-4 rank, FP64 (current practice) | 4.16% | 0.735 | 0/0/0/780 | yes | -0.89 | 0.2058 / 0.1160 / 0.937 |

Matern 3/2, nugget 0.2, range 0.1, N = 10,000 (e_max = 4.50e-03):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 4.42% | 1.000 | 0/0/0/780 | yes | +0.06 | 0.4203 / 0.2359 / 0.927 |
| rank then lowest admissible precision (budget) | 1.83% | 0.414 | 140/361/237/42 | yes | +0.01 | 0.4202 / 0.2358 / 0.927 |
| joint rank and precision (budget) | 1.79% | 0.405 | 143/389/206/42 | yes | +0.04 | 0.4203 / 0.2359 / 0.927 |
| rank rule, every factor in FP16 | 1.10% | 0.250 | 0/780/0/0 | yes | +0.30 | 0.4203 / 0.2359 / 0.923 |
| rank rule, every factor in FP32 | 2.21% | 0.500 | 0/0/780/0 | yes | +0.06 | 0.4203 / 0.2359 / 0.927 |
| rank rule, precision by relative tolerance 1e-2 | 1.26% | 0.284 | 0/574/0/206 | yes | +0.30 | 0.4203 / 0.2359 / 0.923 |
| rank rule, precision by relative tolerance 1e-3 | 1.67% | 0.378 | 0/374/3/403 | yes | +0.30 | 0.4203 / 0.2359 / 0.923 |
| rank rule, precision by relative tolerance 1e-4 | 3.31% | 0.750 | 0/0/137/643 | yes | +0.06 | 0.4203 / 0.2359 / 0.927 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.61% | 0.137 | 0/779/1/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 1.09% | 0.246 | 0/751/29/0 | yes | -1.72 | 0.4206 / 0.2361 / 0.930 |
| one relative tolerance 1e-4 (BLR) | 3.04% | 0.687 | 0/0/780/0 | yes | -0.12 | 0.4203 / 0.2358 / 0.927 |
| one relative tolerance 1e-5 (BLR) | 4.24% | 0.961 | 0/0/780/0 | yes | +0.04 | 0.4203 / 0.2358 / 0.923 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 27.45% | 6.217 | 209/390/181/0 | yes | +0.02 | 0.4203 / 0.2359 / 0.927 |
| tolerance 1e-4 rank, FP64 (current practice) | 6.07% | 1.375 | 0/0/0/780 | yes | -0.12 | 0.4203 / 0.2358 / 0.927 |

Matern 3/2, nugget 0.2, range 0.3, N = 10,000 (e_max = 4.50e-03):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD | log-lik difference | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 4.59% | 1.000 | 0/0/0/780 | yes | -0.22 | 0.4085 / 0.2304 / 0.937 |
| rank then lowest admissible precision (budget) | 2.23% | 0.485 | 0/78/702/0 | yes | -0.31 | 0.4085 / 0.2304 / 0.937 |
| joint rank and precision (budget) | 2.22% | 0.484 | 0/87/693/0 | yes | -0.30 | 0.4085 / 0.2304 / 0.937 |
| rank rule, every factor in FP16 | 1.15% | 0.250 | 0/780/0/0 | yes | -1.12 | 0.4092 / 0.2309 / 0.940 |
| rank rule, every factor in FP32 | 2.30% | 0.500 | 0/0/780/0 | yes | -0.22 | 0.4085 / 0.2304 / 0.937 |
| rank rule, precision by relative tolerance 1e-2 | 1.15% | 0.250 | 0/780/0/0 | yes | -1.12 | 0.4092 / 0.2309 / 0.940 |
| rank rule, precision by relative tolerance 1e-3 | 1.15% | 0.250 | 0/780/0/0 | yes | -1.12 | 0.4092 / 0.2309 / 0.940 |
| rank rule, precision by relative tolerance 1e-4 | 2.44% | 0.531 | 0/0/696/84 | yes | -0.22 | 0.4085 / 0.2304 / 0.937 |
| one relative tolerance 1e-2 for rank and precision (BLR) | 0.45% | 0.097 | 0/780/0/0 | no | fails |  |
| one relative tolerance 1e-3 (BLR) | 0.77% | 0.167 | 0/765/15/0 | no | fails |  |
| one relative tolerance 1e-4 (BLR) | 2.08% | 0.453 | 0/0/780/0 | yes | +0.29 | 0.4085 / 0.2304 / 0.937 |
| one relative tolerance 1e-5 (BLR) | 2.97% | 0.648 | 0/0/780/0 | yes | -0.10 | 0.4085 / 0.2304 / 0.937 |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 45.42% | 9.889 | 0/143/637/0 | yes | +0.08 | 0.4084 / 0.2304 / 0.937 |
| tolerance 1e-4 rank, FP64 (current practice) | 4.16% | 0.907 | 0/0/0/780 | yes | +0.29 | 0.4085 / 0.2304 / 0.937 |

Wind-gust covariance (SC26 fitted parameters, relative nugget 0.93), N = 10,000, tile 250 (bytes and positive definiteness only; the covariance carries no residuals):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD |
|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 2.12% | 1.000 | 0/0/0/780 | yes |
| rank then lowest admissible precision (budget) | 0.50% | 0.236 | 77/92/0/611 | yes |
| joint rank and precision (budget) | 0.48% | 0.225 | 100/69/0/611 | yes |
| rank rule, every factor in FP16 | 0.53% | 0.250 | 0/780/0/0 | yes |
| rank rule, every factor in FP32 | 1.06% | 0.500 | 0/0/780/0 | yes |
| rank rule, precision by relative tolerance 1e-2 | 1.78% | 0.839 | 3/22/0/755 | yes |
| rank rule, precision by relative tolerance 1e-3 | 2.11% | 0.997 | 1/3/0/776 | yes |
| rank rule, precision by relative tolerance 1e-4 | 2.12% | 1.000 | 0/0/0/780 | yes |
| one relative tolerance 1e-2 for rank and precision (BLR) | 3.75% | 1.768 | 8/772/0/0 | yes |
| one relative tolerance 1e-3 (BLR) | 6.91% | 3.257 | 1/669/110/0 | yes |
| one relative tolerance 1e-4 (BLR) | 14.23% | 6.712 | 0/9/771/0 | yes |
| one relative tolerance 1e-5 (BLR) | 17.55% | 8.277 | 0/1/779/0 | yes |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 13.32% | 6.282 | 729/51/0/0 | yes |
| tolerance 1e-4 rank, FP64 (current practice) | 28.48% | 13.435 | 0/0/0/780 | yes |

Wind-gust covariance (SC26 fitted parameters, relative nugget 0.93), N = 20,000, tile 250 (bytes and positive definiteness only; the covariance carries no residuals):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD |
|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 1.60% | 1.000 | 0/0/0/3160 | yes |
| rank then lowest admissible precision (budget) | 0.40% | 0.249 | 118/254/2/2786 | yes |
| joint rank and precision (budget) | 0.39% | 0.241 | 143/231/0/2786 | yes |
| rank rule, every factor in FP16 | 0.40% | 0.250 | 0/3160/0/0 | yes |
| rank rule, every factor in FP32 | 0.80% | 0.500 | 0/0/3160/0 | yes |
| rank rule, precision by relative tolerance 1e-2 | 0.63% | 0.392 | 5/187/0/2968 | yes |
| rank rule, precision by relative tolerance 1e-3 | 1.59% | 0.995 | 0/9/0/3151 | yes |
| rank rule, precision by relative tolerance 1e-4 | 1.60% | 1.000 | 0/1/0/3159 | yes |
| one relative tolerance 1e-2 for rank and precision (BLR) | 3.88% | 2.429 | 17/3143/0/0 | yes |
| one relative tolerance 1e-3 (BLR) | 5.91% | 3.700 | 0/2844/316/0 | yes |
| one relative tolerance 1e-4 (BLR) | 12.72% | 7.967 | 0/16/3144/0 | yes |
| one relative tolerance 1e-5 (BLR) | 15.54% | 9.732 | 0/1/3159/0 | yes |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 13.37% | 8.370 | 2941/219/0/0 | yes |
| tolerance 1e-4 rank, FP64 (current practice) | 25.47% | 15.946 | 0/0/0/3160 | yes |

Wind-gust covariance (SC26 fitted parameters, relative nugget 0.93), N = 40,000, tile 250 (bytes and positive definiteness only; the covariance carries no residuals):

| rule | bytes / dense | / rank only | tiles FP8/FP16/FP32/FP64 | PD |
|---|---|---|---|---|
| rank rule, FP64 (Obj. 1) | 1.47% | 1.000 | 0/0/0/12720 | yes |
| rank then lowest admissible precision (budget) | 0.44% | 0.303 | 356/656/71/11637 | yes |
| joint rank and precision (budget) | 0.35% | 0.239 | 415/668/0/11637 | yes |
| rank rule, every factor in FP16 | 0.37% | 0.250 | 0/12720/0/0 | yes |
| rank rule, every factor in FP32 | 0.73% | 0.500 | 0/0/12720/0 | yes |
| rank rule, precision by relative tolerance 1e-2 | 0.54% | 0.368 | 2/600/0/12118 | yes |
| rank rule, precision by relative tolerance 1e-3 | 1.21% | 0.827 | 0/68/13/12639 | yes |
| rank rule, precision by relative tolerance 1e-4 | 1.47% | 0.999 | 0/1/7/12712 | yes |
| one relative tolerance 1e-2 for rank and precision (BLR) | 3.76% | 2.560 | 76/12641/3/0 | yes |
| one relative tolerance 1e-3 (BLR) | 5.22% | 3.558 | 0/11784/936/0 | yes |
| one relative tolerance 1e-4 (BLR) | 11.22% | 7.646 | 0/18/12702/0 | yes |
| one relative tolerance 1e-5 (BLR) | 13.46% | 9.175 | 0/0/12720/0 | yes |
| block mixed precision under the budget (full tiles, lowest admissible precision; the construction of [27]) | 13.13% | 8.951 | 12078/642/0/0 | yes |
| tolerance 1e-4 rank, FP64 (current practice) | 22.44% | 15.295 | 0/0/0/12720 | yes |

Findings (nuggets 0.005 to 0.2, N = 10,000): the budget rules (rank then precision, joint) are positive definite in all 12 cases and within 0.09 of the rank-only log-likelihood, with the same held-out scores. Every factor in FP16 is not positive definite in 3 of 6 settings and, where it is, moves the log-likelihood by 0.2 to 0.9 from rank-only. A relative rounding tolerance of 1e-2 or 1e-3 on the budget rank is not positive definite in 6 of 12 cases; 1e-4 is safe but stores 1.0 to 1.8 times the budget rule's bytes. One relative tolerance for rank and precision (the BLR rule) is not positive definite in 11 of 24 cases, and at nugget 0.005 no tolerance from 1e-2 to 1e-5 is both positive definite and within 5 of the exact log-likelihood at both ranges. Block mixed precision under the same budget is accurate (within 0.1) but stores 10 to 20 times the budget rule's bytes on the Matern fields and 27 to 34 times on the wind-gust covariance. Every factor in FP32 is safe here and coincides with the budget rule's choice at ranges 0.3 and 1; the budget rule finds the FP8 and FP16 tiles at range 0.1 (bytes 0.41 to 0.47 of rank-only against 0.50) and on the wind gusts (0.21 to 0.25 against 0.50).

**16b. Maximum likelihood with the rounded factors re-chosen at every candidate theta** (`codes/40_joint_fit_pilot.py`; same field, seed, dispersed start theta_true * (1.5, 0.6, 2.0) and Nelder-Mead settings as section 1; e_max follows theta through m(theta) = sigma2 eta, so ranks and precisions change along the path). Estimates relative to the dense fit, exact log-likelihood at each estimate relative to the dense fit's, held-out scores (300 points), bytes at the estimate.

| field | fit | sigma2, range, nugget | estimate vs dense | exact log-lik loss vs dense fit | RMSE / CRPS / 90% cover | off-diag bytes / dense | precision mix (tiles) | evals, failures |
|---|---|---|---|---|---|---|---|---|
| Matern, range 0.3, N = 10,000 | dense | 1.3940, 0.3239, 0.0360 | reference | 0.000 | 0.2056 / 0.1158 / 0.937 | 100% |  | 80, 0 |
| Matern, range 0.3, N = 10,000 | rank_only | 1.3956, 0.3263, 0.0360 | within 0.7% | 0.021 | 0.2056 / 0.1158 / 0.937 | 5.88% | 0/0/0/780 | 80, 0 |
| Matern, range 0.3, N = 10,000 | joint | 1.3956, 0.3263, 0.0360 | within 0.7% | 0.021 | 0.2056 / 0.1158 / 0.937 | 2.94% | 0/0/780/0 | 81, 0 |
| Matern, range 0.1, N = 10,000 | dense | 1.3713, 0.1146, 0.0370 | reference | 0.000 | 0.2190 / 0.1222 / 0.930 | 100% |  | 85, 0 |
| Matern, range 0.1, N = 10,000 | rank_only | 1.3631, 0.1142, 0.0373 | within 0.6% | 0.001 | 0.2190 / 0.1222 / 0.930 | 6.13% | 0/0/0/780 | 86, 0 |
| Matern, range 0.1, N = 10,000 | joint | 1.3772, 0.1145, 0.0369 | within 0.4% | 0.018 | 0.2190 / 0.1222 / 0.930 | 2.77% | 40/345/392/3 | 88, 0 |
| Matern, range 0.3, N = 20,000 | dense | 1.3603, 0.3364, 0.0361 | reference | 0.000 | 0.2294 / 0.1297 / 0.893 | 100% |  | 89, 0 |
| Matern, range 0.3, N = 20,000 | rank_only | 1.3747, 0.3365, 0.0357 | within 1.2% | 0.030 | 0.2294 / 0.1297 / 0.893 | 4.86% | 0/0/0/3160 | 92, 0 |
| Matern, range 0.3, N = 20,000 | joint | 1.3747, 0.3365, 0.0357 | within 1.2% | 0.030 | 0.2294 / 0.1297 / 0.893 | 2.43% | 0/0/3160/0 | 92, 0 |

The compressed fits reach the dense estimates to within 1.2% on every parameter with the same held-out scores, lose at most 0.03 in exact log-likelihood, never fail to factorize, and take the same number of evaluations as the dense fit. The joint rule's factor is 2.4 to 2.9% of the dense off-diagonal bytes (0.45 to 0.50 of rank-only).

**16c. Wind-gust maximum likelihood with the rounded factors** (`codes/41_windgust_fit.py`): random subset of N = 10,000 station-hours of the SC26 record (seed 20261020, the subset of section 13) plus 300 held-out station-hours; gust = linear mean in the 17 WRF predictors of the SC26 application (ordinary least squares on the subset, held fixed; R^2 = 0.31) + space-time GP residual with the Gneiting-Matern(3/2, beta = 0) covariance of section 13 and a nugget, theta = (sigma2, r_s km, r_t h, relative nugget), fitted by Nelder-Mead from the SC26 estimates (2.07, 19.7, 3.61, 0.93) with up to three restarts from the result (the standard remedy for premature Nelder-Mead termination, applied to every rule, dense included). The likelihood ridge between sigma2 and the nugget is flat under the linear mean, and the first Nelder-Mead pass of the two precision rules stalled 14 units short of the optimum before the first restart carried them to it; the dense fit itself needed 292 evaluations. Rows `warm`: the same fits started at the dense estimate; row `at dense estimate`: each rule's log-likelihood and prediction evaluated at the dense estimate without fitting.

| start | fit | sigma2, r_s, r_t, nugget | own log-lik at estimate | RMSE / CRPS / 90% cover | off-diag bytes / dense | tiles FP8/FP16/FP32/FP64 (rank 0 counted as FP64) | mean rank | evals, failures |
|---|---|---|---|---|---|---|---|---|
| SC26 estimates | dense | 1.725, 82.430, 4.628, 1.882 | -21690.761 | 1.9379 / 1.0800 / 0.913 | 100% |  |  | 292, 0 |
| SC26 estimates | rank_only | 1.725, 83.011, 4.636, 1.884 | -21690.751 | 1.9382 / 1.0802 / 0.913 | 5.01% | 0/0/0/780 | 6.3 | 346, 0 |
| SC26 estimates | rank_then_precision | 1.780, 78.693, 4.590, 1.797 | -21690.706 | 1.9355 / 1.0786 / 0.913 | 1.23% | 42/159/1/578 | 6.2 | 360, 0 |
| SC26 estimates | joint | 1.717, 81.175, 4.746, 1.893 | -21690.725 | 1.9363 / 1.0793 / 0.913 | 1.21% | 45/157/0/578 | 6.2 | 399, 0 |
| warm (dense estimate) | rank_only | 1.725, 82.941, 4.626, 1.884 | -21690.753 | 1.9382 / 1.0802 / 0.913 | 5.01% | 0/0/0/780 | 6.3 | 145, 0 |
| warm (dense estimate) | rank_then_precision | 1.720, 83.444, 4.668, 1.892 | -21690.729 | 1.9384 / 1.0803 / 0.913 | 1.26% | 40/161/1/578 | 6.3 | 150, 0 |
| warm (dense estimate) | joint | 1.711, 83.803, 4.692, 1.907 | -21690.699 | 1.9384 / 1.0804 / 0.913 | 1.24% | 47/155/0/578 | 6.3 | 151, 0 |
| at dense estimate | dense | 1.725, 82.441, 4.628, 1.882 | -21690.761 | 1.9379 / 1.0800 / 0.913 | | | | 1, 0 |
| at dense estimate | rank_only | 1.725, 82.441, 4.628, 1.882 | -21690.774 | 1.9379 / 1.0800 / 0.913 | | | | 1, 0 |
| at dense estimate | rank_then_precision | 1.725, 82.441, 4.628, 1.882 | -21690.774 | 1.9379 / 1.0801 / 0.913 | | | | 1, 0 |
| at dense estimate | joint | 1.725, 82.441, 4.628, 1.882 | -21690.767 | 1.9379 / 1.0801 / 0.913 | | | | 1, 0 |

On the real data every compressed fit, from the SC26 start or from the dense estimate, ends at the dense optimum (own log-likelihood within 0.06 of the dense 21690.76; the dense fit's held-out RMSE 1.938 m/s, CRPS 1.080 m/s and coverage 0.913 are reproduced to the third digit), with 0 factorization failures, the two precision rules at 1.2% of the dense off-diagonal bytes (45 tiles in FP8, 157 in FP16, 578 dropped at rank 0 for the joint rule) against 5.0% for the rank rule alone. The parameters along the ridge differ by up to 5% between fits with the same log-likelihood, a property of the data under the linear mean, not of the approximation. The fitted nugget (1.8 to 1.9 relative) is larger than the SC26 value (0.93) because the linear mean explains less than the SC26 neural-network mean. N = 20,000 fits and the rounded-factor chains (16d) are added below as they finish.

Reproduce (from codes/, Storrs):
  for eta in 0.005 0.05 0.2; do for r in 0.3 0.1; do sbatch --job-name=joint-tol-eta$eta-r$r storrs_cpu.sbatch "18_joint_rank_precision_pilot.py --n 10000 --eta $eta --range $r --mp_eps 1e-2,1e-3,1e-4 --blr_eps 1e-2,1e-3,1e-4,1e-5 --extra_rows --out ../results/joint_tol/eta${eta}_r$r"; done; done
  for n in 10000 20000 40000; do sbatch --job-name=gust-tol-n$n storrs_cpu_big.sbatch "39_joint_windgust.py --n $n --b 250 --mp_eps 1e-2,1e-3,1e-4 --blr_eps 1e-2,1e-3,1e-4,1e-5 --extra_rows --out ../results/joint_windgust_tol"; done
  sbatch --job-name=joint-fit-n10000 storrs_cpu.sbatch "40_joint_fit_pilot.py --n 10000 --out ../results/joint_fit"; sbatch --job-name=joint-fit-n10000-r01 storrs_cpu.sbatch "40_joint_fit_pilot.py --n 10000 --range 0.1 --out ../results/joint_fit_range0.1"; sbatch --job-name=joint-fit-n20000 storrs_cpu_big.sbatch "40_joint_fit_pilot.py --n 20000 --out ../results/joint_fit"
  for rule in dense rank_only rank_then_precision joint; do sbatch --job-name=gust-fit-n10000-$rule storrs_cpu.sbatch "41_windgust_fit.py --n 10000 --rules $rule --restarts 3 --out ../results/windgust_fit/$rule"; done
  for rule in rank_only rank_then_precision joint; do sbatch --job-name=gust-fitw-n10000-$rule storrs_cpu.sbatch "41_windgust_fit.py --n 10000 --rules $rule --restarts 2 --start 1.7253,82.4407,4.6283,1.8822 --out ../results/windgust_fit/warm/$rule"; done
  sbatch --job-name=gust-eval-n10000 storrs_cpu.sbatch "41_windgust_fit.py --n 10000 --rules dense,rank_only,rank_then_precision,joint --eval_at 1.7253,82.4407,4.6283,1.8822 --out ../results/windgust_fit/eval"
  (chains, 16d) k=1; for cfg in "101 1.3,0.8,1.5" "102 0.6,1.4,0.5" "103 2.0,0.5,2.0" "104 0.8,1.2,0.8" "105 1.3,0.8,1.5" "106 0.6,1.4,0.5" "107 2.0,0.5,2.0" "108 0.8,1.2,0.8"; do set -- $cfg; for rule in joint rtp; do sbatch --requeue --job-name=mcmc-$rule-c$k storrs_cpu.sbatch "38_tlr_cholesky_mcmc.py --n 5000 --rule $rule:0.9 --iters 4000 --burn 1000 --chain_seed $1 --start_mult $2 --out ../results/mcmc_tlrchol/${rule}_0.9/c$k"; done; k=$((k+1)); done
  (per evaluation at N = 10,000: dense 3 to 5 s, compressed 15 to 30 s on 16 cores; the chains run 4 to 11 s per iteration on shared nodes and resume from ckpt_n5000.npz when resubmitted)

**16d. Metropolis-Hastings driven by the rounded factors** (`codes/38_tlr_cholesky_mcmc.py --rule joint:0.9`; same data, prior, dispersed starts and self-tuned proposal as section 14; N = 5,000, b = 250; 4,000 iterations, burn-in 1,000; the rank and the storage precision of every tile re-chosen under the budget at every proposal, Sigma^TLR assembled from the rounded factors and factorized densely). Four of the eight chains launched had finished by 14:00 on 2026-10-24 (c2, c5, c6, c7; 4.1 to 5.4 s per iteration on shared 16-core nodes); the others, and the eight `rtp:0.9` (rank then lowest admissible precision) chains, are added when they finish. Pooled over the four chains (12,000 draws) against the exact posterior of section 14 (four dense chains, 32,000 draws):

| | sigma2 | range | nugget |
|---|---|---|---|
| pooled mean, rounded factors | 1.527 | 0.3474 | 0.0411 |
| pooled mean, exact | 1.485 | 0.3456 | 0.0420 |
| mean difference in exact posterior SD | +0.05 | +0.03 | -0.05 |
| SD ratio over exact | 1.05 | 1.03 | 0.99 |
| split R-hat / bulk ESS (4 chains) | 1.016 / 254 | 1.014 / 256 | 1.015 / 252 |

Monte Carlo standard error of the pooled means, in exact posterior SD: 0.06 for every parameter, so the 0.03 to 0.05 differences are within Monte Carlo error. Acceptance 0.30 to 0.32, 0 factorization failures in 16,000 proposals, held-out RMSE 0.234 / CRPS 0.132 / 90% coverage 0.94 at the posterior mean (the exact chains: 0.234 / 0.132 / 0.94), stored lower triangle including the dense FP64 diagonal tiles 0.13 of the dense factor at the posterior mean. For reference the rank-only TLR Cholesky chains of section 14 differ from exact by 0.02 to 0.03 SD with SD ratios 0.98 to 0.99 at ESS near 750. Outputs: `mcmc_tlrchol/joint_0.9/c*/`, `logs/mcmc-joint-c*.out`.

**16e. The rounded factorization with recompression** (`codes/17_tlr_cholesky_pilot.py --rules adaptive:0.9,mixed:0.9,fixed:1e-4`; rule `mixed` = the adaptive rank rule plus the storage precision of every low-rank factor pair chosen under the same budget, FP8 E4M3 / FP16 / FP32 / FP64, rounded at the initial compression, after every recompression and after every panel solve; FP64 arithmetic; the budget left for rounding is e_max minus the truncation error of the tile, and e_max for the factor tiles after the panel solve). Nothing dense is formed; this is the Objective 3 algorithm inside the TLR Cholesky factorization of section 4, the first time anything rounded went through a recompressing factorization in these pilots. Same field and seed as section 4. Factor bytes count the dense FP64 diagonal tiles and 2 b r p / 8 per off-diagonal factor tile, relative to the dense FP64 factor (N(N+1)/2 values x 8 bytes).

| N | rule | completed | factor bytes / dense factor | factor tiles FP8/FP16/FP32/FP64 | mean rank | log-lik difference at theta_true | RMSE / CRPS / 90% cover |
|---|---|---|---|---|---|---|---|
| 10,000 | adaptive 0.9, FP64 | yes | 9.3% | 0/0/0/780 | 5.6 | +0.26 | 0.2058 / 0.1159 / 0.937 |
| 10,000 | mixed 0.9 | yes | 6.5% | 49/545/186/0 | 5.8 | +0.43 | 0.2058 / 0.1159 / 0.930 |
| 10,000 | fixed 1e-4, FP64 | yes | 11.3% | 0/0/0/780 | 8.1 | -2.59 | 0.2056 / 0.1159 / 0.937 |

Inside the factorization most factor tiles qualify for FP16 (their norms are smaller than those of the covariance tiles), so the rounded factor takes 0.70 of the rank-only factor's bytes, less than the 0.5 of the explicitly formed matrix would suggest only because the dense FP64 diagonal tiles (2.5% of the dense factor at N = 10,000, b = 250) are now a third of what is stored. The rounding adds 0.17 to the log-likelihood difference (0.43 against 0.26, both within the tau/sqrt(2) = 0.64 scale of section 12 for this field) and changes the 90% coverage by two of the 300 held-out points. N = 20,000 and 40,000 rows and the maximum likelihood fit with the mixed rule (`tlr_cholesky_mixed/tlr_cholesky_n{N}.json`, jobs tlrchol-mixed-n{10000,20000,40000}) are added as they finish. Outputs: `tlr_cholesky_mixed/rules_only/tlr_cholesky_n10000.json`, `logs/tlrchol-mixed-*.out`.

**16c, N = 20,000 (added as they finished).** Dense fit from the SC26 start (240 evaluations, restart confirmed): theta = (2.617, 34.490, 5.389, 0.895), own log-likelihood -42732.52, held-out RMSE 1.970 / CRPS 1.084 / 90% coverage 0.883. Joint rule started at the dense estimate: theta = (2.617, 34.490, 5.388, 0.895), own log-likelihood -42733.12, RMSE 1.970 / CRPS 1.084 / coverage 0.883, 0.56% of the dense off-diagonal bytes, tiles FP8/FP16/FP32/FP64 107/332/0/2721, 118 evaluations, 0 failures. The cold-start compressed fits at N = 20,000 (about 100 s per evaluation on shared nodes) and the remaining warm-start fits are added when they finish (`windgust_fit/*/windgust_fit_n20000.json`).
