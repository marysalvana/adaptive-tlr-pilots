#!/usr/bin/env Rscript
# Vecchia at the storage of the adaptive TLR factor (2026-10-21); companion of 24_screening_stress.py / 25.
#
# Vecchia stores N (m + 1) values; the adaptive TLR factor of 24 stores factor_frac * N (N + 1) / 2.  This script sets
# m_match = round(factor_frac * (N + 1) / 2 - 1) so that both use the same memory, and evaluates the Vecchia
# log-likelihood at the TRUE parameters (zero mean, max-min ordering, as in 25; mean over 8 seeded orderings) for m_match and 250, then
# predicts the held-out points (mean only) at m_match.  Writes vecchia_matched.json in the scenario folder.
#
# Usage:  Rscript 26_vecchia_matched_storage.R ../results/screening_stress/nu25_r1_e005
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
dir <- commandArgs(trailingOnly = TRUE)[1]
st <- fromJSON(file.path(dir, "stress.json"))
N <- st$settings$n; nu <- st$settings$nu; th0 <- st$theta_true
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"
cp0 <- c(th0[1], th0[2] / sqrt(2 * nu), th0[3])
obs <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))
hold <- read.csv(file.path(dir, sprintf("sim_n%d_holdout.csv", N)))
locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z
locs_p <- as.matrix(hold[, c("x", "y")]); yp <- hold$z
m_match <- as.integer(round(st$tlr_true$factor_frac * (N + 1) / 2 - 1))
K <- 8L                                    # order_maxmin is randomized: mean over orderings set.seed(1..K)
ords <- lapply(seq_len(K), function(s) { set.seed(s); order_maxmin(locs) })
res <- list(m_match = m_match, tlr_factor_frac = st$tlr_true$factor_frac, exact_loglik_true = st$exact_true$loglik, runs = list())
for (m in sort(unique(c(250L, m_match)))) {
  t0 <- proc.time()[["elapsed"]]
  lls <- sapply(ords, function(o) vecchia_meanzero_loglik(cp0, covfun, y[o], locs[o, , drop = FALSE],
                                                          find_ordered_nn(locs[o, ], m))$loglik)
  ll <- mean(lls); ex <- st$exact_true$loglik
  r <- list(m = m, stored_frac = 2 * (m + 1) / (N + 1), loglik = ll, loglik_err = ll - ex,
            loglik_err_min = min(lls) - ex, loglik_err_max = max(lls) - ex, orderings = K,
            seconds = (proc.time()[["elapsed"]] - t0) / K)
  if (m == m_match) {
    set.seed(1)
    mu <- predictions(locs_pred = locs_p, X_pred = matrix(1, nrow(locs_p), 1), y_obs = y, locs_obs = locs,
                      X_obs = matrix(1, N, 1), beta = 0, covparms = cp0, covfun_name = covfun, m = m)
    r$rmse <- sqrt(mean((yp - mu)^2))
  }
  res$runs[[length(res$runs) + 1]] <- r
  cat(sprintf("%s m=%d stored=%.3f loglik error=%.2f (%.0fs)\n", basename(dir), m, r$stored_frac, r$loglik_err, r$seconds))
}
write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(dir, "vecchia_matched.json"))
