#!/usr/bin/env Rscript
# Vecchia side of the screening-stress head-to-head (2026-10-21); companion of 24_screening_stress.py.
#
# For one scenario folder written by 24 (sim_n{N}_obs.csv, sim_n{N}_holdout.csv, stress.json):
#  (1) at the TRUE parameters, the Vecchia log-likelihood (GpGp, max-min ordering, zero mean, as the exact one in 24)
#      for m = 5, 10, 20, 30, 60, 120 neighbors, and Vecchia prediction of the held-out points (mean from
#      predictions(), sd from conditional simulations), so the likelihood error and the predictive scores can be
#      read against m with the parameters held fixed;
#  (2) Vecchia maximum-likelihood fits with m = 30 and 60 (intercept, as in 12_vecchia_baseline.R).
# Parameterization: 24 uses x = sqrt(2 nu) d / r; GpGp uses d / a, so a = r / sqrt(2 nu); nugget relative in both.
#
# Usage:  Rscript 25_vecchia_stress.R ../results/screening_stress/nu25_r1_e005
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
st <- fromJSON(file.path(dir, "stress.json"))
N <- st$settings$n; nu <- st$settings$nu; th0 <- st$theta_true
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"
scale <- sqrt(2 * nu)
obs  <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))
hold <- read.csv(file.path(dir, sprintf("sim_n%d_holdout.csv", N)))
locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z
locs_p <- as.matrix(hold[, c("x", "y")]); yp <- hold$z
X <- matrix(1, N, 1); Xp <- matrix(1, nrow(locs_p), 1)
cp0 <- c(th0[1], th0[2] / scale, th0[3])

skill <- function(mu, sig) {
  z <- (yp - mu) / sig
  list(rmse = sqrt(mean((yp - mu)^2)),
       crps = mean(sig * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi))),
       cover90 = mean(abs(z) <= qnorm(0.95)))
}

# GpGp's order_maxmin is randomized: the Vecchia log-likelihood at fixed parameters depends on the ordering drawn.
# Every value below is the mean over orderings drawn with set.seed(1..K); min and max are kept.
K <- 8L
ords <- lapply(seq_len(K), function(s) { set.seed(s); order_maxmin(locs) })
fixed <- list()
for (m in c(5L, 10L, 20L, 30L, 60L, 120L)) {
  t0 <- proc.time()[["elapsed"]]
  lls <- sapply(ords, function(o) vecchia_meanzero_loglik(cp0, covfun, y[o], locs[o, , drop = FALSE],
                                                          find_ordered_nn(locs[o, ], m))$loglik)
  ll <- mean(lls)
  t_ll <- (proc.time()[["elapsed"]] - t0) / K
  set.seed(1)
  mu <- predictions(locs_pred = locs_p, X_pred = Xp, y_obs = y, locs_obs = locs, X_obs = X, beta = 0,
                    covparms = cp0, covfun_name = covfun, m = m)
  set.seed(20261020)
  sims <- cond_sim(locs_pred = locs_p, X_pred = Xp, y_obs = y, locs_obs = locs, X_obs = X, beta = 0,
                   covparms = cp0, covfun_name = covfun, m = m, nsims = 200)
  fixed[[length(fixed) + 1]] <- list(m = m, loglik = ll, loglik_min = min(lls), loglik_max = max(lls), orderings = K,
                                     seconds_loglik = t_ll,
                                     skill = skill(mu, apply(sims, 1, sd)))
  cat(sprintf("%s m=%3d loglik(true)=%.3f (%.1fs)\n", basename(dir), m, ll, t_ll))
}

fits <- list()
for (m in c(30L, 60L)) {
  t0 <- proc.time()[["elapsed"]]
  set.seed(1)
  fit <- fit_model(y, locs, X = X, covfun_name = covfun, m_seq = c(10L, m), silent = TRUE)
  cp <- fit$covparms
  mu <- predictions(fit, locs_pred = locs_p, X_pred = Xp, m = m)
  set.seed(20261020)
  sims <- cond_sim(fit, locs_pred = locs_p, X_pred = Xp, m = m, nsims = 200)
  fits[[length(fits) + 1]] <- list(m = m, theta_hat = list(sigma2 = cp[1], range = scale * cp[2], eta = cp[3]),
                                   beta = fit$betahat, vecchia_loglik = fit$loglik,
                                   seconds = proc.time()[["elapsed"]] - t0, skill = skill(mu, apply(sims, 1, sd)))
  cat(sprintf("%s fit m=%d theta_hat=(%.4f, %.4f, %.4f)\n", basename(dir), m, cp[1], scale * cp[2], cp[3]))
}
write(toJSON(list(fixed_theta = fixed, fits = fits), auto_unbox = TRUE, pretty = TRUE, digits = 8),
      file.path(dir, "vecchia.json"))
