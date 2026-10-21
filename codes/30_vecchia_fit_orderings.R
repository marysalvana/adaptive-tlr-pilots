#!/usr/bin/env Rscript
# Vecchia fits over seeded orderings (2026-10-21).  GpGp's fit_model orders the points with the randomized
# order_maxmin, so the Vecchia estimate depends on the ordering drawn.  For one screening field this fits Vecchia
# (fit_model, grouped, intercept, as in 12/25) with m = 30 and 60 under orderings set.seed(1..K) and writes the estimates to
# vecchia_fit_orderings.json; 27_inference_headtohead.py fitorders scores each on the exact likelihood.
# Usage:  Rscript 30_vecchia_fit_orderings.R ../results/screening_stress/nu25_r3_e005 [K=8]
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE); dir <- a[1]; K <- if (length(a) >= 2) as.integer(a[2]) else 8L
st <- fromJSON(file.path(dir, "stress.json")); N <- st$settings$n; nu <- st$settings$nu
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"; sc <- sqrt(2 * nu)
obs <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N))); locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z
X <- matrix(1, N, 1); out <- list()
for (m in c(30L, 60L)) for (k in seq_len(K)) {
  set.seed(k); t0 <- proc.time()[["elapsed"]]
  fit <- fit_model(y, locs, X = X, covfun_name = covfun, m_seq = c(10L, m), silent = TRUE)
  cp <- fit$covparms
  out[[length(out) + 1]] <- list(m = m, seed = k, theta_hat = list(sigma2 = cp[1], range = sc * cp[2], eta = cp[3]),
                                 seconds = proc.time()[["elapsed"]] - t0)
  cat(sprintf("%s m=%d seed=%d theta_hat=(%.4f, %.4f, %.4f)\n", basename(dir), m, k, cp[1], sc * cp[2], cp[3]))
}
write(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(dir, "vecchia_fit_orderings.json"))
