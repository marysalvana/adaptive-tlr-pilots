#!/usr/bin/env Rscript
# Vecchia (GpGp) baseline for the rank-adaptive TLR pilot (2026-10-20).
#
# Fits the same simulated Matern(3/2) field with nugget that 11_tlr_rank_rules.py writes to
# results/sim_n{N}_obs.csv, by Vecchia's approximation with m nearest neighbors
# (GpGp, Guinness 2018), and predicts the same held-out points.  Reports the estimates of
# (sigma2, range, nugget), wall time, peak memory, and RMSE / CRPS / 90% coverage, so that
# the dense, fixed-tolerance TLR, adaptive TLR, and Vecchia fits can be read side by side.
#
# Usage:  Rscript 12_vecchia_baseline.R 10000 [m=30] [../results]
# Reproducible: GpGp's fit is deterministic given the data and the neighbor ordering.
suppressPackageStartupMessages(library(GpGp))
args <- commandArgs(trailingOnly = TRUE)
N   <- if (length(args) >= 1) as.integer(args[1]) else 10000L
m   <- if (length(args) >= 2) as.integer(args[2]) else 30L
dir <- if (length(args) >= 3) args[3] else "../results"

obs  <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))
hold <- read.csv(file.path(dir, sprintf("sim_n%d_holdout.csv", N)))
locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z
locs_p <- as.matrix(hold[, c("x", "y")]); yp <- hold$z

X <- matrix(1, nrow = N, ncol = 1); Xp <- matrix(1, nrow = nrow(locs_p), ncol = 1)   # intercept only
t0 <- proc.time()[["elapsed"]]
fit <- fit_model(y, locs, X = X, covfun_name = "matern15_isotropic",
                 m_seq = c(10L, m), silent = TRUE)
t_fit <- proc.time()[["elapsed"]] - t0
# GpGp's matern15_isotropic parameters: (variance, range, nugget) with nugget relative to variance
cp <- fit$covparms
# GpGp's range parameterization: cov = sigma2 (1 + d/range) exp(-d/range); the rank-rule script
# uses sqrt(3) d / r, so r_proposal = sqrt(3) * range_GpGp.
theta_hat <- c(sigma2 = cp[1], range = sqrt(3) * cp[2], eta = cp[3])

t1 <- proc.time()[["elapsed"]]
pred <- predictions(fit, locs_pred = locs_p, X_pred = Xp, m = m)
t_pred <- proc.time()[["elapsed"]] - t1
# predictive standard deviations from GpGp's conditional simulations
set.seed(20261020)
sims <- cond_sim(fit, locs_pred = locs_p, X_pred = Xp, m = m, nsims = 200)
sig <- apply(sims, 1, sd); mu <- rowMeans(sims)
z <- (yp - mu) / sig
crps <- mean(sig * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi)))
res <- list(N = N, m = m, theta_hat = as.list(theta_hat), loglik = fit$loglik,
            seconds_fit = t_fit, seconds_predict = t_pred,
            peak_memory_mb = as.numeric(sum(gc()[, 6])),
            skill = list(rmse = sqrt(mean((yp - mu)^2)), crps = crps,
                         cover90 = mean(abs(z) <= qnorm(0.95))),
            stored_values_vecchia = N * (m + 1),
            note = "Vecchia likelihood (GpGp matern15_isotropic, m nearest neighbors, max-min ordering); predictive sd from 200 conditional simulations")
cat(jsonlite::toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = 6), "\n",
    file = file.path(dir, sprintf("vecchia_n%d_m%d.json", N, m)))
cat(sprintf("Vecchia m=%d: theta_hat = (%.4f, %.4f, %.4f), fit %.0fs, RMSE %.4f, CRPS %.4f, cover90 %.3f\n",
            m, theta_hat[1], theta_hat[2], theta_hat[3], t_fit, res$skill$rmse, res$skill$crps, res$skill$cover90))
