#!/usr/bin/env Rscript
# Zero-mean Vecchia fits over seeded orderings (2026-10-23).
#
# The exact and adaptive-TLR fits of 24/27 are zero-mean (three covariance parameters); the Vecchia fits of
# 25/30 used GpGp's fit_model, which profiles out an intercept.  Scoring an intercept fit on the zero-mean exact
# likelihood charges Vecchia for the mean it was allowed to fit.  This script fits the SAME zero-mean model by
# Vecchia's approximation, with GpGp's own ingredients: order_maxmin (randomized, so set.seed(1..K)),
# find_ordered_nn, group_obs (the grouped likelihood of Guinness 2018, fit_model's default), and
# vecchia_grouped_meanzero_loglik, maximized over log(sigma2, range, nugget) by optim (Nelder-Mead from GpGp's
# get_start_parms, then BFGS polish; both restarted until the likelihood stops improving).  Output has the same
# format as 30_vecchia_fit_orderings.R so that 27_inference_headtohead.py can score it on the exact likelihood.
# Usage:  Rscript 34_vecchia_fit_meanzero.R <field dir> [K=8] [out=vecchia_fit_orderings_meanzero.json]
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE); dir <- a[1]
K <- if (length(a) >= 2) as.integer(a[2]) else 8L
outname <- if (length(a) >= 3) a[3] else "vecchia_fit_orderings_meanzero.json"
st <- fromJSON(file.path(dir, "stress.json")); N <- st$settings$n; nu <- st$settings$nu
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"; sc <- sqrt(2 * nu)
obs <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N))); locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z

# starting values as fit_model uses them (get_start_parms), on the GpGp scale (range = r / sqrt(2 nu))
sp <- get_start_parms(y, matrix(1, N, 1), locs, covfun)$start_parms
sp <- sp[1:3]                                     # (variance, range, nugget) for the isotropic Matern kernels

fit_one <- function(ord, m) {
  yo <- y[ord]; lo <- locs[ord, , drop = FALSE]
  NNarray <- find_ordered_nn(lo, m); NNlist <- group_obs(NNarray)
  nll <- function(lp) { v <- -vecchia_grouped_meanzero_loglik(exp(lp), covfun, yo, lo, NNlist)$loglik
                        if (!is.finite(v)) 1e10 else v }
  x <- log(sp); best <- nll(x)
  for (it in 1:8) {                               # Nelder-Mead restarts until no improvement, then BFGS polish
    r <- optim(x, nll, method = "Nelder-Mead", control = list(maxit = 2000, reltol = 1e-10))
    improved <- r$value < best - 1e-6; x <- r$par; best <- r$value
    if (!improved) break
  }
  r <- try(optim(x, nll, method = "BFGS", control = list(maxit = 200, reltol = 1e-12)), silent = TRUE)
  if (!inherits(r, "try-error") && is.finite(r$value) && r$value <= best) { x <- r$par; best <- r$value }
  list(covparms = exp(x), loglik = -best)
}

out <- list()
for (m in c(30L, 60L)) for (k in seq_len(K)) {
  set.seed(k); ord <- order_maxmin(locs); t0 <- proc.time()[["elapsed"]]
  f <- fit_one(ord, m); cp <- f$covparms
  out[[length(out) + 1]] <- list(m = m, seed = k, mean = "zero",
                                 theta_hat = list(sigma2 = cp[1], range = sc * cp[2], eta = cp[3]),
                                 vecchia_loglik = f$loglik, seconds = proc.time()[["elapsed"]] - t0)
  cat(sprintf("%s m=%d seed=%d zero-mean theta_hat=(%.4f, %.4f, %.4f) vecchia loglik=%.3f (%.0fs)\n",
              basename(dir), m, k, cp[1], sc * cp[2], cp[3], f$loglik, proc.time()[["elapsed"]] - t0))
}
write(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(dir, outname))
