#!/usr/bin/env Rscript
# Vecchia log-likelihood on the parameter grid of 27_inference_headtohead.py (2026-10-21).
# Ungrouped (vecchia_meanzero_loglik) and grouped (vecchia_grouped_meanzero_loglik, Guinness 2018) response Vecchia,
# max-min ordering (8 seeded orderings), m = 30 and 60, zero mean as the exact likelihood in 24/27.  Writes vecchia_grid.csv in the folder.
# Usage:  Rscript 28_vecchia_grid.R ../results/screening_stress/nu25_r3_e005
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
dir <- commandArgs(trailingOnly = TRUE)[1]
st <- fromJSON(file.path(dir, "stress.json")); N <- st$settings$n; nu <- st$settings$nu
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"; sc <- sqrt(2 * nu)
obs <- read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))
locs <- as.matrix(obs[, c("x", "y")]); y <- obs$z
G <- read.csv(file.path(dir, "grid.csv"))
# order_maxmin is randomized: one column per ordering, drawn with set.seed(1..K); 27 scores each and averages
K <- 8L
out <- data.frame(row = seq_len(nrow(G)))
for (k in seq_len(K)) {
  set.seed(k); ord <- order_maxmin(locs); lo <- locs[ord, ]; yo <- y[ord]
  for (m in c(30L, 60L)) {
    NN <- find_ordered_nn(lo, m); NNl <- group_obs(NN)
    u <- g <- numeric(nrow(G))
    for (i in seq_len(nrow(G))) {
      cp <- c(G$sigma2[i], G$range[i] / sc, G$eta[i])
      u[i] <- vecchia_meanzero_loglik(cp, covfun, yo, lo, NN)$loglik
      g[i] <- vecchia_grouped_meanzero_loglik(cp, covfun, yo, lo, NNl)$loglik
    }
    out[[sprintf("ungrouped_m%d_o%d", m, k)]] <- u; out[[sprintf("grouped_m%d_o%d", m, k)]] <- g
  }
}
write.csv(out[, -1], file.path(dir, "vecchia_grid.csv"), row.names = FALSE)
cat(basename(dir), "Vecchia grid done\n")
