#!/usr/bin/env Rscript
# KL divergence of Vecchia's approximation from the exact Gaussian model (2026-10-21); companion of 31_kl_tlr.py.
# At the true parameters, KL(N(0, Sigma) || N(0, Sigma_tilde)) = 0.5 [tr(Sigma_tilde^{-1} Sigma) - N + log det Sigma_tilde
# - log det Sigma], with Sigma_tilde^{-1} = B^T B and B the sparse inverse Cholesky factor of GpGp's vecchia_Linv (response
# Vecchia, max-min ordering), the measure of Guinness (2018) and Katzfuss and Guinness (2021).  The max-min ordering is
# randomized, so each m is evaluated under orderings set.seed(1..K); mean, min and max are kept.  Writes kl_vecchia.json.
# Usage:  Rscript 32_kl_vecchia.R ../results/screening_stress/nu25_r3_e005 [K=8] [matched]
suppressPackageStartupMessages({library(GpGp); library(jsonlite); library(Matrix)})
a <- commandArgs(trailingOnly = TRUE); dir <- a[1]; K <- if (length(a) >= 2) as.integer(a[2]) else 8L
# optional third argument: "matched" evaluates m = 250 and the m that matches the adaptive TLR storage (from
# vecchia_matched.json of 26) instead of 10/30/60/120, and writes kl_vecchia_matched.json
mode <- if (length(a) >= 3) a[3] else "ladder"
st <- fromJSON(file.path(dir, "stress.json")); N <- st$settings$n; nu <- st$settings$nu; th <- st$theta_true
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"; cp <- c(th[1], th[2] / sqrt(2 * nu), th[3])
covf <- get(covfun)
locs <- as.matrix(read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))[, c("x", "y")])
res <- list()
for (k in seq_len(K)) {
  set.seed(k); ord <- order_maxmin(locs); lo <- locs[ord, ]
  Sig <- covf(cp, lo); logdet <- 2 * sum(log(diag(chol(Sig))))
  ms <- if (mode == "matched") sort(unique(c(250L, fromJSON(file.path(dir, "vecchia_matched.json"))$m_match))) else c(10L, 30L, 60L, 120L)
  for (m in ms) {
    t0 <- proc.time()[["elapsed"]]
    NN <- find_ordered_nn(lo, m); Li <- vecchia_Linv(cp, covfun, lo, NN)
    ok <- !is.na(NN)
    B <- sparseMatrix(i = row(NN)[ok], j = NN[ok], x = Li[ok], dims = c(N, N))
    BS <- as.matrix(B %*% Sig)
    ii <- row(NN)[ok]; jj <- NN[ok]
    tr <- sum(BS[cbind(ii, jj)] * Li[ok]); rm(BS)     # tr(B Sigma B^T) = sum over the nonzeros of B of (B Sigma) * B
    logdet_t <- -2 * sum(log(Li[, 1]))
    kl <- 0.5 * (tr - N + logdet_t - logdet)
    res[[length(res) + 1]] <- list(m = m, seed = k, kl = kl, seconds = proc.time()[["elapsed"]] - t0)
    cat(sprintf("%s seed=%d m=%d KL=%.3f (%.0fs)\n", basename(dir), k, m, kl, proc.time()[["elapsed"]] - t0))
  }
}
summ <- lapply(split(res, sapply(res, `[[`, "m")), function(r) { v <- sapply(r, `[[`, "kl"); list(m = r[[1]]$m, mean = mean(v), min = min(v), max = max(v), orderings = length(v)) })
write(toJSON(list(runs = res, summary = unname(summ)), auto_unbox = TRUE, pretty = TRUE, digits = 8), file.path(dir, if (mode == "matched") "kl_vecchia_matched.json" else "kl_vecchia.json"))
