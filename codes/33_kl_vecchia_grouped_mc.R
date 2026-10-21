#!/usr/bin/env Rscript
# KL divergence of GROUPED response Vecchia (GpGp's fit_model default, group = TRUE; Guinness 2018) from the exact model,
# by Monte Carlo (2026-10-21).  GpGp does not expose the grouped inverse factor, so KL(p || q) = E_p[log p(y) - log q(y)]
# is estimated from R draws y ~ N(0, Sigma) at the true parameters: the exact log-likelihood from a dense Cholesky, the
# grouped Vecchia one from vecchia_grouped_meanzero_loglik.  The ungrouped Vecchia log-likelihood is evaluated on the same
# draws, so its Monte Carlo KL can be checked against the exact ungrouped KL of 32_kl_vecchia.R.  Orderings set.seed(1..K).
# Writes kl_vecchia_grouped_mc.json.
# Usage:  Rscript 33_kl_vecchia_grouped_mc.R ../results/screening_stress/nu25_r3_e005 [R=20] [K=4]
suppressPackageStartupMessages({library(GpGp); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE); dir <- a[1]
R <- if (length(a) >= 2) as.integer(a[2]) else 20L; K <- if (length(a) >= 3) as.integer(a[3]) else 4L
st <- fromJSON(file.path(dir, "stress.json")); N <- st$settings$n; nu <- st$settings$nu; th <- st$theta_true
covfun <- if (nu == 1.5) "matern15_isotropic" else "matern25_isotropic"; cp <- c(th[1], th[2] / sqrt(2 * nu), th[3])
covf <- get(covfun)
locs <- as.matrix(read.csv(file.path(dir, sprintf("sim_n%d_obs.csv", N)))[, c("x", "y")])
L <- t(chol(covf(cp, locs))); logdet <- 2 * sum(log(diag(L)))
set.seed(20261021); Z <- matrix(rnorm(N * R), N, R); Y <- L %*% Z
ll_exact <- -0.5 * (colSums(Z^2) + logdet + N * log(2 * pi))         # exact log-likelihood of each draw
rm(L)
out <- list()
for (k in seq_len(K)) {
  set.seed(k); ord <- order_maxmin(locs); lo <- locs[ord, ]
  for (m in c(30L, 60L)) {
    NN <- find_ordered_nn(lo, m); NNl <- group_obs(NN)
    du <- dg <- numeric(R)
    for (r in seq_len(R)) {
      yo <- Y[ord, r]
      du[r] <- ll_exact[r] - vecchia_meanzero_loglik(cp, covfun, yo, lo, NN)$loglik
      dg[r] <- ll_exact[r] - vecchia_grouped_meanzero_loglik(cp, covfun, yo, lo, NNl)$loglik
    }
    out[[length(out) + 1]] <- list(m = m, seed = k, kl_ungrouped_mc = mean(du), se_ungrouped = sd(du) / sqrt(R),
                                   kl_grouped_mc = mean(dg), se_grouped = sd(dg) / sqrt(R))
    cat(sprintf("%s seed=%d m=%d KL ungrouped %.2f (se %.2f), grouped %.2f (se %.2f)\n", basename(dir), k, m,
                mean(du), sd(du) / sqrt(R), mean(dg), sd(dg) / sqrt(R)))
  }
}
summ <- lapply(split(out, sapply(out, `[[`, "m")), function(r) list(m = r[[1]]$m,
  kl_grouped = mean(sapply(r, `[[`, "kl_grouped_mc")), kl_ungrouped_mc = mean(sapply(r, `[[`, "kl_ungrouped_mc")),
  draws = R, orderings = length(r)))
write(toJSON(list(runs = out, summary = unname(summ)), auto_unbox = TRUE, pretty = TRUE, digits = 8),
      file.path(dir, "kl_vecchia_grouped_mc.json"))
