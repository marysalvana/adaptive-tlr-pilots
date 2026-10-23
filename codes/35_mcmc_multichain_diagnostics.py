"""Multi-chain diagnostics for the Objective 2 pilot (2026-10-23).

Four random-walk Metropolis chains per likelihood (exact, adaptive TLR tau = 0.9, fixed tolerance 1e-4) at
N = 5,000: chain 1 is the committed chain of results/mcmc_long/<rule>/ (chain seed 20261021, start
1.3, 0.8, 1.5 times theta_true); chains 2-4 are results/mcmc_multi/<rule>/c{2,3,4}/ from
19_tlr_mcmc_gpu.py with --chain_seed 102/103/104 and dispersed starts (0.6,1.4,0.5), (2.0,0.5,2.0),
(0.8,1.2,0.8).  All chains share the data, prior, proposal covariance and burn-in.

Per likelihood and parameter: pooled posterior mean and SD over the four chains, rank-normalized split
R-hat and bulk effective sample size of Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021,
Bayesian Analysis), Monte Carlo standard errors of the mean and of the SD (batch means over chains), and
the differences from the exact chains in exact posterior SD units.

Usage:  python3 35_mcmc_multichain_diagnostics.py [--out ../results/mcmc_multi]
"""
import argparse, json, os
import numpy as np
from scipy import stats

RULES = {"dense": "exact", "adaptive_0.9": "adaptive TLR (tau = 0.9)", "fixed_1e-4": "fixed tolerance 1e-4"}
FILES = {"dense": "mcmc_n5000_dense_chain.npy", "adaptive_0.9": "mcmc_n5000_adaptive_0.9_chain.npy",
         "fixed_1e-4": "mcmc_n5000_fixed_0.0001_chain.npy"}
PARAMS = ["sigma2", "range", "eta"]


def split_rhat_ess(chains):
    """Rank-normalized split R-hat and bulk ESS (Vehtari et al. 2021) for chains of shape (M, n)."""
    M, n = chains.shape
    half = n // 2
    x = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)       # 2M split chains
    r = stats.rankdata(x, method="average").reshape(x.shape)
    z = stats.norm.ppf((r - 3 / 8) / (x.size + 1 / 4))                             # rank-normalized
    m, nn = z.shape
    means = z.mean(axis=1); vars_ = z.var(axis=1, ddof=1)
    B = nn * means.var(ddof=1); W = vars_.mean()
    var_hat = (nn - 1) / nn * W + B / nn
    rhat = np.sqrt(var_hat / W)
    # bulk ESS from the rank-normalized split chains, Geyer initial monotone sequence on the pooled autocorrelation
    acov = np.zeros(nn)
    for row in z:
        rr = row - row.mean()
        f = np.fft.rfft(rr, 2 * nn)
        ac = np.fft.irfft(f * np.conj(f))[:nn] / nn
        acov += ac
    acov /= m
    rho = 1 - (W - acov) / var_hat
    # initial monotone positive sequence
    tau = -1.0; t = 0
    while t + 1 < nn:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2 * pair; t += 2
    ess = m * nn / max(tau, 1e-12)
    return float(rhat), float(min(ess, m * nn * 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/mcmc_multi")
    ap.add_argument("--long", default="../results/mcmc_long")
    ap.add_argument("--burn", type=int, default=2000)
    a = ap.parse_args()
    res = {}
    for rule, label in RULES.items():
        paths = [os.path.join(a.long, rule, FILES[rule])] + [os.path.join(a.out, rule, f"c{k}", FILES[rule]) for k in (2, 3, 4)]
        chains = []
        for p in paths:
            if os.path.exists(p):
                c = np.load(p)
                if c.shape[0] > a.burn:
                    chains.append(np.exp(c[a.burn:]))          # chains are stored as log(theta); summaries on the natural scale
        if not chains:
            print(f"{rule}: no chains found"); continue
        n = min(c.shape[0] for c in chains)
        X = np.stack([c[:n] for c in chains])                                        # (M, n, 3)
        M = X.shape[0]
        rec = dict(label=label, chains=M, draws_per_chain=int(n), per_parameter={})
        for j, name in enumerate(PARAMS):
            ch = X[:, :, j]
            rhat, ess = split_rhat_ess(ch)
            pooled = ch.reshape(-1)
            chain_means = ch.mean(axis=1); chain_sds = ch.std(axis=1, ddof=1)
            rec["per_parameter"][name] = dict(
                mean=float(pooled.mean()), sd=float(pooled.std(ddof=1)), rhat=rhat, ess_bulk=ess,
                mc_se_mean=float(pooled.std(ddof=1) / np.sqrt(max(ess, 1))),
                chain_means=chain_means.tolist(), chain_sds=chain_sds.tolist(),
                se_of_sd_across_chains=float(chain_sds.std(ddof=1) / np.sqrt(M)) if M > 1 else None)
        res[rule] = rec
    if "dense" in res:
        d = res["dense"]["per_parameter"]
        for rule in res:
            if rule == "dense":
                continue
            for name in PARAMS:
                t = res[rule]["per_parameter"][name]
                t["mean_diff_in_exact_sd"] = (t["mean"] - d[name]["mean"]) / d[name]["sd"]
                t["sd_ratio_over_exact"] = t["sd"] / d[name]["sd"]
                t["mean_diff_over_mc_se"] = (t["mean"] - d[name]["mean"]) / np.hypot(t["mc_se_mean"], d[name]["mc_se_mean"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "multichain_diagnostics_n5000.json"), "w"), indent=1)
    for rule, rec in res.items():
        print(f"{rec['label']}: {rec['chains']} chains x {rec['draws_per_chain']} draws")
        for name in PARAMS:
            t = rec["per_parameter"][name]
            extra = ""
            if "mean_diff_in_exact_sd" in t:
                extra = (f" | mean diff {t['mean_diff_in_exact_sd']:+.3f} exact SD ({t['mean_diff_over_mc_se']:+.1f} MC SE)"
                         f", SD ratio {t['sd_ratio_over_exact']:.2f}")
            print(f"  {name:7s} mean {t['mean']:.4f} sd {t['sd']:.4f} R-hat {t['rhat']:.3f} ESS {t['ess_bulk']:.0f} "
                  f"MC SE {t['mc_se_mean']:.4f}{extra}")


if __name__ == "__main__":
    main()
