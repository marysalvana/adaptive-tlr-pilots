"""Oscillation of the TLR log-likelihood error over the exact posterior (2026-10-23, Objective 2).

The total variation bound of Obj. 2 needs sup |ell^TLR - ell| over the region where the posterior
concentrates; a constant shift leaves pi^TLR unchanged, so only half the oscillation of d(theta) =
ell^TLR(theta) - ell(theta) matters:  ||pi^TLR - pi||_TV <= tanh(delta / 2), delta = (max d - min d) / 2
over that region (plus the posterior mass outside it).  This script evaluates d(theta) at draws of
the exact chain of the MCMC pilot (same data: seed, N, b as 19_tlr_mcmc_gpu.py), for the assembled
Sigma^TLR (as in the pilot, exact tile SVDs) and for the TLR Cholesky factorization with
recompression of 17_tlr_cholesky_pilot.py, for each rule.

Usage:  python3 37_posterior_oscillation.py --n 5000 --chain ../results/mcmc_long/dense/mcmc_n5000_dense_chain.npy
"""
import argparse, json, os, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(here, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rules = load("rules", "11_tlr_rank_rules.py")
tlr = load("tlr", "17_tlr_cholesky_pilot.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--chain", type=str, default="../results/mcmc_long/dense/mcmc_n5000_dense_chain.npy")
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--rules", type=str, default="adaptive:0.9,fixed:1e-4")
    ap.add_argument("--out", type=str, default="../results/posterior_oscillation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)

    # data: identical to 19_tlr_mcmc_gpu.py for the same seed and N
    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs = perm[:a.n]
    S, y = S_all[obs], y_all[obs]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    log(f"simulated N={a.n}; tiles b={a.b}, nB={len(bounds)}")

    chain = np.load(a.chain)[a.burn:]
    idx = np.linspace(0, len(chain) - 1, a.draws).round().astype(int)
    draws = np.exp(chain[idx])
    post_mean = np.exp(chain).mean(0)
    thetas = np.vstack([theta_true, post_mean, draws])
    labels = ["theta_true", "posterior_mean"] + ["draw"] * len(draws)
    specs = [(s.split(":")[0], float(s.split(":")[1])) for s in a.rules.split(",") if s]
    res = {f"{r}_{v:g}": dict(assembled=[], cholesky=[]) for r, v in specs}
    ll_exact = []
    t0 = time.time()
    for k, th in enumerate(thetas):
        K = rules.cov_full(S, th)
        ll, _ = rules.loglik_dense(K, y); ll_exact.append(ll)
        for r, v in specs:
            name = f"{r}_{v:g}"
            Kt = K.copy(); rules.truncate_tiles(Kt, bounds, r, v, th[0] * th[2])
            lla, _ = rules.loglik_dense(Kt, y); del Kt
            llc, fac, info = tlr.tlr_loglik(S, y, th, bounds, r, v)
            res[name]["assembled"].append(None if lla is None else float(lla - ll))
            res[name]["cholesky"].append(None if llc is None else float(llc - ll))
        if k % 25 == 0 or k < 2:
            log(f"{k:4d} {labels[k]:14s} theta={np.round(th,4).tolist()} " +
                " ".join(f"{n}: asm {res[n]['assembled'][-1]} chol {res[n]['cholesky'][-1]}" for n in res) + f" ({time.time()-t0:.0f}s)")
        if (k + 1) % 10 == 0:                                            # partial save (2026-10-24): a timed-out job keeps its draws
            json.dump(dict(settings=vars(a), partial=True, n_done=k + 1, labels=labels[:k + 1], thetas=thetas[:k + 1].tolist(),
                           loglik_exact=ll_exact, rules={n: dict(per_theta=d) for n, d in res.items()}),
                      open(os.path.join(a.out, f"posterior_oscillation_n{a.n}_partial.json"), "w"))
    out = dict(settings=vars(a), theta_true=theta_true.tolist(), posterior_mean=post_mean.tolist(), n_draws=len(draws),
               labels=labels, thetas=thetas.tolist(), loglik_exact=ll_exact, rules={})
    for name, d in res.items():
        rec = dict(per_theta=d)
        for var in ("assembled", "cholesky"):
            vals = np.array([x for x in d[var][2:] if x is not None], float)      # draws only
            fails = sum(x is None for x in d[var][2:])
            if len(vals):
                delta = (vals.max() - vals.min()) / 2
                rec[var] = dict(n=int(len(vals)), failures=int(fails), min=float(vals.min()), max=float(vals.max()),
                                mean=float(vals.mean()), sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                                half_oscillation=float(delta), tv_bound_half_oscillation=float(np.tanh(delta / 2)),
                                sup_abs=float(np.abs(vals).max()), tv_bound_sup=float(np.tanh(np.abs(vals).max() / 2)),
                                at_theta_true=d[var][0], at_posterior_mean=d[var][1])
            else:
                rec[var] = dict(n=0, failures=int(fails))
        out["rules"][name] = rec
        log(f"{name}: " + " | ".join(f"{var}: n={rec[var].get('n')} fail={rec[var].get('failures')} range=[{rec[var].get('min')}, {rec[var].get('max')}] "
                                      f"half-osc={rec[var].get('half_oscillation')} TV<={rec[var].get('tv_bound_half_oscillation')}" for var in ("assembled", "cholesky")))
    path = os.path.join(a.out, f"posterior_oscillation_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1)
    log("saved " + path)


if __name__ == "__main__":
    main()
