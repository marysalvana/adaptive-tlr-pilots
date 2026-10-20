"""MCMC pilot (2026-10-20): Metropolis-Hastings for a Matern(3/2) Gaussian process with
nugget, run twice on the same simulated field with the same seed and proposal: once with the
exact (dense) likelihood and once with the adaptive-TLR likelihood (tau = 0.9,
ranks re-chosen at every proposal).  Reports posterior means, standard deviations, and 95%
credible intervals of (sigma2, range, eta), the difference of posterior means in units of the
dense posterior SD, acceptance rates, factorization failures, and held-out prediction from the
posterior mean.  Flat priors on log(theta) inside wide bounds; random-walk proposals on
log(theta) whose scale is tuned to 25-35% acceptance during burn-in and then frozen.

Usage:  python3 16_tlr_mcmc_pilot.py --n 5000 --iters 3000 --burn 500 --out ../results
Reproducible: fixed seeds; settings recorded in the JSON output.
"""
import argparse, json, os, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)
LO, HI = np.log([0.1, 0.02, 0.001]), np.log([10.0, 2.0, 1.0])


def run_chain(S, y, bounds, rule, value, start, iters, burn, seed, scale0=0.08):
    rng = np.random.default_rng(seed)
    cur = np.array(start, float); scale = scale0
    cur_ll = -rules.neg_loglik(cur, S, y, rule, value, bounds, {"evals": 0, "failures": 0})
    chain = np.zeros((iters, 3)); acc = 0; fail = 0; ll_trace = np.zeros(iters)
    t0 = time.time()
    for it in range(iters):
        prop = cur + scale * rng.standard_normal(3)
        if np.all(prop > LO) and np.all(prop < HI):
            trace = {"evals": 0, "failures": 0}
            ll = -rules.neg_loglik(prop, S, y, rule, value, bounds, trace)
            fail += trace["failures"]
            if trace["failures"] == 0 and np.log(rng.random()) < ll - cur_ll:
                cur, cur_ll = prop, ll; acc += 1
        chain[it] = cur; ll_trace[it] = cur_ll
        if it < burn and (it + 1) % 50 == 0:               # Robbins-Monro style tuning toward 0.3
            rate = acc / (it + 1); scale *= np.exp(0.5 * (rate - 0.3))
        if (it + 1) % 250 == 0:
            print(f"  {rule:8s} it={it+1} acc={acc/(it+1):.2f} scale={scale:.3f} theta={np.round(np.exp(cur),3).tolist()} ({time.time()-t0:.0f}s)", flush=True)
    return chain, ll_trace, acc / iters, fail, scale, time.time() - t0


def summarize(chain, burn):
    th = np.exp(chain[burn:])
    return dict(mean=th.mean(0).tolist(), sd=th.std(0, ddof=1).tolist(),
                ci95=np.percentile(th, [2.5, 97.5], axis=0).T.tolist(), n_post=int(len(th)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=str, default="../results")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    print(f"simulated N={a.n} (+{a.n_pred} held out); tiles b={a.b}, nB={len(bounds)}", flush=True)
    start = np.log(theta_true * np.array([1.3, 0.8, 1.5]))     # start away from the truth
    out = dict(settings=vars(a), theta_true=theta_true.tolist(), chains={})
    for name, rule, value in [("dense", "dense", None), ("adaptive_tlr", "adaptive", a.tau)]:
        chain, llt, acc, fail, scale, sec = run_chain(S, y, bounds, rule, value, start, a.iters, a.burn, a.seed + 1)
        summ = summarize(chain, a.burn)
        th_mean = np.array(summ["mean"])
        pr = rules.predict(S, y, Sp, th_mean, rule, value, bounds)
        sk = rules.skill(yp, *pr) if pr is not None else None
        stored = None
        if rule == "adaptive":
            K, info = rules.make_cov(S, th_mean, rule, value, bounds); stored = info[0] / (a.n * a.n); del K
        out["chains"][name] = dict(summary=summ, acceptance=acc, factorization_failures=fail, proposal_scale=scale,
                                   seconds=sec, skill_at_posterior_mean=sk, stored_fraction_at_posterior_mean=stored)
        np.save(os.path.join(a.out, f"mcmc_n{a.n}_{name}_chain.npy"), chain)
        print(f"{name}: mean={np.round(th_mean,4).tolist()} sd={np.round(summ['sd'],4).tolist()} acc={acc:.2f} failures={fail} skill={sk} stored={stored} ({sec:.0f}s)", flush=True)
    d = out["chains"]["dense"]["summary"]; t = out["chains"]["adaptive_tlr"]["summary"]
    out["posterior_mean_difference_in_dense_SD"] = (np.abs(np.array(t["mean"]) - np.array(d["mean"])) / np.array(d["sd"])).tolist()
    out["posterior_sd_ratio_tlr_over_dense"] = (np.array(t["sd"]) / np.array(d["sd"])).tolist()
    print("posterior mean difference in dense SD units:", np.round(out["posterior_mean_difference_in_dense_SD"], 3).tolist())
    json.dump(out, open(os.path.join(a.out, f"tlr_mcmc_pilot_n{a.n}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, f"tlr_mcmc_pilot_n{a.n}.json"))


if __name__ == "__main__":
    main()
