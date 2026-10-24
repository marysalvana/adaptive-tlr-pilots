"""Maximum likelihood with rounded factors (2026-10-24, Objective 3).

The Obj. 3 pilot of 18_joint_rank_precision_pilot.py chose rank and storage precision under the budget at the true
parameters only.  This script fits the Matern model by maximum likelihood with the assembled Sigma^TLR re-chosen at
every candidate theta (budget e_max = tau b m(theta) / N, rank and precision per tile), three ways: dense FP64, the
rank rule alone in FP64 (Obj. 1), and the joint rank-and-precision rule (FP8/FP16/FP32/FP64 storage), and compares the
estimates and the held-out prediction scores.  Same data, seed, start and optimizer settings as 11_tlr_rank_rules.py.

Usage:  python3 40_joint_fit_pilot.py --n 10000 --out ../results/preliminary/joint_fit
Reproducible: fixed seeds; settings recorded in the JSON output.
"""
import argparse, json, os, time, importlib.util
import numpy as np
from scipy import optimize
from scipy.linalg import cho_solve

here = os.path.dirname(os.path.abspath(__file__))
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(here, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rules = load("rules", "11_tlr_rank_rules.py")
jp = load("jp", "18_joint_rank_precision_pilot.py")


def assemble_rule(S, theta, bounds, rule, tau, eps, n):
    """Dense Sigma^TLR under `rule` with rank and precision re-chosen at theta; returns K, bytes, mix, ranks."""
    emax = tau * (bounds[0][1] - bounds[0][0]) * theta[0] * theta[2] / n
    def pick(i, j, T):
        opts, svd = jp.tile_options(T, emax, eps)
        r, p, bb, err = opts[rule]
        return r, p, bb, err, svd
    return jp.assemble(S, theta, bounds, rule, pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--rules", type=str, default="dense,rank_only,joint")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--out", type=str, default="../results/preliminary/joint_fit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)

    rng = np.random.default_rng(a.seed)                      # identical field to 18_joint_rank_precision_pilot.py
    theta_true = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    log(f"simulated N={a.n}; tiles b={a.b}, nB={len(bounds)}")

    def cov(theta, rule):
        if rule == "dense":
            return rules.cov_full(S, theta), None
        K, byt, mix, ranks, errs = assemble_rule(S, theta, bounds, rule, a.tau, a.eps, a.n)
        return K, dict(offdiag_bytes=int(byt), precision_mix_tiles={int(k): int(v) for k, v in mix.items()},
                       mean_rank=float(np.mean(ranks)), max_rank=int(max(ranks)))

    def predict(theta, rule):
        K, info = cov(theta, rule)
        ll, cf = rules.loglik_dense(K, y)
        if ll is None:
            return None, None, info
        Kp = theta[0] * rules.matern32(rules.dist(Sp, S), theta[1])
        alpha = cho_solve(cf, y, check_finite=False); mu = Kp @ alpha
        W = cho_solve(cf, Kp.T, check_finite=False)
        var = theta[0] * (1 + theta[2]) - (Kp * W.T).sum(1)
        return float(ll), rules.skill(yp, mu, np.sqrt(np.maximum(var, 1e-12))), info

    out = dict(settings=vars(a), theta_true=theta_true.tolist(), fits={})
    dense_bytes_off = sum((i1 - i0) * (j1 - j0) * 8 for i, (i0, i1) in enumerate(bounds) for (j0, j1) in bounds[:i])
    for rule in [r for r in a.rules.split(",") if r]:
        trace = dict(evals=0, failures=0); t0 = time.time()
        def nll(logth):
            K, _ = cov(np.exp(logth), rule)
            ll, _ = rules.loglik_dense(K, y); trace["evals"] += 1
            if ll is None:
                trace["failures"] += 1; return 1e12
            if trace["evals"] % 10 == 0:
                log(f"  {rule} eval {trace['evals']} theta={np.round(np.exp(logth),4).tolist()} nll={-ll:.3f} ({time.time()-t0:.0f}s)")
            return -ll
        x0 = np.log(theta_true * np.array([1.5, 0.6, 2.0]))     # dispersed start, as in 11_tlr_rank_rules.py
        res = optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=0.05, maxiter=a.maxiter))
        th = np.exp(res.x)
        ll_hat, sk, info = predict(th, rule)
        rec = dict(theta_hat=th.tolist(), loglik_at_theta_hat=ll_hat, evals=trace["evals"], failures=trace["failures"],
                   seconds=time.time() - t0, skill=sk, storage_at_theta_hat=info)
        if info is not None:
            rec["storage_at_theta_hat"]["offdiag_bytes_frac_of_dense"] = info["offdiag_bytes"] / dense_bytes_off
        out["fits"][rule] = rec
        log(f"FIT {rule:10s} theta_hat={np.round(th,4).tolist()} loglik={ll_hat} evals={trace['evals']} fail={trace['failures']} skill={sk} storage={info} ({time.time()-t0:.0f}s)")
    if "dense" in out["fits"]:
        thd = np.array(out["fits"]["dense"]["theta_hat"])
        for rule, rec in out["fits"].items():
            rec["theta_hat_rel_to_dense"] = (np.array(rec["theta_hat"]) / thd - 1).tolist()
            # exact log-likelihood at each rule's estimate, for the loss relative to the dense fit
            ll_exact, _ = rules.loglik_dense(rules.cov_full(S, np.array(rec["theta_hat"])), y)
            rec["exact_loglik_at_theta_hat"] = None if ll_exact is None else float(ll_exact)
            rec["exact_loglik_loss_vs_dense_fit"] = None if ll_exact is None else float(out["fits"]["dense"]["loglik_at_theta_hat"] - ll_exact)
            log(f"{rule:10s} theta_hat/dense-1={np.round(rec['theta_hat_rel_to_dense'],4).tolist()} exact loglik loss vs dense fit={rec['exact_loglik_loss_vs_dense_fit']}")
    path = os.path.join(a.out, f"joint_fit_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1); log("saved " + path)


if __name__ == "__main__":
    main()
