"""Standard errors for the rank-rule pilot (2026-10-20), so the pilot can be scored against the
criterion "reproduce the dense estimates within one standard error".

For the dense fit and the adaptive-TLR fit (tau = 0.9) at their maximum-likelihood estimates
in results/tlr_rank_rules_n{N}.json, the observed information is computed by
central finite differences of the log-likelihood in log(theta) (step h), inverted, and
transformed to the theta scale (SE_theta = theta * SE_log).  Reports the standard errors and
the differences between the adaptive and dense estimates in dense standard-error units.

Usage:  python3 15_pilot_standard_errors.py --n 10000 --out ../results
Reproducible: deterministic given the CSV and JSON written by 11_tlr_rank_rules.py.
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)


def observed_information(logth, S, y, rule, value, bounds, h):
    """Negative Hessian of the log-likelihood in log(theta), by central differences."""
    f = lambda z: -rules.neg_loglik(z, S, y, rule, value, bounds, {"evals": 0, "failures": 0})
    p = len(logth); H = np.zeros((p, p)); f0 = f(logth)
    for i in range(p):
        ei = np.zeros(p); ei[i] = h
        H[i, i] = (f(logth + ei) - 2 * f0 + f(logth - ei)) / h ** 2
        for j in range(i):
            ej = np.zeros(p); ej[j] = h
            H[i, j] = H[j, i] = (f(logth + ei + ej) - f(logth + ei - ej) - f(logth - ei + ej) + f(logth - ei - ej)) / (4 * h ** 2)
    return -H, f0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--h", type=float, default=0.02, help="finite-difference step in log(theta)")
    ap.add_argument("--out", type=str, default="../results")
    a = ap.parse_args()
    obs = np.loadtxt(os.path.join(a.out, f"sim_n{a.n}_obs.csv"), delimiter=",", skiprows=1)
    S, y = obs[:, :2], obs[:, 2]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    res = json.load(open(os.path.join(a.out, f"tlr_rank_rules_n{a.n}.json")))
    fits = {(m["rule"], m["value"]): m for m in res["mle"]}
    dense = fits[("dense", None)]; adapt = fits[("adaptive", 0.9)]
    out = dict(n=a.n, b=a.b, step=a.h, fits={})
    se_dense = None
    for name, m, rule, value in [("dense", dense, "dense", None), ("adaptive_0.9", adapt, "adaptive", 0.9)]:
        th = np.array(m["theta_hat"]); t0 = time.time()
        I, ll = observed_information(np.log(th), S, y, rule, value, bounds, a.h)
        cov_log = np.linalg.inv(I)
        se_log = np.sqrt(np.diag(cov_log)); se = th * se_log
        out["fits"][name] = dict(theta_hat=th.tolist(), loglik=float(ll), se_theta=se.tolist(), se_log=se_log.tolist(),
                                 info_log=I.tolist(), seconds=time.time() - t0)
        print(f"{name:13s} theta_hat={np.round(th,4).tolist()} SE={np.round(se,4).tolist()} ({time.time()-t0:.0f}s)", flush=True)
        if name == "dense":
            se_dense = se
    d = np.abs(np.array(adapt["theta_hat"]) - np.array(dense["theta_hat"]))
    out["adaptive_minus_dense_in_dense_SE"] = (d / se_dense).tolist()
    out["note"] = "observed information by central differences in log(theta); SE_theta = theta * SE_log"
    print("adaptive - dense, in dense SE units:", np.round(d / se_dense, 3).tolist())
    json.dump(out, open(os.path.join(a.out, f"pilot_standard_errors_n{a.n}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, f"pilot_standard_errors_n{a.n}.json"))


if __name__ == "__main__":
    main()
