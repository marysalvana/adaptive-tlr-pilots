"""Head-to-head with Vecchia at a fixed N (2026-10-21): where do the estimates sit on the exact likelihood?

For the simulated field of 17_tlr_cholesky_pilot.py at a given N (sim_n{N}_obs.csv / sim_n{N}_holdout.csv in the
results folder), this script evaluates the exact (dense) log-likelihood and the adaptive-TLR log-likelihood
(17's tile Cholesky with recompression, tau = 0.9) at several parameter points: the true parameters, the dense
maximum-likelihood estimate (from tlr_cholesky_n{N}.json if its MLE was run), and the Vecchia estimates with 30
and 60 neighbors (vecchia_n{N}_m{30,60}.json from 12_vecchia_baseline.R).  It reports, for each point, the
exact log-likelihood, its gap to the best point, the adaptive-TLR log-likelihood, and held-out prediction skill
of the exact model at that point.  A Vecchia estimate that sits many log-likelihood units below the dense fit
is one the exact model rejects, whatever the prediction skill; the TLR column shows whether the compressed
likelihood ranks the points the same way.

Usage:  python3 23_vecchia_headtohead.py --n 40000 --dir ../results/tlr_cholesky [--dense 1]
"""
import argparse, json, os, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)
spec17 = importlib.util.spec_from_file_location("tlr", os.path.join(here, "17_tlr_cholesky_pilot.py"))
tlr = importlib.util.module_from_spec(spec17); spec17.loader.exec_module(tlr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--dense", type=int, default=1, help="evaluate the exact log-likelihood (needs N^2 memory)")
    ap.add_argument("--dir", type=str, default="../results/tlr_cholesky")
    ap.add_argument("--mle_json", type=str, default=None, help="JSON with the dense MLE (default: <dir>/tlr_cholesky_n{N}.json)")
    a = ap.parse_args()
    obs = np.loadtxt(os.path.join(a.dir, f"sim_n{a.n}_obs.csv"), delimiter=",", skiprows=1)
    hold = np.loadtxt(os.path.join(a.dir, f"sim_n{a.n}_holdout.csv"), delimiter=",", skiprows=1)
    S, y, Sp, yp = obs[:, :2], obs[:, 2], hold[:, :2], hold[:, 2]
    assert S.shape[0] == a.n
    bounds = rules.tile_bounds(a.n, a.b)
    if a.n > 30000:
        rules.cov_full = tlr.cov_full_lowmem
    points = {"true": [1.0, 0.3, 0.05]}
    mle_json = a.mle_json or os.path.join(a.dir, f"tlr_cholesky_n{a.n}.json")
    if os.path.exists(mle_json):
        r = json.load(open(mle_json))
        for x in r.get("mle", []):
            if x["rule"] == "dense": points["dense MLE"] = x["theta_hat"]
            if x["rule"] == "adaptive": points["adaptive-TLR MLE"] = x["theta_hat"]
        if "settings" in r: points["true"] = [r["settings"]["sigma2"], r["settings"]["range"], r["settings"]["eta"]]
    for m in (30, 60):
        f = os.path.join(a.dir, f"vecchia_n{a.n}_m{m}.json")
        if os.path.exists(f):
            v = json.load(open(f))["theta_hat"]; points[f"Vecchia m={m}"] = [v["sigma2"], v["range"], v["eta"]]
    out = dict(n=a.n, b=a.b, tau=a.tau, points={})
    print(f"N={a.n}: evaluating {list(points)}", flush=True)
    for name, th in points.items():
        th = np.array(th, float); rec = dict(theta=th.tolist())
        t0 = time.time()
        if a.dense:
            K = rules.cov_full(S, th)
            ll, cf = rules.loglik_dense(K, y)
            rec["exact_loglik"] = None if ll is None else float(ll)
            if cf is not None:
                Kp = th[0] * rules.matern32(rules.dist(Sp, S), th[1])
                from scipy.linalg import cho_solve
                alpha = cho_solve(cf, y, check_finite=False); mu = Kp @ alpha
                W = cho_solve(cf, Kp.T, check_finite=False); var = th[0] * (1 + th[2]) - (Kp * W.T).sum(1)
                rec["skill_exact_model"] = rules.skill(yp, mu, np.sqrt(np.maximum(var, 1e-12)))
            del K, cf
            rec["seconds_exact"] = time.time() - t0
        t0 = time.time()
        llt, fac, info = tlr.tlr_loglik(S, y, th, bounds, "adaptive", a.tau)
        rec["tlr_loglik"] = None if llt is None else float(llt)
        rec["tlr_completed"] = bool(info["completed"])
        if info["completed"]:
            rec["tlr_factor_frac"] = info["stored_factor"] / (a.n * (a.n + 1) // 2)
            rec["skill_tlr"] = rules.skill(yp, *tlr.tlr_predict(S, y, Sp, th, bounds, fac))
        rec["seconds_tlr"] = time.time() - t0
        out["points"][name] = rec
        print(f"  {name:18s} theta={np.round(th,4).tolist()} exact={rec.get('exact_loglik')} tlr={rec.get('tlr_loglik')} "
              f"completed={rec['tlr_completed']} skill={rec.get('skill_exact_model') or rec.get('skill_tlr')}", flush=True)
    ref = max((p.get("exact_loglik") if p.get("exact_loglik") is not None else p.get("tlr_loglik") or -np.inf) for p in out["points"].values())
    for name, p in out["points"].items():
        base = p.get("exact_loglik") if p.get("exact_loglik") is not None else p.get("tlr_loglik")
        p["gap_to_best"] = None if base is None else float(ref - base)
        print(f"  gap below the best point: {name:18s} {p['gap_to_best']}", flush=True)
    path = os.path.join(a.dir, f"vecchia_headtohead_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1); print("saved", path)


if __name__ == "__main__":
    main()
