"""Wind-gust maximum likelihood fit with rounded factors (2026-10-24, Objective 3).

Real-data companion of 40_joint_fit_pilot.py.  A random subset of N station-hours of the SC26 wind-gust record (Jahan
et al. 2025; the 2017-2019 storms with WRF predictors) is modeled as gust = linear mean in the 17 WRF predictors of the
SC26 application (ordinary least squares on the training subset, held fixed) + space-time Gaussian process residual with
the Gneiting-Matern(3/2, beta = 0) covariance of 39_joint_windgust.py and a nugget, theta = (sigma2, r_s, r_t, eta).
The parameters are fitted by Nelder-Mead four ways, with the compressed matrix re-chosen at every candidate theta:
dense FP64; the rank rule of Obj. 1 in FP64 (rank_only); rank then the lowest admissible precision (rank_then_precision);
and the joint rank-and-precision rule (joint).  300 held-out station-hours are predicted by kriging from each fit.
Reports estimates, held-out RMSE, CRPS and 90% coverage, evaluations, factorization failures and bytes at the estimate.

Usage:  python3 41_windgust_fit.py --n 10000 --out ../results/preliminary/windgust_fit
Reproducible: fixed seeds (the subset is drawn as in 39 for the same seed and N); settings recorded in the JSON output.
"""
import argparse, json, os, time, importlib.util
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.linalg import cho_solve

here = os.path.dirname(os.path.abspath(__file__))
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(here, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rules = load("rules", "11_tlr_rank_rules.py")
jp = load("jp", "18_joint_rank_precision_pilot.py")
wg = load("wg", "39_joint_windgust.py")
PRED = ['PSFC(Pa)', 'T2(C)', 'WS_950mb(m/s)', 'WS_850mb(m/s)', 'WS_10m(m/s)', 'PBLH(km)', 'Terrain_height(m)', 'Ustar(m/s)',
        'SfcWG_UPP(m/s)', 'WG_AFWA(m/s)', 'WG_ECMWF(m/s)', 'Temp_grad(1km_sfc)(C/km)', 'Temp_grad(2km_sfc)(C/km)', 'ELEV(M)',
        'WindDC(cos)', 'WindDC(sin)', 'yday']
TARGET = 'WG_o(m/s)'


def cov_full(P, th):
    p = dict(sigma2=th[0], r_s=th[1], r_t=th[2])
    n = len(P); K = np.empty((n, n))
    for i0 in range(0, n, 2000):
        K[i0:i0 + 2000] = wg.cov_block(P[i0:i0 + 2000], P, p)
    K[np.diag_indices_from(K)] += th[0] * th[3]
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--rules", type=str, default="dense,rank_only,rank_then_precision,joint")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=0, help="Nelder-Mead restarts from its own result until the improvement is below fatol (2026-10-24)")
    ap.add_argument("--start", type=str, default="", help="sigma2,r_s,r_t,eta: starting point of the optimizer (default: the SC26 estimates) (2026-10-24)")
    ap.add_argument("--eval_at", type=str, default="", help="sigma2,r_s,r_t,eta: evaluate each rule's log-likelihood and held-out scores there instead of fitting (2026-10-24)")
    ap.add_argument("--out", type=str, default="../results/preliminary/windgust_fit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)

    d = pd.read_csv(wg.DATA)
    d["DTS"] = pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD, hour=d.HH))
    d["yday"] = np.cos(2 * np.pi * (pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD)).dt.dayofyear - 1) / 365)
    d = d[d[PRED + [TARGET]].notna().all(axis=1)].reset_index(drop=True)
    lat0, lon0, t0 = d.Lat.mean(), d.Lon.mean(), d.DTS.min()
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(len(d), a.n, replace=False))                 # the training subset of 39 for this seed and N
    rest = np.setdiff1d(np.arange(len(d)), sub)
    hold = np.sort(rng.choice(rest, a.n_pred, replace=False))             # held-out station-hours
    dtr, dte = d.iloc[sub], d.iloc[hold]
    X = np.column_stack([np.ones(a.n), dtr[PRED].values]); Xp = np.column_stack([np.ones(a.n_pred), dte[PRED].values])
    mu_, sd_ = X[:, 1:].mean(0), X[:, 1:].std(0); X[:, 1:] = (X[:, 1:] - mu_) / sd_; Xp[:, 1:] = (Xp[:, 1:] - mu_) / sd_
    beta = np.linalg.lstsq(X, dtr[TARGET].values, rcond=None)[0]
    y = dtr[TARGET].values - X @ beta; yp = dte[TARGET].values; mean_p = Xp @ beta
    P = wg.coords(dtr, lat0, lon0, t0); Pp = wg.coords(dte, lat0, lon0, t0)
    order = rules.morton_order(P[:, :2]); P, y = P[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    log(f"wind-gust subset N={a.n} of {len(d)} station-hours (+{a.n_pred} held out); linear mean R^2={1 - y.var() / dtr[TARGET].values.var():.3f}, "
        f"residual variance {y.var():.3f}; tiles b={a.b}, nB={len(bounds)}")
    theta0 = np.array([wg.PAR["sigma2"], wg.PAR["r_s"], wg.PAR["r_t"], wg.PAR["eta"]])      # SC26 estimates as the start
    if a.start:
        theta0 = np.array([float(x) for x in a.start.split(",")])
    dense_bytes_off = sum((i1 - i0) * (j1 - j0) * 8 for i, (i0, i1) in enumerate(bounds) for (j0, j1) in bounds[:i])

    def cov(th, rule):
        K = cov_full(P, th)
        if rule == "dense":
            return K, None
        emax = a.tau * a.b * th[0] * th[3] / a.n
        def pick(i, j, T):
            opts, svd = jp.tile_options(T, emax, a.eps)
            r, p, bb, err = opts[rule]
            return r, p, bb, err, svd
        byt = 0; mix = {p: 0 for p in jp.PREC}; ranks = []
        for i, (i0, i1) in enumerate(bounds):
            for j, (j0, j1) in enumerate(bounds[:i]):
                r, p, bb, err, (U, s, Vt) = pick(i, j, K[i0:i1, j0:j1])
                if r == 0:
                    Tr = np.zeros((i1 - i0, j1 - j0))
                else:
                    Ur = U[:, :r] * np.sqrt(s[:r]); Vr = Vt[:r].T * np.sqrt(s[:r])
                    Tr = jp.stored(Ur, p) @ jp.stored(Vr, p).T
                K[i0:i1, j0:j1] = Tr; K[j0:j1, i0:i1] = Tr.T
                byt += bb; mix[p] += 1; ranks.append(r)
        return K, dict(offdiag_bytes=int(byt), offdiag_bytes_frac_of_dense=byt / dense_bytes_off,
                       precision_mix_tiles={int(k): int(v) for k, v in mix.items()}, mean_rank=float(np.mean(ranks)), max_rank=int(max(ranks)))

    def predict(th, rule):
        K, info = cov(th, rule)
        ll, cf = rules.loglik_dense(K, y)
        if ll is None:
            return None, None, info
        Kp = wg.cov_block(Pp, P, dict(sigma2=th[0], r_s=th[1], r_t=th[2]))
        mu = mean_p + Kp @ cho_solve(cf, y, check_finite=False)
        W = cho_solve(cf, Kp.T, check_finite=False)
        var = th[0] * (1 + th[3]) - (Kp * W.T).sum(1)
        return float(ll), rules.skill(yp, mu, np.sqrt(np.maximum(var, 1e-12))), info

    out = dict(settings=vars(a), n_data=int(len(d)), start_theta=theta0.tolist(), beta=beta.tolist(), fits={})
    if a.eval_at:
        th = np.array([float(x) for x in a.eval_at.split(",")]); out["eval_at"] = th.tolist(); out["evals"] = {}
        for rule in [r for r in a.rules.split(",") if r]:
            t0_ = time.time(); ll, sk, info = predict(th, rule)
            out["evals"][rule] = dict(loglik=ll, skill=sk, storage=info, seconds=time.time() - t0_)
            log(f"EVAL {rule:20s} at theta={np.round(th,4).tolist()}: loglik={ll} skill={sk} storage={info} ({time.time()-t0_:.0f}s)")
        path = os.path.join(a.out, f"windgust_eval_n{a.n}.json"); json.dump(out, open(path, "w"), indent=1); log("saved " + path); return
    for rule in [r for r in a.rules.split(",") if r]:
        trace = dict(evals=0, failures=0); t0_ = time.time()
        def nll(logth):
            K, _ = cov(np.exp(logth), rule)
            ll, _ = rules.loglik_dense(K, y); trace["evals"] += 1
            if ll is None:
                trace["failures"] += 1; return 1e12
            if trace["evals"] % 10 == 0:
                log(f"  {rule} eval {trace['evals']} theta={np.round(np.exp(logth),4).tolist()} nll={-ll:.3f} ({time.time()-t0_:.0f}s)")
            return -ll
        res = optimize.minimize(nll, np.log(theta0), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=0.05, maxiter=a.maxiter))
        for k in range(a.restarts):                       # restart from the result: a standard remedy for premature Nelder-Mead convergence on a ridge
            prev = res.fun
            res = optimize.minimize(nll, res.x, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=0.05, maxiter=a.maxiter))
            log(f"  {rule} restart {k+1}: nll {prev:.3f} -> {res.fun:.3f}")
            if prev - res.fun < 0.05: break
        th = np.exp(res.x)
        ll_hat, sk, info = predict(th, rule)
        out["fits"][rule] = dict(theta_hat=th.tolist(), loglik_at_theta_hat=ll_hat, evals=trace["evals"], failures=trace["failures"],
                                 seconds=time.time() - t0_, skill=sk, storage_at_theta_hat=info)
        log(f"FIT {rule:20s} theta_hat={np.round(th,4).tolist()} loglik={ll_hat} evals={trace['evals']} fail={trace['failures']} skill={sk} storage={info} ({time.time()-t0_:.0f}s)")
    if "dense" in out["fits"]:
        thd = np.array(out["fits"]["dense"]["theta_hat"])
        for rule, rec in out["fits"].items():
            rec["theta_hat_rel_to_dense"] = (np.array(rec["theta_hat"]) / thd - 1).tolist()
            ll_exact, _ = rules.loglik_dense(cov_full(P, np.array(rec["theta_hat"])), y)
            rec["exact_loglik_at_theta_hat"] = None if ll_exact is None else float(ll_exact)
            rec["exact_loglik_loss_vs_dense_fit"] = None if ll_exact is None else float(out["fits"]["dense"]["loglik_at_theta_hat"] - ll_exact)
            log(f"{rule:20s} theta_hat/dense-1={np.round(rec['theta_hat_rel_to_dense'],4).tolist()} exact loglik loss vs dense fit={rec['exact_loglik_loss_vs_dense_fit']}")
    path = os.path.join(a.out, f"windgust_fit_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1); log("saved " + path)


if __name__ == "__main__":
    main()
