"""Rank-adaptive TLR pilot experiment (2026-10-20).

Compares three ways of choosing the rank of every off-diagonal covariance tile of a
Matern (nu = 3/2) Gaussian process with nugget, on a simulated field:

  dense     the exact covariance (reference);
  fixed     fixed relative Frobenius tolerance eps per tile, as in ExaGeoStat/HiCMA;
  adaptive  the condition-aware absolute budget
                e_max = tau * b * m(theta) / N,   m(theta) = sigma2 * eta (the nugget),
            which guarantees ||Delta Sigma||_F <= tau * m(theta) <= tau * lambda_min(Sigma),
            so Sigma^TLR stays positive definite (Weyl), for any theta the optimizer visits.

For each rule the script reports stored values (memory) relative to dense, whether the
Cholesky factorization of Sigma^TLR succeeds, the log-likelihood error at the true
parameters, the maximum-likelihood estimates of (sigma2, range, eta) obtained by
optimizing the TLR likelihood (ranks re-chosen at every theta), and prediction at
held-out points (RMSE, CRPS, 90% interval coverage).  Everything runs on a CPU in
NumPy/SciPy; Sigma^TLR is assembled densely so that the factorization itself is exact
and the only approximation is the tile truncation (the quantity the rules control).

Usage:  python3 11_tlr_rank_rules.py --n 10000 --out ../results
Reproducible: fixed seeds; all settings are recorded in the JSON output.
"""
import argparse, json, os, time
import numpy as np
from scipy import optimize
from scipy.special import erf
from scipy.linalg import cho_factor, cho_solve, LinAlgError

SQRT3 = np.sqrt(3.0)


# ----------------------------------------------------------------------------- model
def matern32(h, r):
    x = SQRT3 * h / r
    return (1.0 + x) * np.exp(-x)


def morton_order(xy, order=10):
    n = 1 << order
    x = np.clip((xy[:, 0] * n).astype(np.int64), 0, n - 1).astype(np.uint64)
    y = np.clip((xy[:, 1] * n).astype(np.int64), 0, n - 1).astype(np.uint64)

    def spread(v):
        v = v & np.uint64(0xFFFFFFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
        return v
    return np.argsort(spread(x) | (spread(y) << np.uint64(1)), kind="stable")


def dist(A, B):
    d2 = (A[:, None, 0] - B[None, :, 0]) ** 2 + (A[:, None, 1] - B[None, :, 1]) ** 2
    return np.sqrt(np.maximum(d2, 0.0))


def cov_full(S, theta):
    sigma2, r, eta = theta
    K = sigma2 * matern32(dist(S, S), r)
    K[np.diag_indices_from(K)] += sigma2 * eta
    return K


# ------------------------------------------------------------------------ tile rules
def tile_bounds(n, b):
    bnd = list(range(0, n, b)) + [n]
    return [(bnd[i], bnd[i + 1]) for i in range(len(bnd) - 1)]


def truncate_tiles(K, bounds, rule, value, m_theta):
    """Replace every off-diagonal tile of K (in place, symmetric) by its rank-r_ij
    truncation.  rule='fixed': relative Frobenius tolerance `value` = eps.
    rule='adaptive': absolute budget e_max = value * b * m_theta / N with value = tau.
    Returns (stored values, max relative tile error, max absolute tile error, ranks)."""
    n = K.shape[0]
    b = max(e - s for s, e in bounds)
    stored = sum((e - s) ** 2 for s, e in bounds)                 # dense diagonal tiles
    ranks = []
    emax = value * b * m_theta / n
    max_rel = 0.0; max_abs = 0.0; fro2 = 0.0
    for i, (i0, i1) in enumerate(bounds):
        for j, (j0, j1) in enumerate(bounds[:i]):
            T = K[i0:i1, j0:j1]
            U, s, Vt = np.linalg.svd(T, full_matrices=False)
            tail = np.sqrt(np.cumsum(s[::-1] ** 2)[::-1])        # tail[r] = ||T - T_r||_F for rank r
            tail = np.append(tail, 0.0)
            if rule == "fixed":
                thr = value * np.sqrt((s ** 2).sum())
            else:
                thr = emax
            r = int(np.argmax(tail <= thr))                       # smallest rank meeting the threshold
            Tr = (U[:, :r] * s[:r]) @ Vt[:r] if r > 0 else np.zeros_like(T)
            err = tail[r]
            K[i0:i1, j0:j1] = Tr
            K[j0:j1, i0:i1] = Tr.T
            stored += 2 * r * max(i1 - i0, j1 - j0)
            ranks.append(r)
            fro2 += 2 * err ** 2
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, err / max(np.sqrt((s ** 2).sum()), 1e-300))
    return stored, max_rel, max_abs, np.sqrt(fro2), ranks


# -------------------------------------------------------------------------- inference
def loglik_dense(K, y):
    try:
        c, low = cho_factor(K, lower=True, check_finite=False)
    except LinAlgError:
        return None, None
    alpha = cho_solve((c, low), y, check_finite=False)
    ld = 2.0 * np.log(np.diag(c)).sum()
    n = len(y)
    return -0.5 * (y @ alpha + ld + n * np.log(2 * np.pi)), (c, low)


def make_cov(S, theta, rule, value, bounds):
    K = cov_full(S, theta)
    info = None
    if rule != "dense":
        info = truncate_tiles(K, bounds, rule, value, theta[0] * theta[2])
    return K, info


def neg_loglik(logth, S, y, rule, value, bounds, trace):
    theta = np.exp(logth)
    K, info = make_cov(S, theta, rule, value, bounds)
    ll, _ = loglik_dense(K, y)
    trace["evals"] += 1
    if ll is None:
        trace["failures"] += 1
        return 1e12
    return -ll


def crps_gauss(y, mu, sig):
    z = (y - mu) / sig
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
    return (sig * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))).mean()


def predict(S, y, Sp, theta, rule, value, bounds):
    K, info = make_cov(S, theta, rule, value, bounds)
    ll, cf = loglik_dense(K, y)
    if cf is None:
        return None
    Kp = theta[0] * matern32(dist(Sp, S), theta[1])
    alpha = cho_solve(cf, y, check_finite=False)
    mu = Kp @ alpha
    W = cho_solve(cf, Kp.T, check_finite=False)
    var = theta[0] * (1 + theta[2]) - (Kp * W.T).sum(1)
    return mu, np.sqrt(np.maximum(var, 1e-12))


def skill(yp, mu, sig):
    z = np.abs(yp - mu) / sig
    return dict(rmse=float(np.sqrt(((yp - mu) ** 2).mean())), crps=float(crps_gauss(yp, mu, sig)),
                cover90=float((z <= 1.6448536).mean()))


# -------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250, help="tile size")
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3, help="Matern range (strong dependence)")
    ap.add_argument("--eta", type=float, default=0.05, help="nugget relative to sigma2")
    ap.add_argument("--eps", type=str, default="1e-1,1e-2,1e-3,1e-4")
    ap.add_argument("--tau", type=str, default="0.5,0.9,0.99")
    ap.add_argument("--mle", type=str, default="dense,fixed:1e-2,fixed:1e-3,adaptive:0.9",
                    help="comma list of rules to fit by maximum likelihood; '' to skip")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=str, default="../results")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta])
    n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    # simulate the field exactly (dense Cholesky of the full covariance incl. held-out points)
    t0 = time.time()
    K_all = cov_full(S_all, theta_true)
    L = np.linalg.cholesky(K_all)
    y_all = L @ rng.standard_normal(n_tot)
    del K_all, L
    perm = rng.permutation(n_tot)
    obs, hold = perm[:a.n], perm[a.n:]
    S, y = S_all[obs], y_all[obs]
    Sp, yp = S_all[hold], y_all[hold]
    order = morton_order(S)
    S, y = S[order], y[order]
    bounds = tile_bounds(a.n, a.b)
    print(f"simulated N={a.n} (+{a.n_pred} held out) in {time.time()-t0:.1f}s; tiles b={a.b}, nB={len(bounds)}", flush=True)

    out = dict(settings=vars(a), theta_true=theta_true.tolist(), n_tiles=len(bounds),
               dense_values=a.n * a.n, rules=[])

    # reference likelihood and nugget bound
    K, _ = make_cov(S, theta_true, "dense", None, bounds)
    ll_dense, cf = loglik_dense(K, y)
    m_theta = theta_true[0] * theta_true[2]
    pr = predict(S, y, Sp, theta_true, "dense", None, bounds)
    out["dense_loglik_true"] = float(ll_dense)
    out["m_theta"] = float(m_theta)
    out["dense_skill_true_theta"] = skill(yp, *pr)
    print(f"dense: loglik(theta_true)={ll_dense:.3f}, skill={out['dense_skill_true_theta']}", flush=True)
    del K, cf

    rules = [("fixed", float(e)) for e in a.eps.split(",")] + [("adaptive", float(t)) for t in a.tau.split(",")]
    for rule, value in rules:
        t0 = time.time()
        K, (stored, max_rel, max_abs, fro, ranks) = make_cov(S, theta_true, rule, value, bounds)
        ll, cf = loglik_dense(K, y)
        rec = dict(rule=rule, value=value, stored_values=int(stored), memory_frac=stored / (a.n * a.n),
                   mean_rank=float(np.mean(ranks)), max_rank=int(np.max(ranks)), min_rank=int(np.min(ranks)),
                   max_tile_rel_err=float(max_rel), max_tile_abs_err=float(max_abs), delta_fro=float(fro),
                   delta_fro_over_m=float(fro / m_theta), pd=ll is not None,
                   loglik_err=None if ll is None else float(ll - ll_dense))
        if ll is not None:
            rec["skill_true_theta"] = skill(yp, *predict(S, y, Sp, theta_true, rule, value, bounds))
        rec["seconds"] = time.time() - t0
        out["rules"].append(rec)
        print(f"{rule:8s} {value:<7g} mem={rec['memory_frac']:.3f} rank(mean/max)={rec['mean_rank']:.1f}/{rec['max_rank']} "
              f"||dS||_F/m={rec['delta_fro_over_m']:.3g} PD={rec['pd']} dloglik={rec['loglik_err']} "
              f"skill={rec.get('skill_true_theta')} ({rec['seconds']:.0f}s)", flush=True)
        del K, cf

    # maximum likelihood under each rule (ranks re-chosen at every theta)
    out["mle"] = []
    for spec in [s for s in a.mle.split(",") if s]:
        rule, value = (spec.split(":") + [None])[:2]
        value = None if value is None else float(value)
        trace = dict(evals=0, failures=0)
        t0 = time.time()
        x0 = np.log(theta_true * np.array([1.5, 0.6, 2.0]))     # dispersed start
        res = optimize.minimize(neg_loglik, x0, args=(S, y, rule, value, bounds, trace), method="Nelder-Mead",
                                options=dict(xatol=1e-3, fatol=0.05, maxiter=300))
        th = np.exp(res.x)
        pr = predict(S, y, Sp, th, rule, value, bounds)
        rec = dict(rule=rule, value=value, theta_hat=th.tolist(), neg_loglik=float(res.fun),
                   evals=trace["evals"], failures=trace["failures"], seconds=time.time() - t0,
                   skill=None if pr is None else skill(yp, *pr))
        if rule != "dense":
            _, info = make_cov(S, th, rule, value, bounds)
            rec["memory_frac_at_theta_hat"] = info[0] / (a.n * a.n)
        out["mle"].append(rec)
        print(f"MLE {spec:14s} theta_hat={np.round(th,4).tolist()} evals={trace['evals']} fail={trace['failures']} "
              f"skill={rec['skill']} ({rec['seconds']:.0f}s)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"tlr_rank_rules_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1)
    # data for the Vecchia baseline (R / GpGp)
    np.savetxt(os.path.join(a.out, f"sim_n{a.n}_obs.csv"), np.column_stack([S, y]), delimiter=",", header="x,y,z", comments="")
    np.savetxt(os.path.join(a.out, f"sim_n{a.n}_holdout.csv"), np.column_stack([Sp, yp]), delimiter=",", header="x,y,z", comments="")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
