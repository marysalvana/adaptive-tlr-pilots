"""TLR Cholesky pilot (2026-10-21): the rank rules inside an actual tile low-rank factorization.

Unlike 11_tlr_rank_rules.py, which assembles Sigma^TLR densely and factorizes it exactly, this
script never forms the dense matrix.  Every off-diagonal tile is stored as a low-rank pair
(U, V) and the Cholesky factorization proceeds tile by tile (right-looking, as in HiCMA /
ExaGeoStat): dense POTRF on the diagonal tile, a triangular solve that leaves each low-rank
tile's rank unchanged, and Schur-complement updates whose low-rank sums are recompressed
under the same rule that chose the initial ranks.  So the rule is applied where the software
applies it, and the factorization can fail (an indefinite Schur complement) in exactly the
way a fixed tolerance can make it fail.

Rules (as in 11_tlr_rank_rules.py):
  fixed     relative Frobenius tolerance eps per tile (initial compression and recompression);
  adaptive  absolute budget e_max = tau * b * m(theta) / N, m(theta) = sigma2 * eta.

Reported per rule, at the true parameters and after a maximum-likelihood fit with the TLR
likelihood: whether the factorization completed, stored values of the factor (diagonal tiles
dense, off-diagonal tiles 2 b r_ij after recompression) relative to a dense factor, the
log-likelihood error against the dense Cholesky, wall time, the estimates, and prediction at
held-out points through TLR triangular solves (RMSE, CRPS, 90% coverage).

Usage:  python3 17_tlr_cholesky_pilot.py --n 10000 --out ../results/tlr_cholesky
Reproducible: fixed seeds; settings recorded in the JSON output.
"""
import argparse, json, os, time, importlib.util
import numpy as np
from scipy import optimize
from scipy.linalg import cholesky, solve_triangular, LinAlgError

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)


# ------------------------------------------------------------------ low-rank helpers
def truncate_svd(T, rule, thr_rel, thr_abs):
    """Rank-r truncation of a small dense tile by the rule; returns (U, V) with T ~ U V^T."""
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    tail = np.append(np.sqrt(np.cumsum(s[::-1] ** 2)[::-1]), 0.0)
    thr = thr_rel * np.sqrt((s ** 2).sum()) if rule == "fixed" else thr_abs
    r = int(np.argmax(tail <= thr))
    return U[:, :r] * s[:r], Vt[:r].T


def recompress(U, V, rule, thr_rel, thr_abs):
    """Recompress U V^T (U: b x k, V: b x k) to the smallest rank meeting the rule."""
    if U.shape[1] == 0:
        return U, V
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V)
    W = Ru @ Rv.T                                    # k x k core
    Uw, s, Vwt = np.linalg.svd(W)
    tail = np.append(np.sqrt(np.cumsum(s[::-1] ** 2)[::-1]), 0.0)
    thr = thr_rel * np.sqrt((s ** 2).sum()) if rule == "fixed" else thr_abs
    r = int(np.argmax(tail <= thr))
    return Qu @ (Uw[:, :r] * s[:r]), Qv @ Vwt[:r].T


# ------------------------------------------------------------------ TLR Cholesky
def tlr_cholesky(S, theta, bounds, rule, value):
    """Factorize Sigma(theta) ~ L L^T tile by tile without forming Sigma.
    Returns (L_diag, L_off, info) or (None, None, info) if a diagonal tile is not positive definite.
    L_off[(i, j)] = (U, V) with L_ij ~ U V^T, i > j."""
    n = S.shape[0]
    nB = len(bounds)
    b = max(e - s for s, e in bounds)
    sigma2, r, eta = theta
    m_theta = sigma2 * eta
    thr_rel = value if rule == "fixed" else None
    thr_abs = value * b * m_theta / n if rule == "adaptive" else None
    # initial tiles
    A_diag = []
    A_off = {}
    stored0 = 0
    for i, (i0, i1) in enumerate(bounds):
        D = sigma2 * rules.matern32(rules.dist(S[i0:i1], S[i0:i1]), r)
        D[np.diag_indices_from(D)] += m_theta
        A_diag.append(D)
        stored0 += (i1 - i0) ** 2
        for j, (j0, j1) in enumerate(bounds[:i]):
            T = sigma2 * rules.matern32(rules.dist(S[i0:i1], S[j0:j1]), r)
            A_off[(i, j)] = truncate_svd(T, rule, thr_rel, thr_abs)
            stored0 += 2 * A_off[(i, j)][0].shape[1] * max(i1 - i0, j1 - j0)
    info = dict(stored_sigma=int(stored0), max_rank_sigma=max(u.shape[1] for u, _ in A_off.values()) if A_off else 0)
    # right-looking factorization
    L_diag = [None] * nB
    L_off = {}
    for k in range(nB):
        try:
            Lkk = cholesky(A_diag[k], lower=True, check_finite=False)
        except LinAlgError:
            info.update(completed=False, failed_at=k)
            return None, None, info
        L_diag[k] = Lkk
        # panel: L_ik = A_ik L_kk^{-T} = U (L_kk^{-1} V)^T
        for i in range(k + 1, nB):
            U, V = A_off.pop((i, k))
            if U.shape[1]:
                V = solve_triangular(Lkk, V, lower=True, check_finite=False)
            L_off[(i, k)] = (U, V)
        # trailing update A_ij -= L_ik L_jk^T, j <= i
        for i in range(k + 1, nB):
            Ui, Vi = L_off[(i, k)]
            if Ui.shape[1] == 0:
                continue
            for j in range(k + 1, i + 1):
                Uj, Vj = L_off[(j, k)]
                if Uj.shape[1] == 0:
                    continue
                W = Vi.T @ Vj                                   # r_i x r_j
                if i == j:
                    A_diag[i] -= (Ui @ W) @ Ui.T
                else:
                    Uij, Vij = A_off[(i, j)]
                    Unew = np.hstack([Uij, -(Ui @ W)])
                    Vnew = np.hstack([Vij, Uj])
                    A_off[(i, j)] = recompress(Unew, Vnew, rule, thr_rel, thr_abs)
    stored = sum(d.shape[0] ** 2 for d in L_diag) + sum(2 * u.shape[1] * u.shape[0] for u, _ in L_off.values())
    ranks = [u.shape[1] for u, _ in L_off.values()]
    info.update(completed=True, stored_factor=int(stored), mean_rank_factor=float(np.mean(ranks)) if ranks else 0.0,
                max_rank_factor=int(max(ranks)) if ranks else 0)
    return L_diag, L_off, info


def tlr_forward(L_diag, L_off, bounds, B):
    """Solve L Z = B (B: n x m) by tile forward substitution."""
    Z = np.zeros_like(B)
    for k, (k0, k1) in enumerate(bounds):
        rhs = B[k0:k1].copy()
        for j, (j0, j1) in enumerate(bounds[:k]):
            U, V = L_off[(k, j)]
            if U.shape[1]:
                rhs -= U @ (V.T @ Z[j0:j1])
        Z[k0:k1] = solve_triangular(L_diag[k], rhs, lower=True, check_finite=False)
    return Z


def tlr_backward(L_diag, L_off, bounds, Z):
    """Solve L^T X = Z by tile backward substitution (L_ij^T = V U^T)."""
    nB = len(bounds)
    X = np.zeros_like(Z)
    for k in range(nB - 1, -1, -1):
        k0, k1 = bounds[k]
        rhs = Z[k0:k1].copy()
        for i in range(k + 1, nB):
            i0, i1 = bounds[i]
            U, V = L_off[(i, k)]
            if U.shape[1]:
                rhs -= V @ (U.T @ X[i0:i1])
        X[k0:k1] = solve_triangular(L_diag[k], rhs, lower=True, trans="T", check_finite=False)
    return X


def tlr_loglik(S, y, theta, bounds, rule, value):
    L_diag, L_off, info = tlr_cholesky(S, theta, bounds, rule, value)
    if L_diag is None:
        return None, None, info
    z = tlr_forward(L_diag, L_off, bounds, y[:, None])[:, 0]
    ld = 2.0 * sum(np.log(np.diag(d)).sum() for d in L_diag)
    ll = -0.5 * (z @ z + ld + len(y) * np.log(2 * np.pi))
    return float(ll), (L_diag, L_off), info


def tlr_predict(S, y, Sp, theta, bounds, fac):
    L_diag, L_off = fac
    Kp = theta[0] * rules.matern32(rules.dist(Sp, S), theta[1])          # n_pred x n
    alpha = tlr_backward(L_diag, L_off, bounds, tlr_forward(L_diag, L_off, bounds, y[:, None]))[:, 0]
    mu = Kp @ alpha
    W = tlr_forward(L_diag, L_off, bounds, Kp.T)                          # n x n_pred
    var = theta[0] * (1 + theta[2]) - (W ** 2).sum(0)
    return mu, np.sqrt(np.maximum(var, 1e-12))


def neg_loglik_tlr(logth, S, y, bounds, rule, value, trace):
    theta = np.exp(logth)
    trace["evals"] += 1
    if rule == "dense":
        K = rules.cov_full(S, theta)
        ll, _ = rules.loglik_dense(K, y)
    else:
        ll, _, _ = tlr_loglik(S, y, theta, bounds, rule, value)
    if ll is None:
        trace["failures"] += 1
        return 1e12
    return -ll


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--rules", type=str, default="fixed:1e-2,fixed:1e-3,fixed:1e-4,fixed:1e-5,adaptive:0.9",
                    help="rules evaluated at the true theta")
    ap.add_argument("--mle", type=str, default="dense,fixed:1e-4,adaptive:0.9",
                    help="rules fitted by maximum likelihood with the TLR Cholesky likelihood; '' to skip")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--dense", type=int, default=1, help="1: compute the dense reference (needs N^2 memory); 0: skip it (large N)")
    ap.add_argument("--out", type=str, default="../results/tlr_cholesky")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta])
    n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    t0 = time.time()
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    print(f"simulated N={a.n} (+{a.n_pred} held out) in {time.time()-t0:.1f}s; tiles b={a.b}, nB={len(bounds)}", flush=True)
    out = dict(settings=vars(a), theta_true=theta_true.tolist(), n_tiles=len(bounds),
               dense_factor_values=a.n * (a.n + 1) // 2, rules=[], mle=[])

    # dense reference (skipped with --dense 0, where N^2 values do not fit)
    ll_dense = None
    if a.dense:
        t0 = time.time()
        K = rules.cov_full(S, theta_true)
        ll_dense, cf = rules.loglik_dense(K, y)
        pr = rules.predict(S, y, Sp, theta_true, "dense", None, bounds)
        out["dense_loglik_true"] = float(ll_dense); out["dense_seconds"] = time.time() - t0
        out["dense_skill_true_theta"] = rules.skill(yp, *pr)
        print(f"dense: loglik(theta_true)={ll_dense:.3f} skill={out['dense_skill_true_theta']} ({time.time()-t0:.0f}s)", flush=True)
        del K, cf

    # rules at the true theta, through the TLR Cholesky
    for spec in [s for s in a.rules.split(",") if s]:
        rule, value = spec.split(":"); value = float(value)
        t0 = time.time()
        ll, fac, info = tlr_loglik(S, y, theta_true, bounds, rule, value)
        rec = dict(rule=rule, value=value, completed=info["completed"], stored_sigma=info["stored_sigma"],
                   sigma_memory_frac=info["stored_sigma"] / (a.n * a.n), seconds=time.time() - t0)
        if info["completed"]:
            rec.update(stored_factor=info["stored_factor"], factor_memory_frac=info["stored_factor"] / out["dense_factor_values"],
                       mean_rank_factor=info["mean_rank_factor"], max_rank_factor=info["max_rank_factor"],
                       loglik=ll, loglik_err=None if ll_dense is None else ll - ll_dense,
                       skill_true_theta=rules.skill(yp, *tlr_predict(S, y, Sp, theta_true, bounds, fac)))
        else:
            rec.update(failed_at_tile=info["failed_at"])
        out["rules"].append(rec)
        print(f"{rule:8s} {value:<7g} completed={rec['completed']} sigma_mem={rec['sigma_memory_frac']:.3f} "
              f"factor_mem={rec.get('factor_memory_frac')} rank(mean/max)={rec.get('mean_rank_factor')}/{rec.get('max_rank_factor')} "
              f"dloglik={rec.get('loglik_err')} skill={rec.get('skill_true_theta')} ({rec['seconds']:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"tlr_cholesky_n{a.n}.json"), "w"), indent=1)

    # maximum likelihood with the TLR Cholesky likelihood (ranks re-chosen at every theta)
    for spec in [s for s in a.mle.split(",") if s]:
        rule, value = (spec.split(":") + [None])[:2]
        value = None if value is None else float(value)
        trace = dict(evals=0, failures=0)
        t0 = time.time()
        x0 = np.log(theta_true * np.array([1.5, 0.6, 2.0]))
        res = optimize.minimize(neg_loglik_tlr, x0, args=(S, y, bounds, rule, value, trace), method="Nelder-Mead",
                                options=dict(xatol=1e-3, fatol=0.05, maxiter=300))
        th = np.exp(res.x)
        rec = dict(rule=rule, value=value, theta_hat=th.tolist(), neg_loglik=float(res.fun), evals=trace["evals"],
                   failures=trace["failures"], seconds=time.time() - t0, skill=None)
        if rule == "dense":
            pr = rules.predict(S, y, Sp, th, "dense", None, bounds)
            if pr is not None:
                rec["skill"] = rules.skill(yp, *pr)
        else:
            ll, fac, info = tlr_loglik(S, y, th, bounds, rule, value)
            rec["completed_at_theta_hat"] = info["completed"]
            if info["completed"]:
                rec["factor_memory_frac_at_theta_hat"] = info["stored_factor"] / out["dense_factor_values"]
                rec["skill"] = rules.skill(yp, *tlr_predict(S, y, Sp, th, bounds, fac))
        out["mle"].append(rec)
        print(f"MLE {spec:14s} theta_hat={np.round(th,4).tolist()} evals={trace['evals']} fail={trace['failures']} "
              f"factor_mem={rec.get('factor_memory_frac_at_theta_hat')} skill={rec['skill']} ({rec['seconds']:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"tlr_cholesky_n{a.n}.json"), "w"), indent=1)

    np.savetxt(os.path.join(a.out, f"sim_n{a.n}_obs.csv"), np.column_stack([S, y]), delimiter=",", header="x,y,z", comments="")
    np.savetxt(os.path.join(a.out, f"sim_n{a.n}_holdout.csv"), np.column_stack([Sp, yp]), delimiter=",", header="x,y,z", comments="")
    print("saved", os.path.join(a.out, f"tlr_cholesky_n{a.n}.json"))


if __name__ == "__main__":
    main()
