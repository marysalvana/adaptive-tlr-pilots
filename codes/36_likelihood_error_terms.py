"""Decomposition of the TLR log-likelihood error into the terms the Obj. 1 bound uses (2026-10-23).

For the assembled Sigma^TLR of a rank rule at the true parameters, with Delta = Sigma^TLR - Sigma,
alpha = Sigma^{-1} y, z = L^{-1} y (Sigma = L L^T) and E = L^{-1} Delta L^{-T}:

  error        ell^TLR - ell                                   (exact, two dense Cholesky factorizations)
  T1           -1/2 tr(Sigma^{-1} Delta) + 1/2 alpha^T Delta alpha   (first-order term)
  T1 bound     sum_{i>j} ( ||(Sigma^{-1})_{ij}||_F + ||alpha_i|| ||alpha_j|| ) e_ij     (tile by tile, Cauchy-Schwarz),
               with e_ij = ||Delta_ij||_F the actual tile errors, and the a priori version with e_ij replaced
               by the rule's budget e_max
  remainder    error - T1, bounded N-free for the log-determinant part by ||E||_F^2 / (4 (1 - ||E||_2)) and for
               the quadratic form by ||E z||^2 / (2 (1 - ||E||_2)) with ||E z||^2 = (Delta alpha)^T Sigma^{-1} (Delta alpha)
               (the loose version ||E||_2^2 ||z||^2 / (2 (1 - ||E||_2)) is also reported).  All of T1, ||E||_F, ||E||_2 and
               ||E z|| are computable from the TLR factor and the truncation residuals (Hutchinson estimates with TLR solves).
  eta route    the uniform bound 1/2 [ N (-log(1 - eta)) + eta/(1 - eta) ||z||^2 ] with eta = ||E||_2 (actual),
               eta_hat = sqrt(2 sum e_ij^2) / m(theta) (a posteriori), and tau (a priori): this is the O(N tau) number.

Same field as 17/18 (seed 20261020, Morton order, b = 250).  Dense N x N arrays: needs about 5 N^2 doubles.
Usage:  python3 36_likelihood_error_terms.py --n 10000 --rules adaptive:0.9,adaptive:0.5,fixed:1e-4,fixed:1e-5
"""
import argparse, json, os, time, importlib.util
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse.linalg import eigsh

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)


def tile_norms(D, bounds):
    """||D_ij||_F for i > j and the per-block norms of a vector are computed by the caller; here: matrix tiles."""
    out = {}
    for i, (i0, i1) in enumerate(bounds):
        for j, (j0, j1) in enumerate(bounds[:i]):
            out[(i, j)] = float(np.linalg.norm(D[i0:i1, j0:j1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--rules", type=str, default="adaptive:0.9,adaptive:0.5,fixed:1e-4,fixed:1e-5")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=str, default="../results/bound_terms")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)

    rng = np.random.default_rng(a.seed)
    theta = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs = perm[:a.n]
    S, y = S_all[obs], y_all[obs]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b); nB = len(bounds); b = a.b
    m_theta = theta[0] * theta[2]
    log(f"simulated N={a.n}; tiles b={b}, nB={nB}; m(theta)={m_theta:.4f}")

    t0 = time.time()
    K = rules.cov_full(S, theta)
    c, low = cho_factor(K, lower=True, check_finite=False)
    ll = -0.5 * (y @ cho_solve((c, low), y, check_finite=False) + 2.0 * np.log(np.diag(c)).sum() + a.n * np.log(2 * np.pi))
    alpha = cho_solve((c, low), y, check_finite=False)
    z = solve_triangular(c, y, lower=True, check_finite=False); z2 = float(z @ z)
    Kinv = cho_solve((c, low), np.eye(a.n), check_finite=False)
    Lchol = np.tril(c)                                     # the factor itself (cho_factor leaves garbage above)
    del c
    Kinv_tiles = tile_norms(Kinv, bounds)
    alpha_blk = [float(np.linalg.norm(alpha[i0:i1])) for (i0, i1) in bounds]
    sumKinv = sum(Kinv_tiles.values()); sumalpha = sum(alpha_blk[i] * alpha_blk[j] for i in range(nB) for j in range(i))
    log(f"dense: loglik {ll:.3f}, ||z||^2 {z2:.1f}, inverse and factor in {time.time()-t0:.0f}s")

    out = dict(settings=vars(a), theta=theta.tolist(), m_theta=m_theta, dense_loglik=float(ll), z2=z2,
               sum_Kinv_tile_norms=sumKinv, sum_alpha_block_products=sumalpha, rules={})
    for spec_ in [s for s in a.rules.split(",") if s]:
        rule, value = spec_.split(":"); value = float(value)
        name = f"{rule}_{value:g}"
        t0 = time.time()
        Kt = K.copy()
        stored, max_rel, max_abs, fro, ranks = rules.truncate_tiles(Kt, bounds, rule, value, m_theta)
        emax = value * b * m_theta / a.n if rule == "adaptive" else None
        llt, cf = rules.loglik_dense(Kt, y)
        if llt is None:
            out["rules"][name] = dict(rule=rule, value=value, completed=False, stored_frac=stored / a.n ** 2,
                                      max_abs_tile_err=max_abs, sqrt2sum_e2=fro)
            log(f"{name}: assembled Sigma^TLR not positive definite"); continue
        del cf
        Kt -= K                                            # Delta
        D = Kt
        e_tiles = tile_norms(D, bounds)
        err = float(llt - ll)
        tr_term = float((Kinv * D).sum())                  # tr(Sigma^{-1} Delta)
        q_term = float(alpha @ (D @ alpha))
        T1 = -0.5 * tr_term + 0.5 * q_term
        B_ld = sum(Kinv_tiles[k] * e_tiles[k] for k in e_tiles)          # |1/2 tr| <= 1/2 * 2 * sum_{i>j} ...
        B_q = sum(alpha_blk[i] * alpha_blk[j] * e_tiles[(i, j)] for (i, j) in e_tiles)
        # E = L^{-1} Delta L^{-T}
        E = solve_triangular(Lchol, D, lower=True, check_finite=False)
        E = solve_triangular(Lchol, E.T, lower=True, check_finite=False)
        E_F = float(np.linalg.norm(E))
        E_2 = float(abs(eigsh(E, k=1, which="LM", return_eigenvectors=False)[0]))
        del E
        eta_hat = fro / m_theta
        w = D @ alpha                                      # quadratic-form remainder: 1/2 z^T E^2 (I+E)^{-1} z <= 1/2 ||E z||^2 / (1 - ||E||_2), ||E z||^2 = (Delta alpha)^T Sigma^{-1} (Delta alpha)
        Ez2 = float(w @ (Kinv @ w))
        def eta_route(eta):
            if eta >= 1: return None
            return float(0.5 * (a.n * (-np.log1p(-eta)) + eta / (1 - eta) * z2))
        rec = dict(rule=rule, value=value, completed=True, stored_frac=stored / a.n ** 2, mean_rank=float(np.mean(ranks)),
                   max_rank=int(max(ranks)), e_max=emax, max_abs_tile_err=max_abs, max_rel_tile_err=max_rel,
                   sqrt2sum_e2=fro, eta_hat=eta_hat,
                   error=err, first_order=T1, first_order_trace_part=-0.5 * tr_term, first_order_quadratic_part=0.5 * q_term,
                   remainder=err - T1,
                   T1_bound_tilewise=B_ld + B_q, T1_bound_tilewise_logdet=B_ld, T1_bound_tilewise_quadratic=B_q,
                   T1_bound_apriori_emax=None if emax is None else emax * (sumKinv + sumalpha),
                   E_fro=E_F, E_2=E_2,
                   remainder_bound_logdet=E_F ** 2 / (4 * (1 - E_2)) if E_2 < 1 else None,
                   remainder_bound_quadratic=Ez2 / (2 * (1 - E_2)) if E_2 < 1 else None, Ez2=Ez2,
                   remainder_bound_quadratic_loose=E_2 ** 2 * z2 / (2 * (1 - E_2)) if E_2 < 1 else None,
                   remainder_bound_logdet_apriori=eta_hat ** 2 / (4 * (1 - eta_hat)) if eta_hat < 1 else None,
                   eta_route_bound_actual=eta_route(E_2), eta_route_bound_eta_hat=eta_route(eta_hat),
                   eta_route_bound_tau=eta_route(value) if rule == "adaptive" else None,
                   seconds=time.time() - t0)
        out["rules"][name] = rec
        log(f"{name}: stored {rec['stored_frac']:.3f} | error {err:+.4f} = T1 {T1:+.4f} + remainder {err-T1:+.4f} | "
            f"T1 tile bound {B_ld+B_q:.3f} (a priori {rec['T1_bound_apriori_emax']}) | ||E||_2 {E_2:.4f} ||E||_F {E_F:.4f} eta_hat {eta_hat:.3f} | "
            f"remainder bounds logdet {rec['remainder_bound_logdet']:.4f} quad {rec['remainder_bound_quadratic']:.4f} | "
            f"eta-route {rec['eta_route_bound_actual']:.1f} (eta_hat: {rec['eta_route_bound_eta_hat']}) ({time.time()-t0:.0f}s)")
        del D
    path = os.path.join(a.out, f"bound_terms_n{a.n}.json")
    json.dump(out, open(path, "w"), indent=1)
    log("saved " + path)


if __name__ == "__main__":
    main()
