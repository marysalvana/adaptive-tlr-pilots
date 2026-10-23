"""MCMC pilot on a GPU (2026-10-21): Metropolis-Hastings for a Matern(3/2) Gaussian process with
nugget, run with the exact (dense) likelihood and with the adaptive-TLR likelihood (and optionally a
fixed-tolerance TLR likelihood) on the same simulated field, seed, start, and proposal.

Same experiment as 16_tlr_mcmc_pilot.py (same data for the same seed and N), written in the style of
the companion mixed-precision Bayesian GP code: PyTorch on one GPU (CPU fallback), the tile SVDs batched
in one call, Cholesky by torch.linalg.cholesky_ex so a failed factorization rejects the proposal,
a checkpoint every `--ckpt_every` iterations with resume, and per-iteration timing.  Reports
posterior means, standard deviations, 95% credible intervals, the difference of posterior means in
dense-posterior-SD units, acceptance rates, factorization failures, held-out prediction from the
posterior mean, and the stored fraction of the TLR covariance at the posterior mean.

Usage:  python3 19_tlr_mcmc_gpu.py --n 5000 --iters 3000 --burn 500 --device cuda --out ../results/mcmc_gpu
Reproducible: fixed seeds; settings recorded in the JSON output.  N must be a multiple of --b.
"""
import argparse, json, os, time, importlib.util
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rules", os.path.join(here, "11_tlr_rank_rules.py"))
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)
LO, HI = np.log([0.1, 0.02, 0.001]), np.log([10.0, 2.0, 1.0])
SQRT3 = 3.0 ** 0.5


class Likelihood:
    """Log-likelihood of y under Sigma(theta) with the given rule, on `device`."""

    def __init__(self, S, y, b, device, dtype=torch.float64, svd_mode="fp32", rand_k=64, rand_power=2, seed=0):
        self.svd_mode = svd_mode; self.rand_k = rand_k; self.rand_power = rand_power; self.last_fallback = 0
        self.n = S.shape[0]; self.b = b; self.nB = self.n // b
        assert self.n % b == 0, "N must be a multiple of the tile size"
        self.dev = device; self.dt = dtype
        St = torch.as_tensor(S, dtype=dtype, device=device)
        d2 = (St[:, None, 0] - St[None, :, 0]) ** 2 + (St[:, None, 1] - St[None, :, 1]) ** 2
        self.D = torch.sqrt(torch.clamp(d2, min=0.0))                     # n x n distances
        self.y = torch.as_tensor(y, dtype=dtype, device=device)
        self.eye = torch.eye(self.n, dtype=dtype, device=device)
        ii, jj = np.tril_indices(self.nB, k=-1)                            # off-diagonal tiles (i > j)
        self.ti = torch.as_tensor(ii, device=device); self.tj = torch.as_tensor(jj, device=device)
        self.last = {}
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.omega = torch.randn(len(ii), b, rand_k, generator=g, dtype=dtype).to(device)   # fixed test matrix: reproducible

    def cov(self, theta):
        sigma2, r, eta = theta
        x = SQRT3 * self.D / r
        return sigma2 * ((1.0 + x) * torch.exp(-x)) + sigma2 * eta * self.eye

    def truncate(self, K, rule, value, m_theta):
        """Replace every off-diagonal tile by a rank-r truncation (batched); returns stored values.
        svd_mode 'fp64': batched double-precision SVD (exact, slow on GPUs without fast FP64).
        svd_mode 'fp32': the singular vectors come from a single-precision batched SVD; the
        truncated tile is the double-precision projection T V_r V_r^T onto the leading r right
        singular vectors, and its error ||T - T_r||_F is computed in double precision, so the rank
        is the smallest one whose *actual* error meets the rule (verified, not estimated)."""
        b, nB = self.b, self.nB
        T = K.view(nB, b, nB, b).permute(0, 2, 1, 3)[self.ti, self.tj]      # (ntile, b, b)
        fro = torch.sqrt((T ** 2).sum((1, 2)))
        if rule == "fixed":
            thr = value * fro
        else:
            thr = torch.full((T.shape[0],), value * b * m_theta / self.n, dtype=self.dt, device=self.dev)
        if self.svd_mode == "fp64":
            U, s, Vh = torch.linalg.svd(T, full_matrices=False)
            tail2 = torch.flip(torch.cumsum(torch.flip(s ** 2, [1]), 1), [1])
            tail = torch.sqrt(torch.cat([tail2, torch.zeros_like(tail2[:, :1])], 1))
            ranks = (tail <= thr[:, None]).to(torch.int64).argmax(1)
            keep = (torch.arange(b, device=self.dev)[None, :] < ranks[:, None]).to(self.dt)
            Tr = (U * (s * keep)[:, None, :]) @ Vh
        elif self.svd_mode == "fp32":
            _, _, Vh32 = torch.linalg.svd(T.to(torch.float32), full_matrices=False)
            V = Vh32.transpose(1, 2).to(self.dt)                              # (ntile, b, b), columns ~orthonormal
            V, _ = torch.linalg.qr(V)                                          # exactly orthonormal in FP64
            C = T @ V                                                          # (ntile, b, b): coefficients
            cum = torch.cumsum((C ** 2).sum(1), 1)                             # ||T V_{1..r}||_F^2
            err2 = torch.clamp(fro[:, None] ** 2 - cum, min=0.0)               # exact ||T - T V_r V_r^T||_F^2
            err = torch.sqrt(torch.cat([fro[:, None] ** 2, err2], 1).clamp(min=0.0))   # err[:, r], r = 0..b
            ranks = (err <= thr[:, None]).to(torch.int64).argmax(1)
            keep = (torch.arange(b, device=self.dev)[None, :] < ranks[:, None]).to(self.dt)
            Tr = (C * keep[:, None, :]) @ V.transpose(1, 2)
        else:
            # randomized range finder (Halko-Martinsson-Tropp) with power iterations, batched matmuls only:
            # Q spans the leading k right singular directions; the truncated tile is T V_r V_r^T with V_r
            # the leading r directions of the small projected matrix, and the error is exact in FP64.
            k = self.rand_k
            Om = self.omega[:T.shape[0], :, :k]                                 # fixed Gaussian test matrix
            Y = T.transpose(1, 2) @ (T @ Om)                                    # (ntile, b, k): (T^T T) Om
            for _ in range(self.rand_power):
                Q, _ = torch.linalg.qr(Y); Y = T.transpose(1, 2) @ (T @ Q)
            Q, _ = torch.linalg.qr(Y)                                           # (ntile, b, k) orthonormal
            Bs = T @ Q                                                          # (ntile, b, k) = T Q
            # small eigen-problem of Q^T T^T T Q (k x k) gives the leading right singular directions in span(Q)
            G = Bs.transpose(1, 2) @ Bs                                         # (ntile, k, k)
            evals, evecs = torch.linalg.eigh(G)                                 # ascending
            evecs = torch.flip(evecs, [2])                                      # descending
            V = Q @ evecs                                                       # (ntile, b, k)
            C = T @ V                                                           # coefficients, (ntile, b, k)
            cum = torch.cumsum((C ** 2).sum(1), 1)                              # ||T V_{1..r}||_F^2
            err2 = torch.clamp(fro[:, None] ** 2 - cum, min=0.0)
            err = torch.sqrt(torch.cat([fro[:, None] ** 2, err2], 1))           # err[:, r], r = 0..k
            ok = (err <= thr[:, None])
            ranks = ok.to(torch.int64).argmax(1)
            need_full = ~ok.any(1)                                              # budget not met within k directions
            keep = (torch.arange(k, device=self.dev)[None, :] < ranks[:, None]).to(self.dt)
            Tr = (C * keep[:, None, :]) @ V.transpose(1, 2)
            if bool(need_full.any()):                                           # exact fallback for those tiles
                idx = torch.nonzero(need_full)[:, 0]
                U, s, Vh = torch.linalg.svd(T[idx], full_matrices=False)
                tail2 = torch.flip(torch.cumsum(torch.flip(s ** 2, [1]), 1), [1])
                tail = torch.sqrt(torch.cat([tail2, torch.zeros_like(tail2[:, :1])], 1))
                r_f = (tail <= thr[idx][:, None]).to(torch.int64).argmax(1)
                keep_f = (torch.arange(b, device=self.dev)[None, :] < r_f[:, None]).to(self.dt)
                Tr[idx] = (U * (s * keep_f)[:, None, :]) @ Vh
                ranks[idx] = r_f
            self.last_fallback = int(need_full.sum().item())
        Kt = K.view(nB, b, nB, b).permute(0, 2, 1, 3).clone()
        Kt[self.ti, self.tj] = Tr
        Kt[self.tj, self.ti] = Tr.transpose(1, 2)
        K = Kt.permute(0, 2, 1, 3).reshape(self.n, self.n)
        stored = nB * b * b + 2 * b * int(ranks.sum().item())
        return K, stored, ranks

    def __call__(self, theta, rule, value):
        t0 = time.time()
        K = self.cov(theta)
        stored = self.n * self.n
        if rule != "dense":
            K, stored, ranks = self.truncate(K, rule, value, theta[0] * theta[2])
        L, info = torch.linalg.cholesky_ex(K)
        if int(info.item()) != 0:
            self.last = dict(ok=False, stored=stored, time=time.time() - t0)
            return -float("inf")
        z = torch.linalg.solve_triangular(L, self.y[:, None], upper=False)[:, 0]
        ll = -0.5 * (float(z @ z) + 2.0 * float(torch.log(torch.diag(L)).sum()) + self.n * np.log(2 * np.pi))
        self.last = dict(ok=True, stored=stored, time=time.time() - t0, L=L)
        return ll

    def predict(self, Sp, theta, rule, value):
        ll = self(theta, rule, value)
        if not self.last["ok"]:
            return None
        L = self.last["L"]
        Spt = torch.as_tensor(Sp, dtype=self.dt, device=self.dev)
        # cross covariances
        d2 = (Spt[:, None, 0] - self.Sxy[None, :, 0]) ** 2 + (Spt[:, None, 1] - self.Sxy[None, :, 1]) ** 2
        x = SQRT3 * torch.sqrt(torch.clamp(d2, min=0.0)) / theta[1]
        Kp = theta[0] * ((1.0 + x) * torch.exp(-x))                       # n_pred x n
        alpha = torch.cholesky_solve(self.y[:, None], L)[:, 0]
        mu = Kp @ alpha
        W = torch.linalg.solve_triangular(L, Kp.T, upper=False)
        var = theta[0] * (1 + theta[2]) - (W ** 2).sum(0)
        return mu.cpu().numpy(), torch.sqrt(torch.clamp(var, min=1e-12)).cpu().numpy()


def ess(x):
    """Effective sample size of a 1-d chain by the initial positive sequence estimator (Geyer 1992)."""
    x = np.asarray(x, float) - x.mean(); n = len(x)
    if n < 10 or x.var() == 0: return float(n)
    f = np.fft.rfft(x, 2 * n); acov = np.fft.irfft(f * np.conj(f))[:n] / n; rho = acov / acov[0]
    s = 0.0
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0: break
        s += pair
    return float(n / max(1e-12, 2 * s - 1))


def run_chain(lik, rule, value, start, iters, burn, seed, ckpt, log, scale0=0.08, ckpt_every=100, prop_chol=None):
    rng = np.random.default_rng(seed)
    cur = np.array(start, float); scale = scale0
    Lp = np.eye(3) if prop_chol is None else prop_chol
    chain = np.zeros((iters, 3)); lls = np.zeros(iters); times = np.zeros(iters)
    acc = 0; fail = 0; it0 = 0
    if ckpt and os.path.exists(ckpt):
        ck = np.load(ckpt, allow_pickle=True)
        it0 = int(ck["it"]); chain[:it0] = ck["chain"][:it0]; lls[:it0] = ck["lls"][:it0]; times[:it0] = ck["times"][:it0]
        cur = ck["cur"]; scale = float(ck["scale"]); acc = int(ck["acc"]); fail = int(ck["fail"])
        rng.bit_generator.state = ck["rng_state"].item()
        log(f"  {rule}: resumed at iteration {it0}")
    cur_ll = lik(np.exp(cur), rule, value)
    assert np.isfinite(cur_ll), "initial state: factorization failed"
    t_start = time.time()
    for it in range(it0, iters):
        t0 = time.time()
        prop = cur + scale * (Lp @ rng.standard_normal(3))
        if np.all(prop > LO) and np.all(prop < HI):
            ll = lik(np.exp(prop), rule, value)
            if not lik.last["ok"]:
                fail += 1
            elif np.log(rng.random()) < ll - cur_ll:
                cur, cur_ll = prop, ll; acc += 1
        chain[it] = cur; lls[it] = cur_ll; times[it] = time.time() - t0
        if it < burn and (it + 1) % 50 == 0:
            rate = acc / (it + 1); scale *= np.exp(0.5 * (rate - 0.3))
        if (it + 1) % 250 == 0:
            log(f"  {rule:8s} it={it+1} acc={acc/(it+1):.2f} scale={scale:.3f} theta={np.round(np.exp(cur),3).tolist()} "
                f"{np.mean(times[max(0,it-249):it+1]):.2f}s/it ({time.time()-t_start:.0f}s)")
        if ckpt and (it + 1) % ckpt_every == 0:
            np.savez(ckpt, it=it + 1, chain=chain, lls=lls, times=times, cur=cur, scale=scale, acc=acc, fail=fail,
                     rng_state=np.array(rng.bit_generator.state, dtype=object))
    return chain, lls, acc / iters, fail, scale, float(times.sum())


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
    ap.add_argument("--chains", type=str, default="dense,adaptive:0.9", help="comma list, e.g. dense,adaptive:0.9,fixed:1e-4")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--rand_k", type=int, default=64)
    ap.add_argument("--prop_cov", type=str, default=None, help=".npy 3x3 covariance of log(theta) from a pilot chain; proposal = scale * chol(cov) N(0,I) with scale tuned in burn-in from 2.38/sqrt(3)")
    ap.add_argument("--svd", type=str, default="fp32", choices=["fp32", "fp64", "rand"], help="rank decision from an FP32 SVD with FP64 projection and verified error, a full FP64 SVD, or a randomized range finder (k=--rand_k, batched matmuls, exact fallback) with verified error")
    ap.add_argument("--out", type=str, default="../results/mcmc_gpu")
    ap.add_argument("--chain_seed", type=int, default=None, help="seed of the chain's proposals and starting jitter (default --seed + 1); the data seed is --seed")
    ap.add_argument("--start_mult", type=str, default="1.3,0.8,1.5", help="starting value as multipliers of theta_true (dispersed starts for several chains)")
    a = ap.parse_args()
    a.chain_seed = a.seed + 1 if a.chain_seed is None else a.chain_seed
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)
    dev = torch.device(a.device)
    log(f"device {dev}" + (f" {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    # data: identical to 16_tlr_mcmc_pilot.py for the same seed and N
    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    lik = Likelihood(S, y, a.b, dev, svd_mode=a.svd, rand_k=a.rand_k, seed=a.seed)
    lik.Sxy = torch.as_tensor(S, dtype=torch.float64, device=dev)
    log(f"simulated N={a.n} (+{a.n_pred} held out); tiles b={a.b}, nB={a.n // a.b}")
    t0 = time.time(); ll0 = lik(theta_true, "dense", None); t_dense = time.time() - t0
    t0 = time.time(); ll1 = lik(theta_true, "adaptive", 0.9); t_ad = time.time() - t0
    log(f"loglik(theta_true): dense {ll0:.3f} in {t_dense:.2f}s; adaptive 0.9 {ll1:.3f} in {t_ad:.2f}s "
        f"(stored {lik.last['stored']/(a.n*a.n):.3f} of dense)")

    start = np.log(theta_true * np.array([float(v) for v in a.start_mult.split(",")]))
    prop_chol = None; scale0 = 0.08
    if a.prop_cov:
        prop_chol = np.linalg.cholesky(np.load(a.prop_cov)); scale0 = 2.38 / np.sqrt(3)
        log(f"proposal shaped by {a.prop_cov}, initial scale {scale0:.3f}")
    out_path = os.path.join(a.out, f"tlr_mcmc_pilot_n{a.n}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else dict(settings=vars(a), theta_true=theta_true.tolist(), chains={})
    out["settings"] = vars(a)
    for spec_ in [s for s in a.chains.split(",") if s]:
        rule, value = (spec_.split(":") + [None])[:2]; value = None if value is None else float(value)
        name = rule if rule == "dense" else f"{rule}_{value:g}"
        if name in out["chains"] and out["chains"][name].get("complete"):
            log(f"{name}: already complete, skipping"); continue
        ckpt = os.path.join(a.out, f"ckpt_n{a.n}_{name}.npz")
        try:
            chain, lls, acc, fail, scale, sec = run_chain(lik, rule, value, start, a.iters, a.burn, a.chain_seed, ckpt, log,
                                                          scale0=scale0, ckpt_every=a.ckpt_every, prop_chol=prop_chol)
        except AssertionError as e:                       # the rule cannot even start: indefinite at the start state
            out["chains"][name] = dict(rule=rule, value=value, complete=True, could_not_start=True, reason=str(e),
                                       start_theta=np.exp(start).tolist())
            log(f"{name}: could not start ({e})"); json.dump(out, open(out_path, "w"), indent=1); continue
        summ = summarize(chain, a.burn); th_mean = np.array(summ["mean"])
        summ["ess"] = [ess(chain[a.burn:, k]) for k in range(3)]
        summ["mc_se_of_mean"] = (np.array(summ["sd"]) / np.sqrt(np.maximum(summ["ess"], 1.0))).tolist()
        np.save(os.path.join(a.out, f"mcmc_n{a.n}_{name}_logcov.npy"), np.cov(chain[a.burn:].T))
        pr = lik.predict(Sp, th_mean, rule, value)
        sk = rules.skill(yp, *pr) if pr is not None else None
        stored = None
        if rule != "dense":
            lik(th_mean, rule, value); stored = lik.last["stored"] / (a.n * a.n)
        out["chains"][name] = dict(rule=rule, value=value, summary=summ, acceptance=acc, factorization_failures=fail,
                                   proposal_scale=scale, seconds=sec, seconds_per_iteration=sec / a.iters,
                                   skill_at_posterior_mean=sk, stored_fraction_at_posterior_mean=stored, complete=True)
        np.save(os.path.join(a.out, f"mcmc_n{a.n}_{name}_chain.npy"), chain)
        log(f"{name}: mean={np.round(th_mean,4).tolist()} sd={np.round(summ['sd'],4).tolist()} ess={np.round(summ['ess'],0).tolist()} "
            f"mc_se={np.round(summ['mc_se_of_mean'],4).tolist()} acc={acc:.2f} failures={fail} skill={sk} stored={stored} ({sec:.0f}s, {sec/a.iters:.2f}s/it)")
        json.dump(out, open(out_path, "w"), indent=1)
    if "dense" in out["chains"]:
        d = out["chains"]["dense"]["summary"]
        for name, c in out["chains"].items():
            if name == "dense": continue
            t = c["summary"]
            c["posterior_mean_difference_in_dense_SD"] = (np.abs(np.array(t["mean"]) - np.array(d["mean"])) / np.array(d["sd"])).tolist()
            c["posterior_sd_ratio_over_dense"] = (np.array(t["sd"]) / np.array(d["sd"])).tolist()
            log(f"{name}: posterior mean difference in dense SD units: {np.round(c['posterior_mean_difference_in_dense_SD'],3).tolist()}; "
                f"SD ratio {np.round(c['posterior_sd_ratio_over_dense'],3).tolist()}")
        json.dump(out, open(out_path, "w"), indent=1)
    log("saved " + out_path)


if __name__ == "__main__":
    main()
