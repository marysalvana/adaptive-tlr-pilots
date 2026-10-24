"""Metropolis-Hastings driven by the TLR Cholesky likelihood itself (2026-10-23, Objective 2).

The MCMC pilot of 19_tlr_mcmc_gpu.py assembled Sigma^TLR densely and shaped its proposal from the exact
chain's posterior covariance.  Here (1) the likelihood is the TLR Cholesky factorization with recompression
of 17_tlr_cholesky_pilot.py (nothing dense is formed; a failed factorization rejects the proposal), and
(2) the proposal is tuned from the chain itself: a diagonal random walk on log(theta) whose scale is
adapted during burn-in, re-shaped once at half burn-in by the covariance of the draws so far, and held
fixed after burn-in (so the chain is a fixed-kernel Metropolis-Hastings sampler from then on).  No exact
chain is used.  The same data as the pilot (seed, N, b), dispersed starts, checkpoint and resume.

Usage:
  python3 38_tlr_cholesky_mcmc.py --n 5000 --rule adaptive:0.9 --iters 10000 --burn 2000 --chain_seed 101 --start_mult 1.3,0.8,1.5 --out ../results/preliminary/mcmc_tlrchol/adaptive_0.9/c1
  python3 38_tlr_cholesky_mcmc.py --pool --out ../results/preliminary/mcmc_tlrchol        # pooled summaries, split R-hat, ESS
"""
import argparse, json, os, glob, time, importlib.util
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(here, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rules = load("rules", "11_tlr_rank_rules.py")
tlr = load("tlr", "17_tlr_cholesky_pilot.py")
jp = load("jp", "18_joint_rank_precision_pilot.py")      # 2026-10-24: rule joint:tau (rank and storage precision under the budget, Obj. 3)
from scipy.linalg import cho_solve
LO, HI = np.log([0.1, 0.02, 0.001]), np.log([10.0, 2.0, 1.0])


def simulate(a):
    rng = np.random.default_rng(a.seed)
    theta_true = np.array([a.sigma2, a.range, a.eta]); n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    L = np.linalg.cholesky(rules.cov_full(S_all, theta_true)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    return theta_true, S, y, Sp, yp


class Lik:
    def __init__(self, S, y, bounds, rule, value):
        self.S, self.y, self.bounds, self.rule, self.value = S, y, bounds, rule, value
        self.last = {}
    def __call__(self, theta):
        t0 = time.time()
        if self.rule == "dense":
            ll, cf = rules.loglik_dense(rules.cov_full(self.S, theta), self.y)
            self.last = dict(ok=ll is not None, time=time.time() - t0, cf=cf)
            return -np.inf if ll is None else ll
        if self.rule in ("joint", "rtp"):
            K, cf, info = self.rounded(theta)
            ll, cf = rules.loglik_dense(K, self.y)
            self.last = dict(ok=ll is not None, time=time.time() - t0, cf=cf, info=info)
            return -np.inf if ll is None else ll
        ll, fac, info = tlr.tlr_loglik(self.S, self.y, theta, self.bounds, self.rule, self.value)
        self.last = dict(ok=ll is not None, time=time.time() - t0, fac=fac, info=info)
        return -np.inf if ll is None else ll
    def rounded(self, theta):
        """Sigma^TLR assembled from the rounded factors, rank and precision chosen under the budget at theta."""
        n = len(self.y); b = self.bounds[0][1] - self.bounds[0][0]
        emax = self.value * b * theta[0] * theta[2] / n
        def pick(i, j, T):
            opts, svd = jp.tile_options(T, emax, 1e-4)
            r, p, bb, err = opts["joint" if self.rule == "joint" else "rank_then_precision"]
            return r, p, bb, err, svd
        K, byt, mix, ranks, errs = jp.assemble(self.S, theta, self.bounds, self.rule, pick)
        diag = sum((i1 - i0) ** 2 * 8 for i0, i1 in self.bounds)
        info = dict(offdiag_bytes=int(byt), precision_mix_tiles={int(k): int(v) for k, v in mix.items()}, mean_rank=float(np.mean(ranks)),
                    stored_factor=(byt + diag) / 8.0)        # in FP64-entry units, as tlr_loglik reports (whole matrix, both triangles / 2)
        return K, None, info
    def predict(self, Sp, theta):
        if self.rule == "dense":
            return rules.predict(self.S, self.y, Sp, theta, "dense", None, self.bounds)
        if self.rule in ("joint", "rtp"):
            K, _, info = self.rounded(theta)
            ll, cf = rules.loglik_dense(K, self.y)
            if ll is None: return None, None
            self.last = dict(ok=True, info=info)
            Kp = theta[0] * rules.matern32(rules.dist(Sp, self.S), theta[1])
            mu = Kp @ cho_solve(cf, self.y, check_finite=False)
            W = cho_solve(cf, Kp.T, check_finite=False)
            var = theta[0] * (1 + theta[2]) - (Kp * W.T).sum(1)
            return mu, np.sqrt(np.maximum(var, 1e-12))
        ll, fac, info = tlr.tlr_loglik(self.S, self.y, theta, self.bounds, self.rule, self.value)
        if ll is None: return None, None
        self.last = dict(ok=True, info=info)
        return tlr.tlr_predict(self.S, self.y, Sp, theta, self.bounds, fac)


def run_chain(lik, start, iters, burn, seed, ckpt, log, ckpt_every=100):
    rng = np.random.default_rng(seed)
    cur = np.array(start, float); scale = 0.08; Lp = np.eye(3); shaped = False
    chain = np.zeros((iters, 3)); lls = np.zeros(iters); times = np.zeros(iters)
    acc = 0; acc_win = 0; fail = 0; it0 = 0
    if ckpt and os.path.exists(ckpt):
        ck = np.load(ckpt, allow_pickle=True)
        it0 = int(ck["it"]); chain[:it0] = ck["chain"][:it0]; lls[:it0] = ck["lls"][:it0]; times[:it0] = ck["times"][:it0]
        cur = ck["cur"]; scale = float(ck["scale"]); Lp = ck["Lp"]; shaped = bool(ck["shaped"]); acc = int(ck["acc"]); fail = int(ck["fail"])
        acc_win = int(ck["acc_win"]); rng.bit_generator.state = ck["rng_state"].item()
        log(f"  resumed at iteration {it0}")
    cur_ll = lik(np.exp(cur))
    assert np.isfinite(cur_ll), "initial state: factorization failed"
    t_start = time.time()
    for it in range(it0, iters):
        t0 = time.time()
        prop = cur + scale * (Lp @ rng.standard_normal(3))
        if np.all(prop > LO) and np.all(prop < HI):
            ll = lik(np.exp(prop))
            if not lik.last["ok"]:
                fail += 1
            elif np.log(rng.random()) < ll - cur_ll:
                cur, cur_ll = prop, ll; acc += 1; acc_win += 1
        chain[it] = cur; lls[it] = cur_ll; times[it] = time.time() - t0
        if it < burn:
            if (it + 1) % 50 == 0:                                   # Robbins-Monro on the scale, target acceptance 0.3
                rate = acc_win / 50; acc_win = 0; scale *= np.exp(0.5 * (rate - 0.3))
            if (it + 1) == burn // 2 and not shaped:                  # shape from the chain's own draws, no exact chain
                cov = np.cov(chain[burn // 4:it + 1].T) + 1e-8 * np.eye(3)
                Lp = np.linalg.cholesky(cov); scale = 2.38 / np.sqrt(3); shaped = True
                log(f"  proposal re-shaped at iteration {it+1} from the chain's covariance; scale {scale:.3f}")
        if (it + 1) % 250 == 0:
            log(f"  it={it+1} acc={acc/(it+1):.2f} scale={scale:.3f} theta={np.round(np.exp(cur),3).tolist()} "
                f"fail={fail} {np.mean(times[max(0,it-249):it+1]):.2f}s/it ({time.time()-t_start:.0f}s)")
        if ckpt and (it + 1) % ckpt_every == 0:
            np.savez(ckpt, it=it + 1, chain=chain, lls=lls, times=times, cur=cur, scale=scale, Lp=Lp, shaped=shaped,
                     acc=acc, acc_win=acc_win, fail=fail, rng_state=np.array(rng.bit_generator.state, dtype=object))
    return chain, lls, acc / iters, fail, scale, Lp, float(times.sum())


def ess1(x):
    x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n] / n; rho = ac / ac[0]
    tau = 1.0; t = 1
    while t + 1 < n and rho[t] + rho[t + 1] > 0:
        tau += 2 * (rho[t] + rho[t + 1]); t += 2
    return n / tau


def pool(a):
    diag = load("diag", "35_mcmc_multichain_diagnostics.py")
    out = dict(burn=a.burn, rules={})
    exact = None
    for rd in sorted(glob.glob(os.path.join(a.out, "*"))):
        if not os.path.isdir(rd): continue
        name = os.path.basename(rd)
        files = sorted(glob.glob(os.path.join(rd, "c*", f"chain_n{a.n}.npy")))
        if not files: continue
        chains = np.stack([np.load(f)[a.burn:] for f in files])                     # (M, n, 3), log scale
        th = np.exp(chains)
        pooled = th.reshape(-1, 3)
        summ = dict(n_chains=len(files), files=files, mean=pooled.mean(0).tolist(), sd=pooled.std(0, ddof=1).tolist(),
                    ci95=np.percentile(pooled, [2.5, 97.5], axis=0).T.tolist(),
                    chain_means=th.mean(1).tolist(), chain_sds=th.std(1, ddof=1).tolist())
        rh, es = zip(*[diag.split_rhat_ess(th[:, :, k]) for k in range(3)])
        summ["split_rhat"] = list(rh); summ["bulk_ess"] = list(es)
        summ["mc_se_of_mean"] = (np.array(summ["sd"]) / np.sqrt(np.maximum(es, 1.0))).tolist()
        meta = [json.load(open(os.path.join(os.path.dirname(f), f"tlr_mcmc_n{a.n}.json"))) for f in files]
        summ["acceptance"] = [m["acceptance"] for m in meta]; summ["factorization_failures"] = [m["factorization_failures"] for m in meta]
        summ["seconds_per_iteration"] = [m["seconds_per_iteration"] for m in meta]
        summ["skill_at_posterior_mean"] = [m.get("skill_at_posterior_mean") for m in meta]
        summ["stored_fraction_at_posterior_mean"] = [m.get("stored_fraction_at_posterior_mean") for m in meta]
        out["rules"][name] = summ
        if name == "dense": exact = summ
        print(f"{name}: mean {np.round(summ['mean'],4).tolist()} sd {np.round(summ['sd'],4).tolist()} R-hat {np.round(rh,3).tolist()} ESS {np.round(es,0).tolist()} "
              f"acc {np.round(summ['acceptance'],2).tolist()} failures {summ['factorization_failures']}")
    if exact:
        for name, s in out["rules"].items():
            if name == "dense": continue
            s["mean_difference_in_exact_sd"] = ((np.array(s["mean"]) - np.array(exact["mean"])) / np.array(exact["sd"])).tolist()
            s["sd_ratio_over_exact"] = (np.array(s["sd"]) / np.array(exact["sd"])).tolist()
            print(f"{name}: mean difference in exact SD {np.round(s['mean_difference_in_exact_sd'],3).tolist()}; SD ratio {np.round(s['sd_ratio_over_exact'],3).tolist()}")
    path = os.path.join(a.out, f"pooled_n{a.n}.json"); json.dump(out, open(path, "w"), indent=1); print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--rule", type=str, default="adaptive:0.9", help="dense, adaptive:tau, fixed:eps, joint:tau or rtp:tau (rounded factors under the budget, joint or rank-then-precision, 2026-10-24)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--chain_seed", type=int, default=101)
    ap.add_argument("--start_mult", type=str, default="1.3,0.8,1.5")
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--out", type=str, default="../results/preliminary/mcmc_tlrchol/adaptive_0.9/c1")
    a = ap.parse_args()
    if a.pool:
        pool(a); return
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)
    theta_true, S, y, Sp, yp = simulate(a)
    bounds = rules.tile_bounds(a.n, a.b)
    rule, value = (a.rule.split(":") + [None])[:2]; value = None if value is None else float(value)
    lik = Lik(S, y, bounds, rule, value)
    log(f"simulated N={a.n} (+{a.n_pred} held out); tiles b={a.b}, nB={len(bounds)}; rule {rule} {value}")
    t0 = time.time(); ll0 = lik(theta_true); log(f"loglik(theta_true) = {ll0:.3f} in {time.time()-t0:.2f}s")
    start = np.log(theta_true * np.array([float(v) for v in a.start_mult.split(",")]))
    path = os.path.join(a.out, f"tlr_mcmc_n{a.n}.json")
    if os.path.exists(path) and json.load(open(path)).get("complete"):
        log("already complete"); return
    ckpt = os.path.join(a.out, f"ckpt_n{a.n}.npz")
    chain, lls, acc, fail, scale, Lp, sec = run_chain(lik, start, a.iters, a.burn, a.chain_seed, ckpt, log, a.ckpt_every)
    th = np.exp(chain[a.burn:])
    summ = dict(mean=th.mean(0).tolist(), sd=th.std(0, ddof=1).tolist(), ci95=np.percentile(th, [2.5, 97.5], axis=0).T.tolist(),
                n_post=int(len(th)), ess=[float(ess1(chain[a.burn:, k])) for k in range(3)])
    th_mean = np.array(summ["mean"])
    mu, sig = lik.predict(Sp, th_mean)
    sk = rules.skill(yp, mu, sig) if mu is not None else None
    stored = None
    if rule != "dense" and lik.last.get("info"):
        stored = lik.last["info"]["stored_factor"] / (a.n * (a.n + 1) / 2)
    out = dict(settings=vars(a), theta_true=theta_true.tolist(), rule=rule, value=value, start_theta=np.exp(start).tolist(),
               summary=summ, acceptance=acc, factorization_failures=fail, final_scale=scale, proposal_chol=Lp.tolist(),
               seconds=sec, seconds_per_iteration=sec / a.iters, skill_at_posterior_mean=sk,
               stored_fraction_at_posterior_mean=stored, complete=True)
    np.save(os.path.join(a.out, f"chain_n{a.n}.npy"), chain)
    json.dump(out, open(path, "w"), indent=1)
    log(f"done: mean={np.round(th_mean,4).tolist()} sd={np.round(summ['sd'],4).tolist()} ess={np.round(summ['ess'],0).tolist()} "
        f"acc={acc:.2f} failures={fail} skill={sk} stored={stored} ({sec:.0f}s, {sec/a.iters:.2f}s/it)")
    log("saved " + path)


if __name__ == "__main__":
    main()
