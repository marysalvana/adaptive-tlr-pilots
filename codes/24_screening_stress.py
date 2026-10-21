"""Screening-stress head-to-head (2026-10-21): adaptive TLR and Vecchia where Vecchia's screening assumption weakens.

The pilot field of 17_tlr_cholesky_pilot.py (Matern 3/2, range 0.3 on the unit square, relative nugget 0.05) is the
setting in which Vecchia's approximation is known to be accurate.  This script repeats the comparison on fields with a
longer range, a smoother kernel (Matern 5/2), and a larger nugget, each at N = 10,000 so that the exact (dense)
likelihood is available as the reference.

Stage "sim" (one scenario per call): simulate the field by a dense Cholesky; evaluate at the true parameters the exact
log-likelihood, the adaptive-TLR log-likelihood through 17's tile Cholesky (tau = 0.9; ranks, factor storage, error
against exact), and held-out prediction skill of both; fit the exact maximum-likelihood estimate.  Writes
<out>/<tag>/sim_n{N}_obs.csv, _holdout.csv and stress.json.

Stage "eval" (after 25_vecchia_stress.R): score each Vecchia estimate on the exact likelihood (gap below the exact MLE)
and with exact-model prediction at that estimate; adds the results to stress.json.

Usage:  python3 24_screening_stress.py sim  --nu 2.5 --range 1.0 --eta 0.05 --tag nu25_r1_e005
        python3 24_screening_stress.py eval --tag nu25_r1_e005
Reproducible: fixed seed; settings recorded in stress.json.
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np
from scipy import optimize

here = os.path.dirname(os.path.abspath(__file__))
spec17 = importlib.util.spec_from_file_location("tlr", os.path.join(here, "17_tlr_cholesky_pilot.py"))
tlr = importlib.util.module_from_spec(spec17); spec17.loader.exec_module(tlr)
rules = tlr.rules                                   # the helpers 17 itself uses; the kernel is patched below


def matern(nu):
    if nu == 1.5:
        c = np.sqrt(3.0)
        def k(h, r):
            x = c * h / r
            return (1.0 + x) * np.exp(-x)
    elif nu == 2.5:
        c = np.sqrt(5.0)
        def k(h, r):
            x = c * h / r
            return (1.0 + x + x * x / 3.0) * np.exp(-x)
    else:
        raise ValueError("nu must be 1.5 or 2.5")
    return k


def set_kernel(nu):
    rules.matern32 = matern(nu)                     # used by cov_full, predict, tlr_cholesky and tlr_predict


def neg_ll_exact(logth, S, y, trace):
    trace["evals"] += 1
    ll, _ = rules.loglik_dense(rules.cov_full(S, np.exp(logth)), y)
    if ll is None:
        trace["failures"] += 1
        return 1e12
    return -ll


def stage_sim(a, d):
    set_kernel(a.nu)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    th0 = np.array([a.sigma2, a.range, a.eta])
    n_tot = a.n + a.n_pred
    S_all = rng.random((n_tot, 2))
    t0 = time.time()
    L = np.linalg.cholesky(rules.cov_full(S_all, th0)); y_all = L @ rng.standard_normal(n_tot); del L
    perm = rng.permutation(n_tot); obs, hold = perm[:a.n], perm[a.n:]
    S, y, Sp, yp = S_all[obs], y_all[obs], S_all[hold], y_all[hold]
    order = rules.morton_order(S); S, y = S[order], y[order]
    bounds = rules.tile_bounds(a.n, a.b)
    np.savetxt(os.path.join(d, f"sim_n{a.n}_obs.csv"), np.column_stack([S, y]), delimiter=",", header="x,y,z", comments="")
    np.savetxt(os.path.join(d, f"sim_n{a.n}_holdout.csv"), np.column_stack([Sp, yp]), delimiter=",", header="x,y,z", comments="")
    out = dict(settings=vars(a), theta_true=th0.tolist(), sim_seconds=time.time() - t0)
    path = os.path.join(d, "stress.json")
    print(f"[{a.tag}] simulated N={a.n} in {out['sim_seconds']:.0f}s", flush=True)

    t0 = time.time()
    ll_ex, cf = rules.loglik_dense(rules.cov_full(S, th0), y); del cf
    out["exact_true"] = dict(loglik=float(ll_ex), skill=rules.skill(yp, *rules.predict(S, y, Sp, th0, "dense", None, bounds)),
                             seconds=time.time() - t0)
    print(f"[{a.tag}] exact loglik(true)={ll_ex:.3f} skill={out['exact_true']['skill']}", flush=True)

    t0 = time.time()
    ll, fac, info = tlr.tlr_loglik(S, y, th0, bounds, "adaptive", a.tau)
    rec = dict(tau=a.tau, completed=bool(info["completed"]), seconds=time.time() - t0)
    if info["completed"]:
        rec.update(loglik=ll, loglik_err=ll - ll_ex, factor_frac=info["stored_factor"] / (a.n * (a.n + 1) // 2),
                   mean_rank=info["mean_rank_factor"], max_rank=info["max_rank_factor"],
                   skill=rules.skill(yp, *tlr.tlr_predict(S, y, Sp, th0, bounds, fac)))
    del fac
    out["tlr_true"] = rec
    print(f"[{a.tag}] adaptive TLR: {rec}", flush=True)
    json.dump(out, open(path, "w"), indent=1)

    if a.mle:
        t0 = time.time(); trace = dict(evals=0, failures=0)
        x0 = np.log(th0 * np.array([1.5, 0.6, 2.0]))
        res = optimize.minimize(neg_ll_exact, x0, args=(S, y, trace), method="Nelder-Mead",
                                options=dict(xatol=1e-3, fatol=0.05, maxiter=400))
        th = np.exp(res.x)
        out["exact_mle"] = dict(theta_hat=th.tolist(), loglik=float(-res.fun), evals=trace["evals"], failures=trace["failures"],
                                seconds=time.time() - t0,
                                skill=rules.skill(yp, *rules.predict(S, y, Sp, th, "dense", None, bounds)))
        print(f"[{a.tag}] exact MLE {np.round(th, 4).tolist()} loglik={-res.fun:.3f} ({out['exact_mle']['seconds']:.0f}s)", flush=True)
        json.dump(out, open(path, "w"), indent=1)


def stage_eval(a, d):
    path = os.path.join(d, "stress.json")
    out = json.load(open(path))
    st = out["settings"]
    set_kernel(st["nu"])
    n = st["n"]
    obs = np.loadtxt(os.path.join(d, f"sim_n{n}_obs.csv"), delimiter=",", skiprows=1)
    hold = np.loadtxt(os.path.join(d, f"sim_n{n}_holdout.csv"), delimiter=",", skiprows=1)
    S, y, Sp, yp = obs[:, :2], obs[:, 2], hold[:, :2], hold[:, 2]
    bounds = rules.tile_bounds(n, st["b"])
    vf = json.load(open(os.path.join(d, "vecchia.json")))
    best = out.get("exact_mle", {}).get("loglik", out["exact_true"]["loglik"])
    out["vecchia_fixed_theta"] = vf["fixed_theta"]
    for r in out["vecchia_fixed_theta"]:
        r["loglik_err"] = r["loglik"] - out["exact_true"]["loglik"]
    out["vecchia_fits"] = []
    for f in vf["fits"]:
        th = np.array([f["theta_hat"]["sigma2"], f["theta_hat"]["range"], f["theta_hat"]["eta"]])
        ll, cf = rules.loglik_dense(rules.cov_full(S, th), y); del cf
        g = dict(f, exact_loglik_at_theta_hat=None if ll is None else float(ll),
                 gap_below_exact_mle=None if ll is None else float(best - ll),
                 noise_var=float(th[0] * th[2]),
                 skill_exact_model_at_theta_hat=rules.skill(yp, *rules.predict(S, y, Sp, th, "dense", None, bounds)))
        out["vecchia_fits"].append(g)
        print(f"[{a.tag}] Vecchia m={f['m']} theta_hat={np.round(th, 4).tolist()} gap below exact MLE={g['gap_below_exact_mle']}", flush=True)
    json.dump(out, open(path, "w"), indent=1)


def stage_report(a):
    """Markdown table over every scenario folder in --out (log-likelihood errors at the true parameters)."""
    rows = []
    for tag in sorted(os.listdir(a.out)):
        f = os.path.join(a.out, tag, "stress.json")
        if not os.path.exists(f): continue
        o = json.load(open(f)); st = o["settings"]
        if "vecchia_fixed_theta" not in o: continue
        v = {r["m"]: r["loglik_err"] for r in o["vecchia_fixed_theta"]}
        mf = os.path.join(a.out, tag, "vecchia_matched.json")
        mm = json.load(open(mf)) if os.path.exists(mf) else None
        mt = [r for r in mm["runs"] if r["m"] == mm["m_match"]][0] if mm else None
        t = o["tlr_true"]; fits = {g["m"]: g for g in o["vecchia_fits"]}
        rows.append((st["nu"], st["range"], st["eta"], t, v, mt, fits, o))
    print("| nu | range | eta | adaptive TLR: storage, max rank, error | Vecchia error m = 10 / 30 / 60 / 120 | Vecchia at TLR storage: m, error | exact CRPS / TLR / Vecchia m=30 | Vecchia fit gap below exact MLE, m = 30 / 60 |")
    print("|---|---|---|---|---|---|---|---|")
    for nu, r, e, t, v, mt, fits, o in sorted(rows, key=lambda x: (x[2], x[0], x[1])):
        crps30 = [x["skill"]["crps"] for x in o["vecchia_fixed_theta"] if x["m"] == 30][0]
        vm = f"{mt['m']}, {mt['loglik_err']:+.2f}" if mt else "-"
        print(f"| {nu:g} | {r:g} | {e:g} | {100*t['factor_frac']:.1f}%, {t['max_rank']}, {t['loglik_err']:+.2f} | "
              + " / ".join(f"{v[m]:+.1f}" for m in (10, 30, 60, 120))
              + f" | {vm} | {o['exact_true']['skill']['crps']:.4f} / {t['skill']['crps']:.4f} / {crps30:.4f} | "
              + f"{fits[30]['gap_below_exact_mle']:.2f} / {fits[60]['gap_below_exact_mle']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sim", "eval", "report"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--n_pred", type=int, default=300)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--nu", type=float, default=1.5)
    ap.add_argument("--sigma2", type=float, default=1.0)
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--mle", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=str, default="../results/screening_stress")
    a = ap.parse_args()
    if a.stage == "report":
        return stage_report(a)
    d = os.path.join(a.out, a.tag)
    stage_sim(a, d) if a.stage == "sim" else stage_eval(a, d)


if __name__ == "__main__":
    main()
