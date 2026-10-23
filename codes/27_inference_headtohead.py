"""Inference-level head-to-head (2026-10-21): adaptive TLR and Vecchia on the exact likelihood, per screening field.

For one scenario folder of 24_screening_stress.py (N = 10,000, exact likelihood available):
 (1) a verified exact maximum-likelihood estimate: Nelder-Mead, restarted until it stops improving, from the best of
     several starts (the truth, 24's exact fit, the Vecchia fits of 25 and of 30 under 8 orderings, a previous run);
 (2) the adaptive-TLR maximum-likelihood estimate (tau = 0.9, 17's tile Cholesky with recompression), started from the
     same point; its exact log-likelihood gives the gap below the exact MLE, as for Vecchia's fits;
 (3) the shape error of each approximate likelihood: on a grid of parameters around the exact MLE (Sobol points in a
     box of +-0.3 in each log parameter), the spread of l_approx(theta) - l_exact(theta) after removing its mean, so a
     constant shift (which leaves the MLE and the posterior unchanged) does not count.  Vecchia's values on the grid
     come from 28_vecchia_grid.R (grouped and ungrouped, m = 30 and 60).

Stages:  python3 27_inference_headtohead.py fit  --tag T   (exact and TLR MLE, grid exact and TLR values, grid.csv)
         Rscript 28_vecchia_grid.R <dir>                     (Vecchia on the grid -> vecchia_grid.csv)
         python3 27_inference_headtohead.py score --tag T   (gaps and shape errors -> inference.json)
         python3 27_inference_headtohead.py report          (markdown table over all fields)
"""
import argparse, json, os, sys, time, importlib.util
import numpy as np
from scipy import optimize
from scipy.stats import qmc

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(here, "24_screening_stress.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
tlr, rules = ss.tlr, ss.rules


def load(d):
    out = json.load(open(os.path.join(d, "stress.json"))); st = out["settings"]; n = st["n"]
    ss.set_kernel(st["nu"])
    obs = np.loadtxt(os.path.join(d, f"sim_n{n}_obs.csv"), delimiter=",", skiprows=1)
    return out, st, obs[:, :2], obs[:, 2], rules.tile_bounds(n, st["b"])


def ll_exact(S, y, th):
    ll, _ = rules.loglik_dense(rules.cov_full(S, np.asarray(th)), y)
    return None if ll is None else float(ll)


def ll_tlr(S, y, th, bounds, tau, trace=None):
    """Adaptive-TLR log-likelihood; None if the factorization fails or, at an extreme trial point of the optimizer, a
    tile SVD does not converge (counted in trace["numerical_errors"], so such points stay visible)."""
    try:
        ll, _, info = tlr.tlr_loglik(S, y, np.asarray(th), bounds, "adaptive", tau)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        if trace is not None: trace["numerical_errors"] = trace.get("numerical_errors", 0) + 1
        return None, dict(completed=False)
    return (None if ll is None else float(ll)), info


def nm(f, x0, trace, restarts=6, tol=0.01):
    """Nelder-Mead in log parameters, restarted from its own result (fresh simplex of +-10% per parameter) until a
    restart improves the log-likelihood by less than tol: on flat likelihoods (large nugget, long range) a single run can
    stop at its starting point."""
    def obj(x):
        trace["evals"] += 1
        v = f(np.exp(x))
        if v is None:
            trace["failures"] += 1; return 1e12
        return -v
    x = np.log(np.asarray(x0, float)); best = obj(x); trace["restarts"] = 0
    for k in range(restarts):
        simplex = np.vstack([x] + [x + 0.1 * np.eye(3)[j] for j in range(3)])
        r = optimize.minimize(obj, x, method="Nelder-Mead",
                              options=dict(xatol=1e-4, fatol=1e-3, maxiter=600, initial_simplex=simplex))
        trace["restarts"] = k + 1
        improved = best - r.fun
        if r.fun < best: x, best = r.x, r.fun
        if improved < tol: break
    return np.exp(x), -float(best)


def stage_fit(a, d):
    out, st, S, y, bounds = load(d)
    tau = st["tau"]
    res = dict(tau=tau)
    starts = {"truth": out["theta_true"]}
    if "exact_mle" in out: starts["24 exact fit"] = out["exact_mle"]["theta_hat"]
    vf = json.load(open(os.path.join(d, "vecchia.json")))
    for f in vf["fits"]:
        starts[f"Vecchia m={f['m']} fit"] = [f["theta_hat"]["sigma2"], f["theta_hat"]["range"], f["theta_hat"]["eta"]]
    fo = os.path.join(d, "vecchia_fit_orderings.json")             # Vecchia fits under 8 orderings (30), when present
    if os.path.exists(fo):
        for r in json.load(open(fo)):
            starts[f"Vecchia m={r['m']} ordering {r['seed']}"] = [r["theta_hat"]["sigma2"], r["theta_hat"]["range"], r["theta_hat"]["eta"]]
    if os.path.exists(os.path.join(d, "inference.json")):          # a previous run's estimates, when present
        old = json.load(open(os.path.join(d, "inference.json")))
        starts["previous exact MLE"] = old["exact_mle"]["theta_hat"]; starts["previous TLR MLE"] = old["tlr_mle"]["theta_hat"]
    sv = {k: ll_exact(S, y, v) for k, v in starts.items()}
    best = max(sv, key=lambda k: -np.inf if sv[k] is None else sv[k])
    x0 = np.array(starts[best]); res["starts"] = dict(values=sv, chosen=best)
    print(f"[{a.tag}] exact log-likelihood at the starts {sv}; start from {best}", flush=True)
    t0 = time.time(); tr = dict(evals=0, failures=0)
    th_e, l_e = nm(lambda th: ll_exact(S, y, th), x0, tr)
    res["exact_mle"] = dict(theta_hat=th_e.tolist(), loglik=l_e, evals=tr["evals"], restarts=tr["restarts"], seconds=time.time() - t0)
    print(f"[{a.tag}] verified exact MLE {np.round(th_e, 4).tolist()} loglik={l_e:.3f} ({tr['evals']} evals)", flush=True)
    t0 = time.time(); tr = dict(evals=0, failures=0)
    th_t, l_t = nm(lambda th: ll_tlr(S, y, th, bounds, tau, tr)[0], x0, tr)
    l_t_exact = ll_exact(S, y, th_t)
    _, info = ll_tlr(S, y, th_t, bounds, tau)
    res["tlr_mle"] = dict(theta_hat=th_t.tolist(), tlr_loglik=l_t, exact_loglik=l_t_exact, evals=tr["evals"], restarts=tr["restarts"],
                          failures=tr["failures"], numerical_errors=tr.get("numerical_errors", 0), seconds=time.time() - t0,
                          factor_frac=info["stored_factor"] / (st["n"] * (st["n"] + 1) // 2) if info["completed"] else None)
    print(f"[{a.tag}] adaptive-TLR MLE {np.round(th_t, 4).tolist()} exact loglik there={l_t_exact:.3f} "
          f"failures={tr['failures']}", flush=True)
    # grid around the exact MLE
    z = qmc.Sobol(3, scramble=True, seed=20261021).random(a.grid)
    G = th_e * np.exp(0.3 * (2 * z - 1))
    G = np.vstack([th_e, G])
    rows = []
    for th in G:
        le = ll_exact(S, y, th); lt, info = ll_tlr(S, y, th, bounds, tau)
        rows.append([*th, le, lt])
    rows = np.array(rows, float)
    np.savetxt(os.path.join(d, "grid.csv"), rows, delimiter=",", header="sigma2,range,eta,exact,tlr", comments="")
    json.dump(res, open(os.path.join(d, "inference.json"), "w"), indent=1)
    print(f"[{a.tag}] grid of {len(G)} points written", flush=True)


def stage_score(a, d):
    out = json.load(open(os.path.join(d, "stress.json")))
    res = json.load(open(os.path.join(d, "inference.json")))
    l_best = res["exact_mle"]["loglik"]
    res["tlr_mle"]["gap_below_exact_mle"] = l_best - res["tlr_mle"]["exact_loglik"]
    vf = json.load(open(os.path.join(d, "vecchia.json")))
    S, y = None, None
    res["vecchia_fits"] = []
    for f in out["vecchia_fits"]:
        res["vecchia_fits"].append(dict(m=f["m"], theta_hat=f["theta_hat"], exact_loglik=f["exact_loglik_at_theta_hat"],
                                        gap_below_exact_mle=None if f["exact_loglik_at_theta_hat"] is None
                                        else l_best - f["exact_loglik_at_theta_hat"]))
    g = np.genfromtxt(os.path.join(d, "grid.csv"), delimiter=",", names=True)
    v = np.genfromtxt(os.path.join(d, "vecchia_grid.csv"), delimiter=",", names=True)
    shape = {}
    def spread(approx):
        dlt = approx - g["exact"]; ok = np.isfinite(dlt)
        dlt = dlt[ok] - dlt[ok].mean()
        return dict(sd=float(dlt.std()), max_abs=float(np.abs(dlt).max()), points=int(ok.sum()))
    shape["adaptive TLR"] = spread(g["tlr"])
    groups = {}                                       # Vecchia: one column per seeded ordering (28); mean, min, max of the SD
    for col in v.dtype.names:
        groups.setdefault(col.rsplit("_o", 1)[0], []).append(spread(v[col]))
    for name, lst in groups.items():
        sds = [x["sd"] for x in lst]
        shape[name] = dict(sd=float(np.mean(sds)), sd_min=float(min(sds)), sd_max=float(max(sds)), orderings=len(lst),
                           max_abs=float(np.mean([x["max_abs"] for x in lst])), points=lst[0]["points"])
    res["shape_error"] = shape
    res["grid_note"] = "Sobol points in exact_mle * exp(+-0.3) per parameter, plus the exact MLE; spread of l_approx - l_exact after removing its mean"
    json.dump(res, open(os.path.join(d, "inference.json"), "w"), indent=1)
    print(f"[{a.tag}] gaps: TLR {res['tlr_mle']['gap_below_exact_mle']:.2f}, Vecchia "
          + ", ".join(f"m={x['m']} {x['gap_below_exact_mle']:.2f}" for x in res["vecchia_fits"])
          + "; shape SD " + ", ".join(f"{k} {s['sd']:.2f}" for k, s in shape.items()), flush=True)


def stage_fitorders(a, d):
    """Score the Vecchia fits of 30_vecchia_fit_orderings.R (one per seeded ordering) on the exact likelihood."""
    out, st, S, y, bounds = load(d)
    res = json.load(open(os.path.join(d, "inference.json"))); l_best = res["exact_mle"]["loglik"]
    fo_name = a.fo_file or "vecchia_fit_orderings.json"           # 34_vecchia_fit_meanzero.R writes vecchia_fit_orderings_meanzero.json
    key_suffix = "" if fo_name == "vecchia_fit_orderings.json" else "_meanzero"
    rows = json.load(open(os.path.join(d, fo_name)))
    summ = {}
    for r in rows:
        th = [r["theta_hat"]["sigma2"], r["theta_hat"]["range"], r["theta_hat"]["eta"]]
        le = ll_exact(S, y, th); r["exact_loglik"] = le; r["gap_below_exact_mle"] = None if le is None else l_best - le
        summ.setdefault(r["m"], []).append(r["gap_below_exact_mle"])
    res["vecchia_fit_orderings" + key_suffix] = rows
    res["vecchia_fit_gap_over_orderings" + key_suffix] = {str(m): dict(mean=float(np.mean(g)), min=float(min(g)), max=float(max(g)), orderings=len(g))
                                             for m, g in summ.items()}
    json.dump(res, open(os.path.join(d, "inference.json"), "w"), indent=1)
    print(f"[{a.tag}] Vecchia fit gap over orderings: " + ", ".join(f"m={m} mean {np.mean(g):.2f} min {min(g):.2f} max {max(g):.2f}"
          for m, g in summ.items()) + f"; TLR {res['tlr_mle']['gap_below_exact_mle']:.2f}", flush=True)


def stage_report(a):
    print("| nu | range | eta | gap below exact MLE: adaptive TLR / Vecchia m=30 / m=60 | shape error SD: adaptive TLR / Vecchia m=30 / m=60 (ungrouped; grouped) |")
    print("|---|---|---|---|---|")
    rows = []
    for tag in sorted(os.listdir(a.out)):
        f = os.path.join(a.out, tag, "inference.json")
        if not os.path.exists(f) or "shape_error" not in json.load(open(f)): continue
        r = json.load(open(f)); st = json.load(open(os.path.join(a.out, tag, "stress.json")))["settings"]
        v = {x["m"]: x["gap_below_exact_mle"] for x in r["vecchia_fits"]}; sh = r["shape_error"]
        rows.append((st["eta"], st["nu"], st["range"], r["tlr_mle"]["gap_below_exact_mle"], v, sh))
    for eta, nu, rg, gt, v, sh in sorted(rows):
        print(f"| {nu:g} | {rg:g} | {eta:g} | {gt:.2f} / {v[30]:.2f} / {v[60]:.2f} | {sh['adaptive TLR']['sd']:.2f} / "
              f"{sh['ungrouped_m30']['sd']:.2f}; {sh['grouped_m30']['sd']:.2f} / {sh['ungrouped_m60']['sd']:.2f}; {sh['grouped_m60']['sd']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "score", "fitorders", "report"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--out", type=str, default="../results/screening_stress")
    ap.add_argument("--fo_file", type=str, default=None, help="fitorders: Vecchia fit file to score (default vecchia_fit_orderings.json; 34 writes vecchia_fit_orderings_meanzero.json)")
    a = ap.parse_args()
    if a.stage == "report":
        return stage_report(a)
    d = os.path.join(a.out, a.tag)
    {"fit": stage_fit, "score": stage_score, "fitorders": stage_fitorders}[a.stage](a, d)


if __name__ == "__main__":
    main()
