"""Compare a rerun of run_screening_stress.sh with the committed results (2026-10-21).

For every field, prints the committed and rerun values of the numbers the README quotes: exact and
adaptive-TLR log-likelihood at the truth, TLR storage, Vecchia log-likelihood errors (m = 30, 60, matched storage),
fit gaps below the exact MLE, and shape errors, and flags any difference above the tolerance.  The simulated
fields come from fixed seeds, so the data should agree to rounding; optimizer results (MLEs, gaps) can move slightly
with the BLAS library and thread count, hence the looser tolerance on those.

Usage:  python3 29_check_reproduction.py ../results/screening_stress_rerun [--ref ../results/screening_stress]
"""
import argparse, json, os


def get(d, tag):
    r = {}
    f = os.path.join(d, tag, "stress.json")
    if not os.path.exists(f): return r
    s = json.load(open(f))
    r["exact loglik at truth"] = s["exact_true"]["loglik"]
    r["TLR loglik error at truth"] = s["tlr_true"].get("loglik_err")
    r["TLR storage fraction"] = s["tlr_true"].get("factor_frac")
    for v in s.get("vecchia_fixed_theta", []):
        if v["m"] in (30, 60): r[f"Vecchia m={v['m']} error at truth"] = v["loglik_err"]
    f = os.path.join(d, tag, "vecchia_matched.json")
    if os.path.exists(f):
        m = json.load(open(f)); x = [u for u in m["runs"] if u["m"] == m["m_match"]][0]
        r["Vecchia error at TLR storage"] = x["loglik_err"]
    f = os.path.join(d, tag, "inference.json")
    if os.path.exists(f):
        i = json.load(open(f))
        r["exact MLE loglik"] = i["exact_mle"]["loglik"]
        if "gap_below_exact_mle" in i["tlr_mle"]: r["TLR fit gap"] = i["tlr_mle"]["gap_below_exact_mle"]
        for v in i.get("vecchia_fits", []): r[f"Vecchia m={v['m']} fit gap"] = v["gap_below_exact_mle"]
        for k, v in i.get("shape_error", {}).items(): r[f"shape SD {k}"] = v["sd"]
        for m, v in i.get("vecchia_fit_gap_over_orderings", {}).items(): r[f"Vecchia m={m} fit gap, mean of orderings"] = v["mean"]
    f = os.path.join(d, tag, "kl_tlr.json")
    if os.path.exists(f): r["KL adaptive TLR"] = json.load(open(f))["kl"]
    f = os.path.join(d, tag, "kl_vecchia.json")
    if os.path.exists(f):
        for x in json.load(open(f))["summary"]: r[f"KL Vecchia m={x['m']} (mean of orderings)"] = x["mean"]
    f = os.path.join(d, tag, "kl_vecchia_matched.json")
    if os.path.exists(f):
        for x in json.load(open(f))["summary"]: r[f"KL Vecchia m={x['m']} (matched run)"] = x["mean"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rerun")
    ap.add_argument("--ref", default="../results/screening_stress")
    ap.add_argument("--tol_data", type=float, default=1e-6, help="absolute tolerance, quantities at fixed parameters")
    ap.add_argument("--tol_opt", type=float, default=0.1, help="absolute tolerance, optimizer results (MLEs, gaps)")
    a = ap.parse_args()
    bad = 0
    for tag in sorted(t for t in os.listdir(a.ref) if os.path.isdir(os.path.join(a.ref, t)) and t != "logs"):
        ref, new = get(a.ref, tag), get(a.rerun, tag)
        print(f"== {tag}")
        for k, v in ref.items():
            w = new.get(k)
            opt = ("gap" in k) or ("MLE" in k)
            tol = a.tol_opt if opt else a.tol_data * max(1.0, abs(v))
            ok = w is not None and v is not None and abs(w - v) <= tol
            bad += not ok
            print(f"   {'ok ' if ok else 'DIFF'} {k:34s} committed {v!s:>22} rerun {w!s:>22}")
    print(f"{bad} differences above tolerance")


if __name__ == "__main__":
    main()
