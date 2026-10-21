"""KL divergence of the adaptive TLR approximation from the exact Gaussian model (2026-10-21).

For a screening field of 24_screening_stress.py (N = 10,000), at the true parameters:
    KL(N(0, Sigma) || N(0, Sigma_tilde)) = 0.5 [ tr(Sigma_tilde^{-1} Sigma) - N + log det Sigma_tilde - log det Sigma ],
with Sigma_tilde = L L^T from 17's tile Cholesky with recompression (adaptive tau = 0.9).  The trace is
||L^{-1} C||_F^2 with C the exact Cholesky factor, computed by TLR forward substitution.  This is the accuracy measure
used for Vecchia approximations (Guinness 2018; Katzfuss and Guinness 2021); 32_kl_vecchia.R gives Vecchia's.
Writes kl_tlr.json in the field folder.

Usage:  python3 31_kl_tlr.py --tag nu25_r3_e005 [--out ../results/screening_stress]
"""
import argparse, json, os, time, importlib.util
import numpy as np
from scipy.linalg import cholesky

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ss", os.path.join(here, "24_screening_stress.py"))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
tlr, rules = ss.tlr, ss.rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="../results/screening_stress")
    a = ap.parse_args()
    d = os.path.join(a.out, a.tag)
    st = json.load(open(os.path.join(d, "stress.json"))); s = st["settings"]; th = np.array(st["theta_true"])
    ss.set_kernel(s["nu"]); n = s["n"]
    S = np.loadtxt(os.path.join(d, f"sim_n{n}_obs.csv"), delimiter=",", skiprows=1)[:, :2]
    bounds = rules.tile_bounds(n, s["b"])
    t0 = time.time()
    C = cholesky(rules.cov_full(S, th), lower=True, overwrite_a=True, check_finite=False)
    logdet = 2.0 * np.log(np.diag(C)).sum()
    Ld, Lo, info = tlr.tlr_cholesky(S, th, bounds, "adaptive", s["tau"])
    logdet_t = 2.0 * sum(np.log(np.diag(x)).sum() for x in Ld)
    X = tlr.tlr_forward(Ld, Lo, bounds, C); del C
    tr = float((X * X).sum()); del X
    kl = 0.5 * (tr - n + logdet_t - logdet)
    res = dict(method="adaptive TLR", tau=s["tau"], kl=kl, trace=tr, logdet_exact=logdet, logdet_approx=logdet_t,
               factor_frac=info["stored_factor"] / (n * (n + 1) // 2), seconds=time.time() - t0)
    json.dump(res, open(os.path.join(d, "kl_tlr.json"), "w"), indent=1)
    print(f"[{a.tag}] KL adaptive TLR = {kl:.4f} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
