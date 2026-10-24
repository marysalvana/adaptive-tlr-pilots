"""Joint rank and storage-precision choice on the wind-gust space-time covariance (2026-10-23, Objective 3).

Same comparison as 18_joint_rank_precision_pilot.py (rank only / rank then precision / joint / fixed tolerance,
one error budget per tile e_max = tau b m(theta) / N, m(theta) = sigma2 * eta) but on the covariance matrix of the
SC26 wind-gust application: a random subset of the 131,889 station-hour points of the 2017-2019 storms
(hpc4ees data/train_data.csv; coordinates in km from the mean station and hours from the first observation,
as in the SC26 notebook), Morton order by spatial location, and the Gneiting-Matern(3/2) covariance with
beta = 0 at the parameters fitted by mixed precision in the SC26 paper (par_mp of app2_forecast_skill.npz:
r_s = 19.72 km, r_t = 3.61 h, eta = 0.928, sigma2 = 2.07).  No residuals are needed: the tile choices and the
positive definiteness of the assembled Sigma^TLR depend only on the covariance.

Usage:  python3 39_joint_windgust.py --n 10000 --b 250 --out ../results/preliminary/joint_windgust
"""
import argparse, json, os, time, importlib.util
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, LinAlgError

here = os.path.dirname(os.path.abspath(__file__))
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(here, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rules = load("rules", "11_tlr_rank_rules.py")
joint = load("joint", "18_joint_rank_precision_pilot.py")
SQRT3 = np.sqrt(3.0)
DATA = os.environ.get("GUST_CSV", os.path.join(here, "..", "data", "windgust_train_data.csv"))   # copy of hpc4ees/data/train_data.csv (Jahan et al. 2025 wind gusts + WRF predictors), 2026-10-23
PAR = dict(r_s=19.7246432, r_t=3.6102287, eta=0.92818591, sigma2=2.06869252)      # SC26 mixed-precision fit


def coords(d, lat0, lon0, t0):
    xk = (d.Lon.values - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); yk = (d.Lat.values - lat0) * 110.57
    th = (d.DTS - t0).dt.total_seconds().values / 3600.0
    return np.stack([xk, yk, th], 1).astype(np.float64)


def cov_block(Pi, Pj, p):
    """sigma2 * M_{3/2}(h / r_s) / psi(v), psi(v) = 1 + (v / r_t)^2 (beta = 0), no nugget."""
    h = np.sqrt(((Pi[:, None, :2] - Pj[None, :, :2]) ** 2).sum(-1))
    v = np.abs(Pi[:, None, 2] - Pj[None, :, 2]); psi = 1.0 + (v / p["r_t"]) ** 2
    x = SQRT3 * h / p["r_s"]
    return p["sigma2"] * (1.0 + x) * np.exp(-x) / psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--b", type=int, default=250)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--mp_eps", type=str, default="", help="precision by relative rounding tolerance on the budget rank (2026-10-24)")
    ap.add_argument("--blr_eps", type=str, default="", help="rank and precision by one relative tolerance (2026-10-24)")
    ap.add_argument("--extra_rows", action="store_true", help="uniform_16, uniform_32 and block_mp rows (2026-10-24)")
    ap.add_argument("--out", type=str, default="../results/preliminary/joint_windgust")
    a = ap.parse_args()
    joint.MP_EPS = [float(x) for x in a.mp_eps.split(",") if x]; joint.BLR_EPS = [float(x) for x in a.blr_eps.split(",") if x]; joint.EXTRA_ROWS = a.extra_rows
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)
    d = pd.read_csv(DATA)
    d["DTS"] = pd.to_datetime(dict(year=d.YYYY, month=d.MM, day=d.DD, hour=d.HH))
    lat0, lon0, t0 = d.Lat.mean(), d.Lon.mean(), d.DTS.min()
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(len(d), a.n, replace=False))
    P = coords(d.iloc[sub], lat0, lon0, t0)
    order = rules.morton_order(P[:, :2]); P = P[order]
    bounds = rules.tile_bounds(a.n, a.b); b = a.b
    m_theta = PAR["sigma2"] * PAR["eta"]; emax = a.tau * b * m_theta / a.n
    log(f"wind-gust subset N={a.n} of {len(d)} station-hours; {d.Station_x.nunique()} stations; tiles b={b}, nB={len(bounds)}; "
        f"m(theta)={m_theta:.3f}, e_max={emax:.3e}")
    t0_ = time.time()
    K = np.empty((a.n, a.n))
    for i0 in range(0, a.n, 2000):
        K[i0:i0 + 2000] = cov_block(P[i0:i0 + 2000], P, PAR)
    K[np.diag_indices_from(K)] += m_theta
    log(f"covariance in {time.time()-t0_:.0f}s; off-diagonal tile norms: " +
        str(np.percentile([np.linalg.norm(K[i0:i1, j0:j1]) for i, (i0, i1) in enumerate(bounds) for (j0, j1) in bounds[:i]], [0, 10, 50, 90, 100]).round(3).tolist()))
    dense_bytes_off = sum((i1 - i0) * (j1 - j0) * 8 for i, (i0, i1) in enumerate(bounds) for (j0, j1) in bounds[:i])
    t0_ = time.time(); choices = {}
    for i, (i0, i1) in enumerate(bounds):
        for j, (j0, j1) in enumerate(bounds[:i]):
            choices[(i, j)] = joint.tile_options(K[i0:i1, j0:j1], emax, a.eps)
    log(f"per-tile choices in {time.time()-t0_:.0f}s")
    out = dict(settings=vars(a), parameters=PAR, m_theta=m_theta, e_max=emax, dense_offdiag_bytes=int(dense_bytes_off), rules={})
    for rule in ["fixed", "rank_only", "rank_then_precision", "joint"] + [f"mp_tol_{e:g}" for e in joint.MP_EPS] + [f"blr_{e:g}" for e in joint.BLR_EPS] + (["uniform_16", "uniform_32", "block_mp"] if joint.EXTRA_ROWS else []):
        t0_ = time.time()
        Kr = K.copy(); byt = 0; mix = {p: 0 for p in joint.PREC}; ranks = []; errs = []
        for i, (i0, i1) in enumerate(bounds):
            for j, (j0, j1) in enumerate(bounds[:i]):
                opts, (U, s, Vt) = choices[(i, j)]
                r, p, bb, err = opts[rule]
                if r == 0:
                    Tr = np.zeros((i1 - i0, j1 - j0))
                else:
                    Ur = U[:, :r] * np.sqrt(s[:r]); Vr = Vt[:r].T * np.sqrt(s[:r])
                    Tr = joint.stored(Ur, p) @ joint.stored(Vr, p).T
                Kr[i0:i1, j0:j1] = Tr; Kr[j0:j1, i0:i1] = Tr.T
                byt += bb; mix[p] += 1; ranks.append(r); errs.append(err)
        try:
            cho_factor(Kr, lower=True, check_finite=False); pd_ = True
        except LinAlgError:
            pd_ = False
        del Kr
        rec = dict(offdiag_bytes=int(byt), offdiag_bytes_frac_of_dense=byt / dense_bytes_off, precision_mix_tiles=mix,
                   mean_rank=float(np.mean(ranks)), max_rank=int(max(ranks)), max_tile_err=float(max(errs)),
                   budget_met_all_tiles=bool(max(errs) <= emax * (1 + 1e-9)), pd=pd_, seconds=time.time() - t0_)
        out["rules"][rule] = rec
        log(f"{rule:20s} bytes={rec['offdiag_bytes_frac_of_dense']:.4f} of dense | mix={mix} | rank(mean/max)={rec['mean_rank']:.1f}/{rec['max_rank']} "
            f"| budget met={rec['budget_met_all_tiles']} PD={pd_} ({rec['seconds']:.0f}s)")
    ro = out["rules"]["rank_only"]["offdiag_bytes"]
    for rule in out["rules"]:
        out["rules"][rule]["bytes_frac_of_rank_only"] = out["rules"][rule]["offdiag_bytes"] / ro
    path = os.path.join(a.out, f"joint_windgust_n{a.n}_b{b}.json"); json.dump(out, open(path, "w"), indent=1)
    log(f"joint / rank-only bytes = {out['rules']['joint']['bytes_frac_of_rank_only']:.3f} | rank-then-precision / rank-only = "
        f"{out['rules']['rank_then_precision']['bytes_frac_of_rank_only']:.3f}")
    log("saved " + path)


if __name__ == "__main__":
    main()
