"""Multi-GPU tile Cholesky for the preliminary results of the CDS&E proposal (2026-10-20).

Factorizes the covariance of a Matern (nu = 3/2) Gaussian process with nugget, built
tile by tile, with the block rows of the factor distributed cyclically over the GPUs
visible to PyTorch (or over a list of pseudo-devices on a CPU for testing).  Each GPU
owns the tiles L_{ik} of its block rows i; at panel k the owner factorizes L_{kk},
every GPU receives block row k (the tiles L_{kj}, j <= k, that the trailing update
needs), and updates its own rows independently.  This is the same tile algorithm as the
proposal's Objective pages (dense tiles here; the rank and precision rules act on the
tiles L_{ik} and do not change the communication pattern).  Reports wall time, peak
memory per GPU, and the log-determinant against a single-device factorization.

Usage:
  python3 13_multigpu_tile_cholesky.py --n 40000 --b 1000 --devices cuda:0,cuda:1
  python3 13_multigpu_tile_cholesky.py --n 4000 --b 500 --devices cpu,cpu   # logic test
Reproducible: fixed seed; settings recorded in the JSON output.
"""
import argparse, json, math, os, time
import numpy as np
import torch

SQRT3 = math.sqrt(3.0)


def cov_tile(S, I, J, theta, diag):
    sigma2, r, eta = theta
    A, B = S[I[0]:I[1]], S[J[0]:J[1]]
    d2 = (A[:, None, 0] - B[None, :, 0]) ** 2 + (A[:, None, 1] - B[None, :, 1]) ** 2
    x = SQRT3 * torch.sqrt(d2.clamp_min(0)) / r
    K = sigma2 * (1 + x) * torch.exp(-x)
    if diag:
        K.diagonal().add_(sigma2 * eta)
    return K


def xfer(t, dst):
    """Copy a tile to another device.  GPU-to-GPU copies are staged through host memory: on the
    UConn Storrs A30 nodes a direct peer copy (torch .to between two CUDA devices) returned
    wrong data in PyTorch 2.5.1 although peer access was reported as available."""
    if t.device.type == "cuda" and dst.type == "cuda" and t.device != dst:
        return t.cpu().to(dst)
    return t.to(dst)


def tile_cholesky_multi(S_dev, bounds, theta, devices, owner):
    """S_dev[d] = site coordinates on device d.  owner[i] = device index of block row i.
    Returns (dict of factor tiles keyed (i, k) living on their owner device, logdet)."""
    nb = len(bounds)
    L = {}
    logdet = torch.zeros((), dtype=torch.float64)
    for k in range(nb):
        dk = owner[k]
        # panel: S_kk - sum_j L_kj L_kj^T on the owner of row k
        Skk = cov_tile(S_dev[dk], bounds[k], bounds[k], theta, True)
        for j in range(k):
            Lkj = L[(k, j)]
            Skk.addmm_(Lkj, Lkj.T, alpha=-1.0)
        Lkk, info = torch.linalg.cholesky_ex(Skk)
        if int(info) != 0:
            raise RuntimeError(f"nonpositive pivot in block {k}")
        L[(k, k)] = Lkk
        logdet = logdet + 2.0 * torch.log(torch.diagonal(Lkk)).sum().cpu()
        # broadcast block row k (L_kj, j <= k) to every other device once; synchronous copies,
        # because a non-blocking cross-device copy is not ordered before the receiving GPU's trailing update
        row_k = {d: {j: xfer(L[(k, j)], devices[d]) for j in range(k + 1)} for d in range(len(devices)) if d != dk}
        row_k[dk] = {j: L[(k, j)] for j in range(k + 1)}
        # trailing rows, each on its owner
        for i in range(k + 1, nb):
            di = owner[i]
            M = cov_tile(S_dev[di], bounds[i], bounds[k], theta, False)
            for j in range(k):
                M.addmm_(L[(i, j)], row_k[di][j].T, alpha=-1.0)
            # L_ik = M L_kk^{-T}, solved as L_kk X = M^T on contiguous operands (the CUDA kernel
            # mishandled the transposed-view right-hand form in PyTorch 2.5)
            L[(i, k)] = torch.linalg.solve_triangular(row_k[di][k], M.T.contiguous(), upper=False, left=True).T.contiguous()
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)
    return L, float(logdet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--b", type=int, default=500)
    ap.add_argument("--devices", type=str, default="cpu,cpu")
    ap.add_argument("--range", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--check_single", action="store_true", help="also factor on the first device alone and compare")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=str, default="../results/preliminary")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    devices = [torch.device(d) for d in a.devices.split(",")]
    rng = np.random.default_rng(a.seed)
    S = torch.tensor(rng.random((a.n, 2)), dtype=torch.float64)
    # Morton order so that tiles are spatially compact
    # inline Morton order (same as 11_tlr_rank_rules.morton_order)
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
    S = S[torch.tensor(morton_order(S.numpy()))]
    bnd = list(range(0, a.n, a.b)) + [a.n]
    bounds = [(bnd[i], bnd[i + 1]) for i in range(len(bnd) - 1)]
    theta = (1.0, a.range, a.eta)
    S_dev = [S.to(d) for d in devices]
    owner = [i % len(devices) for i in range(len(bounds))]

    for d in devices:
        if d.type == "cuda":
            torch.cuda.reset_peak_memory_stats(d)
    t0 = time.perf_counter()
    L, ld = tile_cholesky_multi(S_dev, bounds, theta, devices, owner)
    t_multi = time.perf_counter() - t0
    peaks = [torch.cuda.max_memory_allocated(d) / 1e9 if d.type == "cuda" else None for d in devices]
    stored_gb = sum(t.numel() * 8 for t in L.values()) / 1e9
    print(f"multi-device ({len(devices)} x {devices[0].type}): N={a.n} b={a.b} nB={len(bounds)} time={t_multi:.1f}s "
          f"logdet={ld:.6f} factor={stored_gb:.2f} GB peak/GPU={peaks}", flush=True)
    out = dict(settings=vars(a), n_tiles=len(bounds), devices=[str(d) for d in devices], seconds_multi=t_multi,
               logdet_multi=ld, factor_gb=stored_gb, peak_gb_per_device=peaks)
    del L
    if a.check_single:
        d0 = devices[0]
        if d0.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(d0)
        t0 = time.perf_counter()
        L1, ld1 = tile_cholesky_multi([S_dev[0]], bounds, theta, [d0], [0] * len(bounds))
        t_single = time.perf_counter() - t0
        out.update(seconds_single=t_single, logdet_single=ld1, logdet_abs_diff=abs(ld1 - ld),
                   peak_gb_single=torch.cuda.max_memory_allocated(d0) / 1e9 if d0.type == "cuda" else None)
        print(f"single-device: time={t_single:.1f}s logdet={ld1:.6f} |diff|={abs(ld1-ld):.2e} speedup={t_single/t_multi:.2f}x", flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = f"n{a.n}_b{a.b}_{len(devices)}x{devices[0].type}"
    json.dump(out, open(os.path.join(a.out, f"multigpu_tile_cholesky_{tag}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, f"multigpu_tile_cholesky_{tag}.json"))


if __name__ == "__main__":
    main()
