"""Pilot results figure (2026-10-21): the adaptive rank rule inside a TLR Cholesky factorization with
recompression, against fixed tolerances, as N grows past the dense limit.

Reads results/tlr_cholesky/tlr_cholesky_n{N}.json (codes/17_tlr_cholesky_pilot.py) and draws
two panels against N, stored values of the Cholesky factor relative to a dense factor and the log-likelihood
error at the true parameters, and a third head to head with Vecchia at N = 10,000 (results/screening_stress,
codes 24-26): log-likelihood error at the true parameters against stored values, Vecchia traced over its neighbor count
and adaptive TLR one point per field, on the pilot field and a smoother, longer-range field (SS_FIELDS overrides).
A tolerance whose factorization did not complete is marked with a
cross on the axis.  Colors: Okabe-Ito blue (adaptive), vermillion (eps = 1e-4), gray (eps = 1e-5);
identity is also carried by the marker shape and direct labels.

Run:  python3 codes/20_pilot_results_figure.py
Output: results/pilot_results.pdf (and .png)
"""
import glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, VERM, GRAY, INK, RULE = "#0072B2", "#D55E00", "#6E6E6E", "#1a1a1a", "#c9c8c2"
SERIES = [("adaptive", 0.9, "adaptive, $\\tau=0.9$", BLUE, "o"),
          ("fixed", 1e-4, "fixed $\\varepsilon=10^{-4}$", VERM, "s"),
          ("fixed", 1e-5, "fixed $\\varepsilon=10^{-5}$", GRAY, "^")]

rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "tlr_cholesky", "tlr_cholesky_n*.json"))):
    r = json.load(open(f)); n = r["settings"]["n"]
    for x in r["rules"]:
        rows[(x["rule"], x["value"], n)] = x
Ns = sorted({k[2] for k in rows})
RES = os.path.join(ROOT, "results")
est = {}                                                   # (method, N) -> theta_hat
for f in glob.glob(os.path.join(RES, "tlr_cholesky", "tlr_cholesky_n*.json")):
    r = json.load(open(f)); n = r["settings"]["n"]
    for x in r.get("mle", []):
        if x["rule"] in ("dense", "adaptive"): est[(x["rule"], n)] = x["theta_hat"]
for f in glob.glob(os.path.join(RES, "tlr_cholesky", "vecchia_n*_m*.json")):
    v = json.load(open(f)); est[(f"vecchia{v['m']}", v["N"])] = [v["theta_hat"]["sigma2"], v["theta_hat"]["range"], v["theta_hat"]["eta"]]
se = {}
for f in glob.glob(os.path.join(RES, "pilot_standard_errors_n*.json")):
    s = json.load(open(f))
    n = s.get("N") or s.get("n")
    if "se_theta" in s: se[n] = s["se_theta"]                                   # 20,000 file (finite differences)
    elif isinstance(s.get("fits"), dict) and "se_theta" in s["fits"].get("dense", {}): se[n] = s["fits"]["dense"]["se_theta"]   # script 15 output
GREEN = "#009E73"
EST_SERIES = [("dense", "dense fit", INK, "D"), ("adaptive", "adaptive TLR fit", BLUE, "o"),
              ("vecchia30", "Vecchia, 30 neighbors", VERM, "s"), ("vecchia60", "Vecchia, 60 neighbors", GRAY, "^")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "axes.titlesize": 7.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})
VEC = "#009E73"                                            # Vecchia: its own color (vermillion is the 1e-4 tolerance)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.6, 1.95), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.15]))
for ax in (a1, a2, a3):
    ax.grid(axis="y", color=RULE, lw=0.4, alpha=0.6); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
for ax in (a1, a2):
    ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([("" if n == 60000 else f"{n//1000}k") for n in Ns], fontsize=6.5)
    ax.minorticks_off(); ax.set_xlabel("$N$", labelpad=1); ax.set_xlim(Ns[0] / 1.3, Ns[-1] * 1.3)
    ax.axvline(27440, color=INK, lw=0.6, ls=(0, (3, 3)), alpha=0.6)   # dense FP64 fit limit, wind-gust study, one 80 GB GPU
a1.set_title("(a) factor storage"); a1.set_ylabel("% of a dense factor", labelpad=1)
a2.set_title("(b) likelihood error"); a2.set_ylabel("|log-likelihood error|", labelpad=1); a2.set_yscale("log")
a3.set_title("(c) vs. Vecchia, $N=10{,}000$")

LBL = {("adaptive", 0.9): "adaptive", ("fixed", 1e-4): "$\\varepsilon=10^{-4}$", ("fixed", 1e-5): "$\\varepsilon=10^{-5}$"}
for rule, val, label, col, mk in SERIES:
    xs, ys, es, fails = [], [], [], []
    for n in Ns:
        x = rows.get((rule, val, n))
        if x is None: continue
        if x["completed"]:
            xs.append(n); ys.append(100 * x["factor_memory_frac"])
            if x.get("loglik_err") is not None: es.append((n, abs(x["loglik_err"])))
        else:
            fails.append(n)
    kw = dict(color=col, lw=1.3, marker=mk, ms=4, mec="white", mew=0.7, zorder=3)
    a1.plot(xs, ys, **kw)
    if es: a2.plot([e[0] for e in es], [e[1] for e in es], **kw)
    for n in fails:
        for ax, y in ((a1, 0.6), (a2, 0.045)):
            ax.plot([n], [y], marker="x", color=col, ms=5.5, mew=1.4, ls="none", zorder=4)
    if xs:                                                 # direct label left of the first point
        a1.annotate(LBL[(rule, val)], (xs[0], ys[0]), xytext=(-5, 0), textcoords="offset points", color=col,
                    fontsize=6.3, va="center", ha="right")
a1.set_ylim(0, 15); a1.set_xlim(Ns[0] / 2.6, Ns[-1] * 1.3)
a1.text(27440 * 1.08, 14.6, "dense fit limit,\none 80 GB GPU", fontsize=5.8, color=INK, va="top")
a1.text(60000, 9.6, "$\\times$: factorization\nfailed", fontsize=5.8, color=VERM, ha="center", va="top")
a2.set_ylim(0.03, 40)
a2.text(95000, 4.0, "100k: no dense\nreference", ha="right", va="center", fontsize=5.8, color=BLUE)

# (c) head to head with Vecchia on all nine screening fields at N = 10,000 (codes 24, 26, 31, 32): KL divergence of each
# approximation from the exact Gaussian model at the true parameters (Guinness 2018; Katzfuss and Guinness 2021).
# Rows: fields (smoothness nu, range, relative nugget).  Blue: adaptive TLR.  Green: grouped response Vecchia (GpGp's
# fitting default, max-min ordering; Monte Carlo KL of 33) with 30 (filled square) and 60 (open square) neighbors, and
# ungrouped Vecchia with the neighbors that match the adaptive TLR storage (diamond, exact KL of 32); means over orderings.
SS = os.environ.get("SS_DIR", os.path.join(RES, "screening_stress"))
fields = []
for tag in sorted(os.listdir(SS)):
    d = os.path.join(SS, tag)
    if not os.path.exists(os.path.join(d, "kl_vecchia.json")): continue
    st = json.load(open(os.path.join(d, "stress.json")))["settings"]
    kt = json.load(open(os.path.join(d, "kl_tlr.json")))["kl"]
    kv = {x["m"]: x["mean"] for x in json.load(open(os.path.join(d, "kl_vecchia.json")))["summary"]}
    gf = os.path.join(d, "kl_vecchia_grouped_mc.json")       # grouped Vecchia (GpGp's fitting default), Monte Carlo KL (33)
    if os.path.exists(gf):
        kv.update({x["m"]: x["kl_grouped"] for x in json.load(open(gf))["summary"]})
    km = None
    if os.path.exists(os.path.join(d, "kl_vecchia_matched.json")):
        mm = json.load(open(os.path.join(d, "vecchia_matched.json")))["m_match"]
        km = {x["m"]: x["mean"] for x in json.load(open(os.path.join(d, "kl_vecchia_matched.json")))["summary"]}.get(mm)
    fields.append(((st["nu"], st["eta"], st["range"]), kt, kv, km))
fields.sort(key=lambda x: x[0])
NU = {1.5: "3/2", 2.5: "5/2"}
for i, (key, kt, kv, km) in enumerate(fields):
    yv = len(fields) - 1 - i
    a3.plot([kv[30]], [yv], marker="s", ms=3.8, color=VEC, ls="none", zorder=3)
    a3.plot([kv[60]], [yv], marker="s", ms=3.8, color=VEC, mfc="white", mew=0.9, ls="none", zorder=3)
    if km is not None:
        a3.plot([km], [yv], marker="D", ms=3.4, color=VEC, mfc="white", mew=0.9, ls="none", zorder=3)
    a3.plot([kt], [yv], marker="o", ms=4.5, color=BLUE, mec="white", mew=0.6, ls="none", zorder=4)
a3.set_yticks(range(len(fields)))
a3.set_yticklabels([("pilot  " if (nu, eta, rg) == (1.5, 0.05, 0.3) else "") + f"{NU[nu]}  {rg:g}  {eta:g}" for (nu, eta, rg), _, _, _ in fields][::-1], fontsize=5.6)
a3.set_ylim(-4.4, len(fields) - 0.3)
a3.set_xscale("log"); a3.set_xlim(0.02, 400); a3.set_xticks([0.1, 1, 10, 100]); a3.set_xticklabels(["0.1", "1", "10", "100"], fontsize=6.5)
a3.minorticks_off(); a3.grid(axis="x", color=RULE, lw=0.4, alpha=0.6); a3.grid(axis="y", visible=False)
a3.set_xlabel("KL divergence (0 = exact)", labelpad=1); a3.set_ylabel("")
a3.text(0.97, 0.03, "rows: smoothness $\\nu$, range, nugget\nblue: adaptive TLR\ngreen: grouped Vecchia, $m=30$ (filled),\n$m=60$ (open); ungrouped Vecchia at\nthe same storage as TLR (diamond)",
        transform=a3.transAxes, fontsize=5.4, ha="right", va="bottom", color=INK, linespacing=1.15)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(os.environ.get("FIG_OUT", os.path.join(ROOT, "results")), f"pilot_results.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
print("sizes", Ns, "->", os.path.join(os.environ.get("FIG_OUT", os.path.join(ROOT, "results")), "pilot_results.pdf"))
