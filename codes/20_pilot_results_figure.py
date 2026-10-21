"""Pilot results figure (2026-10-21): the adaptive rank rule inside a TLR Cholesky factorization with
recompression, against fixed tolerances, as N grows past the dense limit.

Reads results/tlr_cholesky/tlr_cholesky_n{N}.json (codes/17_tlr_cholesky_pilot.py) and draws
two panels: stored values of the Cholesky factor relative to a dense factor, and the log-likelihood error
at the true parameters, both against N.  A tolerance whose factorization did not complete is marked with a
cross on the axis.  Colors: Okabe-Ito blue (adaptive), vermillion (eps = 1e-4), gray (eps = 1e-5);
identity is also carried by the marker shape and direct labels.

Run:  python3 codes/20_pilot_results_figure.py
Output: results/pilot_results.pdf (and .png)
"""
import glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.25))
for ax in (a1, a2):
    ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([f"{n//1000}k" for n in Ns])
    ax.minorticks_off(); ax.grid(axis="y", color=RULE, lw=0.4, alpha=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("$N$")
a1.set_ylabel("factor stored values, % of dense")
a2.set_ylabel("log-likelihood error, true parameters")
a2.set_yscale("log")
for rule, val, label, col, mk in SERIES:
    xs, ys, es, fails = [], [], [], []
    for n in Ns:
        x = rows.get((rule, val, n))
        if x is None: continue
        if x["completed"]:
            xs.append(n); ys.append(100 * x["factor_memory_frac"])
            if x.get("loglik_err") is not None:
                es.append((n, abs(x["loglik_err"])))
        else:
            fails.append(n)
    a1.plot(xs, ys, color=col, lw=1.4, marker=mk, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
    if es:
        a2.plot([e[0] for e in es], [e[1] for e in es], color=col, lw=1.4, marker=mk, ms=4.5, mec="white", mew=0.8, label=label, zorder=3)
    for n in fails:                                       # failed factorization: cross on the axis
        for ax, y in ((a1, 0.5), (a2, 0.05)):
            ax.plot([n], [y], marker="x", color=col, ms=7, mew=1.6, ls="none", zorder=4)
    if xs:                                                # direct label at the last point
        a1.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=6.5, va="center")
a1.set_ylim(0, 15); a2.set_ylim(0.03, 30)
a1.text(0.03, 0.04, "$\\times$ factorization failed", transform=a1.transAxes, fontsize=6.5, color=INK)
a2.text(0.03, 0.04, "$\\times$ factorization failed", transform=a2.transAxes, fontsize=6.5, color=INK)
for ax in (a1, a2):                                       # dense FP64 fit limit of the wind-gust study on one 80 GB GPU
    ax.axvline(27440, color=INK, lw=0.6, ls=(0, (3, 3)), alpha=0.7)
a1.text(27440 * 1.06, 14.3, "dense fit limit,\none 80 GB GPU", fontsize=6, color=INK, va="top")
fig.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "results", f"pilot_results.{ext}"), dpi=300, bbox_inches="tight")
print("sizes", Ns, "-> overleaf/figures/pilot_results.pdf")
