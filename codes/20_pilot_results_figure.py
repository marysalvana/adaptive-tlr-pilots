"""Pilot results figure (2026-10-21): the adaptive rank rule inside a TLR Cholesky factorization with
recompression, against fixed tolerances, as N grows past the dense limit.

Reads results/tlr_cholesky/tlr_cholesky_n{N}.json (codes/17_tlr_cholesky_pilot.py) and draws
three panels against N: stored values of the Cholesky factor relative to a dense factor; the log-likelihood error
at the true parameters; and the fitted nugget (the well-identified parameter) of the dense fit (with its standard error
where computed, pilot_standard_errors_n{N}.json), the adaptive-TLR fit, and Vecchia with 30 and 60 neighbors
(vecchia_n{N}_m{30,60}.json), against the true value: the head-to-head on estimates.  A tolerance whose factorization did not complete is marked with a
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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.6, 2.3))
for ax in (a1, a2, a3):
    ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([("" if n == 60000 else f"{n//1000}k") for n in Ns], fontsize=6.5)
    ax.minorticks_off(); ax.grid(axis="y", color=RULE, lw=0.4, alpha=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("$N$"); ax.set_xlim(Ns[0] / 1.35, Ns[-1] * 1.35)
a1.set_ylabel("factor stored values, % of dense")
a2.set_ylabel("log-likelihood error")
a2.set_yscale("log")
a3.set_ylabel("fitted nugget $\\eta$")
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
# sizes without an exact log-likelihood (no dense factor): say so in the error panel
no_ref = [n for n in Ns if all(rows.get((r, v, n), {}).get("loglik_err") is None for r, v, _, _, _ in SERIES if rows.get((r, v, n), {}).get("completed"))]
for n in no_ref:
    a2.text(n, 0.45, "no dense\nreference", ha="center", va="center", fontsize=6, color=BLUE)
# one legend for both panels: the three rules and the failure marker (its color is the rule's)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=col, marker=mk, ms=4.5, mec="white", mew=0.8, lw=1.4, label=label) for _, _, label, col, mk in SERIES]
handles.append(Line2D([], [], color=INK, marker="x", ms=6, mew=1.4, ls="none", label="factorization failed (color = rule)"))
# third panel: the fitted nugget, head to head
a3.axhline(0.05, color=GREEN, lw=1.0, ls=(0, (4, 2)), zorder=1)
a3.text(Ns[0] / 1.3, 0.0515, "truth", color=GREEN, fontsize=6, va="bottom", ha="left")
h3 = [Line2D([], [], color=GREEN, lw=1.0, ls=(0, (4, 2)), label="truth")]
for key, label, col, mk in EST_SERIES:
    xs = [n for n in Ns if (key, n) in est]; ys = [est[(key, n)][2] for n in xs]
    kw = dict(color=col, marker=mk, ms=5.5 if key == "dense" else 4.5, lw=1.2, zorder=3)
    if key == "dense": kw.update(mfc="white", mew=1.2)                       # hollow, so it stays visible under the adaptive marker
    else: kw.update(mec="white", mew=0.8)
    a3.plot(xs, ys, label=label, **kw)
    h3.append(Line2D([], [], label=label, **{k: v for k, v in kw.items() if k != "zorder"}))
for n, s in se.items():
    if ("dense", n) in est:
        a3.errorbar([n], [est[("dense", n)][2]], yerr=[s[2]], color=INK, capsize=2, lw=0.8, zorder=2)
a3.set_ylim(0.02, 0.11)
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.34, -0.02), handletextpad=0.5, columnspacing=1.2)
fig.legend(handles=h3, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.09), handletextpad=0.5, columnspacing=1.2)
for ax in (a1, a2):                                       # dense FP64 fit limit of the wind-gust study on one 80 GB GPU
    ax.axvline(27440, color=INK, lw=0.6, ls=(0, (3, 3)), alpha=0.7)
a1.text(27440 * 1.06, 14.3, "dense fit limit,\none 80 GB GPU", fontsize=6, color=INK, va="top")
fig.tight_layout(w_pad=1.6, rect=(0, 0.12, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "results", f"pilot_results.{ext}"), dpi=300, bbox_inches="tight")
print("sizes", Ns, "-> overleaf/figures/pilot_results.pdf")
