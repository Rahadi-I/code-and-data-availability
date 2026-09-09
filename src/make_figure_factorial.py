"""Factorial figure: (A) 3x2 cell means of F1 with SMOTE reference, (B) per-dataset gain of the best cell over SMOTE,
(C) dispersion vs MMD per cell. Usage: python make_figure_factorial.py results/results_v07_factorial.csv fig_factorial_v07"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
G = ["gpf", "sig", "dtw"]; GL = {"gpf": "cross-product\nareas (GPF-2)", "sig": "signature\n(level 3)", "dtw": "DTW\ndistance"}
W = ["plain", "aligned"]; WL = {"plain": "plain convex step", "aligned": "DTW-aligned step"}
cell = lambda g, w: f"guide={g} | walk={w}"
q = d.pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")
DS = sorted(d.dataset.unique()); N = len(q)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1.4, 1]}); fig.subplots_adjust(wspace=0.35)

# A. cell means with SMOTE and none reference lines
a = ax[0]; x = np.arange(len(G)); wdt = 0.36
col = {"plain": "#9ec5f4", "aligned": "#2a78d6"}
for i, w in enumerate(W):
    means = [q[cell(g, w)].mean() for g in G]; sds = [q[cell(g, w)].std() / np.sqrt(N) for g in G]
    a.bar(x + (i - 0.5) * wdt, means, wdt, yerr=sds, color=col[w], label=WL[w], capsize=2, error_kw={"lw": 0.8})
    for xi, m in zip(x, means): a.text(xi + (i - 0.5) * wdt, m + 0.012, f"{m:.2f}", ha="center", fontsize=7.5)
a.axhline(q["SMOTE-raw"].mean(), color="#eb6834", lw=1.5, ls="--", label=f"SMOTE, raw space ({q['SMOTE-raw'].mean():.2f})")
a.axhline(q["none"].mean(), color="#777", lw=1, ls=":", label=f"no oversampling ({q['none'].mean():.2f})")
a.set_xticks(x); a.set_xticklabels([GL[g] for g in G], fontsize=8); a.set_ylabel(f"minority F1, mean over {N} dataset×IR blocks (± s.e.)")
a.set_ylim(0.4, 0.8); a.legend(fontsize=7.5, frameon=False, loc="lower left")
a.set_title("A.  Guidance space × synthesis rule", loc="left", weight="bold", fontsize=10)

# B. per-dataset gain over SMOTE for each aligned cell
b = ax[1]; yy = np.arange(len(DS))
gd = d[d.IR.isin([10, 20])].pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean").reindex(DS)
colors = {"gpf": "#eda100", "sig": "#1baf7a", "dtw": "#e87ba4"}
for i, g in enumerate(G):
    gain = gd[cell(g, "aligned")] - gd["SMOTE-raw"]
    b.barh(yy + (i - 1) * 0.27, gain.values, height=0.26, color=colors[g], label=f"{GL[g].replace(chr(10), ' ')} + aligned step")
gain0 = gd[cell("gpf", "plain")] - gd["SMOTE-raw"]
b.scatter(gain0.values, yy, marker="|", s=120, color="#444", label="GPF guidance + plain step", zorder=3)
b.axvline(0, color="#777", lw=1); b.set_yticks(yy); b.set_yticklabels(DS, fontsize=8); b.invert_yaxis()
b.set_xlabel("F1 gain over SMOTE (raw space), IR 10 & 20"); b.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
b.set_title("B.  Where the aligned, geometry-guided walk gains", loc="left", weight="bold", fontsize=10)

# C. dispersion vs MMD
c = ax[2]
for g in G:
    for w in W:
        s = d[d.method == cell(g, w)]
        c.scatter(s.disp_ratio.mean(), s.mmd_syn.mean(), s=70, color=colors[g], edgecolor="white", lw=1.2,
                  marker="o" if w == "aligned" else "s", zorder=3)
s = d[d.method == "SMOTE-raw"]; c.scatter(s.disp_ratio.mean(), s.mmd_syn.mean(), s=80, color="#eb6834", edgecolor="white", marker="D", zorder=3)
c.annotate("SMOTE, raw", (s.disp_ratio.mean(), s.mmd_syn.mean()), xytext=(6, -10), textcoords="offset points", fontsize=7.5)
for g in G:
    for w in W:
        s = d[d.method == cell(g, w)]
        c.annotate(f"{g}/{w}", (s.disp_ratio.mean(), s.mmd_syn.mean()), xytext=(5, 3), textcoords="offset points", fontsize=7)
c.axvline(1, ls="--", color="#777", lw=1); c.set_xlabel("dispersion ratio  tr Cov(synthetic) / tr Cov(real minority)"); c.set_ylabel("MMD (RBF), synthetic → held-out minority")
c.set_title("C.  Dispersion and admissibility per cell", loc="left", weight="bold", fontsize=10); c.grid(alpha=0.25)
c.text(0.02, 0.97, "circles: aligned step · squares: plain step", transform=c.transAxes, fontsize=7.5, va="top")
fig.suptitle(f"Factorial: three guidance spaces × two synthesis rules on {len(DS)} datasets, IR {{5,10,20}}, standard splits", weight="bold", fontsize=11, y=1.02)
fig.savefig(f"results/{out}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"results/{out}.pdf", bbox_inches="tight")

# stats to stdout
meth = [m for m in q.columns if m != "none"]
print("mean rank:\n", q[meth].rank(axis=1, ascending=False).mean().round(2).sort_values().to_string())
for m in meth:
    if m != "SMOTE-raw":
        print(f"{m:28s} vs SMOTE Δ={(q[m]-q['SMOTE-raw']).mean():+.3f} p={wilcoxon(q[m], q['SMOTE-raw']).pvalue:.4f}")
al = q[[cell(g, 'aligned') for g in G]].mean(1); pl = q[[cell(g, 'plain') for g in G]].mean(1)
print(f"alignment main effect Δ={(al-pl).mean():+.3f} p={wilcoxon(al, pl).pvalue:.4f}")
for g in G:
    print(f"guidance {g}: mean F1 {(q[cell(g,'plain')].mean()+q[cell(g,'aligned')].mean())/2:.3f}")
print("saved", out)
