"""20-dataset factorial figure (cells run on all datasets: SMOTE, gpf/plain, gpf/aligned, sig/aligned; dtw/aligned on the 15 with L<=500)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
q = d.pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")
C = {"SMOTE-raw": ("SMOTE, raw space", "#eb6834"), "guide=gpf | walk=plain": ("GPF-guided, plain step", "#9ec5f4"),
     "guide=gpf | walk=aligned": ("GPF-guided, aligned step", "#2a78d6"), "guide=sig | walk=aligned": ("signature-guided, aligned step", "#1baf7a"),
     "guide=dtw | walk=aligned": ("DTW-guided, aligned step*", "#e87ba4")}
DS = sorted(d.dataset.unique())
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1, 1.5, 1]}); fig.subplots_adjust(wspace=0.38)
# A
a = ax[0]; names = list(C); base = q["SMOTE-raw"]
vals, errs, cols, labs = [], [], [], []
for m in names:
    if m == "SMOTE-raw":
        s = q[[m]].dropna(); vals.append(float(s[m].mean())); errs.append(0.0)
    else:
        s = q[[m, "SMOTE-raw"]].dropna(); diff = s[m] - s["SMOTE-raw"]
        vals.append(float(s[m].mean())); errs.append(float(diff.std() / np.sqrt(len(s))))
    cols.append(C[m][1]); labs.append(C[m][0] + f"\n(N={len(s)})")
a.bar(list(range(len(names))), vals, color=cols, yerr=errs, capsize=2, error_kw={"lw": 0.8})
for i, (v, e) in enumerate(zip(vals, errs)): a.text(i, v + e + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
a.axhline(q["none"].mean(), color="#777", lw=1, ls=":"); a.text(len(names) - 0.5, q["none"].mean() + 0.006, "no oversampling (0.460)", ha="right", va="bottom", fontsize=7, color="#555", bbox=dict(fc="white", ec="none", pad=1.2))
a.set_xticks(range(len(names))); a.set_xticklabels(labs, fontsize=7, rotation=25, ha="right"); a.set_ylim(0.4, 0.76)
a.set_ylabel("minority F1, mean over dataset×IR blocks\n(error bar: s.e. of the paired difference to SMOTE)")
a.set_title("A.  Cell means, 20 datasets × IR {5,10,20}", loc="left", weight="bold", fontsize=10)
a.text(0.0, -0.26, "* DTW guidance on the 15 datasets with L ≤ 500", transform=a.transAxes, fontsize=7, color="#555")
# B
b = ax[1]; yy = np.arange(len(DS))
gd = d[d.IR.isin([10, 20])].pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean").reindex(DS)
order = (gd["guide=sig | walk=aligned"] - gd["SMOTE-raw"]).sort_values(ascending=False).index; gd = gd.reindex(order); DSo = list(order)
for i, m in enumerate(["guide=gpf | walk=aligned", "guide=sig | walk=aligned"]):
    b.barh(yy + (i - 0.5) * 0.36, (gd[m] - gd["SMOTE-raw"]).values, height=0.34, color=C[m][1], label=C[m][0])
b.scatter((gd["guide=gpf | walk=plain"] - gd["SMOTE-raw"]).values, yy, marker="|", s=110, color="#444", label="GPF-guided, plain step", zorder=3)
b.axvline(0, color="#777", lw=1); b.set_yticks(yy); b.set_yticklabels(DSo, fontsize=7.5); b.invert_yaxis()
b.set_xlabel("F1 gain over SMOTE (raw space), mean of IR 10 and 20"); b.legend(fontsize=7.5, frameon=False, loc="lower right")
b.set_title("B.  Per-dataset gain of the aligned, geometry-guided walk", loc="left", weight="bold", fontsize=10)
# C
c = ax[2]
for m in names:
    s = d[d.method == m]; x, y = s.disp_ratio.median(), s.mmd_syn.median()
    c.scatter(x, y, s=80, color=C[m][1], edgecolor="white", lw=1.2, zorder=3, marker="D" if m == "SMOTE-raw" else "o")
    lab = C[m][0].replace("*", "")
    if m == "SMOTE-raw": xy, ha = (8, -3), "left"
    elif "plain" in m: xy, ha = (-8, -3), "right"
    elif "gpf" in m: xy, ha = (-8, -3), "right"
    elif "sig" in m: xy, ha = (0, -14), "center"
    else: xy, ha = (8, -3), "left"
    c.annotate(lab, (x, y), xytext=xy, textcoords="offset points", fontsize=7, ha=ha)
c.axvline(1, ls="--", color="#777", lw=1); c.margins(x=0.18, y=0.12); c.set_xlabel("dispersion ratio, median over runs"); c.set_ylabel("MMD (RBF) synthetic → held-out minority, median")
c.set_title("C.  Dispersion and admissibility", loc="left", weight="bold", fontsize=10); c.grid(alpha=0.25)
fig.savefig(f"results/{out}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"results/{out}.pdf", bbox_inches="tight"); print("saved", out)
