"""Friedman test + Nemenyi critical-difference diagram over datasets (one block = dataset, averaged over seeds).
Usage: python stats_cd.py results/results_v05_standard.csv fig_cd_v05 [IR]"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

src, out = sys.argv[1], sys.argv[2]
IRsel = [int(sys.argv[3])] if len(sys.argv) > 3 else [5, 10, 20]
d = pd.read_csv(src); d = d[d.IR.isin(IRsel)]
import os
METHODS = [m for m in (os.environ.get("METHODS").split(";;") if os.environ.get("METHODS") else ["SMOTE-raw", "chain-mean-GPF (PSSTO-style)", "KNNOR-GPF (M=1)", "KNNOR-GPF (M calibrated)",
                       "hybrid GPF-nbrs/raw-walk (M calibrated)", "hybrid GPF-nbrs/raw-walk (M raw-calibrated)"]) if m in set(d.method)]
SHORT = {"SMOTE-raw": "SMOTE (raw)", "chain-mean-GPF (PSSTO-style)": "chain mean (GPF, PSSTO-style)", "KNNOR-GPF (M=1)": "KNNOR-GPF, M=1",
         "KNNOR-GPF (M calibrated)": "KNNOR-GPF, M calibrated", "hybrid GPF-nbrs/raw-walk (M calibrated)": "hybrid, GPF-calibrated M",
         "hybrid GPF-nbrs/raw-walk (M raw-calibrated)": "hybrid, raw-calibrated M",
         "KNNOR-official-GPF (randmx=1)": "reference implementation, GPF, default bound", "KNNOR-official-GPF (randmx=M*)": "reference implementation, GPF, calibrated bound", "KNNOR-official-raw (randmx=1)": "reference implementation, raw, default bound",
         "guide=gpf | walk=plain": "GPF-guided, plain step", "guide=gpf | walk=aligned": "GPF-guided, aligned step",
         "guide=sig | walk=plain": "signature-guided, plain step", "guide=sig | walk=aligned": "signature-guided, aligned step",
         "guide=dtw | walk=plain": "DTW-guided, plain step", "guide=dtw | walk=aligned": "DTW-guided, aligned step"}
# Nemenyi q_alpha (two-tailed, alpha=0.05) for k = 2..10
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

piv = d.pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")[METHODS].dropna()
N, k = piv.shape
ranks = piv.rank(axis=1, ascending=False)
avg = ranks.mean().sort_values()
stat, p = friedmanchisquare(*[piv[m].values for m in METHODS])
CD = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
print(f"blocks N={N} (dataset×IR), k={k} methods; Friedman chi2={stat:.2f}, p={p:.2e}; Nemenyi CD (α=0.05) = {CD:.3f}")
print(avg.round(3).to_string())
print("\npairwise Wilcoxon (signed-rank) vs SMOTE-raw:")
for m in METHODS:
    if m != "SMOTE-raw":
        print(f"  {SHORT[m]:32s} ΔF1={(piv[m]-piv['SMOTE-raw']).mean():+.3f}  p={wilcoxon(piv[m], piv['SMOTE-raw']).pvalue:.4f}")

# CD diagram
fig, ax = plt.subplots(figsize=(8, 0.9 + 0.42 * k))
lo, hi = 1, k
ax.set_xlim(lo - 2.6, hi + 2.6); ax.set_ylim(-0.42 * np.ceil(k / 2) - 0.9, 1.4); ax.axis("off")
ax.plot([lo, hi], [0, 0], color="black", lw=1)
for r in range(lo, hi + 1):
    ax.plot([r, r], [0, 0.15], color="black", lw=1); ax.text(r, 0.3, str(r), ha="center", fontsize=9)
ax.plot([lo, lo + CD], [0.9, 0.9], color="black", lw=2); ax.text(lo + CD / 2, 1.05, f"CD = {CD:.2f}", ha="center", fontsize=9)
items = list(avg.items())
for i, (m, r) in enumerate(items):
    left = i < np.ceil(k / 2)                                  # best half on the left, rest on the right
    y = -0.5 - 0.42 * (i if left else i - int(np.ceil(k / 2)))
    xt = lo - 0.2 if left else hi + 0.2
    ax.plot([r, r], [0, y], color="#555", lw=0.8); ax.plot([r, xt], [y, y], color="#555", lw=0.8)
    ax.text(xt - 0.05 if left else xt + 0.05, y, f"{SHORT[m]}  ({r:.2f})", va="center", ha="right" if left else "left", fontsize=8.5)
# cliques: consecutive methods within CD
sorted_r = [r for _, r in items]; yb = -0.25; drawn = []
for i in range(k):
    j = i
    while j + 1 < k and sorted_r[j + 1] - sorted_r[i] <= CD: j += 1
    if j > i and not any(a <= i and b >= j for a, b in drawn):
        ax.plot([sorted_r[i] - 0.04, sorted_r[j] + 0.04], [yb, yb], color="black", lw=3, solid_capstyle="butt"); yb -= 0.12; drawn.append((i, j))
ax.set_title(f"Critical-difference diagram, F1 (minority) · N = {N} dataset×IR blocks · IR ∈ {{{', '.join(map(str, IRsel))}}} · Friedman p = {p:.1e}", fontsize=9.5)
fig.savefig(f"results/{out}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"results/{out}.pdf", bbox_inches="tight")
print("saved", out)
