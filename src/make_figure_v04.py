"""v0.4 figure across ~20 datasets. Usage: python make_figure_v04.py results/results_v04_pca.csv fig_go_no_go_v04_pca"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
COL = {"hybrid GPF-nbrs/raw-walk (M raw-calibrated)": "#008300", "hybrid GPF-nbrs/raw-walk (M calibrated)": "#e87ba4", "SMOTE-raw": "#2a78d6", "chain-mean-GPF (PSSTO-style)": "#eb6834", "KNNOR-GPF (M=1)": "#1baf7a", "KNNOR-GPF (M calibrated)": "#eda100"}
LAB = {"SMOTE-raw": "SMOTE, raw space", "KNNOR-raw (M=1)": "KNNOR, raw, M=1", "chain-mean-GPF (PSSTO-style)": "chain mean in GPF (PSSTO-style)",
       "KNNOR-GPF (M=1)": "KNNOR in GPF, M=1 (default bound)", "KNNOR-GPF (M=3/2)": "KNNOR in GPF, M=3/2", "KNNOR-GPF (M calibrated)": "KNNOR in GPF, M calibrated (ours)", "hybrid GPF-nbrs/raw-walk (M calibrated)": "hybrid: GPF neighbours, raw walk (ours)", "hybrid GPF-nbrs/raw-walk (M raw-calibrated)": "hybrid, raw-calibrated M (ours)"}
ORDER = [m for m in ["SMOTE-raw", "KNNOR-raw (M=1)", "chain-mean-GPF (PSSTO-style)", "KNNOR-GPF (M=1)", "KNNOR-GPF (M=3/2)", "KNNOR-GPF (M calibrated)", "hybrid GPF-nbrs/raw-walk (M calibrated)", "hybrid GPF-nbrs/raw-walk (M raw-calibrated)"] if m in set(d.method)]
meta = d.groupby("dataset").agg(H=("H", "first"), ch=("n_channels", "first"), L=("length", "first")).sort_values("H")
DS = list(meta.index)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.5, wspace=0.38)

# A. per-dataset F1 at IR=20 (hardest): four oversamplers + none, datasets sorted by H
ax = fig.add_subplot(gs[0, :])
sub = d[d.IR == 20]
yy = np.arange(len(DS))
none = sub[sub.method == "none"].groupby("dataset").f1.mean().reindex(DS)
ax.scatter(none.values, yy, marker="|", s=160, color="#555", label="no oversampling", zorder=2)
for m in [m for m in ["SMOTE-raw", "chain-mean-GPF (PSSTO-style)", "KNNOR-GPF (M=1)", "KNNOR-GPF (M calibrated)", "hybrid GPF-nbrs/raw-walk (M calibrated)", "hybrid GPF-nbrs/raw-walk (M raw-calibrated)"] if m in ORDER]:
    v = sub[sub.method == m].groupby("dataset").f1.mean().reindex(DS)
    ax.scatter(v.values, yy, s=42, color=COL[m], edgecolor="white", lw=0.8, label=LAB[m], zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{n}  ({int(meta.loc[n,'ch'])} ch, H={meta.loc[n,'H']:.2f})" for n in DS], fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("F1 (minority) at imbalance ratio 20, mean over 5 seeds"); ax.grid(axis="x", alpha=0.25)
ax.set_title("A.  Minority F1 at IR = 20 across datasets (sorted by roughness exponent H; random forest in the same feature space for every method)", loc="left", fontsize=10, weight="bold")
ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))

# B. mean rank by IR
ax = fig.add_subplot(gs[1, 0])
for i, IR in enumerate([5, 10, 20]):
    piv = d[d.IR == IR].pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean")[ORDER]
    rk = piv.rank(axis=1, ascending=False).mean()
    ax.barh(np.arange(len(ORDER)) + (i - 1) * 0.27, [rk[m] for m in ORDER], height=0.26,
            color=[COL.get(m, "#9a9a9a") for m in ORDER], alpha=[0.45, 0.7, 1.0][i], label=f"IR = {IR}")
ax.set_yticks(np.arange(len(ORDER))); ax.set_yticklabels([LAB[m] for m in ORDER], fontsize=8); ax.invert_yaxis()
ax.set_xlim(1, len(ORDER)); ax.set_xlabel(f"mean rank over {len(DS)} datasets (1 = best)")
ax.legend(fontsize=7.5, frameon=False, loc="lower right", title="bar shade", title_fontsize=7.5)
ax.set_title("B.  Mean rank by F1", loc="left", fontsize=10, weight="bold")

# C. dispersion vs MMD
ax = fig.add_subplot(gs[1, 1])
for m in ORDER:
    s = d[d.method == m]
    x, y = s.disp_ratio.mean(), s.mmd_syn.mean()
    ax.scatter(x, y, s=70, color=COL.get(m, "#9a9a9a"), edgecolor="white", lw=1.2, zorder=3)
    OFF = {"SMOTE-raw": ((8, -3), "left"), "KNNOR-GPF (M=1)": ((8, -3), "left"), "chain-mean-GPF (PSSTO-style)": ((8, -3), "left"),
           "KNNOR-GPF (M calibrated)": ((-6, 9), "left"), "hybrid GPF-nbrs/raw-walk (M calibrated)": ((0, 8), "center"),
           "hybrid GPF-nbrs/raw-walk (M raw-calibrated)": ((0, -13), "center")}
    xy, ha = OFF.get(m, ((6, 4), "left"))
    ax.annotate(LAB[m].replace(" (default bound)", "").replace(" (ours)", "").replace("hybrid: GPF neighbours, raw walk", "hybrid: GPF neighbours,\nraw walk"), (x, y), xytext=xy, textcoords="offset points", fontsize=7, ha=ha, va="bottom" if xy[1] > 4 else ("top" if xy[1] < -4 else "center"))
ax.axvline(1, ls="--", color="#777", lw=1); ax.margins(y=0.12)
ax.set_xlabel("dispersion ratio  tr Cov(synthetic) / tr Cov(real minority)"); ax.set_ylabel("MMD (RBF), synthetic → held-out minority")
ax.set_xlim(0, max(1.15, d.disp_ratio.max() * 0 + (2.7 if "hybrid GPF-nbrs/raw-walk (M calibrated)" in ORDER else 1.15))); ax.grid(alpha=0.25)
ax.set_title("C.  Dispersion vs distribution match", loc="left", fontsize=10, weight="bold")

# D. gain of ours over SMOTE-raw vs H (C3 prediction)
ax = fig.add_subplot(gs[1, 2])
g = d[d.IR.isin([10, 20])].pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean")
gain = (g["KNNOR-GPF (M calibrated)"] - g["SMOTE-raw"]).reindex(DS)
space = (d[d.method == "none"].groupby("dataset").f1.mean() - d[d.method == "none (raw-flat RF)"].groupby("dataset").f1.mean()).reindex(DS)
sc = ax.scatter(space.values, gain.values, c=meta.H.values, cmap="Blues", s=60, edgecolor="#444", lw=0.6, vmin=0, vmax=1)
for n in DS:
    if abs(gain[n]) > 0.04 or abs(space[n]) > 0.3:
        OFFD = {"Libras": ((4, -9), "left"), "Epilepsy": ((-4, -9), "right"), "ArrowHead": ((-4, -9), "right"), "EthanolConcentration": ((4, -9), "left"),
                "BasicMotions": ((0, 7), "center"), "ERing": ((4, 2), "left"), "RacketSports": ((4, 3), "left"), "Handwriting": ((4, 3), "left"), "JapaneseVowels": ((4, -9), "left"),
                "GunPoint": ((4, 3), "left"), "Cricket": ((4, 3), "left"), "OSULeaf": ((4, 3), "left"), "NATOPS": ((4, 3), "left"), "ArticularyWordRecognition": ((4, 3), "left")}
        xy, ha = OFFD.get(n, ((4, 3), "left"))
        ax.annotate(n if n != "ArticularyWordRecognition" else "ArticularyWordRec.", (space[n], gain[n]), xytext=xy, textcoords="offset points", fontsize=6.5, ha=ha)
ax.margins(x=0.15, y=0.1)
ax.axhline(0, color="#777", lw=1); ax.axvline(0, color="#777", lw=1)
ax.set_xlabel("signal in the feature space:  F1(GPF, none) − F1(raw, none)"); ax.set_ylabel("F1 gain of ours over SMOTE-raw (IR 10 & 20)")
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("H", fontsize=8)
ax.set_title("D.  Where does the geometric walk win?", loc="left", fontsize=10, weight="bold")

import os
if not os.environ.get("PAPER"): fig.suptitle(f"Go/no-go — {len(DS)} datasets · {src.split('/')[-1]}", fontsize=12, weight="bold", y=0.995)
fig.savefig(f"results/{out}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"results/{out}.pdf", bbox_inches="tight")
print("saved", out)
