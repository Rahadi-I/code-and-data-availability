"""Go/no-go figure from results/results_v03_select64.csv (English labels, paper-ready)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("results/results_v03_select64.csv")
METHODS = {
    "SMOTE-raw": ("SMOTE, raw space", "#2a78d6"),
    "chain-mean-GPF (PSSTO-style)": ("chain mean in GPF space (PSSTO-style)", "#eb6834"),
    "KNNOR-GPF (M=1)": ("KNNOR in GPF space, M = 1 (published)", "#1baf7a"),
    "KNNOR-GPF (M calibrated)": ("KNNOR in GPF space, M calibrated (ours)", "#eda100"),
}
DATASETS = ["JapaneseVowels", "BasicMotions", "GunPoint", "ItalyPowerDemand", "ArrowHead", "OSULeaf"]
CH = {"JapaneseVowels": 12, "BasicMotions": 6, "GunPoint": 1, "ItalyPowerDemand": 1, "ArrowHead": 1, "OSULeaf": 1}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1], hspace=0.6, wspace=0.75)

# Row 1: F1 vs IR per dataset
for i, ds in enumerate(DATASETS):
    ax = fig.add_subplot(gs[0, i])
    sub = d[d.dataset == ds]
    H = sub.H.iloc[0]
    for m, (lab, col) in METHODS.items():
        g = sub[sub.method == m].groupby("IR").f1
        mu, sd = g.mean(), g.std()
        ax.plot(mu.index, mu.values, "-o", color=col, ms=4, lw=1.8, label=lab)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.10, lw=0)
    g0 = sub[sub.method == "none"].groupby("IR").f1.mean()
    ax.plot(g0.index, g0.values, "--", color="#777", lw=1.2, label="no oversampling")
    ax.set_xticks([5, 10, 20]); ax.set_ylim(0, 1.02)
    ax.set_title(f"{ds}\n{CH[ds]} ch, H = {H:.2f}", fontsize=9)
    ax.set_xlabel("imbalance ratio")
    if i == 0: ax.set_ylabel("F1 (minority), mean ± sd over 5 seeds")
    ax.grid(alpha=0.25)
fig.text(0.5, 0.94, "A.  Minority-class F1 against constructed imbalance ratio  (random forest on 64 selected multi-scale GPF features in every case)",
         ha="center", fontsize=10, weight="bold")

# Row 2 left: mean rank
ax = fig.add_subplot(gs[1, 0:2])
order = ["SMOTE-raw", "KNNOR-raw (M=1)", "chain-mean-GPF (PSSTO-style)", "KNNOR-GPF (M=1)", "KNNOR-GPF (M=3/2)", "KNNOR-GPF (M calibrated)"]
piv = d.pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")[order]
rank = piv.rank(axis=1, ascending=False).mean().sort_values()
labels = {"SMOTE-raw": "SMOTE, raw", "KNNOR-raw (M=1)": "KNNOR, raw, M=1", "chain-mean-GPF (PSSTO-style)": "chain mean, GPF",
          "KNNOR-GPF (M=1)": "KNNOR, GPF, M=1", "KNNOR-GPF (M=3/2)": "KNNOR, GPF, M=3/2", "KNNOR-GPF (M calibrated)": "KNNOR, GPF, M calibrated"}
cols = [METHODS.get(m, ("", "#9a9a9a"))[1] for m in rank.index]
ax.barh([labels[m] for m in rank.index][::-1], rank.values[::-1], color=cols[::-1], height=0.6)
for y, v in enumerate(rank.values[::-1]):
    ax.text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlim(1, 6); ax.set_xlabel("mean rank over 18 dataset × IR cells (1 = best)")
ax.set_title("B.  Overall ranking by F1", loc="left", fontsize=10, weight="bold")

# Row 2 middle: dispersion vs energy distance (the C2 claim)
ax = fig.add_subplot(gs[1, 2:4])
for m in order:
    sub = d[d.method == m]
    x, y = sub.disp_ratio.mean(), sub.energy_syn.mean()
    col = METHODS.get(m, ("", "#9a9a9a"))[1]
    ax.scatter(x, y, s=70, color=col, edgecolor="white", lw=1.2, zorder=3)
    off = {"KNNOR-GPF (M=1)": (-6, -12), "KNNOR-GPF (M=3/2)": (6, 6), "KNNOR-GPF (M calibrated)": (6, -10), "chain-mean-GPF (PSSTO-style)": (-6, 8), "SMOTE-raw": (6, -10), "KNNOR-raw (M=1)": (6, 4)}[m]
    ax.annotate(labels[m], (x, y), xytext=off, textcoords="offset points", fontsize=7.5, ha="left" if off[0] > 0 else "right")
ax.axvline(1.0, color="#777", ls="--", lw=1); ax.text(0.99, 2.0, "target: variance\nretained = 1 ", fontsize=7.5, color="#555", va="top", ha="right")
ax.set_xlabel("dispersion ratio  tr Cov(synthetic) / tr Cov(real minority)")
ax.set_ylabel("energy distance,\nsynthetic → held-out real minority")
ax.set_xlim(0, 1.15); ax.grid(alpha=0.25)
ax.set_title("C.  Calibrating M recovers dispersion and moves the\n     synthetic set toward the held-out minority (C2)", loc="left", fontsize=10, weight="bold")

# Row 2 right: GPF space vs raw space without oversampling (does the space carry signal?)
ax = fig.add_subplot(gs[1, 4:6])
raw = d[d.method == "none (raw-flat RF)"].groupby("dataset").f1.mean().reindex(DATASETS)
gpf = d[d.method == "none"].groupby("dataset").f1.mean().reindex(DATASETS)
yy = np.arange(len(DATASETS))
ax.barh(yy + 0.18, raw.values, height=0.34, color="#9a9a9a", label="raw flattened series")
ax.barh(yy - 0.18, gpf.values, height=0.34, color="#2a78d6", label="GPF feature space")
ax.set_yticks(yy); ax.set_yticklabels([f"{ds} (H={d[d.dataset==ds].H.iloc[0]:.2f})" for ds in DATASETS], fontsize=8)
ax.yaxis.tick_right(); ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True); ax.tick_params(axis="y", length=0)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("F1 without oversampling, mean over IR and seeds")
ax.set_xlim(0, 1.0); ax.legend(fontsize=7.5, loc="upper left", frameon=False, bbox_to_anchor=(0.0, -0.16), ncol=2)
ax.set_title("D.  Does the feature space carry signal? (C3)", loc="left", fontsize=10, weight="bold")

h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Go/no-go prototype v0.3 — KNNOR walk inside GPF space vs raw-space and PSSTO-style oversampling", fontsize=12, weight="bold", y=0.995)
fig.savefig("results/fig_go_no_go_v03.png", dpi=180, bbox_inches="tight")
fig.savefig("results/fig_go_no_go_v03.pdf", bbox_inches="tight")
print("saved")
