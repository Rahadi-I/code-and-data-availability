import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv("results/frontier_M.csv")
g = d.groupby("M").agg(L_k=("L_k","first"), disp_walk=("disp_walk","mean"), disp_f=("disp_filtered","mean"),
                       accept=("accept","mean"), energy=("energy","mean"), f1=("f1","mean"), f1sd=("f1","std"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); fig.subplots_adjust(wspace=0.55)
a = ax[0]
a.plot(g.index, g.L_k, "--", color="#555", lw=1.5, label="closed form $L_k$ (i.i.d., k = 5)")
a.plot(g.index, g.disp_walk, "-o", color="#2a78d6", ms=4, label="bare walk (measured)")
a.plot(g.index, g.disp_f, "-o", color="#eda100", ms=4, label="after local α + purity filter")
a.axhline(1, color="#999", lw=0.8); a.axvline(1.0, color="#1baf7a", lw=0.8, ls=":"); a.axvline(1.5, color="#999", lw=0.8, ls=":")
a.text(0.97, 1.32, "M = 1\n(default)", fontsize=7, color="#1baf7a", ha="right"); a.text(1.53, 0.05, "M = 3/2\n(theory)", fontsize=7, color="#555", ha="left")
a.set_xlabel("step bound M"); a.set_ylabel("variance retained  tr Cov(syn) / tr Cov(real)"); a.set_ylim(0, 2.4)
a.legend(fontsize=7, frameon=False, loc="upper left"); a.set_title("A.  Contraction vs M", loc="left", weight="bold", fontsize=10)
b = ax[1]
b.plot(g.index, g.accept, "-o", color="#eb6834", ms=4, label="purity-filter acceptance rate")
b.set_xlabel("step bound M"); b.set_ylabel("acceptance rate"); b.set_ylim(0, 1)
b2 = b.twinx(); b2.plot(g.index, g.energy, "-s", color="#9a9a9a", ms=4, label="energy distance to held-out minority")
b2.set_ylabel("energy distance"); b2.spines["top"].set_visible(False)
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower left"); b.set_title("B.  Acceptance and admissibility vs M", loc="left", weight="bold", fontsize=10)
c = ax[2]
for ds, s in d.groupby("dataset"):
    m = s.groupby("M").f1.mean(); c.plot(m.index, m.values - m.values[0], "-", lw=1, alpha=0.5, label=ds)
c.plot(g.index, g.f1 - g.f1.iloc[0], "-o", color="black", ms=4, lw=2, label="mean")
c.axhline(0, color="#999", lw=0.8); c.set_xlabel("step bound M"); c.set_ylabel("F1 change relative to M = 0.5")
c.legend(fontsize=6.5, frameon=False, ncol=2); c.set_title("C.  Downstream F1 vs M (IR = 10)", loc="left", weight="bold", fontsize=10)
import os
if not os.environ.get("PAPER"): fig.suptitle("The acceptance–dispersion frontier in M (6 datasets × 3 seeds, GPF space, k = 5)", weight="bold", fontsize=11)
fig.savefig("results/fig_frontier_M.png", dpi=170, bbox_inches="tight"); fig.savefig("results/fig_frontier_M.pdf", bbox_inches="tight")
print(g.round(3))
