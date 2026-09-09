"""Fig. 1 of the paper: the problem (A-C), the closed-form contraction in M (D), the validity gate (E) and the pipeline (F)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from knnor import L_k, L_k_shared_alpha

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 3, figsize=(14, 9)); fig.subplots_adjust(wspace=0.32, hspace=0.42)
BLUE, GREEN, ORANGE, NAVY, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#1f3a5f", "#777"

# A. raw-space interpolation of two shifted peaks
a = ax[0, 0]; t = np.arange(101)
x1 = np.exp(-((t - 32) / 8) ** 2); x2 = np.exp(-((t - 62) / 8) ** 2)
a.plot(t, x1, color=BLUE, lw=2, label="minority sample 1"); a.plot(t, x2, color=GREEN, lw=2, label="minority sample 2")
a.plot(t, 0.5 * (x1 + x2), "--", color=ORANGE, lw=2, label="convex midpoint (SMOTE / KNNOR)")
a.annotate("two half-peaks:\nno physical counterpart", (63, 0.5), xytext=(74, 0.72), fontsize=8, color="#444",
           arrowprops=dict(arrowstyle="-", color="#777"), ha="left"); a.set_ylim(0, 1.5)
a.set_xlabel("time"); a.set_ylabel("value"); a.legend(fontsize=7.5, frameon=False, loc="upper left")
a.set_title("A.  Raw-space interpolation of misaligned series", loc="left", weight="bold", fontsize=9.5)

# B. path space: same loop shifted in phase
b = ax[0, 1]; th = np.linspace(0, 2 * np.pi, 200)
b.plot(np.cos(th), 0.6 * np.sin(th), color=BLUE, lw=2, label="path 1 (a loop)")
b.plot(np.cos(th), 0.6 * np.sin(th), "--", color=GREEN, lw=2, dashes=(4, 3), label="path 2 = same loop, phase-shifted")
ph = 2 * np.pi / 3; m = 0.5 * (np.c_[np.cos(th), 0.6 * np.sin(th)] + np.c_[np.cos(th + ph), 0.6 * np.sin(th + ph)])
b.fill(m[:, 0], m[:, 1], color=ORANGE, alpha=0.2); b.plot(m[:, 0], m[:, 1], color=ORANGE, lw=2, label="pointwise mean of the two paths")
b.text(0, -0.95, "area of the mean path = 25% of the true area\nmean of the two signed areas = 100%", ha="center", fontsize=8, color="#444")
b.set_xlim(-1.35, 1.35); b.set_ylim(-1.15, 1.05); b.set_aspect("equal"); b.set_xlabel("channel 1"); b.set_ylabel("channel 2")
b.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
b.set_title("B.  In path space the same loop is one point,\n     but the mean of the paths is not", loc="left", weight="bold", fontsize=9.5)

# C. variance retained vs k for each convex rule
c = ax[0, 2]; ks = np.arange(1, 9)
c.axhline(1, color=GREY, lw=0.8, ls=":")
c.plot(ks, [L_k(1.5, k) for k in ks], "-o", color=GREEN, ms=4, lw=2, label="KNNOR, M = 3/2 (calibrated)")
c.plot(ks, [2 / 3] * len(ks), "-o", color=BLUE, ms=4, lw=2, label="SMOTE (k = 1 case)")
c.plot(ks, [L_k(1.0, k) for k in ks], "-o", color=ORANGE, ms=4, lw=2, label="KNNOR, M = 1 (default bound)")
c.plot(ks, [1 / k for k in ks], "-s", color=NAVY, ms=4, lw=2, label="chain mean of k (PSSTO-style)")
c.set_xlabel("neighbours visited, k"); c.set_ylabel("variance retained  tr Cov(syn) / tr Cov(real)"); c.set_ylim(0, 1.55)
c.legend(fontsize=7.5, frameon=False, loc="upper right")
c.set_title("C.  Every convex rule contracts the minority", loc="left", weight="bold", fontsize=9.5)

# D. closed form in M: i.i.d. alpha vs shared alpha (reference implementation), k = 5
d = ax[1, 0]; Ms = np.linspace(0.3, 2.4, 120)
d.axhline(1, color=GREY, lw=0.8, ls=":")
d.plot(Ms, [L_k(M, 5) for M in Ms], color=BLUE, lw=2, label="independent step sizes, $L_5(M)$")
d.plot(Ms, [L_k_shared_alpha(M, 5, n=20001) for M in Ms], color=ORANGE, lw=2, label="one shared step size, $L_5^{\\mathrm{sh}}(M)$")
d.axvline(1.0, color=GREY, lw=0.8, ls="--"); d.axvline(1.5, color=GREY, lw=0.8, ls="--"); d.axvline(1.55, color=GREY, lw=0.8, ls="--")
d.scatter([1.0, 1.0], [L_k(1.0, 5), L_k_shared_alpha(1.0, 5)], color=[BLUE, ORANGE], zorder=3, s=30)
d.text(0.97, 1.25, "M = 1\n(default)", ha="right", fontsize=7.5, color="#444"); d.text(1.58, 0.12, "M = 3/2 and M ≈ 1.55:\nfull variance retained", ha="left", fontsize=7.5, color="#444")
d.text(0.96, L_k(1.0, 5) + 0.07, f"{L_k(1.0, 5):.2f}", fontsize=7.5, color=BLUE, ha="right"); d.text(0.96, L_k_shared_alpha(1.0, 5) - 0.16, f"{L_k_shared_alpha(1.0, 5):.2f}", fontsize=7.5, color=ORANGE, ha="right")
d.set_xlabel("step bound M"); d.set_ylabel("variance retained, k = 5"); d.set_ylim(0, 2.1)
d.legend(fontsize=7.5, frameon=False, loc="upper left")
d.set_title("D.  The contraction has a closed form in M", loc="left", weight="bold", fontsize=9.5)

# E. roughness gate
e = ax[1, 1]; w = np.array([4, 8, 16, 32, 64, 128]); rng = np.random.default_rng(0)
e.loglog(w, (w / 4) ** 0.98, "-o", color=GREEN, lw=2, ms=4, label="smooth path: H = 0.98"); e.loglog(w, (w / 4) ** 0.48, "-o", color=ORANGE, lw=2, ms=4, label="diffusive path: H = 0.48")
e.text(11, 11, "features carry\nthe shape", color=GREEN, fontsize=8, ha="left"); e.text(45, 1.6, "features are\nnoise-dominated", color=ORANGE, fontsize=8, ha="left")
e.set_xlabel("lag w"); e.set_ylabel("mean chord  |X(t+w) − X(t)|, normalised"); e.legend(fontsize=7.5, frameon=False, loc="upper left")
e.set_title("E.  Roughness exponent H as a validity gate", loc="left", weight="bold", fontsize=9.5)

# F. pipeline: guidance vs synthesis
f = ax[1, 2]; f.axis("off"); f.set_xlim(0, 10); f.set_ylim(0, 10)
def box(x, y, w, h, text, fc="#eef2f7", ec=NAVY, bold=False):
    f.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2", fc=fc, ec=ec, lw=1.2))
    f.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.9, weight="bold" if bold else "normal")
def arrow(x0, y0, x1, y1): f.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#555", lw=1))
box(1.0, 8.6, 8.0, 1.1, "imbalanced multivariate time series")
f.text(2.5, 7.85, "GUIDANCE\n(geometric space)", ha="center", va="center", fontsize=6.8, weight="bold", color=GREEN)
f.text(7.5, 7.85, "SYNTHESIS\n(raw space)", ha="center", va="center", fontsize=6.8, weight="bold", color=BLUE)
box(0.1, 6.1, 4.7, 1.2, "neighbours + origin filter in\nsignature / GPF / DTW space", fc="#e6f7ef", ec=GREEN)
box(0.1, 4.5, 4.7, 1.2, "step bound M* calibrated\n(dispersion → 1), local α", fc="#e6f7ef", ec=GREEN)
box(0.1, 2.9, 4.7, 1.2, "purity vote\n(acceptance reported)", fc="#e6f7ef", ec=GREEN)
box(5.3, 4.5, 4.5, 2.8, "k-step convex walk on\nraw sequences, each step\nDTW-aligned\n→ admissible", fc="#e9f1fb", ec=BLUE, bold=False)
box(5.3, 2.9, 4.5, 1.2, "classifier on real +\nsynthetic sequences", fc="#eef2f7")
box(1.0, 0.6, 8.0, 1.5, "reported: dispersion, MMD, Brier / ECE, acceptance;\ngate: class signal of the guiding space")
arrow(3.0, 8.6, 2.5, 8.25); arrow(7.0, 8.6, 7.5, 8.25); arrow(4.8, 6.7, 5.3, 6.4); arrow(4.8, 5.1, 5.3, 5.4); arrow(4.8, 3.5, 5.3, 3.5)
arrow(7.5, 4.5, 7.5, 4.1); arrow(7.5, 2.9, 7.5, 2.1)
f.set_title("F.  Geometry guides, raw space synthesises", loc="left", weight="bold", fontsize=9.5)

fig.savefig("results/fig1_problem_plan.pdf", bbox_inches="tight"); fig.savefig("results/fig1_problem_plan.png", dpi=170, bbox_inches="tight"); print("saved")
