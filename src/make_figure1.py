"""Fig. 1: the proposed framework. Greyscale only, so the figure reads in print.

Faithful to the implementation (aligned.guided_walk_oversample + knnor.calibrate_M):
  * the guidance space supplies neighbours, the origin filter and the purity vote — nothing else;
  * the step bound M* is calibrated apart from it, in phi_GPF, and shared by every guidance space,
    so it enters the synthesis stage from above;
  * the validity gate is a pre-check, so it sits before the geometric space;
  * the diagnostics are measured on the accepted synthetic set, so they leave the synthesis stage below.
Short labels only; detail belongs in the caption.
"""
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

INK, MID_GREY, HAIR = "#1a1a1a", "#5f5f5f", "#b8b8b8"
F_SYN, F_CHIP, F_AUX = "#e9e9e9", "#f4f4f4", "#f4f4f4"
plt.rcParams.update({"font.size": 10})

fig, ax = plt.subplots(figsize=(13.4, 6.4))
ax.set_xlim(0, 136); ax.set_ylim(0, 68); ax.axis("off")

Y, H = 22, 30
MID = Y + H / 2                      # the axis every in/out arrow sits on


def stage(x, y, w, h, title, chips=(), fc="white", tfs=10.5, cfs=8.8, lw=1.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                fc=fc, ec=INK, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - (4.4 if chips else h / 2), title,
            ha="center", va="center", fontsize=tfs, weight="bold", color=INK, zorder=3)
    n = len(chips)
    for i, c in enumerate(chips):
        if n == 1:                                   # single chip: centre it on the flow axis
            cy = y + h / 2
        else:
            lo, hi = y + 3.4, y + h - 8.8
            cy = (lo + hi) / 2 + (n - 1) * 5.9 / 2 - i * 5.9
        ax.add_patch(FancyBboxPatch((x + 2.5, cy - 2.15), w - 5.0, 4.3,
                                    boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=F_CHIP, ec=HAIR, lw=0.9, zorder=3))
        ax.text(x + w / 2, cy, c, ha="center", va="center", fontsize=cfs, color=INK, zorder=4)


def arrow(p0, p1, lw=1.9, ls="-", color=INK):
    ax.annotate("", p1, p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=17))


# ---- main flow -------------------------------------------------------------
stage(1, Y + 9, 15, 12, "Imbalanced\ntime series", tfs=10.2)

gx, gy, gw, gh = 19.5, MID - 8, 16, 16
ax.add_patch(Polygon([(gx + gw / 2, gy + gh), (gx + gw, gy + gh / 2), (gx + gw / 2, gy), (gx, gy + gh / 2)],
                     closed=True, fc="white", ec=INK, lw=1.7, zorder=2))
ax.text(gx + gw / 2, gy + gh / 2, "validity\ngate", ha="center", va="center",
        fontsize=9.6, weight="bold", color=INK, zorder=3)

stage(39, Y, 25, H, "Geometric space", ("cross-product areas", "path signature", "DTW distance"))
stage(69, Y, 25, H, "Guided selection", ("neighbours", "origin filter", "purity vote"))
stage(99, Y, 22, H, "Raw-space\nsynthesis", ("DTW-aligned\nconvex walk",), fc=F_SYN)
stage(126, Y + 9, 9.5, 12, "Balanced\nset", tfs=10.2)

for a, b in ((16, 19.5), (35.5, 39), (64, 69), (94, 99), (121, 126)):
    arrow((a, MID), (b, MID))

# ---- guidance bracket, below the two guidance boxes ------------------------
ax.plot([39, 94], [19, 19], color=INK, lw=1.1)
ax.plot([39, 39], [19, 20.6], color=INK, lw=1.1); ax.plot([94, 94], [19, 20.6], color=INK, lw=1.1)
ax.text(66.5, 16.6, "guidance   —   no synthetic point is created here",
        ha="center", va="top", fontsize=9.4, style="italic", color=INK)

ax.text(gx + gw / 2, gy - 2.4, "is the geometric space\nmore informative than\nthe raw space?",
        ha="center", va="top", fontsize=8.2, color=MID_GREY)

# ---- M* in from above, diagnostics out below: one vertical axis ------------
CX = 99 + 11                                   # centre of the synthesis box
ax.add_patch(FancyBboxPatch((CX - 17, 57.5), 34, 9.0, boxstyle="round,pad=0.5,rounding_size=1.2",
                            fc="white", ec=INK, lw=1.3, zorder=2))
ax.text(CX, 64.0, "step bound $M^{\\ast}$", ha="center", va="center", fontsize=9.4, weight="bold", color=INK, zorder=3)
ax.text(CX, 60.1, "calibrated so the walk preserves\nthe minority covariance",
        ha="center", va="center", fontsize=8.3, color=INK, zorder=3)
arrow((CX, 57.5), (CX, Y + H + 0.6))

ax.add_patch(FancyBboxPatch((CX - 17, 4.0), 34, 9.0, boxstyle="round,pad=0.5,rounding_size=1.2",
                            fc=F_AUX, ec=MID_GREY, lw=1.2, zorder=2))
ax.text(CX, 10.5, "reported every run", ha="center", va="center", fontsize=9.4, weight="bold", color=INK, zorder=3)
ax.text(CX, 6.6, "dispersion · admissibility\ncalibration · acceptance",
        ha="center", va="center", fontsize=8.3, color=INK, zorder=3)
arrow((CX, Y - 0.6), (CX, 13.0), lw=1.4, ls="--", color=MID_GREY)

fig.savefig("results/fig1_framework.pdf", bbox_inches="tight")
fig.savefig("results/fig1_framework.png", dpi=190, bbox_inches="tight")
print("saved")
