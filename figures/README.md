# Figures of the manuscript

Exactly the six figures the paper includes, in the order they appear, each as PDF (used in the
manuscript) and PNG (for quick viewing on GitHub). Nothing else lives here: a figure in this folder
that is not in the paper, or a figure in the paper that is not here, would be a defect.

| file | in the paper | produced by |
|---|---|---|
| `fig1_framework` | Fig. 1 — the framework: geometry guides, the raw space synthesises | `src/make_figure1.py` |
| `fig_cd_v06_official` | Fig. 2 — critical-difference diagram, main comparison (v0.6) | `src/stats_cd.py` |
| `fig_frontier_M` | Fig. 3 — retained variance and accuracy along the step bound M | `src/frontier_M.py`, `src/plot_frontier.py` |
| `fig_factorial_v07_20` | Fig. 4 — guidance space x synthesis rule, 20 datasets (panels A-C) | `src/make_figure_factorial20.py` |
| `fig_cd_v07_20` | Fig. 5 — critical-difference diagram, 20-dataset factorial | `src/stats_cd.py` |
| `fig_go_no_go_v05_standard` | Fig. 6 — the validity gate | `src/make_figure_v04.py` |

Set `PAPER=1` before running the figure scripts to drop the on-figure titles, as in the manuscript.
To regenerate every figure from your own run, see `RUN_LOCALLY.md`.
