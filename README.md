# Figures of the manuscript

Exactly the six figures the paper includes, in the order they appear, each as PDF (used in the
manuscript) and PNG (for quick viewing on GitHub). Nothing else lives here: a figure in this folder
that is not in the paper, or a figure in the paper that is not here, would be a defect.

All six were produced by the run whose CSVs are in `results/` and `v08/results/` — the same run the
manuscript's numbers come from.

| file | in the paper | produced by | from |
|---|---|---|---|
| `fig1_framework` | Fig. 1 — the framework: geometry guides, the raw space synthesises | `src/make_figure1.py` | no data (a drawing) |
| `fig_cd_v06_official` | Fig. 2 — critical-difference diagram, main comparison | `src/stats_cd.py` | `results/results_v06_official_all.csv` |
| `fig_frontier_M` | Fig. 3 — retained variance, acceptance and F1 along the step bound M | `src/frontier_M.py` → `src/plot_frontier.py` | `results/frontier_M.csv` |
| `fig_factorial_v07_20` | Fig. 4 — guidance space × synthesis rule, 20 datasets (panels A–C) | `src/make_figure_factorial20.py` | `results/results_v07_factorial_all.csv` |
| `fig_cd_v07_20` | Fig. 5 — critical-difference diagram, 20-dataset factorial | `src/stats_cd.py` | `results/results_v07_factorial_all.csv` |
| `fig_go_no_go_v05_standard` | Fig. 6 — the validity gate | `src/make_figure_v04.py` | `results/results_v06_official_all.csv` |

Set `PAPER=1` before running the figure scripts to drop the on-figure titles, as in the manuscript.
To regenerate all six from your own run, see `RUN_LOCALLY.md`.
