# Path-geometric guidance with raw-space synthesis — code and data availability

Code, result tables and figures for the manuscript
*"Path-geometric guidance with raw-space synthesis: a calibrated k-step convex walk for oversampling imbalanced time series"*
(S. B. Belhaouari, I. Rahadi; in preparation, 2026).

**Claim in one line:** use a path-geometric space (level-3 signature, cross-product areas, or DTW) only to choose neighbours, origins and step sizes, and synthesise on the raw sequences with a DTW-aligned convex step — the result is admissible by construction and is the best-ranked oversampler on 20 UEA/UCR benchmarks.

## Provenance

**Every number and every figure in the manuscript comes from the CSVs in this repository**, produced on the corresponding author's machine (Windows, Anaconda, Python 3.11) in September 2026. Nothing is carried over from an earlier run.

`src/audit_numbers.py` is the check. It recomputes every quantity the manuscript quotes — both tables, the ranks, the paired tests, the medians, the per-dataset gains — and prints the manuscript's value beside the computed one:

```bash
cd src
python audit_numbers.py
```

Expected: `ALL QUOTED NUMBERS AGREE`. A `MISMATCH` line means the manuscript and these CSVs disagree, which would be a defect worth reporting.

| manuscript item | produced from |
|---|---|
| Table 1, Fig. 2, Fig. 6 | `results/results_v06_official_all.csv` |
| Sec. 5.3, the eight-dataset pilot | `results/results_v07_factorial.csv` |
| Table 2, Fig. 4, Fig. 5 | `results/results_v07_factorial_all.csv` |
| Fig. 3 | `results/frontier_M.csv` |
| Sec. 5.5, robustness | `v08/results/results_v08_*.csv` |
| Appendix A | `results/table_datasets.tex` |
| Propositions 1 and 2, the DTW-aligned step, the signature/Lévy-area identity | no data — `src/verify_claims.py` |

## What is here

| Folder | Contents |
|---|---|
| `src/` | all code (Python 3.11): feature maps, oversamplers, experiment runners, statistics, figure scripts, the claim self-check and the number audit |
| `results/` | one CSV row per fitted model: dataset, seed, imbalance ratio, method, F1 / G-mean / balanced accuracy / AUC / Brier / ECE, dispersion ratio, energy distance and MMD to the held-out minority, purity acceptance, calibrated bound M |
| `figures/` | the six figures of the manuscript, PDF and PNG |
| `v08/` | robustness study of Sec. 5.5: a second downstream classifier, the feature-selection budget, and the dispersion guard — a **negative result**, see `v08/README.md` |
| `paper/` | the compiled manuscript |

The benchmark datasets are those of the UEA and UCR archives (Bagnall et al. 2018; Dau et al. 2019) and are **not redistributed**. Download them from https://www.timeseriesclassification.com/dataset.php (or with `src/download_uea.py`) into `src/data/<Name>/<Name>_TRAIN.ts` / `_TEST.ts`; `.zip` archives placed in `src/data/data-download-ulang-manual/` are read directly.

## Reproducing everything

`RUN_LOCALLY.md` is the full guide. In short, from `src/`:

```bash
python run_all_local.py smoke      # one dataset, a few minutes — check the pipeline runs
python run_all_local.py main       # the 20-dataset factorial, 2–4 h  → Table 2, Figs. 4–5
python run_all_local.py stats      # Friedman + Nemenyi + Wilcoxon
python run_all_local.py figures    # Figs. 1 and 4
python run_all_local.py pilot      # Sec. 5.3, the full 3×2 factorial on 8 datasets
python run_all_local.py compare    # Fig. 2 (and the CSV Fig. 6 needs), 1–2 h
python run_all_local.py gate       # Fig. 6, seconds
python run_all_local.py frontier   # Fig. 3, 30–60 min
python run_all_local.py v08        # Sec. 5.5, several hours
python run_all_local.py v08nsel    # feature budgets 32 and 128
python run_all_local.py v08trim    # the trimming variant of the guard
python run_all_local.py v08stats   # analyse the three above
```

Every run is seeded. With the package versions listed in `RUN_LOCALLY.md` the numbers reproduce to the last digit; with a different scikit-learn expect drift in the third decimal of F1 on the smallest cells (IR = 20 leaves four minority sequences, where one sample changing sides moves F1 by 0.17), with ranks and p-values unchanged.

## Method map

| Component | File | Notes |
|---|---|---|
| degree-2 cross-product features (signed Lévy area, two-scale-debiased unsigned area, multi-scale), roughness exponent H | `src/gpf.py` | |
| level-3 path signature by Chen's identity (numpy only) | `src/signature.py` | antisymmetric level-2 part = Lévy area (checked) |
| k-step convex walk with calibrated bound M, local step bound, purity test; closed forms L_k (i.i.d. steps) and L_k^shared (shared step); chain-mean and SMOTE rules; wrapper reproducing the reference KNNOR implementation (`augmentdata`) with a pass cap | `src/knnor.py` | |
| DTW guidance and the DTW-aligned convex step (20 % Sakoe–Chiba window) | `src/aligned.py` | |
| dataset loaders (UEA/UCR `.ts`, UCR `.tsv`, TSER, zip) | `src/data.py` | |
| main comparison | `src/run_experiment.py` | env: `SPLIT=standard N_SELECT=64 FAST=1 OFFICIAL=1 OUT=...` |
| factorial: guidance space × synthesis rule | `src/run_factorial.py` | env: `SEEDS`, `CELLS`, `OUT`; DTW guidance skipped when L > 500 or > 800 train series |
| frontier in M | `src/frontier_M.py`, `src/plot_frontier.py` | writes `src/results/frontier_M.csv` |
| Friedman + Nemenyi critical-difference diagram, paired Wilcoxon | `src/stats_cd.py` | `METHODS="a;;b;;c"` selects methods |
| figures | `src/make_figure*.py` | `PAPER=1` drops the on-figure titles |
| self-check of every analytical claim (no data needed, ~10 s) | `src/verify_claims.py` | exits non-zero on disagreement |
| recomputes every number the manuscript quotes | `src/audit_numbers.py` | exits non-zero on disagreement |
| one driver for all of the above | `src/run_all_local.py` | no environment variables, works on Windows and Linux alike |

## Headline result

20 datasets (16 UEA + 4 univariate), IR ∈ {5, 10, 20}, three seeds, N = 60 dataset×IR blocks:

| method | mean rank | mean F1 | ΔF1 vs SMOTE | p |
|---|---|---|---|---|
| signature-guided, aligned step | **1.91** | 0.648 | +0.047 | 2.9e-4 |
| cross-product-area-guided, aligned step | 2.26 | 0.643 | +0.042 | 1.4e-3 |
| SMOTE, raw space | 2.88 | 0.601 | — | — |
| cross-product-area-guided, plain step | 2.96 | 0.613 | +0.012 | 0.75 |

Friedman p = 2.8e-6, Nemenyi CD = 0.61. The synthesis rule alone — aligned against plain step under the same guidance — is worth +0.029 (p = 5.1e-4). On the 15 datasets short enough for DTW guidance, +0.072 (p = 7.6e-5). The ordering survives an L2-logistic downstream classifier and feature budgets of 32, 64 and 128 (`v08/`).

Protocol: archive TRAIN/TEST splits; the largest class is made the minority and subsampled to IR ∈ {5, 10, 20}; random forest (300 trees) on the same 64 ANOVA-selected features for every method.

## Citation
Belhaouari, S. B., Rahadi, I. Path-geometric guidance with raw-space synthesis: a calibrated k-step convex walk for oversampling imbalanced time series. In preparation, 2026.

Authorship and venue are not yet final; this repository accompanies a manuscript under preparation.

## Licence
MIT (code). Result tables and figures: CC BY 4.0.
