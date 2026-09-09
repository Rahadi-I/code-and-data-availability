# Path-geometric guidance with raw-space synthesis — code and data availability

Code, result tables and figures for the manuscript
*"Path-geometric guidance with raw-space synthesis: a calibrated k-step convex walk for oversampling imbalanced time series"*
(S. B. Belhaouari, I. Rahadi; in preparation, 2026).

**Claim in one line:** use a path-geometric space (level-3 signature, cross-product areas, or DTW) only to choose neighbours, origins and step sizes, and synthesise on the raw sequences with a DTW-aligned convex step — the result is admissible by construction and is the best-ranked oversampler on 20 UEA/UCR benchmarks.

## What is here

| Folder | Contents |
|---|---|
| `src/` | all code (Python 3.11): feature maps, oversamplers, experiment runners, statistics and figure scripts |
| `results/` | every fitted model as one CSV row: dataset, seed, imbalance ratio, method, F1 / G-mean / balanced accuracy / AUC / Brier / ECE, dispersion ratio, energy distance and MMD to the held-out minority, purity acceptance, calibrated bound M |
| `figures/` | the figures of the manuscript, as produced by the scripts |
| `v08/` | robustness study: a second downstream classifier, the feature-selection budget, and the dispersion guard (a **negative result** — see `v08/README.md`) |

The benchmark datasets are those of the UEA and UCR archives (Bagnall et al. 2018; Dau et al. 2019) and are **not redistributed**.
Download them from https://www.timeseriesclassification.com/dataset.php (or with `src/download_uea.py`) into `src/data/<Name>/<Name>_TRAIN.ts` / `_TEST.ts`; `.zip` archives placed in `src/data/data-download-ulang-manual/` are read directly.

## Method map

| Component | File | Notes |
|---|---|---|
| degree-2 cross-product features (signed Lévy area, two-scale-debiased unsigned area, multi-scale), roughness exponent H | `src/gpf.py` | |
| level-3 path signature by Chen's identity (numpy only) | `src/signature.py` | antisymmetric level-2 part = Lévy area (checked) |
| k-step convex walk with calibrated bound M, local step bound, purity test; closed forms L_k (i.i.d. steps) and L_k^shared (shared step); chain-mean and SMOTE rules; wrapper reproducing the reference KNNOR implementation (`augmentdata`) with a pass cap | `src/knnor.py` | |
| DTW guidance and the DTW-aligned convex step (20 % Sakoe–Chiba window) | `src/aligned.py` | |
| dataset loaders (UEA/UCR `.ts`, UCR `.tsv`, TSER, zip) | `src/data.py` | |
| main comparison (v0.3–v0.6) | `src/run_experiment.py` | env: `SPLIT=standard N_SELECT=64 FAST=1 OFFICIAL=1 OUT=...` |
| factorial: guidance space × synthesis rule (v0.7) | `src/run_factorial.py` | env: `SEEDS`, `CELLS`, `OUT`; DTW guidance skipped when L > 500 or > 800 train series |
| frontier in M | `src/frontier_M.py`, `src/plot_frontier.py` | |
| Friedman + Nemenyi critical-difference diagram, paired Wilcoxon | `src/stats_cd.py` | `METHODS="a;;b;;c"` selects methods |
| figures | `src/make_figure*.py` | |

## Reproducing the tables

```bash
pip install -r requirements.txt
cd src
SPLIT=standard FAST=1 OFFICIAL=1 N_SELECT=64 OUT=../results/results_v06_official_all.csv \
  python run_experiment.py ArticularyWordRecognition BasicMotions CharacterTrajectories Cricket ERing Epilepsy \
  EthanolConcentration HandMovementDirection Handwriting JapaneseVowels Libras NATOPS RacketSports \
  SelfRegulationSCP1 SelfRegulationSCP2 UWaveGestureLibrary GunPoint ItalyPowerDemand ArrowHead OSULeaf
SEEDS=0,1,2 OUT=../results/results_v07_factorial.csv python run_factorial.py NATOPS JapaneseVowels RacketSports \
  GunPoint Handwriting ArticularyWordRecognition Epilepsy OSULeaf
# confirmatory run, all 20 datasets (results/results_v07_factorial_all.csv, figures/fig_factorial_v07_20.*, fig_cd_v07_20.*)
SEEDS=0,1,2 CELLS="gpf/plain,gpf/aligned,sig/aligned,dtw/aligned" OUT=../results/results_v07_factorial_all.csv \
  python run_factorial.py <all 20 dataset names above>
METHODS="SMOTE-raw;;guide=gpf | walk=plain;;guide=gpf | walk=aligned;;guide=sig | walk=aligned" \
  python stats_cd.py ../results/results_v07_factorial_all.csv fig_cd_v07_20
python make_figure_factorial20.py ../results/results_v07_factorial_all.csv fig_factorial_v07_20
```
Headline (20 datasets, N = 60 dataset×IR blocks): signature-guided walk with DTW-aligned step, mean rank 1.92 vs 2.88 for SMOTE
(Nemenyi CD 0.61, Friedman p = 4e-6), ΔF1 = +0.048 (Wilcoxon p = 1.7e-4); GPF-guided aligned +0.042 (p = 1.4e-3);
aligned vs plain step +0.030 (p = 4e-4); DTW-guided aligned +0.072 on the 15 datasets with L ≤ 500 (p = 7.6e-5).
The ranking survives an L2-logistic downstream classifier and feature budgets of 32, 64 and 128 (`v08/`).
Protocol: archive TRAIN/TEST splits; the largest class is made the minority and subsampled to IR ∈ {5, 10, 20}; 5 seeds (3 in the factorial);
random forest (300 trees) on the same 64 ANOVA-selected features for every method; 2 CPU cores suffice (≈1 h per 20-dataset run,
longer with DTW guidance).

## Citation
Belhaouari, S. B., Rahadi, I. Path-geometric guidance with raw-space synthesis: a calibrated k-step convex walk for oversampling imbalanced time series. In preparation, 2026.

Authorship and venue are not yet final; this repository accompanies a manuscript under preparation.

## Licence
MIT (code). Result tables and figures: CC BY 4.0.
