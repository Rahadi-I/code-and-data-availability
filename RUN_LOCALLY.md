# Reproducing the paper on your own machine

Windows, Anaconda. Everything below assumes the working folder `D:\thesis\paper0_all_start_here\`.
Commands are for **Anaconda Prompt (cmd)**; PowerShell equivalents are noted where they differ.

The point of this file is that you can check the findings yourself, cheapest first. If any step
disagrees with the paper, that is a result — write it down rather than adjusting the code to match.

---

## 0. Environment

```cmd
conda create -n geoover python=3.11 -y
conda activate geoover
pip install "numpy>=1.26" "scipy>=1.11" "pandas>=2.0" "scikit-learn>=1.4" "matplotlib>=3.8" "aeon>=1.0"
pip install augmentdata==0.0.12
```

Versions used to produce the numbers in the paper:

| package | version |
|---|---|
| python | 3.11.15 |
| numpy | 2.3.5 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| pandas | 2.3.3 |
| aeon | 1.5.0 |
| matplotlib | 3.10.9 |

**On determinism.** Every run is seeded, so with the same package versions you should get the same
numbers to the last digit. With a different scikit-learn the random forest's internals may differ
slightly; expect drift in the third decimal of F1, with ranks and p-values essentially unchanged.
If you see more than that, something else is wrong.

---

## 1. Analytical claims — about ten seconds, no data needed

```cmd
cd /d D:\thesis\paper0_all_start_here\code
python verify_claims.py
```

This checks the mathematics of the paper against this code and prints PASS/FAIL per claim:

* Proposition 1: `L_5(1) = 0.502`, `L_5(3/2) = 1` exactly, SMOTE (`k=1`, `M=1`) retains 2/3,
  and the non-monotonicity — `L_5` bottoms out near 0.25 at `M ≈ 0.45`.
* Proposition 1 again, by **Monte Carlo**: a simulated walk on i.i.d. Gaussian neighbours must
  reproduce the closed form at `M = 0.8, 1.0, 1.5, 2.0`. This is the real test of the theory.
* Proposition 2 (one shared step size, the rule in `augmentdata`): `0.473`, `0.906`, and unit
  retention at `M ≈ 1.562`.
* The DTW-aligned step, both numbers quoted in the method section.
* The level-2 signature's antisymmetric part equals the signed cross-product area.
* Two-scale debiasing: it must cut the bias by more than half on a smooth path. The rough-path
  case is printed but **not** asserted — at `H ≈ 1/2` it does not help, and the paper says so.

Expect `ALL CHECKS PASSED` and exit code 0.

```cmd
python aligned.py
```
prints the two DTW-aligned-step numbers on their own.

---

## 2. Data is in place — one minute

```cmd
python data.py
```

Lists every dataset the loaders can see and its shape. You should get the 20 used in the paper:

ArrowHead, ArticularyWordRecognition, BasicMotions, CharacterTrajectories, Cricket, ERing,
Epilepsy, EthanolConcentration, GunPoint, HandMovementDirection, Handwriting, ItalyPowerDemand,
JapaneseVowels, Libras, NATOPS, OSULeaf, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2,
UWaveGestureLibrary.

Layout the loaders expect, all under `code\data\`:

```
data\<Name>\<Name>_TRAIN.ts          UEA multivariate
data\UCRArchive_2018\<Name>\<Name>_TRAIN.tsv
data\Monash_UEA_UCR_Regression_Archive\<Name>\<Name>_TRAIN.ts
```

---

## 3. One dataset end to end — a few minutes

Check the pipeline runs before committing hours to it.

```cmd
set SEEDS=0
set CELLS=gpf/plain,gpf/aligned,sig/aligned
set OUT=..\results\smoke.csv
python run_factorial.py ItalyPowerDemand
```

PowerShell instead of `set`:
```powershell
$env:SEEDS="0"; $env:CELLS="gpf/plain,gpf/aligned,sig/aligned"; $env:OUT="..\results\smoke.csv"
```

One line per (seed, IR) with the F1 of each method. Delete `smoke.csv` afterwards.

---

## 4. The headline result — roughly two to four hours

The 3 × 2 factorial on all 20 datasets, three seeds. This produces the numbers in Table 2 and
Figure 4. On two cores it took 11,620 s; your machine should be faster.

```cmd
set SEEDS=0,1,2
set CELLS=gpf/plain,gpf/aligned,sig/aligned,dtw/aligned
set OUT=..\results\repro_v07.csv
python run_factorial.py ItalyPowerDemand ArrowHead GunPoint BasicMotions ERing JapaneseVowels ^
  Libras RacketSports Epilepsy NATOPS OSULeaf Handwriting HandMovementDirection ^
  ArticularyWordRecognition CharacterTrajectories Cricket UWaveGestureLibrary ^
  SelfRegulationSCP1 SelfRegulationSCP2 EthanolConcentration
```

Put EthanolConcentration last — it is by far the slowest (L = 1751). The CSV is rewritten after
every dataset, so you can inspect partial results and stop early without losing anything.

Then the statistics:

```cmd
set METHODS=SMOTE-raw;;guide=gpf | walk=plain;;guide=gpf | walk=aligned;;guide=sig | walk=aligned
python stats_cd.py ..\results\repro_v07.csv fig_cd_repro
```

**What you should see** (N = 60 dataset × IR blocks):

| method | mean rank | ΔF1 vs SMOTE | p (Wilcoxon) |
|---|---|---|---|
| signature-guided, aligned | 1.92 | +0.048 | 1.7e-4 |
| GPF-guided, aligned | 2.26 | +0.042 | 1.4e-3 |
| SMOTE, raw | 2.88 | — | — |
| GPF-guided, plain | 2.95 | +0.012 | 0.79 |

Friedman p = 4.0e-6, Nemenyi CD = 0.61. The DTW-guided aligned cell covers only the 15 datasets
with L ≤ 500 (it is skipped automatically on the rest): +0.072, p = 7.6e-5.

The single most important comparison is **gpf/aligned versus gpf/plain**: same guidance, different
synthesis rule, +0.030 with p = 4e-4. That isolates the DTW-aligned step from everything else.

---

## 5. Robustness — the v0.8 folder

```cmd
cd /d D:\thesis\paper0_all_start_here\v08
set SEEDS=0,1,2
set CLFS=rf,ridge
set N_SELECT=64
set GUARD=1
set CELLS=gpf/plain,gpf/aligned,sig/aligned
set OUT=results\repro_v08.csv
python run_v08.py <the same 20 dataset names>
python analyse_v08.py
```

Three things at once: the second classifier, the feature budget, and the dispersion guard. The
guard is a **negative result** — it caps the dispersion (373 → 2 on EthanolConcentration) and
improves admissibility, yet leaves F1 unchanged (+0.006, p = 0.49). If your run shows the guard
helping, that contradicts the paper and I would want to know.

For the feature budget, repeat with `set N_SELECT=32` and `set N_SELECT=128` on the eight pilot
datasets (ArticularyWordRecognition, Epilepsy, NATOPS, OSULeaf, GunPoint, Handwriting,
JapaneseVowels, RacketSports).

---

## 6. Figures

```cmd
cd /d D:\thesis\paper0_all_start_here\code
python make_figure1.py                                     # framework diagram
python make_figure_factorial20.py ..\results\repro_v07.csv fig_factorial_repro
python make_figure_v04.py ..\results\results_v05_standard.csv fig_overview_repro
python plot_frontier.py                                    # needs results\frontier_M.csv
```

Set `PAPER=1` before the last two to drop the on-figure titles, as in the manuscript.

---

## If something disagrees

Send me the CSV and the console output. Three numbers in the manuscript have already been
corrected this way — a peak height that no configuration reproduced, a unit-retention point that
was 1.55 instead of 1.562, and a claim that the debiasing removes the noise floor when on rough
paths it does not. Finding a fourth would be a good outcome, not an embarrassing one.
