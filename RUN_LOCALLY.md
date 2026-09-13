# Reproducing the paper on your own machine

The point of this file is that you can check the findings yourself, cheapest first. If any step
disagrees with the paper, that is a result — write it down rather than adjusting the code to match.

Commands are shown for **Anaconda Prompt on Windows**; on Linux or macOS they are identical except
for the `cd` syntax.

---

## 0. Environment

```
conda create -n geoover python=3.11 -y
conda activate geoover
pip install -r requirements.txt
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
The cells at IR = 20 leave four minority sequences, where one sample changing sides moves F1 by
0.17, so that is where any drift shows up first. If you see more than that, something else is wrong.

---

## 1. The analytical claims — ten seconds, no data needed

```
cd src
python verify_claims.py
```

Checks the mathematics of the paper against this code and prints PASS/FAIL per claim:

* Proposition 1: `L_5(1) = 0.502`, `L_5(3/2) = 1` exactly, SMOTE (`k=1`, `M=1`) retains 2/3, and the
  non-monotonicity — `L_5` bottoms out near 0.25 at `M ≈ 0.45`.
* Proposition 1 again, by **Monte Carlo**: a simulated walk on i.i.d. Gaussian neighbours must
  reproduce the closed form at `M = 0.8, 1.0, 1.5, 2.0`. This is the real test of the theory.
* Proposition 2 (one shared step size, the rule in `augmentdata`): `0.473`, `0.906`, and unit
  retention at `M ≈ 1.562`.
* The DTW-aligned step, both numbers quoted in the method section.
* The level-2 signature's antisymmetric part equals the signed cross-product area.
* Two-scale debiasing: it must cut the bias by more than half on a smooth path. The rough-path case
  is printed but **not** asserted — at `H ≈ 1/2` it does not help, and the paper says so.

Expect `ALL CHECKS PASSED` and exit code 0.

## 2. The quoted numbers against the shipped CSVs — seconds, no re-running

```
python audit_numbers.py
```

Recomputes every quantity the manuscript quotes — both tables, ranks, paired tests, medians,
per-dataset gains — from the CSVs in `results/` and `v08/results/`, and prints the manuscript's
value beside the computed one. Expect `ALL QUOTED NUMBERS AGREE`.

After your own run, point it at your CSVs with `--fac`, `--v06`, `--pilot`, `--frontier`, `--v08`.

---

## 3. Data is in place — one minute

```
python data.py
```

Lists every dataset the loaders can see and its shape. The 20 used in the paper:

ArrowHead, ArticularyWordRecognition, BasicMotions, CharacterTrajectories, Cricket, ERing,
Epilepsy, EthanolConcentration, GunPoint, HandMovementDirection, Handwriting, ItalyPowerDemand,
JapaneseVowels, Libras, NATOPS, OSULeaf, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2,
UWaveGestureLibrary.

Layout the loaders expect, all under `src/data/`:

```
data/<Name>/<Name>_TRAIN.ts                              UEA multivariate
data/UCRArchive_2018/<Name>/<Name>_TRAIN.tsv             UCR univariate
data/Monash_UEA_UCR_Regression_Archive/<Name>/...        TSER
data/data-download-ulang-manual/<Name>.zip               read directly
```

---

## 4. Re-running the experiments

One driver does all of it. No environment variables, no line continuations, no difference between
cmd and PowerShell. Run it from `src/`.

| command | what it produces | roughly |
|---|---|---|
| `python run_all_local.py smoke` | one dataset, one seed — checks the pipeline runs | 1 min |
| `python run_all_local.py main` | Table 2, Figs. 4–5 — the 20-dataset factorial | 2–4 h |
| `python run_all_local.py stats` | Friedman + Nemenyi + Wilcoxon on the above | seconds |
| `python run_all_local.py figures` | Fig. 1 and Fig. 4 | seconds |
| `python run_all_local.py pilot` | Sec. 5.3 — the full 3×2 factorial on 8 datasets | ~1 h |
| `python run_all_local.py compare` | Fig. 2, and the CSV Fig. 6 also needs | 1–2 h |
| `python run_all_local.py gate` | Fig. 6, from the CSV `compare` produced | seconds |
| `python run_all_local.py frontier` | Fig. 3 — its own sweep over M | 30–60 min |
| `python run_all_local.py v08` | Sec. 5.5 — the guard and the second classifier | several h |
| `python run_all_local.py v08nsel` | Sec. 5.5 — feature budgets 32 and 128 | ~1 h |
| `python run_all_local.py v08trim` | Sec. 5.5 — the trimming variant of the guard | ~1 h |
| `python run_all_local.py v08stats` | analyses the three v08 runs | seconds |

Results are written as `repro_*.csv` so the shipped CSVs are never overwritten. Each run rewrites
its CSV after every dataset, so you can stop with Ctrl-C and keep what is finished.

**What you should see** from `main` + `stats` (N = 60 dataset × IR blocks):

| method | mean rank | ΔF1 vs SMOTE | p (Wilcoxon) |
|---|---|---|---|
| signature-guided, aligned | 1.91 | +0.047 | 2.9e-4 |
| GPF-guided, aligned | 2.26 | +0.042 | 1.4e-3 |
| SMOTE, raw | 2.88 | — | — |
| GPF-guided, plain | 2.96 | +0.012 | 0.75 |

Friedman p = 2.8e-6, Nemenyi CD = 0.61. The DTW-guided aligned cell covers only the 15 datasets
with L ≤ 500 (it is skipped automatically on the rest): +0.072, p = 7.6e-5.

The single most important comparison is **gpf/aligned versus gpf/plain**: same guidance, different
synthesis rule, +0.029 with p = 5.1e-4. That isolates the DTW-aligned step from everything else.

From `v08` + `v08stats`, the guard is a **negative result** — it caps the dispersion (373 → 2 on
EthanolConcentration) and improves admissibility, yet leaves F1 unchanged (+0.006, p = 0.49). If
your run shows the guard helping, that contradicts the paper and we would want to know.

---

## If something disagrees

Open an issue with the CSV and the console output. Several numbers in the manuscript were corrected
this way before submission — a peak height that no configuration reproduced, a unit-retention point
that was 1.55 instead of 1.562, three mean-F1 values in the pilot section, a dispersion median that
read 2.25 where the data gives 1.70, and a claim that two-scale debiasing removes the noise floor
when on rough paths it does not. Finding another would be a good outcome, not an embarrassing one.
