"""Appendix A: the dataset table. Emits LaTeX rows to results/table_datasets.tex."""
import numpy as np, pandas as pd
from data import load_binary, DATASETS
from gpf import hurst_structure_function

DS = ["ArrowHead", "ArticularyWordRecognition", "BasicMotions", "CharacterTrajectories", "Cricket", "ERing",
      "Epilepsy", "EthanolConcentration", "GunPoint", "HandMovementDirection", "Handwriting", "ItalyPowerDemand",
      "JapaneseVowels", "Libras", "NATOPS", "OSULeaf", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
      "UWaveGestureLibrary"]
UNIV = {"ArrowHead", "GunPoint", "ItalyPowerDemand", "OSULeaf"}

# Class signal of the guiding space: F1(GPF features, no oversampling) - F1(raw flattened,
# no oversampling). Both cells are in the main-comparison CSV, so the Signal column comes from
# the same run as the rest of the paper. Look for it in either layout (repo: ../results).
import os
_cand = [os.path.join(r, n) for r in ("results", os.path.join("..", "results"))
         for n in ("repro_v06.csv", "results_v06_official_all.csv", "results_v05_standard.csv")]
_csv = next((c for c in _cand if os.path.exists(c)), None)
if _csv is None:
    raise SystemExit("no CSV with the 'none' and 'none (raw-flat RF)' cells found; looked in "
                     + ", ".join(_cand))
print(f"[signal] {_csv}")
d = pd.read_csv(_csv)
g = d[d.method == "none"].groupby("dataset").f1.mean()
r = d[d.method == "none (raw-flat RF)"].groupby("dataset").f1.mean()
signal = (g - r)

rows = []
for n in DS:
    X, y, ntr = load_binary(n, return_ntrain=True)
    H, _ = hurst_structure_function(X)
    lab = DATASETS.get(n, {}).get("minority")
    lab = str(lab[0]) if lab else "largest class"
    n_min_tr = int((y[:ntr] == 1).sum()); n_maj_tr = int((y[:ntr] == 0).sum())
    src = "UCR" if n in UNIV else "UEA"
    s = signal.get(n, np.nan)
    rows.append(f"{n} & {src} & {X.shape[1]} & {X.shape[2]} & {ntr} & {len(X)-ntr} & {lab} & {n_min_tr}/{n_maj_tr} & {H:.2f} & "
                + ("---" if not np.isfinite(s) else f"${s:+.2f}$") + r" \\")
    print(rows[-1])

open("results/table_datasets.tex", "w").write("\n".join(rows) + "\n")
print("\nwrote results/table_datasets.tex")
