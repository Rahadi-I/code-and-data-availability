"""Appendix A: the dataset table. Emits LaTeX rows to results/table_datasets.tex."""
import numpy as np, pandas as pd
from data import load_binary, DATASETS
from gpf import hurst_structure_function

DS = ["ArrowHead", "ArticularyWordRecognition", "BasicMotions", "CharacterTrajectories", "Cricket", "ERing",
      "Epilepsy", "EthanolConcentration", "GunPoint", "HandMovementDirection", "Handwriting", "ItalyPowerDemand",
      "JapaneseVowels", "Libras", "NATOPS", "OSULeaf", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
      "UWaveGestureLibrary"]
UNIV = {"ArrowHead", "GunPoint", "ItalyPowerDemand", "OSULeaf"}

# class signal of the guiding space, from the v0.5 run: F1(GPF features, no oversampling) - F1(raw flattened, no oversampling)
d = pd.read_csv("results/results_v05_standard.csv")
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
