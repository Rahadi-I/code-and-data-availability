"""
The acceptance-dispersion frontier in M (challenge C2).
For each M: variance retained by the bare walk (vs closed-form L_k), dispersion after local-alpha + purity filters,
purity-filter acceptance rate, energy distance to held-out minority, and downstream F1.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from data import load_binary
from gpf import GPFExtractor
from knnor import knnor_oversample, L_k
from run_experiment import evaluate, energy_distance

DATASETS = sys.argv[1:] or ["NATOPS", "Epilepsy", "RacketSports", "OSULeaf", "GunPoint", "ERing"]
MS = np.round(np.linspace(0.5, 2.6, 8), 2)
IR, K, SEEDS, N_SELECT = 10, 5, [0, 1, 2], 64
rows = []
for name in DATASETS:
    X, y = load_binary(name)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(X, y))
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        maj, mn_all = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
        n_min = min(int(max(4, round(len(maj) / IR))), len(mn_all))
        mn = rng.choice(mn_all, n_min, replace=False)
        Xmin, Xmaj = Xtr[mn], Xtr[maj]
        ext = GPFExtractor().fit(np.concatenate([Xmin, Xmaj]))
        F_all = ext.transform(np.concatenate([Xmin, Xmaj])); sc = StandardScaler().fit(F_all)
        y_all = np.r_[np.ones(n_min), np.zeros(len(maj))]
        cols = np.argsort(np.nan_to_num(f_classif(sc.transform(F_all), y_all)[0]))[::-1][:N_SELECT]
        feat = lambda Z: sc.transform(ext.transform(Z))[:, cols]
        Fmin, Fmaj, Fte = feat(Xmin), feat(Xmaj), feat(Xte)
        Fte_min = Fte[yte == 1]; n_syn = len(maj) - n_min
        tr_real = np.trace(np.cov(Fmin.T))
        for M in MS:
            S0 = knnor_oversample(Fmin, Fmaj, max(300, n_syn), k=K, M=M, filter_prop=1.0, local_alpha=False, purity=False, rng=seed)
            S, st = knnor_oversample(Fmin, Fmaj, n_syn, k=K, M=M, rng=seed, return_stats=True)
            m = evaluate(np.vstack([Fmin, Fmaj, S]), np.r_[np.ones(n_min), np.zeros(len(maj)), np.ones(len(S))], Fte, yte, seed)
            rows.append(dict(dataset=name, seed=seed, M=M, L_k=L_k(M, K),
                             disp_walk=np.trace(np.cov(S0.T)) / tr_real, disp_filtered=np.trace(np.cov(S.T)) / tr_real,
                             accept=st["accept"], energy=energy_distance(S, Fte_min), f1=m["f1"], gmean=m["gmean"]))
        print(name, seed, "done")
    pd.DataFrame(rows).to_csv("results/frontier_M.csv", index=False)
print("saved results/frontier_M.csv")
