"""
Factorial v0.7: guidance space {GPF-2, signature-3, DTW} x synthesis rule {plain raw walk, DTW-aligned raw walk},
anchors: none, SMOTE-raw. Standard archive splits; evaluation always RF on the same 64 selected GPF features.
"""
import os, sys, time, numpy as np, pandas as pd
os.environ.setdefault("SPLIT", "standard"); os.environ.setdefault("N_SELECT", "64")
import run_experiment as R
from data import load_binary
from gpf import GPFExtractor, hurst_structure_function
from signature import SignatureExtractor
from knnor import calibrate_M, smote_oversample
from aligned import guided_walk_oversample
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

OUT = os.environ.get("OUT", "results/results_v07_factorial.csv")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
IRS = [5, 10, 20]; K = 5; N_SELECT = 64


def select_cols(F_all, y_all, n):
    sc = StandardScaler().fit(F_all)
    Fs = np.nan_to_num(f_classif(sc.transform(F_all), y_all)[0])
    cols = np.argsort(Fs)[::-1][:n] if F_all.shape[1] > n else np.arange(F_all.shape[1])
    return sc, cols


def run_dataset(name, log):
    X, y, ntr = load_binary(name, return_ntrain=True)
    H, _ = hurst_structure_function(X); rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr, te = np.arange(ntr), np.arange(ntr, len(X))
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        maj, mn_all = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
        for IR in IRS:
            n_min = min(int(max(4, round(len(maj) / IR))), len(mn_all))
            mn = rng.choice(mn_all, n_min, replace=False)
            IR_actual = len(maj) / n_min; n_syn = len(maj) - n_min
            Xmin, Xmaj = Xtr[mn], Xtr[maj]; Xtrain = np.concatenate([Xmin, Xmaj])
            y_all = np.r_[np.ones(n_min), np.zeros(len(maj))]
            # evaluation space (GPF-64), fitted on the imbalanced training set
            ext = GPFExtractor().fit(Xtrain); F_all = ext.transform(Xtrain); sc, cols = select_cols(F_all, y_all, N_SELECT)
            feat = lambda Z: sc.transform(ext.transform(Z))[:, cols]
            Fmin, Fmaj, Fte = feat(Xmin), feat(Xmaj), feat(Xte); Fte_min = Fte[yte == 1]
            # signature guidance space
            sx = SignatureExtractor().fit(Xtrain); G_all = sx.transform(Xtrain); ssc, scols = select_cols(G_all, y_all, N_SELECT)
            sfeat = lambda Z: ssc.transform(sx.transform(Z))[:, scols]
            Gmin, Gmaj = sfeat(Xmin), sfeat(Xmaj)
            M_star, _ = calibrate_M(Fmin, Fmaj, k=K, rng=seed)

            def record(method, F_syn, extra=None):
                F_tr = np.vstack([Fmin, Fmaj] + ([F_syn] if F_syn is not None else []))
                y_tr = np.r_[np.ones(len(Fmin)), np.zeros(len(Fmaj)), np.ones(0 if F_syn is None else len(F_syn))]
                m = R.evaluate(F_tr, y_tr, Fte, yte, seed)
                m.update(dataset=name, seed=seed, IR=IR, IR_actual=round(IR_actual, 1), n_min=n_min, n_syn=n_syn, method=method,
                         H=round(H, 3), n_channels=X.shape[1], length=X.shape[2], M=M_star)
                if F_syn is not None:
                    m["energy_syn"] = R.energy_distance(F_syn, Fte_min); m["mmd_syn"] = R.mmd_rbf(F_syn, Fte_min)
                    m["disp_ratio"] = np.trace(np.cov(F_syn.T)) / max(np.trace(np.cov(Fmin.T)), 1e-9)
                if extra: m.update(extra)
                rows.append(m)

            record("none", None)
            S = smote_oversample(Xmin.reshape(n_min, -1), n_syn, k=K, rng=seed).reshape(-1, X.shape[1], X.shape[2])
            record("SMOTE-raw", feat(S))
            guides = {"gpf": {"F_min": Fmin, "F_maj": Fmaj}, "sig": {"F_min": Gmin, "F_maj": Gmaj}, "dtw": {"dtw": True}}
            CELLS = os.environ.get("CELLS")            # e.g. "gpf/plain,gpf/aligned,sig/aligned,dtw/aligned"; default all six
            for gname, guide in guides.items():
                fe = sfeat if gname == "sig" else feat
                if gname == "dtw" and (X.shape[2] > 500 or ntr > 800): continue     # DTW-guided purity is too slow on long or large sets
                for aligned in (False, True):
                    if CELLS and f"{gname}/{'aligned' if aligned else 'plain'}" not in CELLS.split(","): continue
                    t0 = time.time()
                    S, st = guided_walk_oversample(Xmin, Xmaj, n_syn, guide, fe, k=K, M=M_star, aligned=aligned, rng=seed, return_stats=True)
                    record(f"guide={gname} | walk={'aligned' if aligned else 'plain'}", feat(S), {"accept": st["accept"], "secs": round(time.time() - t0, 1)})
            f1s = {r["method"]: r["f1"] for r in rows if r["dataset"] == name and r["seed"] == seed and r["IR"] == IR}
            log(f"{name} s={seed} IR={IR} n_min={n_min} M*={M_star:.2f} " + " ".join(f"{k.replace('guide=','').replace(' | walk=','/')}={v:.2f}" for k, v in f1s.items()))
    return rows


if __name__ == "__main__":
    names = sys.argv[1:]; all_rows = []; t0 = time.time()
    for n in names:
        all_rows += run_dataset(n, print); pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"done in {time.time()-t0:.0f}s")
