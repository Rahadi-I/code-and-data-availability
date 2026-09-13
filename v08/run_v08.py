"""
v0.8 run: the same geometry-guided cells as v0.7, plus (i) the dispersion guard, (ii) a second
downstream classifier, (iii) a configurable feature-selection budget.

Every row carries `clf` (rf | ridge) and `n_select`, so one CSV answers three questions:
  * does the guard remove the over-dispersion failure (EthanolConcentration)?
  * does the ranking survive a linear classifier?
  * does the ranking survive a different number of selected features?

env: OUT, SEEDS, CELLS, CLFS, N_SELECT, GUARD (1 = also emit guarded variants of the aligned cells)
"""
import os, sys, time, numpy as np, pandas as pd

# Find the folder holding the shared modules. It is the parent of this one when v08/ sits inside
# the code folder, a sibling named code/ in the working tree, and a sibling named src/ in the
# published repository -- so look for the file rather than assuming a layout.
_here = os.path.dirname(os.path.abspath(__file__))
_up = os.path.dirname(_here)
for _c in (_up, os.path.join(_up, "code"), os.path.join(_up, "src"), _here):
    if os.path.isfile(os.path.join(_c, "run_experiment.py")):
        sys.path.insert(0, _c)
        break
else:
    sys.exit(f"run_experiment.py not found next to {_here}. Expected it in the code/ or src/ "
             f"folder beside v08/, or in v08/'s parent.")
os.environ.setdefault("SPLIT", "standard")
import run_experiment as R
from data import load_binary
from gpf import GPFExtractor, hurst_structure_function
from signature import SignatureExtractor
from knnor import calibrate_M, smote_oversample
from aligned import guided_walk_oversample
from guard import dispersion_guard, dispersion_trim, TAU
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

OUT = os.environ.get("OUT", "results/results_v08.csv")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
CELLS = os.environ.get("CELLS", "gpf/plain,gpf/aligned,sig/aligned").split(",")
CLFS = os.environ.get("CLFS", "rf,ridge").split(",")
N_SELECT = int(os.environ.get("N_SELECT", "64"))
GUARD = int(os.environ.get("GUARD", "1"))
GMODE = os.environ.get("GUARDMODE", "shrink")     # shrink | trim
IRS = [5, 10, 20]; K = 5


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
            ext = GPFExtractor().fit(Xtrain); F_all = ext.transform(Xtrain); sc, cols = select_cols(F_all, y_all, N_SELECT)
            feat = lambda Z: sc.transform(ext.transform(Z))[:, cols]
            Fmin, Fmaj, Fte = feat(Xmin), feat(Xmaj), feat(Xte); Fte_min = Fte[yte == 1]
            ref_trace = float(np.trace(np.cov(Fmin.T)))
            sx = SignatureExtractor().fit(Xtrain); G_all = sx.transform(Xtrain); ssc, scols = select_cols(G_all, y_all, N_SELECT)
            sfeat = lambda Z: ssc.transform(sx.transform(Z))[:, scols]
            Gmin, Gmaj = sfeat(Xmin), sfeat(Xmaj)
            M_star, _ = calibrate_M(Fmin, Fmaj, k=K, rng=seed)

            def record(method, F_syn, extra=None):
                F_tr = np.vstack([Fmin, Fmaj] + ([F_syn] if F_syn is not None else []))
                y_tr = np.r_[np.ones(len(Fmin)), np.zeros(len(Fmaj)), np.ones(0 if F_syn is None else len(F_syn))]
                base = dict(dataset=name, seed=seed, IR=IR, IR_actual=round(IR_actual, 1), n_min=n_min, n_syn=n_syn,
                            method=method, H=round(H, 3), n_channels=X.shape[1], length=X.shape[2], M=M_star,
                            n_select=N_SELECT, tau=TAU)
                if F_syn is not None:
                    base["energy_syn"] = R.energy_distance(F_syn, Fte_min); base["mmd_syn"] = R.mmd_rbf(F_syn, Fte_min)
                    base["disp_ratio"] = np.trace(np.cov(F_syn.T)) / max(ref_trace, 1e-9)
                if extra: base.update(extra)
                for cname in CLFS:
                    m = R.evaluate(F_tr, y_tr, Fte, yte, seed, clf_name=cname)
                    m.update(base); m["clf"] = cname; rows.append(m)

            record("none", None)
            S = smote_oversample(Xmin.reshape(n_min, -1), n_syn, k=K, rng=seed).reshape(-1, X.shape[1], X.shape[2])
            record("SMOTE-raw", feat(S))
            guides = {"gpf": {"F_min": Fmin, "F_maj": Fmaj}, "sig": {"F_min": Gmin, "F_maj": Gmaj}}
            for gname, guide in guides.items():
                fe = sfeat if gname == "sig" else feat
                for aligned in (False, True):
                    cell = f"{gname}/{'aligned' if aligned else 'plain'}"
                    if cell not in CELLS: continue
                    t0 = time.time()
                    S, st = guided_walk_oversample(Xmin, Xmaj, n_syn, guide, fe, k=K, M=M_star, aligned=aligned, rng=seed, return_stats=True)
                    tag = f"guide={gname} | walk={'aligned' if aligned else 'plain'}"
                    record(tag, feat(S), {"accept": st["accept"], "secs": round(time.time() - t0, 1), "guard": 0})
                    if GUARD:
                        for mode in GMODE.split(","):
                            if mode == "shrink":
                                Sg, gi = dispersion_guard(S, st["origins"], Xmin, feat, ref_trace); amt = gi["c"]
                            else:
                                Sg, gi = dispersion_trim(S, st["origins"], Xmin, feat, ref_trace, rng=seed); amt = gi["trim"]
                            record(f"{tag} | guard-{mode}", feat(Sg),
                                   {"accept": st["accept"], "secs": round(time.time() - t0, 1), "guard": 1, "guard_mode": mode,
                                    "guard_fired": int(gi["fired"]), "guard_amt": round(float(amt), 3),
                                    "rho_before": round(gi["rho_before"], 2), "rho_after": round(gi["rho_after"], 2)})
            f1s = {r["method"]: r["f1"] for r in rows if r["dataset"] == name and r["seed"] == seed and r["IR"] == IR and r["clf"] == CLFS[0]}
            log(f"{name} s={seed} IR={IR} n_min={n_min} M*={M_star:.2f} " +
                " ".join(f"{k.replace('guide=', '').replace(' | walk=', '/').replace(' | guard', '+G')}={v:.2f}" for k, v in f1s.items()))
    return rows


if __name__ == "__main__":
    names = sys.argv[1:]; all_rows = []; t0 = time.time()
    for n in names:
        all_rows += run_dataset(n, print); pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"done in {time.time() - t0:.0f}s")
