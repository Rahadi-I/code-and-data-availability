"""
Go/no-go experiment: does a KNNOR walk inside GPF space, with both second-order corrections,
beat raw-space oversamplers and a PSSTO-style chain-mean rule?

Protocol
  * pooled data -> stratified 50/50 train/test split per seed (5 seeds)
  * training minority subsampled to n_min = max(4, n_maj / IR), IR in {5, 10, 20}
  * every method balances the training set (n_syn = n_maj - n_min); the classifier is always
    a random forest on standardised degree-2 GPF features, so the only difference is WHERE synthesis happens
  * metrics: F1 (minority), G-mean, balanced accuracy, ROC-AUC on the untouched test split;
    energy distance from synthetic to held-out real minority (feature space); dispersion ratio; acceptance rate
"""
import numpy as np, pandas as pd, time, json, sys
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist

from data import load_binary, DATASETS
from gpf import GPFExtractor, hurst_structure_function
from knnor import knnor_oversample, calibrate_M, chain_mean_oversample, smote_oversample, L_k, knnor_hybrid_oversample, calibrate_M_hybrid, knnor_official

IRS = [5, 10, 20]
SEEDS = [0, 1, 2, 3, 4]
K = 5
N_SELECT = int(__import__("os").environ.get("N_SELECT", "0"))   # sparse feature selection (C4); 0 = keep all
WALK_PCA = int(__import__("os").environ.get("WALK_PCA", "0"))   # 1: oversample (and classify) in a PCA space of dim <= n_min/2
OUT = __import__("os").environ.get("OUT", "results/results.csv")
SPLIT = __import__("os").environ.get("SPLIT", "pooled")   # "pooled": stratified 50/50 per seed; "standard": archive TRAIN/TEST split, seed only subsamples the minority
FAST = int(__import__("os").environ.get("FAST", "0"))     # 1: drop the slow / redundant comparators
OFFICIAL = int(__import__("os").environ.get("OFFICIAL", "0"))   # 1: add the authors' augmentdata implementation as baseline
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss


def mmd_rbf(A, B, gamma=None):
    Z = np.vstack([A, B]); D2 = cdist(Z, Z, "sqeuclidean")
    gamma = gamma or 1.0 / np.median(D2[D2 > 0])
    K = np.exp(-gamma * D2); n = len(A)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()


def ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def energy_distance(A, B):
    dab = cdist(A, B).mean()
    daa = cdist(A, A).mean()
    dbb = cdist(B, B).mean()
    return 2 * dab - daa - dbb


def gmean(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return np.sqrt(tp / max(tp + fn, 1) * tn / max(tn + fp, 1))


def evaluate(F_tr, y_tr, F_te, y_te, seed, clf_name="rf"):
    """clf_name: 'rf' = 300-tree random forest (the protocol classifier); 'ridge' = L2-penalised
    logistic regression on standardised features (a linear, high-bias alternative that still gives
    calibrated-ish probabilities, so Brier/ECE stay meaningful)."""
    if clf_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight=None)
    elif clf_name == "ridge":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    else:
        raise ValueError(clf_name)
    clf.fit(F_tr, y_tr)
    p = clf.predict(F_te)
    pr = clf.predict_proba(F_te)[:, 1]
    return dict(f1=f1_score(y_te, p, zero_division=0), gmean=gmean(y_te, p),
                bacc=balanced_accuracy_score(y_te, p), auc=roc_auc_score(y_te, pr),
                brier=brier_score_loss(y_te, pr), ece=ece(np.asarray(y_te), pr))


def run_dataset(name, log):
    X, y, ntr = load_binary(name, return_ntrain=True)
    H, _ = hurst_structure_function(X)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        if SPLIT == "standard":
            tr, te = np.arange(ntr), np.arange(ntr, len(X))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            tr, te = next(sss.split(X, y))
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        maj = np.where(ytr == 0)[0]
        mn_all = np.where(ytr == 1)[0]
        for IR in IRS:
            n_min = int(max(4, round(len(maj) / IR)))
            n_min = min(n_min, len(mn_all))
            mn = rng.choice(mn_all, n_min, replace=False)
            IR_actual = len(maj) / n_min
            n_syn = len(maj) - n_min
            Xmin_raw, Xmaj_raw = Xtr[mn], Xtr[maj]

            # feature space fitted on the imbalanced training set only
            ext = GPFExtractor(debias=True).fit(np.concatenate([Xmin_raw, Xmaj_raw]))
            F_all = ext.transform(np.concatenate([Xmin_raw, Xmaj_raw]))
            sc = StandardScaler().fit(F_all)
            if N_SELECT and F_all.shape[1] > N_SELECT:
                # sparse selection on the imbalanced training set only: ANOVA F-score, top N_SELECT columns
                from sklearn.feature_selection import f_classif
                y_all = np.r_[np.ones(len(Xmin_raw)), np.zeros(len(Xmaj_raw))]
                Fs = np.nan_to_num(f_classif(sc.transform(F_all), y_all)[0])
                cols = np.argsort(Fs)[::-1][:N_SELECT]
            else:
                cols = np.arange(F_all.shape[1])
            if WALK_PCA:
                q = int(min(len(cols), max(4, n_min // 2)))
                pca = PCA(n_components=q, random_state=seed).fit(sc.transform(F_all)[:, cols])
                feat = lambda Z: pca.transform(sc.transform(ext.transform(Z))[:, cols])
            else:
                feat = lambda Z: sc.transform(ext.transform(Z))[:, cols]
            Fmin, Fmaj, Fte = feat(Xmin_raw), feat(Xmaj_raw), feat(Xte)
            Fte_min = Fte[yte == 1]
            ref_ed = energy_distance(Fmin, Fte_min)          # real train minority vs held-out minority

            def record(method, F_syn, extra=None):
                F_tr = np.vstack([Fmin, Fmaj] + ([F_syn] if F_syn is not None else []))
                y_tr = np.r_[np.ones(len(Fmin)), np.zeros(len(Fmaj)), np.ones(0 if F_syn is None else len(F_syn))]
                m = evaluate(F_tr, y_tr, Fte, yte, seed)
                m.update(dataset=name, seed=seed, IR=IR, IR_actual=round(IR_actual, 1), n_min=n_min, n_syn=n_syn,
                         method=method, H=round(H, 3), ref_energy=ref_ed, n_feat=Fmin.shape[1], n_channels=X.shape[1], length=X.shape[2])
                if F_syn is not None:
                    m["energy_syn"] = energy_distance(F_syn, Fte_min)
                    m["mmd_syn"] = mmd_rbf(F_syn, Fte_min)
                    m["disp_ratio"] = np.trace(np.cov(F_syn.T)) / max(np.trace(np.cov(Fmin.T)), 1e-9)
                if extra: m.update(extra)
                rows.append(m)

            # reference: random forest on the raw flattened series (is the GPF space competitive at all?)
            raw_tr = np.vstack([Xmin_raw.reshape(len(Xmin_raw), -1), Xmaj_raw.reshape(len(Xmaj_raw), -1)])
            raw_y = np.r_[np.ones(len(Xmin_raw)), np.zeros(len(Xmaj_raw))]
            m = evaluate(raw_tr, raw_y, Xte.reshape(len(Xte), -1), yte, seed)
            m.update(dataset=name, seed=seed, IR=IR, IR_actual=round(IR_actual, 1), n_min=n_min, n_syn=n_syn,
                     method="none (raw-flat RF)", H=round(H, 3), ref_energy=ref_ed); rows.append(m)
            # 0. no oversampling
            record("none", None)
            # 1. SMOTE in raw sample space (flattened D*L), features computed afterwards
            flat = lambda Z: Z.reshape(len(Z), -1)
            unflat = lambda Z: Z.reshape(len(Z), X.shape[1], X.shape[2])
            S = smote_oversample(flat(Xmin_raw), n_syn, k=K, rng=seed)
            record("SMOTE-raw", feat(unflat(S)))
            # 2. KNNOR in raw space, published bound M=1
            if not FAST:
                S, st = knnor_oversample(flat(Xmin_raw), flat(Xmaj_raw), n_syn, k=K, M=1.0, rng=seed, return_stats=True)
                record("KNNOR-raw (M=1)", feat(unflat(S)), {"accept": st["accept"]})
            # 3. PSSTO-style chain mean in GPF space
            S = chain_mean_oversample(Fmin, n_syn, k=K, rng=seed)
            record("chain-mean-GPF (PSSTO-style)", S)
            # 4. KNNOR in GPF space, published bound M=1
            S, st = knnor_oversample(Fmin, Fmaj, n_syn, k=K, M=1.0, rng=seed, return_stats=True)
            record("KNNOR-GPF (M=1)", S, {"accept": st["accept"], "M": 1.0})
            # 5. KNNOR in GPF space, theoretical M=3/2
            if not FAST:
                S, st = knnor_oversample(Fmin, Fmaj, n_syn, k=K, M=1.5, rng=seed, return_stats=True)
                record("KNNOR-GPF (M=3/2)", S, {"accept": st["accept"], "M": 1.5})
            # 6. ours: M calibrated per dataset so that the walk retains the minority covariance
            M_star, ratios = calibrate_M(Fmin, Fmaj, k=K, rng=seed)
            S, st = knnor_oversample(Fmin, Fmaj, n_syn, k=K, M=M_star, rng=seed, return_stats=True)
            record("KNNOR-GPF (M calibrated)", S, {"accept": st["accept"], "M": M_star})
            # 7. hybrid: neighbours + filters in GPF space, walk in raw space (admissible sequences), M calibrated
            S, st = knnor_hybrid_oversample(flat(Xmin_raw), flat(Xmaj_raw), Fmin, Fmaj, n_syn, feat=lambda Z: feat(unflat(Z)),
                                            k=K, M=M_star, rng=seed, return_stats=True)
            record("hybrid GPF-nbrs/raw-walk (M calibrated)", feat(unflat(S)), {"accept": st["accept"], "M": M_star})
            if not FAST:
                S, st = knnor_hybrid_oversample(flat(Xmin_raw), flat(Xmaj_raw), Fmin, Fmaj, n_syn, feat=lambda Z: feat(unflat(Z)),
                                                k=K, M=1.0, rng=seed, return_stats=True)
                record("hybrid GPF-nbrs/raw-walk (M=1)", feat(unflat(S)), {"accept": st["accept"], "M": 1.0})
            # 9. the authors' published implementation (augmentdata), in GPF space: as published, and with the calibrated bound
            if OFFICIAL:
                S, st = knnor_official(Fmin, Fmaj, n_syn, k=K, randmx=1.0, rng=seed, return_stats=True)
                record("KNNOR-official-GPF (randmx=1)", S, {"accept": st["accept"], "M": 1.0})
                S, st = knnor_official(Fmin, Fmaj, n_syn, k=K, randmx=M_star, rng=seed, return_stats=True)
                record("KNNOR-official-GPF (randmx=M*)", S, {"accept": st["accept"], "M": M_star})
                if X.shape[1] * X.shape[2] <= 3000:      # the authors' pure-python purity test is too slow on long raw vectors
                    S, st = knnor_official(flat(Xmin_raw), flat(Xmaj_raw), n_syn, k=K, randmx=1.0, rng=seed, return_stats=True)
                    record("KNNOR-official-raw (randmx=1)", feat(unflat(S)), {"accept": st["accept"], "M": 1.0})
            # 8. hybrid with its OWN calibration: raw-space walk, M chosen so the feature-space covariance is preserved
            M_h, _ = calibrate_M_hybrid(flat(Xmin_raw), flat(Xmaj_raw), Fmin, Fmaj, feat=lambda Z: feat(unflat(Z)), k=K, rng=seed)
            S, st = knnor_hybrid_oversample(flat(Xmin_raw), flat(Xmaj_raw), Fmin, Fmaj, n_syn, feat=lambda Z: feat(unflat(Z)),
                                            k=K, M=M_h, rng=seed, return_stats=True)
            record("hybrid GPF-nbrs/raw-walk (M raw-calibrated)", feat(unflat(S)), {"accept": st["accept"], "M": M_h})
            f1s = {r["method"]: r["f1"] for r in rows if r["dataset"] == name and r["seed"] == seed and r["IR"] == IR}
            log(f"{name} seed={seed} IR={IR} (n_min={n_min}) H={H:.2f} M*={M_star:.2f} M_h={M_h:.2f} " + " ".join(f"{k.split(' (')[0][:12]}={v:.2f}" for k, v in f1s.items()))
    return rows


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    all_rows = []
    t0 = time.time()
    for n in names:
        all_rows += run_dataset(n, print)
        pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"done in {time.time()-t0:.0f}s")
