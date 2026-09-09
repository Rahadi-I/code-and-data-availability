"""
Dataset loaders for the go/no-go experiments.

Folder layout under code/data/ (all three archives live side by side):
    data/<Name>/<Name>_TRAIN.ts                                   UEA multivariate (from download_uea.py) and bundled copies
    data/UCRArchive_2018/<Name>/<Name>_TRAIN.tsv                  UCR univariate archive (tab-separated: label, values...)
    data/Monash_UEA_UCR_Regression_Archive/<Name>/<Name>_TRAIN.ts TSER regression archive (target is a float)

load_binary(name)      -> X (N, D, L) float, y in {0,1}; minority = the class listed in DATASETS, else the smallest class
load_regression(name)  -> X (N, D, L) float, y float
"""
import os
import numpy as np
from aeon.datasets import load_from_ts_file

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UCR_DIR = os.path.join(DATA_DIR, "UCRArchive_2018")
TSER_DIR = os.path.join(DATA_DIR, "Monash_UEA_UCR_Regression_Archive")
ZIP_DIR = os.path.join(DATA_DIR, "data-download-ulang-manual")     # <Name>.zip straight from timeseriesclassification.com


def _to_equal_length(X, L=None):
    """Handle list-of-arrays (unequal length) by linear resampling every series to L points."""
    if isinstance(X, np.ndarray):
        return X
    L = L or max(x.shape[1] for x in X)
    out = np.empty((len(X), X[0].shape[0], L))
    for i, x in enumerate(X):
        t_old = np.linspace(0, 1, x.shape[1])
        t_new = np.linspace(0, 1, L)
        for d in range(x.shape[0]):
            out[i, d] = np.interp(t_new, t_old, x[d])
    return out


def _read_tsv(path):
    """UCR .tsv: first column = class label, rest = values (NaN padding allowed for unequal length)."""
    M = np.genfromtxt(path, delimiter="\t")
    y = M[:, 0].astype(int).astype(str)
    X = M[:, 1:]
    if np.isnan(X).any():                       # variable-length series padded with NaN
        series = [x[~np.isnan(x)][None, :] for x in X]
        X = _to_equal_length(series)
    else:
        X = X[:, None, :]
    return X, y


def _read_split(name, split):
    """Find the split file for `name` in any of the three archives. Returns X (N, D, L), y (str)."""
    ts = os.path.join(DATA_DIR, name, f"{name}_{split}.ts")
    tsv = os.path.join(UCR_DIR, name, f"{name}_{split}.tsv")
    tser = os.path.join(TSER_DIR, name, f"{name}_{split}.ts")
    if os.path.exists(ts):
        X, y = load_from_ts_file(ts)
        return _to_equal_length(X), np.asarray(y).astype(str)
    if os.path.exists(tsv):
        return _read_tsv(tsv)
    if os.path.exists(tser):
        X, y = load_from_ts_file(tser)
        return _to_equal_length(X), np.asarray(y).astype(str)
    zp = os.path.join(ZIP_DIR, f"{name}.zip")
    if os.path.exists(zp):                       # read <Name>_TRAIN.ts straight out of the archive zip
        import zipfile, tempfile
        with zipfile.ZipFile(zp) as z:
            member = [m for m in z.namelist() if m.endswith(f"{name}_{split}.ts")]
            if member:
                with tempfile.TemporaryDirectory() as td:
                    z.extract(member[0], td)
                    X, y = load_from_ts_file(os.path.join(td, member[0]))
                return _to_equal_length(X), np.asarray(y).astype(str)
    raise FileNotFoundError(f"{name}_{split} not found under {DATA_DIR}, {UCR_DIR}, {TSER_DIR} or {ZIP_DIR}")


# name -> minority class label(s); anything not listed uses the smallest class as minority (one-vs-rest)
DATASETS = {
    "BasicMotions":     {"minority": ["walking"]},         # 6 ch, L=100, 4 classes x 25
    "JapaneseVowels":   {"minority": ["1"]},               # 12 ch, L<=29, 9 speakers
    "GunPoint":         {"minority": ["1"]},               # 1 ch, L=150
    "ItalyPowerDemand": {"minority": ["1"]},               # 1 ch, L=24
    "ArrowHead":        {"minority": ["0"]},               # 1 ch, L=251, 3 classes
    "OSULeaf":          {"minority": ["1"]},               # 1 ch, L=427, 6 classes
}


def load_raw(name, return_ntrain=False):
    Xtr, ytr = _read_split(name, "TRAIN")
    Xte, yte = _read_split(name, "TEST")
    L = max(Xtr.shape[2], Xte.shape[2])
    if Xtr.shape[2] != Xte.shape[2]:            # rare: different padded lengths between splits
        Xtr = _to_equal_length(list(Xtr), L); Xte = _to_equal_length(list(Xte), L)
    X = np.concatenate([Xtr, Xte], axis=0).astype(float)
    y = np.concatenate([ytr, yte])
    X = np.nan_to_num(X, nan=0.0)
    return (X, y, len(Xtr)) if return_ntrain else (X, y)


def load_binary(name, minority=None, return_ntrain=False):
    X, y, ntr = load_raw(name, return_ntrain=True)
    if minority is None:
        minority = DATASETS.get(name, {}).get("minority")
    if minority is None:
        # default: the LARGEST class becomes the minority and is subsampled to the target IR (CFAMG-style
        # constructed imbalance); this keeps n_min controllable on multi-class sets with tiny classes
        labels, counts = np.unique(y, return_counts=True)
        minority = [labels[np.argmax(counts)]]
    yb = np.isin(y, [str(m) for m in minority]).astype(int)
    return (X, yb, ntr) if return_ntrain else (X, yb)


def load_regression(name):
    X, y = load_raw(name)
    return X, y.astype(float)


def list_available():
    """Names present in any archive -> archive tag."""
    out = {}
    for root, tag in ((DATA_DIR, "UEA/bundled"), (UCR_DIR, "UCR"), (TSER_DIR, "TSER")):
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n)
            if os.path.isdir(p) and any(f.startswith(n + "_TRAIN") for f in os.listdir(p)):
                out[n] = tag
    return out


if __name__ == "__main__":
    avail = list_available()
    print(f"{len(avail)} datasets found")
    for n, src in avail.items():
        if src == "TSER" and os.path.getsize(os.path.join(TSER_DIR, n, f"{n}_TRAIN.ts")) > 50e6:
            print(f"{n:32s} {src:11s} (large, skipped in listing)"); continue
        try:
            X, y = load_raw(n)
            if src == "TSER":
                bal = f"target range {y.astype(float).min():.3g}..{y.astype(float).max():.3g}"
            else:
                labels, counts = np.unique(y, return_counts=True)
                bal = f"{len(labels)} classes, smallest={counts.min()}"
            print(f"{n:32s} {src:11s} X={X.shape}  {bal}")
        except Exception as e:
            print(f"{n:32s} {src:11s} ERROR {str(e)[:80]}")
