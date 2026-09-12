"""
Factorial pieces: (i) DTW as a guidance space (neighbours / filter / purity by DTW distance on standardised raw
series); (ii) the DTW-aligned convex walk — each neighbour is warped onto the current path's time axis before the
convex step, so peaks are averaged with peaks rather than with troughs (Fig. A of the problem figure).
"""
import numpy as np
from aeon.distances import dtw_pairwise_distance, dtw_alignment_path, dtw_distance

WINDOW = 0.2


def dtw_matrix(A, B=None):
    """A, B: (N, D, L) -> pairwise DTW distances (N_A, N_B) with a Sakoe-Chiba window."""
    return dtw_pairwise_distance(A, B, window=WINDOW) if B is not None else dtw_pairwise_distance(A, window=WINDOW)


def warp_onto(x, n):
    """Warp neighbour n (D, L) onto the time axis of x (D, L): for each index t of x, the mean of n's samples aligned to t."""
    path, _ = dtw_alignment_path(x, n, window=WINDOW)
    L = x.shape[1]
    out = np.zeros_like(x); cnt = np.zeros(L)
    for i, j in path:
        out[:, i] += n[:, j]; cnt[i] += 1
    return out / np.maximum(cnt, 1)[None, :]


def _step(x, n, a, aligned):
    return x + a * ((warp_onto(x, n) if aligned else n) - x)


def guided_walk_oversample(R_min, R_maj, n_new, guide, feat_eval, k=5, M=1.5, filter_prop=0.8, aligned=False,
                           purity=True, rng=None, return_stats=False):
    """
    R_min, R_maj : raw series (n, D, L).  guide : dict with either
        {"F_min", "F_maj"}  feature matrices (guidance by Euclidean k-NN in that space), or
        {"dtw": True}       guidance by DTW distance on the raw series.
    feat_eval : callable raw (n, D, L) -> evaluation-space features (for the purity test when guidance is a feature space).
    Synthesis is always in raw space; `aligned` switches the DTW-aligned step on.
    """
    rng = np.random.default_rng(rng)
    n_min = len(R_min); k_eff = min(k, n_min - 1)
    if k_eff < 1:
        S = R_min[rng.integers(0, n_min, n_new)]
        return (S, {"accept": 1.0}) if return_stats else S
    if guide.get("dtw"):
        Dmm = dtw_matrix(R_min); np.fill_diagonal(Dmm, np.inf)
        idx = np.argsort(Dmm, axis=1)[:, :k_eff]; kth = np.sort(Dmm, axis=1)[:, k_eff - 1]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(guide["F_min"])
        dist, ind = nn.kneighbors(guide["F_min"]); idx = ind[:, 1:]; kth = dist[:, -1]
    keep = np.where(kth <= np.quantile(kth, filter_prop))[0]
    if len(keep) < 2: keep = np.arange(n_min)
    if purity:
        if guide.get("dtw"):
            R_all = np.concatenate([R_min, R_maj]); y_all = np.r_[np.ones(n_min), np.zeros(len(R_maj))]
        else:
            from sklearn.neighbors import NearestNeighbors
            F_all = np.vstack([guide["F_min"], guide["F_maj"]]); y_all = np.r_[np.ones(n_min), np.zeros(len(R_maj))]
            nn_all = NearestNeighbors(n_neighbors=k_eff).fit(F_all)
    out, out_org, tried, batch = [], [], 0, 32
    max_tries = 20 * n_new + 50
    while len(out) < n_new and tried < max_tries:
        cand, org = [], []
        for _ in range(batch):
            o = keep[rng.integers(0, len(keep))]
            x = R_min[o].copy()
            for j in idx[o]:
                x = _step(x, R_min[j], rng.uniform(0.0, M), aligned)
            cand.append(x); org.append(o)
        cand = np.asarray(cand); org = np.asarray(org); tried += batch
        if purity:
            if guide.get("dtw"):
                Dc = dtw_matrix(cand, R_all); nb = np.argsort(Dc, axis=1)[:, :k_eff]
                lab = y_all[nb].mean(1)
            else:
                lab = y_all[nn_all.kneighbors(feat_eval(cand))[1]].mean(1)
            keep_c = lab >= 0.5; cand = cand[keep_c]; org = org[keep_c]
        out.extend(list(cand)); out_org.extend(list(org))
    if out:
        out = np.asarray(out[:n_new]); out_org = np.asarray(out_org[:n_new])
    else:
        out_org = rng.integers(0, n_min, n_new); out = R_min[out_org]
    if len(out) < n_new:
        fill = rng.integers(0, len(out), n_new - len(out))
        out = np.vstack([out, out[fill]]); out_org = np.r_[out_org, out_org[fill]]
    stats = {"accept": len(out) / max(tried, 1), "origins": np.asarray(out_org)}
    return (out, stats) if return_stats else out


if __name__ == "__main__":
    # Reproduces the two numbers quoted in the paper (Sec. "Geometry as guidance").
    # Both cases use a shift of 15 samples, inside the 20-sample Sakoe-Chiba band of WINDOW=0.2 on L=100.
    t = np.arange(100)
    v = np.exp(-((t - 30) / 6.0) ** 2)[None]                      # origin
    for tag, n in (("identical shape, shifted only", np.exp(-((t - 45) / 6.0) ** 2)[None]),
                   ("also lower and wider", 0.55 * np.exp(-((t - 45) / 10.0) ** 2)[None])):
        plain = 0.5 * (v + n)
        al = _step(v, n, 0.5, True)
        n_peaks = int((np.diff(np.sign(np.diff(plain[0]))) < 0).sum())
        print(f"{tag:32s} plain: {n_peaks} peaks, max {plain.max():.2f}   "
              f"aligned: max {al.max():.2f} at t={al[0].argmax()}")
