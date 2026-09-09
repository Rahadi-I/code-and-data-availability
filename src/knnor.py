"""
KNNOR-style k-step convex walk (Islam, Belhaouari, Rehman, Bensmail, ASOC 2022) with
  * calibrated step bound M   (variance retained L_k = A^k + M/(3-M) (1-A^k), A = 1 - M + M^2/3; L_k = 1 at M = 3/2)
  * per-origin local alpha bound (distance to nearest majority point)
  * purity filter (accept a synthetic point only if its k nearest training neighbours are mostly minority)
Also PSSTO-style chain-mean synthesis and a plain SMOTE, all operating on any feature matrix.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def L_k(M, k):
    A = 1.0 - M + M ** 2 / 3.0
    return A ** k + (M / (3.0 - M)) * (1.0 - A ** k)


def _walk(origin, nbrs, M, rng):
    x = origin.copy()
    for nb in nbrs:
        a = rng.uniform(0.0, M)
        x = x + a * (nb - x)
    return x


def knnor_oversample(X_min, X_maj, n_new, k=5, M=1.5, filter_prop=0.8, local_alpha=True,
                     purity=True, rng=None, return_stats=False):
    """
    Returns synthetic minority points in the same feature space as X_min.
    filter_prop : keep the fraction of minority points whose k-th-neighbour distance is smallest (KNNOR filtering step).
    """
    rng = np.random.default_rng(rng)
    n_min = len(X_min)
    k_eff = min(k, n_min - 1)
    if k_eff < 1:
        return (X_min[rng.integers(0, n_min, n_new)], {"accept": 1.0}) if return_stats else X_min[rng.integers(0, n_min, n_new)]

    nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    dist, idx = nn_min.kneighbors(X_min)
    kth = dist[:, -1]
    keep = np.where(kth <= np.quantile(kth, filter_prop))[0]
    if len(keep) < 2:
        keep = np.arange(n_min)

    if local_alpha or purity:
        nn_maj = NearestNeighbors(n_neighbors=1).fit(X_maj)
        rho = nn_maj.kneighbors(X_min)[0][:, 0]           # distance to nearest majority per minority point
    if purity:
        X_all = np.vstack([X_min, X_maj])
        y_all = np.r_[np.ones(n_min), np.zeros(len(X_maj))]
        nn_all = NearestNeighbors(n_neighbors=k_eff).fit(X_all)

    out, tried = [], 0
    max_tries = 20 * n_new + 50
    while len(out) < n_new and tried < max_tries:
        tried += 1
        o = keep[rng.integers(0, len(keep))]
        nb = X_min[idx[o, 1:]]                 # k nearest minority neighbours, ordered by distance
        x = _walk(X_min[o], nb, M, rng)
        if local_alpha:
            disp = np.linalg.norm(x - X_min[o])
            if disp > rho[o]:                  # do not step past the nearest majority point
                x = X_min[o] + (x - X_min[o]) * (rho[o] / disp)
        if purity:
            nb_lab = y_all[nn_all.kneighbors(x[None])[1][0]]
            if nb_lab.mean() < 0.5:
                continue
        out.append(x)
    out = np.asarray(out) if out else X_min[rng.integers(0, n_min, n_new)]
    if len(out) < n_new:                       # top up by resampling accepted points
        out = np.vstack([out, out[rng.integers(0, len(out), n_new - len(out))]])
    stats = {"accept": len(out) / max(tried, 1), "tried": tried}
    return (out, stats) if return_stats else out


def calibrate_M(X_min, X_maj, k=5, grid=np.linspace(0.6, 2.4, 10), n_probe=None, rng=0):
    """Pick M so that trace Cov(synthetic) / trace Cov(real minority) is closest to 1 (no filters, so the walk alone is measured)."""
    n_probe = n_probe or max(200, 4 * len(X_min))
    tr_real = np.trace(np.cov(X_min.T))
    best, best_gap, ratios = 1.5, np.inf, {}
    for M in grid:
        S = knnor_oversample(X_min, X_maj, n_probe, k=k, M=M, filter_prop=1.0, local_alpha=False, purity=False, rng=rng)
        r = np.trace(np.cov(S.T)) / tr_real
        ratios[float(M)] = float(r)
        if abs(r - 1) < best_gap:
            best, best_gap = float(M), abs(r - 1)
    return best, ratios


def chain_mean_oversample(X_min, n_new, k=5, rng=None):
    """PSSTO-style: synthetic point = mean of a same-class nearest-neighbour chain of length k, then SMOTE-style interpolation."""
    rng = np.random.default_rng(rng)
    n_min = len(X_min)
    k_eff = min(k, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    idx = nn.kneighbors(X_min)[1]
    out = []
    for _ in range(n_new):
        o = rng.integers(0, n_min)
        chain = X_min[idx[o]]                  # origin + k neighbours
        m = chain.mean(0)
        lam = rng.uniform()
        out.append(X_min[o] + lam * (m - X_min[o]))
    return np.asarray(out)


def smote_oversample(X_min, n_new, k=5, rng=None):
    rng = np.random.default_rng(rng)
    n_min = len(X_min)
    k_eff = min(k, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    idx = nn.kneighbors(X_min)[1]
    out = []
    for _ in range(n_new):
        o = rng.integers(0, n_min)
        nb = X_min[idx[o, 1 + rng.integers(0, k_eff)]]
        out.append(X_min[o] + rng.uniform() * (nb - X_min[o]))
    return np.asarray(out)


if __name__ == "__main__":
    # sanity: empirical variance retained vs closed-form L_k on an isotropic Gaussian cloud
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    for M in (1.0, 1.5):
        S = knnor_oversample(X, X + 50, 20000, k=5, M=M, filter_prop=1.0, local_alpha=False, purity=False, rng=1)
        print(f"M={M}: empirical Cov ratio {np.trace(np.cov(S.T))/np.trace(np.cov(X.T)):.3f}   closed-form L_5 = {L_k(M,5):.3f}  (k-NN correlation makes the empirical value milder)")


def knnor_hybrid_oversample(R_min, R_maj, F_min, F_maj, n_new, feat, k=5, M=1.5, filter_prop=0.8,
                            purity=True, rng=None, return_stats=False):
    """
    Geometry-guided walk (addresses C1, admissibility): neighbours, the filtering step and the purity check are
    computed in GPF feature space (F), but the convex walk itself is performed in RAW sequence space (R), so every
    synthetic point is an actual sequence.  `feat` maps raw rows -> feature rows (used for the purity check).
    Returns synthetic points in raw space.
    """
    rng = np.random.default_rng(rng)
    n_min = len(F_min)
    k_eff = min(k, n_min - 1)
    if k_eff < 1:
        S = R_min[rng.integers(0, n_min, n_new)]
        return (S, {"accept": 1.0}) if return_stats else S
    nn_min = NearestNeighbors(n_neighbors=k_eff + 1).fit(F_min)
    dist, idx = nn_min.kneighbors(F_min)
    kth = dist[:, -1]
    keep = np.where(kth <= np.quantile(kth, filter_prop))[0]
    if len(keep) < 2:
        keep = np.arange(n_min)
    if purity:
        F_all = np.vstack([F_min, F_maj]); y_all = np.r_[np.ones(n_min), np.zeros(len(F_maj))]
        nn_all = NearestNeighbors(n_neighbors=k_eff).fit(F_all)
    out, tried, batch = [], 0, 64
    max_tries = 20 * n_new + 50
    while len(out) < n_new and tried < max_tries:
        cand = []
        for _ in range(batch):
            o = keep[rng.integers(0, len(keep))]
            cand.append(_walk(R_min[o], R_min[idx[o, 1:]], M, rng))
        cand = np.asarray(cand); tried += batch
        if purity:
            Fc = feat(cand)
            lab = y_all[nn_all.kneighbors(Fc)[1]].mean(1)
            cand = cand[lab >= 0.5]
        out.extend(list(cand))
    out = np.asarray(out[:n_new]) if out else R_min[rng.integers(0, n_min, n_new)]
    if len(out) < n_new:
        out = np.vstack([out, out[rng.integers(0, len(out), n_new - len(out))]])
    stats = {"accept": len(out) / max(tried, 1)}
    return (out, stats) if return_stats else out


def calibrate_M_hybrid(R_min, R_maj, F_min, F_maj, feat, k=5, grid=np.linspace(0.4, 2.8, 13), n_probe=None, rng=0):
    """M for the raw-space walk chosen so that the FEATURE-space covariance of the synthetic set matches the real minority."""
    n_probe = n_probe or max(200, 4 * len(F_min))
    tr_real = np.trace(np.cov(F_min.T))
    best, best_gap, ratios = 1.0, np.inf, {}
    for M in grid:
        S = knnor_hybrid_oversample(R_min, R_maj, F_min, F_maj, n_probe, feat, k=k, M=M, filter_prop=1.0, purity=False, rng=rng)
        r = np.trace(np.cov(feat(S).T)) / tr_real
        ratios[float(M)] = float(r)
        if abs(r - 1) < best_gap:
            best, best_gap = float(M), abs(r - 1)
    return best, ratios


# ---------------------------------------------------------------------------------------------
# The published implementation (pip: augmentdata, Islam & Belhaouari) — used as the reference baseline.
# Differences from the description above that matter for the derivation:
#   * ONE alpha ~ U(0, randmx) is drawn per synthetic point and reused for all k steps (not one per step);
#   * randmx is multiplied by the acceptance fraction after every pass, so the effective bound decays;
#   * the filtering step keeps the dist_percent = 0.6 fraction of minority points closest to their k-th neighbour;
#   * the purity test requires ALL floor(sqrt(k)) nearest training points to be minority.
# ---------------------------------------------------------------------------------------------
def L_k_shared_alpha(M, k, n=200001):
    """Variance retained when one alpha ~ U(0,M) is shared by all k steps (the augmentdata rule), i.i.d. neighbours."""
    a = np.linspace(0, M, n)[1:]
    return float(np.mean((1 - a) ** (2 * k) + sum((a * (1 - a) ** (k - j)) ** 2 for j in range(1, k + 1))))


def knnor_official(X_min, X_maj, n_new, k=5, randmx=1.0, dist_percent=0.6, rng=None, return_stats=False, max_passes=50):
    """
    Faithful re-implementation of augmentdata.DataAugment.augment (Islam & Belhaouari), with two safety changes only:
      * the purity test (ALL floor(sqrt(k)) nearest training points must be minority) is vectorised;
      * the outer loop stops after `max_passes` passes or when randmx < 1e-4 (the library can loop forever when
        nothing is accepted, because randmx is multiplied by the acceptance fraction after every pass).
    Everything else — one shared alpha per synthetic point, decaying randmx, dist_percent filter — is as published.
    """
    rng = np.random.default_rng(rng)
    x, y = np.asarray(X_min, float), np.asarray(X_maj, float)
    n_min = len(x); k = max(1, min(k, n_min - 1))
    D = np.linalg.norm(x - x[:, None], axis=-1); D.sort(axis=1)
    kth = D[:, k - 1] if k - 1 < D.shape[1] else D[:, -1]
    thr = np.sort(kth)[int(np.floor(dist_percent * n_min))] if int(np.floor(dist_percent * n_min)) < n_min else kth.max()
    data = np.vstack([x, y]); labels = np.r_[np.ones(n_min), np.zeros(len(y))]
    nn_all = NearestNeighbors(n_neighbors=max(1, int(np.floor(np.sqrt(k))))).fit(data)
    nbr_idx = np.argsort(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), axis=1)[:, 1:k + 1]
    N = int(n_new); Q = N // n_min + 1; inc = n_min * Q
    out, rejected, passes = [], 0, 0
    while N > 0 and passes < max_passes:
        randmx = randmx * inc / (n_min * Q); inc = 0; passes += 1
        if randmx < 1e-4: break
        for i in range(n_min):
            if kth[i] > thr or N == 0: continue
            v = x[i]; kv = x[nbr_idx[i]]
            for _ in range(Q):
                if N == 0: break
                a = rng.uniform(0, randmx); m = v.copy()
                for j in range(k): m = m + a * (kv[j] - m)
                lab = labels[nn_all.kneighbors(m[None])[1][0]]
                if np.all(lab == 1):
                    out.append(m); N -= 1; inc += 1
                else:
                    rejected += 1
    out = np.asarray(out) if out else x[rng.integers(0, n_min, n_new)]
    if len(out) < n_new:
        out = np.vstack([out, out[rng.integers(0, len(out), n_new - len(out))]])
    stats = {"accept": len(out) / max(len(out) + rejected, 1), "passes": passes, "randmx_final": randmx}
    return (out, stats) if return_stats else out
