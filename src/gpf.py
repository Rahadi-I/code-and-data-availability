"""
Degree-2 Geometric Path Features (GPF) with the two second-order corrections.

X : array (n_series, n_channels, length)

Features per series (time-augmented path, so every channel also pairs with t):
  - signed Levy area  A_ij = 1/2 * sum_t ( x_i(t) dx_j(t) - x_j(t) dx_i(t) ),   x centred at start
  - unsigned area     U_ij = 1/2 * sum_t | x_i(t) dx_j(t) - x_j(t) dx_i(t) |
  - debiased unsigned U_ij^hat = 2 U_ij(m=2) - U_ij(m=1)     (two-scale estimator; removes c*n*sigma^2 floor)
  - first-order terms: increment (end - start) per channel, path length per channel
Every channel is standardised (train statistics) before areas are computed.
"""
import numpy as np
from itertools import combinations


def _pairs(d):
    return list(combinations(range(d), 2))


def _areas(P):
    """P: (N, L, D) time-augmented, start-centred paths. Returns signed A, unsigned U : (N, n_pairs)."""
    dP = np.diff(P, axis=1)                  # (N, L-1, D)
    Pm = P[:, :-1, :]                        # left endpoint
    prs = _pairs(P.shape[2])
    A = np.empty((P.shape[0], len(prs)))
    U = np.empty_like(A)
    for c, (i, j) in enumerate(prs):
        cross = Pm[:, :, i] * dP[:, :, j] - Pm[:, :, j] * dP[:, :, i]
        A[:, c] = 0.5 * cross.sum(1)
        U[:, c] = 0.5 * np.abs(cross).sum(1)
    return A, U


def _augment(X):
    """(N, D, L) -> (N, L, D+1) with a time channel in [0,1], centred at the start."""
    N, D, L = X.shape
    t = np.linspace(0.0, 1.0, L)[None, :, None]
    P = np.concatenate([np.repeat(t, N, axis=0), np.transpose(X, (0, 2, 1))], axis=2)
    return P - P[:, :1, :]


class GPFExtractor:
    """Multi-scale: areas are computed on the whole path and on dyadic sub-windows (levels 0..levels),
    each window re-centred at its own start, so the space carries local as well as global geometry."""
    def __init__(self, debias=True, levels=3):
        self.debias = debias
        self.levels = levels
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-9
        return self

    def _window_features(self, P):
        P = P - P[:, :1, :]
        A, U1 = _areas(P)
        if self.debias and P.shape[1] >= 5:
            _, U2 = _areas(P[:, ::2, :])     # path subsampled every 2 steps
            U = 2.0 * U2 - U1                # two-scale debiased unsigned area
        else:
            U = U1
        inc = P[:, -1, 1:]                   # end - start per channel
        plen = np.abs(np.diff(P[:, :, 1:], axis=1)).sum(1)
        return np.concatenate([A, U, inc, plen], axis=1)

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        P = _augment(Z)
        L = P.shape[1]
        blocks = []
        for lev in range(self.levels + 1):
            nw = 2 ** lev
            edges = np.linspace(0, L, nw + 1).astype(int)
            for w in range(nw):
                a, b = edges[w], edges[w + 1]
                if b - a < 3:
                    continue
                blocks.append(self._window_features(P[:, a:b, :]))
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def n_features(self, d):
        npairs = len(_pairs(d + 1))
        return (2 * npairs + 2 * d) * (2 ** (self.levels + 1) - 1)


def hurst_structure_function(X, q=2, max_lag_frac=0.25):
    """
    Roughness exponent H via the structure function S_q(w) = E|X_{t+w}-X_t|^q ~ w^{qH}.
    X : (N, D, L). Returns mean H over series and channels, and the per-channel array.
    H -> 1 : smooth curve (areas are geometric); H ~ 0.5 : diffusive; H -> 0 : white noise around a mean.
    """
    N, D, L = X.shape
    lags = np.unique(np.round(np.logspace(0, np.log10(max(2, int(L * max_lag_frac))), 8)).astype(int))
    lags = lags[lags >= 1]
    Hs = np.zeros(D)
    for d in range(D):
        S = []
        for w in lags:
            diff = X[:, d, w:] - X[:, d, :-w]
            S.append(np.mean(np.abs(diff) ** q))
        S = np.asarray(S) + 1e-12
        slope = np.polyfit(np.log(lags), np.log(S), 1)[0]
        Hs[d] = slope / q
    return float(np.clip(Hs.mean(), 0, 1.2)), Hs
