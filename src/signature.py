"""
Truncated path signature (levels 1..3) of time-augmented, standardised paths — numpy only (esig/iisignature
could not be built in this environment). Chen's identity over piecewise-linear segments:
    S(X) = exp(dx_1) ⊗ exp(dx_2) ⊗ ... ⊗ exp(dx_L),  exp(v) = (1, v, v⊗v/2, v⊗v⊗v/6) truncated at level 3.
X : (N, D, L)  ->  features (N, d + d^2 + d^3) with d = D + 1 (time channel).
"""
import numpy as np


def _augment(X):
    N, D, L = X.shape
    t = np.linspace(0.0, 1.0, L)[None, :, None]
    return np.concatenate([np.repeat(t, N, axis=0), np.transpose(X, (0, 2, 1))], axis=2)   # (N, L, d)


def signature_level3(P):
    """P: (N, L, d) paths. Returns (S1, S2, S3) with shapes (N,d), (N,d,d), (N,d,d,d)."""
    N, L, d = P.shape
    dX = np.diff(P, axis=1)                                   # (N, L-1, d)
    S1 = np.zeros((N, d)); S2 = np.zeros((N, d, d)); S3 = np.zeros((N, d, d, d))
    for i in range(L - 1):
        v = dX[:, i, :]
        e2 = np.einsum("na,nb->nab", v, v) / 2.0
        e3 = np.einsum("na,nb,nc->nabc", v, v, v) / 6.0
        # (S ⊗ exp(v)) truncated at level 3
        n3 = S3 + e3 + np.einsum("na,nbc->nabc", S1, e2) + np.einsum("nab,nc->nabc", S2, v)
        n2 = S2 + e2 + np.einsum("na,nb->nab", S1, v)
        n1 = S1 + v
        S1, S2, S3 = n1, n2, n3
    return S1, S2, S3


def signature_level2(P):
    N, L, d = P.shape
    dX = np.diff(P, axis=1); S1 = np.zeros((N, d)); S2 = np.zeros((N, d, d))
    for i in range(L - 1):
        v = dX[:, i, :]
        S2 = S2 + np.einsum("na,nb->nab", v, v) / 2.0 + np.einsum("na,nb->nab", S1, v); S1 = S1 + v
    return S1, S2


class SignatureExtractor:
    """Level-3 signature of the time-augmented standardised path, on the whole series and dyadic halves (level 1)."""
    def __init__(self, levels=1):
        self.levels = levels; self.mu = None; self.sd = None

    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True); self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-9
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        P = _augment(Z); L = P.shape[1]; blocks = []
        for lev in range(self.levels + 1):
            nw = 2 ** lev; edges = np.linspace(0, L, nw + 1).astype(int)
            for w in range(nw):
                a, b = edges[w], edges[w + 1]
                if b - a < 3: continue
                seg = P[:, a:b, :] - P[:, a:a + 1, :]
                if seg.shape[2] > 10:                     # d^3 becomes prohibitive: level 2 only for wide series
                    S1, S2 = signature_level2(seg); blocks += [S1, S2.reshape(len(S2), -1)]
                else:
                    S1, S2, S3 = signature_level3(seg)
                    blocks += [S1, S2.reshape(len(S2), -1), S3.reshape(len(S3), -1)]
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # sanity: level-2 antisymmetric part equals the Lévy area used in gpf.py
    from gpf import _areas
    rng = np.random.default_rng(0); X = rng.normal(size=(4, 2, 60)).cumsum(-1)
    P = _augment(X); P = P - P[:, :1, :]
    S1, S2, S3 = signature_level3(P)
    A, _ = _areas(P)
    levy = 0.5 * (S2[:, 0, 1] - S2[:, 1, 0])
    print("Levy area (sig) ", np.round(levy, 4)); print("Levy area (gpf) ", np.round(A[:, 0], 4))
    print("feature dim, d=3:", SignatureExtractor().fit_transform(X).shape)
