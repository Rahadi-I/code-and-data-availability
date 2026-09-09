"""
Dispersion guard (v0.8).

Motivation. The step bound M* is calibrated so that the BARE walk matches the minority covariance
(Prop. 1). The realised synthetic set, however, passes through an origin filter, a local alpha bound,
a purity vote and a DTW-aligned step, so its realised dispersion can drift far from the target. On one
benchmark (EthanolConcentration) the guided walk over-disperses by an order of magnitude and loses
0.11 minority F1 -- the single largest loss in the study.

The guard. Measure the realised dispersion ratio rho = tr Cov(phi(S)) / tr Cov(phi(X_min)) of the
ACCEPTED synthetic set. If rho <= tau the set is returned unchanged (the guard is inactive). Otherwise
each synthetic path is shrunk along its own walk, back towards the real minority sequence it started
from:  s_i <- o_i + c (s_i - o_i),  c in (0, 1].
Because the path stays on the segment between a real minority sequence and the point the walk reached,
shrinkage cannot leave the region the walk already certified as pure, so admissibility is preserved.
c is found by a two-step secant search on the map c -> rho(c) (GPF features are quadratic in the path,
so rho ~ c^2 is used as the initial guess), capped at 3 feature evaluations.
"""
import numpy as np

TAU = 2.0


def _disp(F_syn, ref_trace):
    return float(np.trace(np.cov(F_syn.T)) / max(ref_trace, 1e-9))


def dispersion_guard(S, origins, R_min, feat, ref_trace, tau=TAU, max_eval=3):
    """S (n,D,L) synthetic; origins (n,) index into R_min; feat: raw -> evaluation features.
    Returns (S_guarded, info). info: rho_before, rho_after, c, fired (bool), n_eval."""
    F = feat(S); rho0 = _disp(F, ref_trace)
    info = {"rho_before": rho0, "rho_after": rho0, "c": 1.0, "fired": False, "n_eval": 1}
    if not np.isfinite(rho0) or rho0 <= tau:
        return S, info
    O = R_min[origins]
    c = float(np.clip(np.sqrt(tau / rho0), 0.05, 1.0))          # rho ~ c^2 for area-type features
    best = None
    for _ in range(max_eval - 1):
        Sc = O + c * (S - O)
        rho = _disp(feat(Sc), ref_trace); info["n_eval"] += 1
        if rho <= tau:
            best = (c, rho, Sc)
            break
        c = float(np.clip(c * np.sqrt(tau / rho), 0.05, 1.0))    # secant step under the same power law
    if best is None:
        Sc = O + c * (S - O); rho = _disp(feat(Sc), ref_trace); info["n_eval"] += 1
        best = (c, rho, Sc)
    info.update(rho_after=best[1], c=best[0], fired=True)
    return best[2], info


def dispersion_trim(S, origins, R_min, feat, ref_trace, tau=TAU, max_trim=0.5, rng=0):
    """Alternative guard: instead of shrinking every path (which, in the limit, turns the synthetic set
    into duplicates of the origins -- see Appendix B.1), DISCARD the most deviant synthetic points and
    refill the quota by resampling the retained ones. Points are ordered by their squared distance to the
    real-minority centroid in the evaluation space; the largest retained prefix with rho <= tau is kept.
    Costs one feature evaluation: all prefix dispersions are obtained from cumulative sums."""
    rng = np.random.default_rng(rng)
    F = feat(S); n = len(F)
    rho0 = _disp(F, ref_trace)
    info = {"rho_before": rho0, "rho_after": rho0, "trim": 0.0, "fired": False, "n_eval": 1}
    if not np.isfinite(rho0) or rho0 <= tau or n < 8:
        return S, info
    order = np.argsort(((F - F.mean(0)) ** 2).sum(1))          # least deviant first
    Fo = F[order]
    c1 = np.cumsum(Fo, axis=0); c2 = np.cumsum(Fo ** 2, axis=0)
    m = np.arange(1, n + 1)[:, None]
    var = (c2 - c1 ** 2 / m) / np.maximum(m - 1, 1)            # unbiased trace of Cov over each prefix
    rho_pref = var.sum(1) / max(ref_trace, 1e-9)
    lo = max(8, int(np.ceil((1 - max_trim) * n)))
    ok = np.where(rho_pref[lo - 1:] <= tau)[0]
    keep_m = (lo + ok[-1]) if len(ok) else lo                  # largest admissible prefix, else the floor
    keep = order[:keep_m]
    fill = rng.integers(0, keep_m, n - keep_m)
    idx = np.r_[keep, keep[fill]]
    info.update(rho_after=float(_disp(F[keep], ref_trace)), trim=round(1 - keep_m / n, 3), fired=True)
    return S[idx], info
