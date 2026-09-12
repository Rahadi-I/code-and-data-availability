"""
Self-check of the analytical claims in the paper. Runs in a few seconds, needs no dataset.

    python verify_claims.py

Every check prints the claim as stated in the manuscript, the value this machine computes, and
PASS/FAIL. A FAIL means the manuscript and this code disagree -- report it, do not paper over it.
"""
import numpy as np
from knnor import L_k, L_k_shared_alpha, _walk
from signature import signature_level2
from gpf import _areas, _augment
from aligned import _step

ok = True


def check(name, claim, got, tol, note=""):
    global ok
    good = abs(got - claim) <= tol
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {name:52s} paper {claim:>8.3f}   computed {got:>8.3f}"
          + (f"   ({note})" if note else ""))


print("=" * 108)
print("1. Proposition 1 - variance retained by a k-step walk with independent step sizes")
print("=" * 108)
check("L_5(M=1), the default bound", 0.502, L_k(1.0, 5), 0.002)
check("L_5(M=3/2), unit retention", 1.000, L_k(1.5, 5), 1e-9, "exact for every k")
check("L_1(M=1) = SMOTE, retains 2/3", 0.667, L_k(1.0, 1), 0.002)
mgrid = np.linspace(0.05, 1.45, 400)
vals = np.array([L_k(m, 5) for m in mgrid])
check("minimum of L_5 over M (non-monotone)", 0.254, vals.min(), 0.004,
      f"attained at M = {mgrid[vals.argmin()]:.2f}")

print()
print("=" * 108)
print("2. Proposition 1, Monte-Carlo - the closed form must match a simulated walk on i.i.d. neighbours")
print("=" * 108)
rng = np.random.default_rng(0)
d, n_pool, n_syn = 12, 4000, 40000
pool = rng.normal(size=(n_pool, d))
tr_real = np.trace(np.cov(pool.T))
for M in (0.8, 1.0, 1.5, 2.0):
    idx = rng.integers(0, n_pool, (n_syn, 6))           # origin + 5 independent neighbours
    S = np.array([_walk(pool[r[0]], pool[r[1:]], M, rng) for r in idx])
    check(f"retained variance at M = {M}", L_k(M, 5), np.trace(np.cov(S.T)) / tr_real, 0.02,
          "simulated, i.i.d. neighbours")

print()
print("=" * 108)
print("3. Proposition 2 - one step size shared by all k steps (the reference implementation's rule)")
print("=" * 108)
check("L_5^sh(M=1)", 0.473, L_k_shared_alpha(1.0, 5), 0.002)
check("L_5^sh(M=3/2)", 0.906, L_k_shared_alpha(1.5, 5), 0.003)
from scipy.optimize import brentq
root = brentq(lambda M: L_k_shared_alpha(M, 5) - 1.0, 1.2, 1.95, xtol=1e-6)
check("M where L_5^sh = 1 (k-dependent)", 1.562, root, 0.003)
print(f"       sharing alpha costs {L_k(1.0,5)-L_k_shared_alpha(1.0,5):.3f} of retained variance at M = 1")

print()
print("=" * 108)
print("4. The DTW-aligned convex step (Sec. 'Geometry as guidance')")
print("=" * 108)
t = np.arange(100)
v = np.exp(-((t - 30) / 6.0) ** 2)[None]
for tag, nb, cp, ca in (("neighbour shifted only", np.exp(-((t - 45) / 6.0) ** 2)[None], 0.50, 1.00),
                        ("neighbour lower and wider", 0.55 * np.exp(-((t - 45) / 10.0) ** 2)[None], 0.53, 0.78)):
    plain = 0.5 * (v + nb); al = _step(v, nb, 0.5, True)
    check(f"{tag}: plain midpoint peak", cp, float(plain.max()), 0.01)
    check(f"{tag}: aligned step peak", ca, float(al.max()), 0.01,
          f"single peak at t = {al[0].argmax()}")

print()
print("=" * 108)
print("5. The level-2 signature's antisymmetric part is the signed Levy area")
print("=" * 108)
rng = np.random.default_rng(3)
P = np.cumsum(rng.normal(size=(1, 60, 2)), axis=1)
P = P - P[:, :1, :]
A, _ = _areas(P)
_, S2all = signature_level2(P); S2 = S2all[0]
lev = 0.5 * (S2[0, 1] - S2[1, 0])
check("0.5 (S_12 - S_21) vs the cross-product area", float(A[0, 0]), float(lev), 1e-6)

print()
print("=" * 108)
print("6. Two-scale debiasing of the unsigned area - works on smooth paths, not on rough ones")
print("=" * 108)


def debias_bias(kind, n=400, sigma=0.05, reps=40, seed=7):
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    base = (np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1)[None] if kind == "smooth"
            else np.cumsum(r.normal(size=(1, n, 2)) * 0.05, axis=1))
    base = base - base[:, :1, :]
    _, Uc = _areas(base)
    e_raw, e_deb = [], []
    for _ in range(reps):
        noisy = base + r.normal(size=base.shape) * sigma
        _, U1 = _areas(noisy); _, U2 = _areas(noisy[:, ::2, :])
        e_raw.append(float(U1[0, 0] - Uc[0, 0]))
        e_deb.append(float(2 * U2[0, 0] - U1[0, 0] - Uc[0, 0]))
    return float(Uc[0, 0]), float(np.mean(e_raw)), float(np.mean(e_deb))


U, b_raw, b_deb = debias_bias("smooth")
print(f"       smooth loop  : true area {U:.3f} | bias raw {b_raw:+.3f} | bias debiased {b_deb:+.3f}")
good = abs(b_deb) < abs(b_raw) / 2
ok &= good
print(f"[{'PASS' if good else 'FAIL'}] on a smooth path the correction cuts the bias by more than half")

U, b_raw, b_deb = debias_bias("rough")
print(f"       diffusive path: true area {U:.3f} | bias raw {b_raw:+.3f} | bias debiased {b_deb:+.3f}")
print("       (reported, not asserted: at H = 1/2 the unsigned area has no finite limit and the")
print("        correction does not help. The manuscript says so in Appendix B.4.)")

print()
print("=" * 108)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - the code and the manuscript disagree")
print("=" * 108)
raise SystemExit(0 if ok else 1)
