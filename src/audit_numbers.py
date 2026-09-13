r"""
Recompute every number the manuscript quotes, from whichever CSVs you point it at, and print the
manuscript's value next to the computed one.

    python audit_numbers.py                 # the published CSVs (everything should say OK)
    python audit_numbers.py --mine          # your own reproduction CSVs

Paths can also be given one by one: --fac, --v06, --pilot, --frontier, --v08. Any CSV that is
missing is reported and its block skipped, never silently substituted.

A line is flagged MISMATCH when the computed value differs from the manuscript by more than the
tolerance shown. Ranks and F1 are compared to 0.01, p-values by order of magnitude, because a
p-value that moves from 1.7e-4 to 2.9e-4 is the same finding while 1e-4 to 0.3 is not.
"""
import os, sys, argparse, math
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
R = os.path.join("..", "results")

ap = argparse.ArgumentParser()
ap.add_argument("--mine", action="store_true", help="use the repro_* CSVs instead of the published ones")
ap.add_argument("--fac"); ap.add_argument("--v06"); ap.add_argument("--pilot")
ap.add_argument("--frontier"); ap.add_argument("--v08")
A = ap.parse_args()

if A.mine:
    P = dict(fac=f"{R}/repro_v07.csv", v06=f"{R}/repro_v06.csv", pilot=f"{R}/repro_v07_pilot.csv",
             frontier="results/frontier_M.csv", v08="../v08/results/repro_v08_main.csv")
else:
    P = dict(fac="results/results_v07_factorial_all.csv", v06="results/results_v06_official_all.csv",
             pilot="results/results_v07_factorial.csv", frontier="results/frontier_M.csv",
             v08="../v08/results/results_v08_main.csv")
for k, v in (("fac", A.fac), ("v06", A.v06), ("pilot", A.pilot), ("frontier", A.frontier), ("v08", A.v08)):
    if v:
        P[k] = v

nbad = 0


def chk(label, paper, got, tol=0.01, kind="num"):
    """kind 'num': absolute tolerance. kind 'p': same order of magnitude."""
    global nbad
    if got is None or (isinstance(got, float) and math.isnan(got)):
        print(f"  {label:52s} paper {paper:>10}   computed     --      NOT AVAILABLE"); nbad += 1; return
    if kind == "p":
        ok = (paper <= 0 and got <= 0) or abs(math.log10(max(got, 1e-300)) - math.log10(max(paper, 1e-300))) < 0.7
        ps, gs = f"{paper:.1e}", f"{got:.1e}"
    else:
        ok = abs(got - paper) <= tol
        ps, gs = f"{paper:.3f}", f"{got:.3f}"
    nbad += 0 if ok else 1
    print(f"  {label:52s} paper {ps:>10}   computed {gs:>10}   {'ok' if ok else 'MISMATCH'}")


def load(key):
    p = P[key]
    if not os.path.exists(p):
        print(f"  [{key}] {p} not found -- block skipped")
        return None
    print(f"  [{key}] {p}")
    return pd.read_csv(p)


def blocks(d, methods, value="f1"):
    return d[d.method.isin(methods)].pivot_table(index=["dataset", "IR"], columns="method",
                                                 values=value, aggfunc="mean")[methods].dropna()


def cd(k, N):
    q = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}[k]
    return q * math.sqrt(k * (k + 1) / (6.0 * N))


SM, GP, GA, SA, DA = ("SMOTE-raw", "guide=gpf | walk=plain", "guide=gpf | walk=aligned",
                      "guide=sig | walk=aligned", "guide=dtw | walk=aligned")

# ============================================================ Sec. 5.4 + Table 2 (factorial)
print("\n" + "=" * 104)
print("Confirmatory run on 20 datasets  --  Sec. 5.4, Table 2, Figs. 4 and 5")
print("=" * 104)
d = load("fac")
blockmed = None
if d is not None:
    p = blocks(d, [SM, GP, GA, SA]); r = p.rank(axis=1, ascending=False).mean()
    print(f"  N = {len(p)} blocks")
    chk("rank, signature-guided aligned", 1.91, r[SA])
    chk("rank, GPF-guided aligned", 2.26, r[GA])
    chk("rank, SMOTE", 2.88, r[SM])
    chk("rank, GPF-guided plain", 2.96, r[GP])
    chk("Nemenyi CD (4 methods)", 0.61, cd(4, len(p)))
    chk("Friedman p", 3e-6, friedmanchisquare(*[p[m] for m in (SM, GP, GA, SA)]).pvalue, kind="p")
    for lab, m, dv, pv in (("sig/aligned", SA, 0.047, 2.9e-4), ("gpf/aligned", GA, 0.042, 1.4e-3),
                           ("gpf/plain", GP, 0.012, 0.75)):
        chk(f"dF1 vs SMOTE, {lab}", dv, (p[m] - p[SM]).mean())
        chk(f"  p vs SMOTE, {lab}", pv, wilcoxon(p[m], p[SM]).pvalue, kind="p")
    chk("aligned vs plain (same guidance)", 0.029, (p[GA] - p[GP]).mean())
    chk("  p, aligned vs plain", 5.1e-4, wilcoxon(p[GA], p[GP]).pvalue, kind="p")
    chk("sig/aligned vs gpf/aligned", 0.005, (p[SA] - p[GA]).mean())
    chk("  p, sig vs gpf", 0.28, wilcoxon(p[SA], p[GA]).pvalue, kind="p")
    for lab, m, v in (("sig/aligned", SA, 0.648), ("gpf/aligned", GA, 0.643),
                      ("SMOTE", SM, 0.601), ("gpf/plain", GP, 0.613)):
        chk(f"mean F1, {lab}", v, p[m].mean())
    chk("mean F1, no oversampling", 0.460, d[d.method == "none"].f1.mean())

    g = blocks(d, [SM, GP, GA, SA], "gmean"); rg = g.rank(axis=1, ascending=False).mean()
    for lab, m, v in (("sig/aligned", SA, 1.98), ("gpf/aligned", GA, 2.27), ("SMOTE", SM, 2.74), ("gpf/plain", GP, 3.01)):
        chk(f"G-mean rank, {lab}", v, rg[m])
    chk("dG-mean vs SMOTE, sig/aligned", 0.034, (g[SA] - g[SM]).mean())
    chk("  p", 5.0e-3, wilcoxon(g[SA], g[SM]).pvalue, kind="p")
    for lab, m, v in (("sig/aligned", SA, 0.708), ("gpf/aligned", GA, 0.701),
                      ("SMOTE", SM, 0.673), ("gpf/plain", GP, 0.673)):
        chk(f"mean G-mean, {lab}", v, g[m].mean())
    chk("mean G-mean, no oversampling", 0.519, d[d.method == "none"].gmean.mean())

    b = blocks(d, [SM, GA, SA], "brier")
    chk("Brier, SMOTE", 0.111, b[SM].mean(), 0.002)
    chk("Brier, sig/aligned", 0.104, b[SA].mean(), 0.002)
    chk("Brier, gpf/aligned", 0.104, b[GA].mean(), 0.002)
    chk("Brier, no oversampling", 0.126, d[d.method == "none"].brier.mean(), 0.002)

    def blockmed(frame, m, col):
        """The paper's medians are over dataset x IR BLOCKS, not over raw rows: average the seeds
        within a block first, then take the median across blocks. Row medians differ by up to 0.2."""
        s = frame[frame.method == m]
        return s.pivot_table(index=["dataset", "IR"], values=col, aggfunc="mean")[col].median()

    for lab, m, dv, mv in (("sig/aligned", SA, 1.31, 0.19), ("gpf/aligned", GA, 1.36, 0.21),
                           ("SMOTE", SM, 0.53, 0.19), ("gpf/plain", GP, 1.62, 0.24)):
        chk(f"median dispersion ratio, {lab}", dv, blockmed(d, m, "disp_ratio"), 0.05)
        chk(f"median MMD, {lab}", mv, blockmed(d, m, "mmd_syn"), 0.02)

    # DTW cell, the 15 datasets where it is affordable
    q = blocks(d, [SM, GP, GA, SA, DA]); rq = q.rank(axis=1, ascending=False).mean()
    print(f"  DTW-affordable blocks: N = {len(q)}")
    chk("dF1 vs SMOTE, dtw/aligned", 0.072, (q[DA] - q[SM]).mean())
    chk("  p", 7.6e-5, wilcoxon(q[DA], q[SM]).pvalue, kind="p")
    chk("dF1 vs SMOTE, sig/aligned on the same blocks", 0.047, (q[SA] - q[SM]).mean())
    for lab, m, v in (("dtw/aligned", DA, 2.34), ("sig/aligned", SA, 2.47), ("gpf/aligned", GA, 2.81),
                      ("SMOTE", SM, 3.56), ("gpf/plain", GP, 3.82)):
        chk(f"rank among 5 on DTW blocks, {lab}", v, rq[m])
    chk("mean F1, dtw/aligned", 0.718, q[DA].mean())
    chk("mean G-mean, dtw/aligned", 0.790, blocks(d, [DA], "gmean")[DA].mean())
    chk("Brier, dtw/aligned", 0.086, blocks(d, [DA], "brier")[DA].mean(), 0.002)
    chk("median dispersion ratio, dtw/aligned", 1.21, blockmed(d, DA, "disp_ratio"), 0.05)
    chk("median MMD, dtw/aligned", 0.14, blockmed(d, DA, "mmd_syn"), 0.02)

    # Panel B: per-dataset gains at IR 10-20
    hi = d[d.IR.isin([10, 20])]
    pv = hi.pivot_table(index="dataset", columns="method", values="f1", aggfunc="mean")
    gain = (pv[SA] - pv[SM]).sort_values()
    print(f"  panel B: beats SMOTE by >0.01 on {(gain > 0.01).sum()} datasets (paper 14), "
          f"by <0.01 on {((gain > 0) & (gain <= 0.01)).sum()} (paper 1), loses on {(gain < 0).sum()} (paper 5)")
    for ds, v in (("EthanolConcentration", -0.11), ("GunPoint", -0.07), ("Cricket", -0.02),
                  ("HandMovementDirection", -0.02), ("ArrowHead", -0.02)):
        chk(f"IR10-20 gain, {ds}", v, gain.get(ds), 0.02)
    for ds, v in (("SelfRegulationSCP1", 0.31), ("ArticularyWordRecognition", 0.22),
                  ("UWaveGestureLibrary", 0.15), ("NATOPS", 0.14)):
        chk(f"IR10-20 gain, {ds}", v, gain.get(ds), 0.03)

# ============================================================ Sec. 5.2 + Table 1 (main comparison)
print("\n" + "=" * 104)
print("Main comparison  --  Sec. 5.2, Table 1, Fig. 2")
print("=" * 104)
d6 = load("v06")
if d6 is not None:
    HY = "hybrid GPF-nbrs/raw-walk (M calibrated)"
    KC = "KNNOR-GPF (M calibrated)"
    OF1, OFM = "KNNOR-official-GPF (randmx=1)", "KNNOR-official-GPF (randmx=M*)"
    CM = "chain-mean-GPF (PSSTO-style)"
    M6 = [SM, CM, OF1, OFM, KC, HY]
    p6 = blocks(d6, M6); r6 = p6.rank(axis=1, ascending=False).mean()
    print(f"  N = {len(p6)} blocks")
    for lab, m, v in (("hybrid, GPF-calibrated M", HY, 2.42), ("SMOTE", SM, 2.72), ("KNNOR-GPF calibrated", KC, 3.08),
                      ("reference impl., calibrated bound", OFM, 3.67), ("chain mean (PSSTO)", CM, 4.03),
                      ("reference impl., default bound", OF1, 5.07)):
        chk(f"rank, {lab}", v, r6[m])
    chk("calibrated bound on the reference implementation", 0.025, (p6[OFM] - p6[OF1]).mean())
    chk("  p", 1e-7, wilcoxon(p6[OFM], p6[OF1]).pvalue, kind="p")
    for lab, m, dv, mv in (("hybrid", HY, 1.70, 0.24), ("SMOTE", SM, 0.52, 0.18), ("KNNOR-GPF calib", KC, 0.46, 0.28),
                           ("reference calib", OFM, 0.61, 0.29), ("chain mean", CM, 0.37, 0.24),
                           ("reference default", OF1, 0.24, 0.39)):
        chk(f"median dispersion, {lab}", dv, blockmed(d6, m, "disp_ratio"), 0.06)
        chk(f"median MMD, {lab}", mv, blockmed(d6, m, "mmd_syn"), 0.03)

# ============================================================ Sec. 5.3 (pilot)
print("\n" + "=" * 104)
print("Pilot on eight datasets  --  Sec. 5.3")
print("=" * 104)
dp = load("pilot")
if dp is not None:
    dp = dp[dp.dataset.isin(["ArticularyWordRecognition", "Epilepsy", "NATOPS", "OSULeaf",
                             "GunPoint", "Handwriting", "JapaneseVowels", "RacketSports"])]
    cells = {g: {w: f"guide={g} | walk={w}" for w in ("plain", "aligned")} for g in ("gpf", "sig", "dtw")}
    have = set(dp.method)
    full = [c for g in cells.values() for c in g.values() if c in have]
    print(f"  cells present: {len(full)} of 6")
    if len(full) == 6:
        pp = blocks(dp, full + [SM])
        print(f"  N = {len(pp)} blocks")
        # Average the three guidance spaces WITHIN each block first, then pair. Stacking the three
        # spaces into 72 pseudo-independent pairs instead would give p = 3e-4, which is not the
        # paper's test.
        al = sum(pp[cells[g]["aligned"]] for g in cells) / 3.0
        pl = sum(pp[cells[g]["plain"]] for g in cells) / 3.0
        chk("aligned - plain, averaged over guidance", 0.035, (al - pl).mean())
        chk("  p (paired on block averages)", 0.015, wilcoxon(al, pl).pvalue, kind="p")
        for g, v in (("dtw", 0.692), ("sig", 0.671), ("gpf", 0.668)):
            chk(f"mean F1, {g}/aligned", v, pp[cells[g]['aligned']].mean())
        for g, dv, pv in (("dtw", 0.081, 0.002), ("sig", 0.060, 0.004), ("gpf", 0.058, 0.015)):
            c = cells[g]['aligned']
            chk(f"dF1 vs SMOTE, {g}/aligned", dv, (pp[c] - pp[SM]).mean())
            chk(f"  p", pv, wilcoxon(pp[c], pp[SM]).pvalue, kind="p")
        c = cells['dtw']['plain']
        chk("dF1 vs SMOTE, dtw/plain", 0.050, (pp[c] - pp[SM]).mean())
        chk("  p", 0.06, wilcoxon(pp[c], pp[SM]).pvalue, kind="p")
    else:
        print("  need all six cells -- run `python run_all_local.py pilot`")

# ============================================================ Sec. 4.x (frontier)
print("\n" + "=" * 104)
print("The frontier in M  --  Fig. 3")
print("=" * 104)
df = load("frontier")
if df is not None:
    gg = df.groupby("M")[["L_k", "disp_walk", "disp_filtered", "accept", "f1"]].mean()
    x = gg.index.values
    m_one = np.interp(1.0, gg.disp_walk.values, x)
    chk("M where the bare walk reaches dispersion 1", 1.8, float(m_one), 0.1)
    contraction = 1 - (gg.disp_filtered / gg.disp_walk).mean()
    chk("extra contraction by the filters (fraction)", 0.40, float(contraction), 0.08)
    chk("acceptance rate (flat)", 0.65, float(gg.accept.mean()), 0.03)
    chk("F1 gain from M=0.5 to M=2.6", 0.09, float(gg.f1.iloc[-1] - gg.f1.iloc[0]), 0.02)

# ============================================================ Sec. 5.5 (robustness)
print("\n" + "=" * 104)
print("Robustness  --  Sec. 5.5")
print("=" * 104)
d8 = load("v08")
if d8 is not None:
    base = d8[d8.guard.fillna(0) == 0]
    pr = blocks(base[base.clf == "ridge"], [SM, GP, GA, SA]); rr = pr.rank(axis=1, ascending=False).mean()
    for lab, m, v in (("sig/aligned", SA, 2.06), ("gpf/aligned", GA, 2.38), ("gpf/plain", GP, 2.60), ("SMOTE", SM, 2.97)):
        chk(f"ridge rank, {lab}", v, rr[m])
    chk("ridge dF1 vs SMOTE, sig/aligned", 0.024, (pr[SA] - pr[SM]).mean())
    chk("  p", 4e-4, wilcoxon(pr[SA], pr[SM]).pvalue, kind="p")
    chk("ridge, aligned vs plain", 0.007, (pr[GA] - pr[GP]).mean())
    chk("  p", 0.069, wilcoxon(pr[GA], pr[GP]).pvalue, kind="p")

    SUF = r" \| guard(?:-\w+)?$"
    g = d8[d8.guard.fillna(0) == 1].copy()
    if "guard_mode" in g.columns and g.guard_mode.notna().any():
        g = g[g.guard_mode.fillna("shrink") == "shrink"]
    g["base_m"] = g.method.str.replace(SUF, "", regex=True)
    b = d8[d8.guard.fillna(0) == 0][["dataset", "seed", "IR", "method", "clf", "f1", "mmd_syn"]]
    j = g.merge(b, left_on=["dataset", "seed", "IR", "base_m", "clf"],
                right_on=["dataset", "seed", "IR", "method", "clf"], suffixes=("_g", "_b"))
    jr = j[j.clf == "rf"]; fired = jr[jr.guard_fired == 1]
    chk("guard fires on fraction of cells", 0.30, float(jr.guard_fired.mean()), 0.03)
    chk("median rho before", 3.7, float(fired.rho_before.median()), 0.2)
    chk("median rho after", 1.6, float(fired.rho_after.median()), 0.2)
    chk("guard dF1 when fired", 0.006, float((fired.f1_g - fired.f1_b).mean()), 0.01)
    chk("  p", 0.49, wilcoxon(fired.f1_g, fired.f1_b).pvalue, kind="p")
    chk("guard dMMD when fired", -0.062, float((fired.mmd_syn_g - fired.mmd_syn_b).mean()), 0.02)
    et = fired[fired.dataset == "EthanolConcentration"]
    if len(et):
        chk("EthanolConcentration, max rho before", 373.0, float(et.rho_before.max()), 40)
        chk("EthanolConcentration, dMMD", -0.154, float((et.mmd_syn_g - et.mmd_syn_b).mean()), 0.03)

print("\n" + "=" * 104)
print("ALL QUOTED NUMBERS AGREE" if nbad == 0 else f"{nbad} value(s) differ from the manuscript -- see MISMATCH above")
print("=" * 104)
raise SystemExit(0 if nbad == 0 else 1)
