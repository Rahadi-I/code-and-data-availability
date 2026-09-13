"""Analysis of the v0.8 run: (1) the dispersion guard, (2) the second classifier, (3) the feature
budget, and (4) the trimming variant of the guard.

Reads the CSVs of the published run by default. To analyse your own reproduction instead, set
V08_MAIN / V08_NSEL32 / V08_NSEL128 / V08_TRIM -- `run_all_local.py v08stats` does that for you."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

MAIN = os.environ.get("V08_MAIN", "results/results_v08_main.csv")
NSEL = {n: os.environ.get(f"V08_NSEL{n}", f"results/results_v08_nsel{n}.csv") for n in (32, 128)}
TRIM = os.environ.get("V08_TRIM", "results/results_v08_trim.csv")

M = ["SMOTE-raw", "guide=gpf | walk=plain", "guide=gpf | walk=aligned", "guide=sig | walk=aligned"]
SH = {m: m.replace("guide=", "").replace(" | walk=", "/") for m in M}
print(f"main CSV: {MAIN}")
d = pd.read_csv(MAIN)
base = d[d.guard.fillna(0) == 0]

print(f"=== datasets: {d.dataset.nunique()}  seeds: {sorted(d.seed.unique())} ===\n")

# ---------- 1. second classifier ----------
print("--- (1) downstream classifier robustness (guard off) ---")
for c in ["rf", "ridge"]:
    p = base[base.clf == c].pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")[M].dropna()
    r = p.rank(axis=1, ascending=False).mean()
    chi, pv = friedmanchisquare(*[p[m] for m in M])
    CD = 2.569 * np.sqrt(4 * 5 / (6.0 * len(p)))
    print(f"  clf={c:5s} N={len(p)}  Friedman p={pv:.2e}  CD={CD:.2f}")
    for m in M:
        line = f"    {SH[m]:14s} rank {r[m]:.2f}  F1 {p[m].mean():.3f}"
        if m != "SMOTE-raw":
            line += f"  Δ={p[m].mean()-p['SMOTE-raw'].mean():+.3f}  p={wilcoxon(p[m], p['SMOTE-raw']).pvalue:.4f}"
        print(line)
    print(f"    aligned vs plain (gpf): {(p[M[2]]-p[M[1]]).mean():+.3f}  p={wilcoxon(p[M[2]], p[M[1]]).pvalue:.4f}")

# ---------- 2. dispersion guard ----------
print("\n--- (2) dispersion guard (tau = 2) ---")
# The guarded cells are named "... | guard" by the original run and "... | guard-<mode>" by the
# current one (GUARDMODE gained a suffix later). Strip either, and keep only the shrink mode --
# section (4) is where trim is analysed.
GUARD_SUFFIX = r" \| guard(?:-\w+)?$"
g = d[d.guard.fillna(0) == 1].copy()
if "guard_mode" in g.columns and g.guard_mode.notna().any():
    g = g[g.guard_mode.fillna("shrink") == "shrink"]
g["base_m"] = g.method.str.replace(GUARD_SUFFIX, "", regex=True)
b = d[d.guard.fillna(0) == 0]
j = g.merge(b[["dataset", "seed", "IR", "method", "clf", "f1", "gmean", "mmd_syn"]],
            left_on=["dataset", "seed", "IR", "base_m", "clf"], right_on=["dataset", "seed", "IR", "method", "clf"],
            suffixes=("_g", "_b"))
for c in ["rf", "ridge"]:
    jc = j[j.clf == c]; fired = jc[jc.guard_fired == 1]
    print(f"  clf={c}: guard fires on {jc.guard_fired.mean()*100:.0f}% of cells ({int(jc.guard_fired.sum())}/{len(jc)});"
          f" overall ΔF1={(jc.f1_g-jc.f1_b).mean():+.4f}, when fired {(fired.f1_g-fired.f1_b).mean():+.3f}"
          f" (p={wilcoxon(fired.f1_g, fired.f1_b).pvalue:.4f} on {len(fired)} cells)" if len(fired) > 5 else "")
print("  rho before -> after, median over fired cells:",
      round(j[j.guard_fired == 1].rho_before.median(), 2), "->", round(j[j.guard_fired == 1].rho_after.median(), 2))
print("\n  per-dataset ΔF1 of the guard (rf, cells where it fired):")
f = j[(j.clf == "rf") & (j.guard_fired == 1)]
if len(f):
    t = f.groupby("dataset").apply(lambda x: pd.Series({"cells": len(x), "rho_med": round(x.rho_before.median(), 1),
                                                        "dF1": round((x.f1_g - x.f1_b).mean(), 3)}), include_groups=False)
    print(t.sort_values("dF1").to_string())

# guarded method vs SMOTE, head to head
print("\n  ranking with the guarded aligned cells (rf):")
dd = d[d.clf == "rf"].copy()
if "guard_mode" in dd.columns:
    dd = dd[(dd.guard.fillna(0) == 0) | (dd.guard_mode.fillna("shrink") == "shrink")]
dd["m2"] = dd.method.str.replace(GUARD_SUFFIX, " | guard", regex=True)   # one name for either run
sel = ["SMOTE-raw", "guide=gpf | walk=plain", "guide=gpf | walk=aligned | guard", "guide=sig | walk=aligned | guard"]
p = dd.pivot_table(index=["dataset", "IR"], columns="m2", values="f1", aggfunc="mean")
if all(s in p.columns for s in sel):
    p = p[sel].dropna(); r = p.rank(axis=1, ascending=False).mean()
    print(f"    N={len(p)}  " + "  ".join(f"{s.replace('guide=','').replace(' | walk=','/').replace(' | guard','+G')}={r[s]:.2f}" for s in sel))
    for s in sel[1:]:
        print(f"    {s.replace('guide=','').replace(' | walk=','/').replace(' | guard','+G'):18s} Δ={p[s].mean()-p['SMOTE-raw'].mean():+.3f} p={wilcoxon(p[s], p['SMOTE-raw']).pvalue:.4f}")

# ---------- 3. feature budget ----------
print("\n--- (3) feature-budget sensitivity (8 pilot datasets, rf) ---")
frames = []
for n in [32, 64, 128]:
    try:
        x = pd.read_csv(NSEL[n]) if n != 64 else base[base.clf == "rf"]
        x = x[x.clf == "rf"] if "clf" in x.columns else x
        frames.append((n, x))
    except Exception:
        pass
pilot = set(pd.read_csv(NSEL[32]).dataset.unique()) if len(frames) else set()
for n, x in frames:
    x = x[x.dataset.isin(pilot)]
    cols = [m for m in M if m in set(x.method)]
    p = x.pivot_table(index=["dataset", "IR"], columns="method", values="f1", aggfunc="mean")
    if "SMOTE-raw" not in p.columns:      # the nsel runs also record SMOTE
        continue
    p = p[cols].dropna()
    if len(p) == 0: continue
    r = p.rank(axis=1, ascending=False).mean()
    out = f"  N_SELECT={n:3d}  N={len(p)}  " + "  ".join(f"{SH[m]}={r[m]:.2f}" for m in cols)
    if "guide=sig | walk=aligned" in cols:
        out += f"   ΔF1(sig/aligned−SMOTE)={p['guide=sig | walk=aligned'].mean()-p['SMOTE-raw'].mean():+.3f}"
    print(out)

# ---------- 4. the trimming variant of the guard ----------
print("\n--- (4) trimming variant vs shrinking (rf) ---")
if not os.path.exists(TRIM):
    print(f"  {TRIM} not found -- skipped.")
else:
    t = pd.read_csv(TRIM)
    t = t[t.clf == "rf"] if "clf" in t.columns else t
    t["base_m"] = t.method.str.replace(r" \| guard-(trim|shrink)$", "", regex=True)
    tb = t[t.guard.fillna(0) == 0][["dataset", "seed", "IR", "method", "f1"]]
    for mode in ("trim", "shrink"):
        tg = t[(t.guard.fillna(0) == 1) & (t.guard_mode == mode)]
        if not len(tg):
            continue
        j2 = tg.merge(tb, left_on=["dataset", "seed", "IR", "base_m"],
                      right_on=["dataset", "seed", "IR", "method"], suffixes=("_g", "_b"))
        fired = j2[j2.guard_fired == 1]
        if len(fired) > 5:
            print(f"  {mode:6s}: fired on {len(fired)} cells / {fired.dataset.nunique()} datasets;"
                  f"  dF1 = {(fired.f1_g - fired.f1_b).mean():+.3f}"
                  f"  (p = {wilcoxon(fired.f1_g, fired.f1_b).pvalue:.1e})")
        else:
            print(f"  {mode:6s}: only {len(fired)} fired cells -- too few to test.")
    print("  The paper reports trim = -0.014 (p = 6e-4) over the 65 cells where it fires, on")
    print("  eight of the nine datasets it was run on:")
    print("  refilling the quota by resampling the survivors reintroduces duplication, so trimming")
    print("  is worse than shrinking. A positive number here would contradict the manuscript.")
