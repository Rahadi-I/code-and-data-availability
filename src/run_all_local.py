r"""
One Python driver for the whole local reproduction. No environment variables, no ^ line
continuations, no PowerShell/cmd differences -- just:

    python run_all_local.py smoke      step 3: one dataset, one seed, a few minutes
    python run_all_local.py main       step 4: the 20-dataset factorial, 2-4 hours
    python run_all_local.py stats      step 4b: Friedman + Nemenyi + Wilcoxon on the CSV from `main`
    python run_all_local.py figures    step 6: Fig. 1 and Fig. 4 from the CSV `main` produced
    python run_all_local.py compare    Fig. 2  -- a separate experiment (run_experiment.py), 1-2 h
    python run_all_local.py gate       Fig. 6  -- reuses the CSV from `compare`, seconds
    python run_all_local.py frontier   Fig. 3  -- its own sweep over M, 30-60 min
    python run_all_local.py all        every step above, in order

Which figure comes from which step:
    Fig. 1 framework        figures       (a drawing, no data)
    Fig. 2 CD, main         compare
    Fig. 3 frontier in M    frontier
    Fig. 4 factorial A-C    main -> figures
    Fig. 5 CD, factorial    main -> stats
    Fig. 6 validity gate    compare -> gate

The robustness study of Sec. 5.5 has no figure; it lives in ..\v08\ and is run from here too:
    python run_all_local.py v08         the main v0.8 run: guard + two classifiers, several hours
    python run_all_local.py v08nsel     feature budgets 32 and 128 on the 8 pilot datasets
    python run_all_local.py v08trim     the trimming variant of the guard, 9 datasets
    python run_all_local.py v08stats    analyse whichever of those three you have

Run it from the folder that contains this file (code\). Results go to ..\results\.
Every step writes its CSV after each dataset, so you can stop with Ctrl-C and keep what is done.
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# The 20 datasets of the paper, slowest last on purpose.
DATASETS = [
    "ItalyPowerDemand", "ArrowHead", "GunPoint", "BasicMotions", "ERing", "JapaneseVowels",
    "Libras", "RacketSports", "Epilepsy", "NATOPS", "OSULeaf", "Handwriting",
    "HandMovementDirection", "ArticularyWordRecognition", "CharacterTrajectories", "Cricket",
    "UWaveGestureLibrary", "SelfRegulationSCP1", "SelfRegulationSCP2", "EthanolConcentration",
]

PILOT8 = ["ArticularyWordRecognition", "Epilepsy", "NATOPS", "OSULeaf",
          "GunPoint", "Handwriting", "JapaneseVowels", "RacketSports"]

METHODS = ("SMOTE-raw;;guide=gpf | walk=plain;;"
           "guide=gpf | walk=aligned;;guide=sig | walk=aligned")


def factorial(names, out, seeds, cells):
    """run_factorial reads OUT and SEEDS at import time, so set them before importing it."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.environ["OUT"] = out
    os.environ["SEEDS"] = seeds
    os.environ["CELLS"] = cells
    import importlib, pandas as pd
    import run_factorial as RF
    importlib.reload(RF)                      # honour OUT/SEEDS if this module was imported before
    rows, t0 = [], time.time()
    for i, n in enumerate(names, 1):
        print(f"\n=== [{i}/{len(names)}] {n} " + "=" * 40, flush=True)
        rows += RF.run_dataset(n, print)
        pd.DataFrame(rows).to_csv(out, index=False)     # written after every dataset
        print(f"--- {n} done, {time.time()-t0:.0f}s elapsed, CSV -> {out}", flush=True)
    print(f"\nfinished {len(names)} datasets in {time.time()-t0:.0f}s -> {out}")
    return out


def need(csv):
    if not os.path.exists(csv):
        sys.exit(f"\n{csv} not found -- run `python run_all_local.py main` first.\n")


def _say(text):
    """Print text that may contain Greek letters on a cp1252 Windows console."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace"))


def sh(script, *args):
    """Run a child script and show ITS error, not a bare CalledProcessError.

    PYTHONIOENCODING/PYTHONUTF8 are forced because stats_cd.py prints 'alpha' and 'Delta' as
    Greek letters. On Windows a child writing to a pipe defaults to cp1252 and dies with
    UnicodeEncodeError before printing anything useful.
    """
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, script, *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    if r.stdout:
        _say(r.stdout)
    if r.returncode:
        _say(r.stderr)
        sys.exit(f"\n{script} failed (exit {r.returncode}). The real error is just above.\n")


def stats(csv, tag="fig_cd_repro"):
    need(csv)
    # stats_cd.py and every make_figure*.py write to "results/<name>" relative to the CURRENT
    # folder, i.e. code\results\ -- not the ..\results\ that holds the CSVs. Create it first,
    # otherwise they die with FileNotFoundError after all the computing is already done.
    os.makedirs("results", exist_ok=True)
    os.environ["METHODS"] = METHODS
    sh("stats_cd.py", csv, tag)
    print(r"  -> code\results\%s.pdf / .png" % tag)


def figures(csv):
    need(csv)
    os.makedirs("results", exist_ok=True)
    os.environ["PAPER"] = "1"
    sh("make_figure1.py")
    sh("make_figure_factorial20.py", csv, "fig_factorial_repro")
    print(r"  -> code\results\fig1_framework.pdf, code\results\fig_factorial_repro.pdf")


# --------------------------------------------------------------------------------------------
# The three figures that do NOT come from the factorial run. They belong to the earlier stages
# of the study, so each needs its own experiment first.
# --------------------------------------------------------------------------------------------

def main_comparison(names=None, out=None, fast=True):
    """Fig. 2 (fig_cd_v06_official): the main comparison -- our walk against the reference KNNOR
    implementation, PSSTO-style chain means, SMOTE and no oversampling. run_experiment.py, not
    run_factorial.py. 5 seeds, ~1-2 hours.

    Leave fast=True: it is what the paper reports. FAST=1 drops three comparators (KNNOR-raw M=1,
    KNNOR-GPF M=3/2, hybrid M=1) that appear in NO figure of the manuscript -- the CSVs behind both
    Fig. 2 and Fig. 6 were produced with FAST=1, and the six bars of Fig. 6's panel A are all
    present under it. fast=False adds two extra bars to that panel, so it produces a figure that
    differs from the published one, at roughly double the runtime. Use it only to look beyond the
    paper, never to reproduce it."""
    names = names or DATASETS
    out = out or os.path.join("..", "results", "repro_v06.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.environ.update(SPLIT="standard", FAST="1" if fast else "0", OFFICIAL="1", N_SELECT="64", OUT=out)
    sh("run_experiment.py", *names)
    os.makedirs("results", exist_ok=True)
    os.environ["METHODS"] = ("SMOTE-raw;;chain-mean-GPF (PSSTO-style);;KNNOR-official-GPF (randmx=1);;"
                             "KNNOR-official-GPF (randmx=M*);;KNNOR-GPF (M calibrated);;"
                             "hybrid GPF-nbrs/raw-walk (M calibrated)")
    sh("stats_cd.py", out, "fig_cd_repro_v06")
    print(r"  -> code\results\fig_cd_repro_v06.pdf")
    return out


def gate(csv=None):
    """Fig. 6 (fig_go_no_go_v05_standard): the validity gate. Reads the SAME CSV as the main
    comparison -- no extra experiment, it just plots a different cut of it."""
    csv = csv or os.path.join("..", "results", "repro_v06.csv")
    need(csv)
    os.makedirs("results", exist_ok=True)
    os.environ["PAPER"] = "1"
    sh("make_figure_v04.py", csv, "fig_gate_repro")
    print(r"  -> code\results\fig_gate_repro.pdf")


def frontier(names=None):
    """Fig. 3 (fig_frontier_M): retained variance, acceptance and F1 along the step bound M.
    Its own sweep over 8 values of M on 6 datasets, 3 seeds. ~30-60 min.
    frontier_M.py writes results/frontier_M.csv relative to THIS folder, and plot_frontier.py
    reads it from there -- so both land in code\results\, not ..\results\."""
    os.makedirs("results", exist_ok=True)
    sh("frontier_M.py", *(names or ["NATOPS", "Epilepsy", "RacketSports", "OSULeaf", "GunPoint", "ERing"]))
    os.environ["PAPER"] = "1"
    sh("plot_frontier.py")
    print(r"  -> code\results\fig_frontier_M.pdf")


# --------------------------------------------------------------------------------------------
# Sec. 5.5, the robustness study. Lives in ..\v08\ : its scripts resolve paths relative to the
# CURRENT folder, so each of these runs with the working directory switched to v08 and switched
# back afterwards. Reproductions are written as repro_* so the published CSVs are never touched.
# --------------------------------------------------------------------------------------------

V08 = next((p for p in (os.path.join(HERE, "..", "v08"), os.path.join(HERE, "v08"))
            if os.path.isfile(os.path.join(p, "run_v08.py"))), os.path.join(HERE, "..", "v08"))
V08_MAIN = "results/repro_v08_main.csv"
V08_NSEL = {32: "results/repro_v08_nsel32.csv", 128: "results/repro_v08_nsel128.csv"}
V08_TRIM = "results/repro_v08_trim.csv"
TRIM_SETS = ["ERing", "GunPoint", "HandMovementDirection", "Handwriting", "JapaneseVowels",
             "Libras", "NATOPS", "OSULeaf", "RacketSports"]


def _in_v08(fn):
    """Run fn with the working directory in v08, then come back."""
    here = os.getcwd()
    os.makedirs(os.path.join(V08, "results"), exist_ok=True)
    os.chdir(V08)
    try:
        return fn()
    finally:
        os.chdir(here)


def v08_main(names=None):
    """The headline v0.8 run: three cells, each aligned cell also in a guarded variant, every row
    scored by BOTH classifiers. 20 datasets x IR {5,10,20} x 3 seeds. Several hours -- the guard
    costs extra feature evaluations and ridge is scored on the same synthetic sets, so the marginal
    cost over the v0.7 factorial is smaller than it looks."""
    def go():
        os.environ.update(SEEDS="0,1,2", CLFS="rf,ridge", N_SELECT="64", GUARD="1",
                          GUARDMODE="shrink", CELLS="gpf/plain,gpf/aligned,sig/aligned", OUT=V08_MAIN)
        sh("run_v08.py", *(names or DATASETS))
        print(f"  -> v08\\{V08_MAIN}")
    _in_v08(go)


def v08_nsel(names=None):
    """Feature-budget sensitivity: N_SELECT 32 and 128 on the 8 pilot datasets, RF only, guard off.
    64 is not re-run -- v08stats reads it out of the main CSV."""
    def go():
        for n in (32, 128):
            os.environ.update(SEEDS="0,1,2", CLFS="rf", N_SELECT=str(n), GUARD="0",
                              CELLS="gpf/aligned,sig/aligned", OUT=V08_NSEL[n])
            print(f"\n=== N_SELECT = {n} " + "=" * 50)
            sh("run_v08.py", *(names or PILOT8))
        print(f"  -> v08\\{V08_NSEL[32]}, v08\\{V08_NSEL[128]}")
    _in_v08(go)


def v08_trim(names=None):
    """The trimming variant: discard the most deviant synthetic points instead of shrinking all of
    them. GUARDMODE takes a comma-separated list and emits one guarded cell per mode, so asking for
    "shrink,trim" -- as the published run did -- puts both on the SAME synthetic sets and makes the
    head-to-head comparison in v08stats block (4) possible. Asking for "trim" alone runs half the
    work and leaves that comparison with nothing to compare against."""
    def go():
        os.environ.update(SEEDS="0,1,2", CLFS="rf", N_SELECT="64", GUARD="1", GUARDMODE="shrink,trim",
                          CELLS="gpf/aligned,sig/aligned", OUT=V08_TRIM)
        sh("run_v08.py", *(names or TRIM_SETS))
        print(f"  -> v08\\{V08_TRIM}")
    _in_v08(go)


def v08_stats():
    """Analyse the reproduction CSVs. Falls back to the published CSV for any piece not yet re-run,
    and says which file each number came from."""
    def go():
        for key, path, pub in (("V08_MAIN", V08_MAIN, "results/results_v08_main.csv"),
                               ("V08_NSEL32", V08_NSEL[32], "results/results_v08_nsel32.csv"),
                               ("V08_NSEL128", V08_NSEL[128], "results/results_v08_nsel128.csv"),
                               ("V08_TRIM", V08_TRIM, "results/results_v08_trim.csv")):
            use = path if os.path.exists(path) else pub
            os.environ[key] = use
            if use is pub or use == pub:
                print(f"  note: {key} falls back to the published {pub} (yours not found)")
        sh("analyse_v08.py")
    _in_v08(go)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    MAIN_CSV = os.path.join("..", "results", "repro_v07.csv")

    if what == "smoke":
        factorial(["ItalyPowerDemand"], os.path.join("..", "results", "smoke.csv"),
                  seeds="0", cells="gpf/plain,gpf/aligned,sig/aligned")
        print("\nSmoke test passed. If the F1 columns look sane, go on to:  python run_all_local.py main")

    elif what == "main":
        factorial(DATASETS, MAIN_CSV, seeds="0,1,2",
                  cells="gpf/plain,gpf/aligned,sig/aligned,dtw/aligned")

    elif what == "stats":
        stats(MAIN_CSV)

    elif what == "figures":
        figures(MAIN_CSV)

    elif what == "compare":          # Fig. 2, and the CSV that Fig. 6 also needs
        main_comparison()

    elif what == "gate":             # Fig. 6, from the CSV `compare` produced
        gate()

    elif what == "frontier":         # Fig. 3, its own sweep over M
        frontier()

    elif what == "v08":              # Sec. 5.5: guard + second classifier
        v08_main()

    elif what == "v08nsel":          # Sec. 5.5: feature budget
        v08_nsel()

    elif what == "v08trim":          # Sec. 5.5: the trimming variant
        v08_trim()

    elif what == "v08stats":
        v08_stats()

    elif what == "all":
        factorial(DATASETS, MAIN_CSV, seeds="0,1,2",
                  cells="gpf/plain,gpf/aligned,sig/aligned,dtw/aligned")
        stats(MAIN_CSV)
        figures(MAIN_CSV)
        main_comparison()
        gate()
        frontier()

    else:
        print(__doc__)
