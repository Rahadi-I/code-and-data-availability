r"""
One Python driver for the whole local reproduction. No environment variables, no ^ line
continuations, no PowerShell/cmd differences -- just:

    python run_all_local.py smoke      step 3: one dataset, one seed, a few minutes
    python run_all_local.py main       step 4: the 20-dataset factorial, 2-4 hours
    python run_all_local.py stats      step 4b: Friedman + Nemenyi + Wilcoxon on the CSV from `main`
    python run_all_local.py figures    step 6
    python run_all_local.py all        main, then stats, then figures

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

    elif what == "all":
        factorial(DATASETS, MAIN_CSV, seeds="0,1,2",
                  cells="gpf/plain,gpf/aligned,sig/aligned,dtw/aligned")
        stats(MAIN_CSV)
        figures(MAIN_CSV)

    else:
        print(__doc__)
