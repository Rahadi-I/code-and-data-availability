"""
Download the 30 UEA multivariate datasets with aeon (run on the workstation, needs internet).
Files land as  data/<Name>/<Name>_TRAIN.ts  and  <Name>_TEST.ts  — the layout data.py already reads.

    cd D:\thesis\paper0_all_start_here\code
    python download_uea.py            # all 30 (≈1 GB, some minutes)
    python download_uea.py BasicMotions NATOPS   # just a few
"""
import os, sys, time
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import multivariate   # the 30 UEA names

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

names = sys.argv[1:] or sorted(multivariate)
ok, failed = [], []
for name in names:
    t = time.time()
    try:
        X, y = load_classification(name, extract_path=DATA_DIR)
        shape = X.shape if hasattr(X, "shape") else f"{len(X)} series, unequal length"
        print(f"{name:26s} OK   {shape}   {time.time()-t:5.1f}s")
        ok.append(name)
    except Exception as e:
        print(f"{name:26s} FAIL {str(e)[:120]}")
        failed.append(name)

print(f"\n{len(ok)} downloaded, {len(failed)} failed: {failed}")
print("Files are under", DATA_DIR)
