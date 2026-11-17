"""Re-profile the lone low points of the profiles of figure 6 with 24 restarts and another seed, and keep the
higher ln L. A point is lone low when it lies more than 0.05 below the mean of its two neighbours.
Rewrites the given file in scans/ in place.

    BINARYFOG_NG15_MODELS=<model files> python repair_examples.py ng15_examples_n80_nt.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import DATA                                                     # noqa: E402
from profile_examples import example_scan                                     # noqa: E402

src = SCANS / sys.argv[1]
raw = json.loads(src.read_text())
models = Path(os.environ["BINARYFOG_NG15_MODELS"])
jobs = int(os.environ.get("JOBS", 28))
data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
for key, e in raw.items():
    model = bf.load_ptarcade(models / f"{key}.py")
    param = e["parameter"]
    grid = np.asarray(e["grid"], float)
    for case in ("power law", "population"):
        y = np.asarray(e[case], float)
        nb = np.full_like(y, -np.inf); nb[1:-1] = 0.5 * (y[:-2] + y[2:])
        low = np.flatnonzero(y < nb - 0.05)
        if not len(low):
            print(f"  {key:6s} {case:10s}: no lone lows", flush=True); continue
        scan = example_scan(data, model, param, grid[low], case, restarts=24, sweeps=0, seed=7)
        new = scan.run(jobs=min(jobs, len(low)))["lnL"].to_numpy()
        for i, v in zip(low, new):
            print(f"  {key:6s} {case:10s} {param} = {grid[i]:+.3f}: {y[i]:+.3f} -> {max(v, y[i]):+.3f}", flush=True)
            e[case][i] = float(max(v, y[i]))
src.write_text(json.dumps(raw, indent=1))
print(f"wrote {src.name}")
