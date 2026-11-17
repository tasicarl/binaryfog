"""Re-profile the suspect points of a stored (alpha, beta/H) scan with three further seeds each and keep the
highest ln L: a profile is a maximum, so where runs disagree the higher one is the better estimate.
Suspect: (a) more than THR (default 0.3) below the median of the eight neighbours, (b) T_reh on a bound of the scan
(1 MeV, 10 GeV), (c) beta/H < 1, where the two T_reh branches (near 1 MeV and at GeV) compete. Run with the
environment of run_scan_marginal.py that wrote the scan; JOBS sets the number of worker processes (default 28).

    FREESPEC=ng15_hd_v2 [PRIOR_SET=<set>] python repair_scan.py scan_<mode>_n120_a-2to3[_<set>]_ng15_hd_v2.csv 0.05
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import run_scan_marginal as r
from paths import SCANS

SEEDS = (1_000_003, 2_000_003, 3_000_003)


def job(args):
    a, b, mode, i = args
    best = None
    for s in SEEDS:
        o = r.profile_point((a, b, mode, i + s))
        best = o if best is None or o["lnL"] > best["lnL"] else best
    return i, best


if __name__ == "__main__":
    path = SCANS / Path(sys.argv[1]).name           # a file of scans/
    mode = path.name.split("_")[1]
    d = pd.read_csv(path)
    ka, kb = np.round(np.log10(d.alpha), 6), np.round(np.log10(d.betaH), 6)
    piv = d.assign(ka=ka, kb=kb).pivot(index="ka", columns="kb", values="lnL")
    med = median_filter(piv.to_numpy(), size=3, mode="nearest")
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3      # how far below the neighbours' median
    low = pd.DataFrame(piv.to_numpy() < med - thr, index=piv.index, columns=piv.columns)
    is_low = np.array([low.at[x, y] for x, y in zip(ka, kb)])
    at_bound = (d.Treh <= r.TREH_MIN * 1.001) | (d.Treh >= r.TREH_MAX * 0.999)
    sel = np.flatnonzero(is_low | at_bound | (d.betaH < 1.0))
    print(f"{path.name}: {len(sel)} suspect points ({is_low.sum()} lone lows, {at_bound.sum()} on a T_reh bound, "
          f"{(d.betaH < 1).sum()} below beta/H = 1)", flush=True)
    t0 = time.perf_counter()
    with mp.Pool(int(os.environ.get("JOBS", 28))) as pool:
        res = pool.map(job, [(d.alpha[i], d.betaH[i], mode, int(i)) for i in sel], chunksize=1)
    gained = []
    for i, o in res:
        if o["lnL"] > d.at[i, "lnL"] + 1e-6:
            gained.append(o["lnL"] - d.at[i, "lnL"])
            for k, v in o.items():
                if k in d.columns and k not in ("alpha", "betaH"):
                    d.at[i, k] = v
    d.to_csv(path, index=False)
    g = np.array(gained) if gained else np.zeros(1)
    print(f"  {len(gained)} points improved, by up to {g.max():.3f} (median {np.median(g):.3f}); "
          f"max ln L {d.lnL.max():.4f}; {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
