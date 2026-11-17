"""
The cells of hardening_map.py (same candidates, start points and profiling) at further binary lifetimes, so that
tauf_lower_bound.py resolves the lifetimes at which the preference crosses 2 and 3 sigma.

    python tauf_fine_cells.py <jobs> <tau_f values in Gyr, comma separated> <prior sets, comma separated> <output>
writes a list of cells (prior set, tau_f, nu_inner, ln L with and without the transition) to scans/<output>.
"""
import json
import sys
import time
from multiprocessing import get_context

import hardening_map as H
from paths import SCANS

if __name__ == "__main__":
    jobs = int(sys.argv[1])
    taus = [float(x) for x in sys.argv[2].split(",")]
    work = [w for tag in sys.argv[3].split(",") for w in H.work_items(tag, dict(H.PANELS)[tag], taus)]
    t0 = time.time()
    with get_context("fork").Pool(jobs) as pool:
        res = pool.map(H.cell, work, chunksize=1)
    out = [dict(prior=w, tau=t, nu=n, lnL_pt=a, lnL_bin=b) for w, t, n, a, b in res]
    (SCANS / sys.argv[4]).write_text(json.dumps(out))
    print(f"{len(out)} cells in {time.time()-t0:.0f} s", flush=True)
