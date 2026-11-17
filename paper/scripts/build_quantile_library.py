"""
The held-out set of inputs/library/qlib.jsonl: 150 held-out models (fresh Latin hypercube), 2000 realisations each,
stored as 13 quantiles of log10 h_c^2 per bin. Needs holodeck.

    python build_quantile_library.py --npoints 150 --nreals 2000 --jobs 24 --out ../inputs/library/qlib.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import holodeck as holo
from holodeck.librarian import DEF_PTA_DUR, DEF_NUM_LOUDEST

from holodeck_runs import PARAM_NAMES, PRIOR_BOX, QUANT

PARAM_SPACE = "PS_Classic_Phenom_Uniform"
_space = None


def _get_space():
    global _space
    if _space is None:
        _space = holo.librarian.param_spaces_dict[PARAM_SPACE]()
    return _space


def one_point(job):
    idx, values, nfreqs, nreals = job
    params = dict(zip(PARAM_NAMES, [float(v) for v in values]))
    t0 = time.perf_counter()
    try:
        sam, hard = _get_space().model_for_params(params)
        out = holo.librarian.lib_tools.run_model(
            sam, hard, pta_dur=DEF_PTA_DUR, nfreqs=nfreqs, nreals=nreals,
            nloudest=DEF_NUM_LOUDEST, gwb_flag=True, singles_flag=False,
            details_flag=False, params_flag=False, log=None)
        x = 2 * np.log10(np.asarray(out["gwb"]))            # log10 h_c^2, (nfreqs, nreals)
        q = np.quantile(x, QUANT, axis=1)                   # (nquant, nfreqs)
        rec = {"i": idx, "params": params,
               "quantiles": [[float(v) for v in row] for row in q],
               "median": [float(v) for v in np.median(x, axis=1)],
               "mean": [float(v) for v in x.mean(axis=1)],
               "std": [float(v) for v in x.std(axis=1)],
               "seconds": time.perf_counter() - t0}
    except Exception as exc:                                # noqa: BLE001
        rec = {"i": idx, "params": params, "error": f"{type(exc).__name__}: {exc}",
               "seconds": time.perf_counter() - t0}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npoints", type=int, default=150)
    ap.add_argument("--nreals", type=int, default=2000)
    ap.add_argument("--nfreqs", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=str, default="qlib.jsonl")
    args = ap.parse_args()

    from scipy.stats import qmc
    lo = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES])
    hi = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])
    values = lo + qmc.LatinHypercube(d=6, seed=args.seed).random(args.npoints) * (hi - lo)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
        print(f"resuming: {len(done)} points already in {out_path}", flush=True)
    jobs = [(i, values[i], args.nfreqs, args.nreals)
            for i in range(args.npoints) if i not in done]
    print(f"{len(jobs)} points, {args.nfreqs} bins, {args.nreals} realisations, "
          f"{args.jobs} workers -> {out_path}", flush=True)

    t0 = time.perf_counter()
    n_err = 0
    with open(out_path, "a", buffering=1) as fh, mp.Pool(args.jobs) as pool:
        for n, rec in enumerate(pool.imap_unordered(one_point, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            n_err += "error" in rec
            if n % 10 == 0 or n == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {n}/{len(jobs)} | {el/60:.1f} min | eta {el/n*(len(jobs)-n)/60:.1f} min"
                      f" | errors {n_err}", flush=True)
    print(f"finished in {(time.perf_counter()-t0)/60:.1f} min, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
