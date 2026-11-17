"""
Profile likelihoods of three models of the NANOGrav 15 yr new-physics search (figure 6).

One parameter per model is scanned on a grid while the model's other parameters and the binaries are
profiled, once with the binaries represented by the published (A, gamma) power law under its Gaussian prior and
once by the binaryfog background under the NANOGrav priors. For the population case the file also keeps
the spectra figure 6 draws: at the best fit and at the two ends of the 2 sigma interval, the new-physics
spectrum with every other parameter at the value the maximisation chose at that point.

    BINARYFOG_NG15_MODELS=<model files> python profile_examples.py --data ng15_cp_v2 --out <file in scans/>

IGW_PARAM sets the scanned parameter of the inflationary model (log10_r by default, n_t for figure 6);
ONLY=<model,...> restricts the run to the listed model files.
"""
import argparse
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402

# model file, scanned parameter, panel title
EXAMPLES = [("dw_ds", "log10_T_star", "dark sector domain walls"),
            ("super", "log10_Gmu", "cosmic superstrings"),
            ("igw", os.environ.get("IGW_PARAM", "log10_r"), "inflationary gravitational waves")]
ONLY = [k for k in os.environ.get("ONLY", "").split(",") if k]
# the profile with inflation scanned in its tensor index n_t, drawn in figure 6
FIGURE_JSON = "ng15_examples_n80_nt.json"
BANDS_JSON = "ng15_examples_n80_nt_bands.json"
# ranges in which the power-law background is profiled; its prior is the published two-dimensional
# Gaussian, and the ranges are wide enough not to cut into it
AGAMMA_NUISANCE = {"smbhb_log10_A": (-18.0, -13.0, "linear"),
                   "smbhb_gamma": (0.0, 7.0, "linear")}


def scan_values(lo, hi, n):
    """n values just inside [lo, hi]: tabulated models raise at the edge of their tables."""
    inset = 1e-6 * (hi - lo)
    return np.linspace(lo + inset, hi - inset, n)


def example_scan(data, model, param, values, case, astro_prior="nanograv", **kw):
    """The profile scan of one model parameter, with the binaries as the (A, gamma) power law
    (case "power law") or as the binaryfog background under astro_prior (case "population")."""
    others = [k for k in model.ranges if k != param]
    nuisance = dict(model.nuisance(*others)) if others else {}
    if case == "power law":
        nuisance.update(AGAMMA_NUISANCE)
        return bf.ProfileScan(bf.Likelihood(data, None, marginalise=False),
                              bf.Sum(model.spectrum, bf.smbhb_power_law), scan={param: values},
                              nuisance=nuisance or None, nuisance_prior=bf.ln_prior_agamma,
                              astro_prior=None, **kw)
    return bf.ProfileScan(bf.Likelihood(data, bf.Background(), marginalise=True), model.spectrum,
                          scan={param: values}, nuisance=nuisance or None, nuisance_prior=None,
                          astro_prior=astro_prior, **kw)


def interval(x, y, thr):
    """The profile interval at Delta ln L = thr, with its ends where the curve crosses the threshold
    (linear interpolation between grid points); also whether it reaches the end of the grid."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    level = y.max() - thr
    ok = np.flatnonzero(y >= level)
    i, j = int(ok.min()), int(ok.max())

    def cross(k, nxt):
        if not 0 <= nxt < len(x):
            return x[k]
        return x[k] + (level - y[k]) * (x[nxt] - x[k]) / (y[nxt] - y[k])

    return cross(i, i - 1), cross(j, j + 1), bool(i == 0 or j == len(x) - 1)


def main(dataset, ngrid, jobs, restarts, out):
    models = Path(os.environ["BINARYFOG_NG15_MODELS"])
    data = bf.FreeSpectrum.load(dataset, root=FREESPEC_ROOT)
    result = {}
    for fname, param, label in EXAMPLES:
        if ONLY and fname not in ONLY:
            continue
        model = bf.load_ptarcade(models / f"{fname}.py")
        lo, hi = model.ranges[param]
        grid = scan_values(lo, hi, ngrid)
        others = [k for k in model.ranges if k != param]
        entry = {"label": label, "parameter": param, "grid": list(map(float, grid)),
                 "prior": [float(lo), float(hi)]}
        for case in ("power law", "population"):
            table = example_scan(data, model, param, grid, case, restarts=restarts, sweeps=2).run(jobs=jobs)
            entry[case] = list(map(float, table["lnL"]))
            print(f"  {label:34s} {case:11s} max ln L {max(entry[case]):+.2f}", flush=True)
            if case == "population":
                # evaluated beyond the measured band at both ends, where figure 6 still draws them
                fdraw = np.geomspace(data.freqs[0] / 1.8, data.freqs[-1] * 1.8, 300)
                best = table.loc[table["lnL"].idxmax()]
                y = np.asarray(entry[case], float)
                ok = y >= y.max() - bf.LEVELS_1D[2]
                shown = [float(grid[ok].min()), float(best[param]), float(grid[ok].max())]
                entry["freqs"] = list(map(float, fdraw))
                entry["freqs_measured"] = list(map(float, data.freqs))
                entry["shown"] = shown
                entry["spectra"] = []
                for v in shown:
                    row = table.iloc[int(np.argmin(np.abs(grid - v)))]
                    kw = {k: float(row[k]) for k in others}
                    kw[param] = float(row[param])
                    entry["spectra"].append(list(map(float,
                        np.asarray(model.spectrum(fdraw, **kw), float))))
                entry["best"] = {k: float(best[k]) for k in table.columns if k != "lnL"}
        result[fname] = entry
        Path(out).write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="ng15_cp_v2")
    ap.add_argument("--ngrid", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 28)))
    ap.add_argument("--restarts", type=int, default=8, help="optimiser restarts per grid point")
    ap.add_argument("--out", required=True, help="output file, written to scans/")
    a = ap.parse_args()
    main(a.data, a.ngrid, a.jobs, a.restarts, SCANS / a.out)
