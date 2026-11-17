"""
What the surrogate's interpolation error does to the profile likelihood of the phase transition (section 4.3).

A contour is drawn relative to the maximum of its own scan, so an error that is the same everywhere cancels. This
measures the part that does not cancel. The same phase-transition scan is run twice over the same plane, the same
data and the same reheating temperatures, and in both runs the astrophysical parameters are profiled over the same
2000 points, the holodeck library the surrogate is fitted to. The two runs differ only in where the background at
those points comes from:

    library     the median and the width holodeck measured there, from 2000 realisations per model
    surrogate   the median and the width the surrogate predicts there

The skew is taken from the surrogate on both sides, since the library files store only the median and the
standard deviation per bin. The 16-to-84 half-width of the library is WIDTH_SCALE times its standard deviation.
Both sides profile over the same finite sample, so the offset from a continuous maximum cancels in their difference.

    python surrogate_vs_library.py scan -n 50 -j 24     writes scans/scan_surrogate_vs_library.csv
    python surrogate_vs_library.py analyse               median difference inside the 1 sigma region
"""
import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

import numpy as np
import pandas as pd

from paths import FREESPEC_ROOT, INPUTS, SCANS
import binaryfog as bf                                                        # noqa: E402
from fopt_model import spectrum_fopt                                          # noqa: E402

LIBRARIES = ("lib6d_n400_r2000_f14.jsonl", "lib6d_n1600_r2000_f14_b.jsonl")
DATASET = "ng15_cp_v2"                 # the data of the paper's phase-transition scan
WIDTH_SCALE = 0.875                    # 16-to-84 half-width per standard deviation of log10 h_c^2
NQ = 101                               # nodes of the warp integral, as in the likelihood
TREH = np.logspace(-3, 0, 60)          # GeV
T1SIGMA = 1.15                         # 1 sigma threshold in ln L for two scanned parameters (Wilks)

_Z = np.linspace(-4.0, 4.0, NQ)
_W = np.exp(-0.5 * _Z ** 2)
_W /= _W.sum()
_STATE = {}


def _nodes(median_log10hc2, half_dex, skew, freqs):
    """The background of each model at each warp node: (models, bins, nodes), in h^2 Omega_GW."""
    c = np.clip(np.asarray(skew, float), 0.02, 0.45)[:, None, None]
    z = np.maximum(_Z[None, None, :], -0.5 / c)         # the warp is held flat below its turnover
    lg = median_log10hc2[:, :, None] + half_dex[:, :, None] * (z + c * z ** 2)
    return bf.hc2_to_h2omega(freqs[None, :, None], lg)


def prepare():
    """The 2000 library points, and their background on both sides of the comparison."""
    data = bf.FreeSpectrum.load(DATASET, root=FREESPEC_ROOT)
    bg = bf.Background(data.freqs)
    theta, med_lib, sig_lib = [], [], []
    for name in LIBRARIES:
        for line in open(INPUTS / "library" / name):
            e = json.loads(line)
            m = np.asarray(e["median"], float)
            s = np.asarray(e["std"], float)
            if not (np.isfinite(m).all() and np.isfinite(s).all() and (s > 0).all()):
                continue
            theta.append([e["params"][k] for k in bf.PARAM_NAMES])
            med_lib.append(m)
            sig_lib.append(s)
    theta = np.asarray(theta)
    med_lib = np.asarray(med_lib)
    half_lib = WIDTH_SCALE * np.asarray(sig_lib)

    pref = 2 * np.pi ** 2 / (3 * bf.background.H100_HZ ** 2) * data.freqs ** 2
    med_sur, half_sur, skew = [], [], []
    for t in theta:
        omega, sigma, c = bg.predict(t)
        med_sur.append(np.log10(np.asarray(omega, float) / pref))
        half_sur.append(np.asarray(sigma, float))
        skew.append(c)
    med_sur, half_sur, skew = np.asarray(med_sur), np.asarray(half_sur), np.asarray(skew)

    lnp = np.array([bf.ln_prior(t, "nanograv") for t in theta])
    keep = np.isfinite(lnp)
    B = {"library": _nodes(med_lib[keep], half_lib[keep], skew[keep], data.freqs),
         "surrogate": _nodes(med_sur[keep], half_sur[keep], skew[keep], data.freqs)}
    return data, theta[keep], B, lnp[keep]


def _cell(job):
    """The profile over the library points, on both sides, at one point of the plane."""
    i, alpha, betaH = job
    data, B, lnp = (_STATE[k] for k in ("data", "B", "lnp"))
    out = {"i": i, "alpha": alpha, "betaH": betaH}
    for side, Bs in B.items():
        best, arg = -np.inf, None
        for t in TREH:
            pt = np.asarray(spectrum_fopt(data.freqs, alpha, betaH, t), float)
            total = np.zeros(Bs.shape[0])
            for b in range(data.nbins):
                lr = data.to_log10rho(pt[b] + Bs[:, b, :], b)
                dens = np.exp(np.interp(lr, data.grid, data.density[b], right=-np.inf))
                with np.errstate(divide="ignore"):
                    total += np.log(dens @ _W)
            total += lnp
            k = int(np.argmax(total))
            if total[k] > best:
                best, arg = float(total[k]), (float(t), k)
        out[f"lnL_{side}"] = best
        out[f"Treh_{side}"], out[f"model_{side}"] = arg
    return out


def _init(state):
    _STATE.update(state)


def analyse(path):
    """Median of surrogate minus library over the cells inside the 1 sigma region of either run."""
    table = pd.read_csv(path).sort_values(["alpha", "betaH"])
    Ls = {s: table.pivot(index="alpha", columns="betaH", values=f"lnL_{s}").to_numpy()
          for s in ("library", "surrogate")}
    inside = np.zeros(Ls["library"].shape, bool)
    for L in Ls.values():
        inside |= L >= np.nanmax(L) - T1SIGMA
    d = Ls["surrogate"] - Ls["library"]
    print(f"  surrogate minus library, median over the {int(inside.sum())} cells inside the 1 sigma region: "
          f"{np.median(d[inside]):+.3f}")


def scan(ngrid, jobs, out):
    data, theta, B, lnp = prepare()
    print(f"  {len(theta)} library points, data {DATASET}, {ngrid}x{ngrid} grid, "
          f"{len(TREH)} reheating temperatures", flush=True)
    alpha = np.logspace(-2, 10, ngrid)          # the plane of the phase-transition scan
    betaH = np.logspace(np.log10(0.7), np.log10(300), ngrid)
    jobs_list = [(i, a, b) for i, a in enumerate(alpha) for b in betaH]
    state = {"data": data, "B": B, "lnp": lnp}
    with mp.Pool(jobs, initializer=_init, initargs=(state,)) as pool:
        rows = []
        for n, r in enumerate(pool.imap_unordered(_cell, jobs_list, chunksize=4), 1):
            rows.append(r)
            if n % 250 == 0:
                print(f"    {n}/{len(jobs_list)}", flush=True)
    pd.DataFrame(rows).sort_values(["alpha", "betaH"]).to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("scan", "analyse"))
    ap.add_argument("-n", "--ngrid", type=int, default=50)
    ap.add_argument("-j", "--jobs", type=int, default=24)
    ap.add_argument("--out", default="scan_surrogate_vs_library.csv", help="scan file in scans/")
    a = ap.parse_args()
    if a.mode == "scan":
        scan(a.ngrid, a.jobs, SCANS / a.out)
    else:
        analyse(SCANS / a.out)
