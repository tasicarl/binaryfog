"""
The floor of the warp at the best fit with a phase transition (section 4.3).

The warp of eq. (2.7) turns over at z = -1/(2c), and the likelihood holds it flat below that, so the faintest
realisations are all mapped to one background value. This compares ln L at the joint best fit of the
phase-transition scan with the floor as implemented and with those realisations given no binary contribution.
The difference is to be compared with the contour thresholds of a two-parameter scan, 1.15 (1 sigma) and
3.09 (2 sigma) in ln L.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from fopt_model import spectrum_fopt                                          # noqa: E402

NQ = 101
_Z = np.linspace(-4.0, 4.0, NQ)
_W = np.exp(-0.5 * _Z ** 2)
_W /= _W.sum()


def ln_like(data, med, sig, c, pt, mode):
    """mode 'floor': the warp as implemented. 'zero': the realisations below the floor contribute nothing."""
    zf = -0.5 / c
    total = 0.0
    for i in range(data.nbins):
        s = max(float(sig[i]), 1e-4)
        z = np.maximum(_Z, zf)
        b = 10.0 ** (np.log10(max(med[i], 1e-300)) + s * (z + c * z ** 2))
        if mode == "zero":
            b = np.where(_Z < zf, 0.0, b)
        v = float(np.dot(_W, np.exp(data.ln_density(data.to_log10rho(pt[i] + b, i), i))))
        if v <= 0:
            return -np.inf
        total += np.log(v)
    return total


if __name__ == "__main__":
    data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT, nbins=14)
    bg = bf.Background(data.freqs)
    scan = pd.read_csv(SCANS / "scan_holodeck_n50_ng15_cp_v2.csv")
    row = scan.loc[scan.lnL.idxmax()]
    print(f"{'case':22s} {'weight below the floor':>22s} {'ln L, floor':>12s} {'ln L, zero':>11s}"
          f" {'difference':>11s}")
    theta = [float(row[k]) for k in bf.PARAM_NAMES]
    med, sig, c = bg.predict(theta)
    pt = np.asarray(spectrum_fopt(data.freqs, row.alpha, row.betaH, row.Treh), float)
    a = ln_like(data, med, sig, float(c), pt, "floor")
    b = ln_like(data, med, sig, float(c), pt, "zero")
    w = float(norm.cdf(-0.5 / float(c)))
    print(f"{'joint best fit':22s} {100 * w:21.1f}% {a:12.4f} {b:11.4f} {a - b:+11.2e}")
