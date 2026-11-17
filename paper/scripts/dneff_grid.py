"""Delta N_eff carried by the gravitational waves of the phase transition on the (alpha, beta/H) grid of
figure 5, at the T_reh the binaryfog profile chose at each point; the hatched region of the figure is where
it exceeds 0.22 (Yeh et al. 2022).

Delta N_eff = (8/7) (11/4)^(4/3) / (h^2 Omega_gamma) x integral of h^2 Omega_GW(f) d ln f, by Simpson's rule
on 100000 points spaced logarithmically from 10^-24.9 Hz to 10^23.0 Hz. This grid is
np.logspace(ln 1.5e-11, ln 1e10), the one of TransitionListener's calc_DNeff_GW, which passes natural
logarithms as decimal exponents; it contains the whole spectrum at every grid point.

JOBS sets the number of worker processes (default 28).
"""
import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import integrate

from paths import SCANS
from baseline import TAG                                                 # noqa: E402
from fopt_model import spectrum_fopt                                     # noqa: E402

THRESHOLD = 0.22
OMEGA_GAMMA_H2 = 2.473e-5
FREQS = np.logspace(np.log(1.5e-11), np.log(1e10), 100000)


def dneff(row):
    alpha, betaH, treh = row
    h2Omega = integrate.simpson(spectrum_fopt(FREQS, alpha, betaH, treh) / FREQS, FREQS)
    return float((8 / 7) * (11 / 4) ** (4 / 3) * h2Omega / OMEGA_GAMMA_H2)


if __name__ == "__main__":
    d = pd.read_csv(SCANS / f"scan_holodeck_n120_a-2to3{TAG}.csv")
    t0 = time.perf_counter()
    with mp.Pool(int(os.environ.get("JOBS", 28))) as pool:
        vals = pool.map(dneff, list(zip(d.alpha, d.betaH, d.Treh)), chunksize=50)
    out = d[["alpha", "betaH", "Treh"]].assign(DNeff_GW=vals)
    out.to_csv(SCANS / f"dneff_n120_a-2to3{TAG}.csv", index=False)
    print(f"done in {time.perf_counter()-t0:.0f} s; DNeff range {min(vals):.2e}-{max(vals):.2e}; "
          f"excluded {np.mean(np.array(vals) > THRESHOLD)*100:.1f}% of the grid")
