"""
holodeck at the two best fits of the phase-transition scan on the CURN free spectrum, at the tau_f = 0.1 Gyr edge
of the library: the binaries alone and the binaries with a transition. 5000 realisations each.
Writes scans/bestfit_holodeck.npz, which plot_realisations.py and realisations_numbers.py read.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
import numpy as np
from paths import SCANS
from holodeck_runs import realisations

# best fits of the binaries alone and with the transition in scan_holodeck_n120_a-2to3_ng15_cp_v2.csv; the
# likelihood is flat along a ridge, so the points are fixed explicitly (parameters in the order of PARAM_NAMES)
POINTS = {"alone": [-2.057398007857129, 11.285746106643874, 8.682245844210364, 0.3211293982177308,
                    0.1000000000009868, -0.4578878583710465],
          "with_pt": [-2.5596156320338133, 10.899643450725389, 8.599910178483668, 0.3198162697732506,
                      0.1000001364677706, -0.2493926883329171]}
PT = [49.828635146473054, 45.57861199974174, 0.0208007760760785]     # alpha, beta/H, T_reh in GeV


def run(name):
    return name, realisations(POINTS[name], nreals=5000)


if __name__ == "__main__":
    with mp.Pool(2) as pool:
        res = dict(pool.map(run, list(POINTS)))
    np.savez(SCANS / "bestfit_holodeck.npz", **{f"log10hc2_{k}": v for k, v in res.items()},
             **{f"theta_{k}": np.array(v) for k, v in POINTS.items()}, pt=np.array(PT))
    print("wrote bestfit_holodeck.npz")
