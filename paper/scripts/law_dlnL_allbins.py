"""Per-bin ln L(law) - ln L(5000 realisations) for the warp, the log-normal and the map-Airy law, in all 14 bins,
at the two holodeck points of bestfit_holodeck.npz (binaries alone; binaries with a phase transition), CURN free
spectrum, NANOGrav priors. At the second point the phase-transition spectrum (fopt_model.py) is added."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.stats import norm
from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf
from binaryfog.background import hc2_to_h2omega
from mapairy import matched
from fopt_model import spectrum_fopt

data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT)
z = np.load(SCANS / "bestfit_holodeck.npz", allow_pickle=True)
q = (np.arange(100000) + 0.5) / 100000
zq = norm.ppf(q)
for tag in ("alone", "with_pt"):
    X, th = z["log10hc2_" + tag], z["theta_" + tag]
    C = float(np.ravel(bf.Background().predict(th)[2])[0])
    npc = np.zeros(data.nbins) if tag == "alone" else spectrum_fopt(data.freqs, *z["pt"])
    rows = []
    for i in range(X.shape[0]):
        y = X[i]
        med = np.median(y)
        half = 0.5 * (np.percentile(y, 84) - np.percentile(y, 16))

        def L(x):
            lg = med + x
            return np.exp(data.ln_density(data.to_log10rho(
                npc[i] + hc2_to_h2omega(np.full(np.shape(x), data.freqs[i]), lg), i), i))
        Lr = L(y - med).mean()
        zz = np.maximum(zq, -0.5 / C)
        _, lgm, cdf = matched(half)
        xs = {"warp": half * (zz + C * zz ** 2), "lognormal": half * zq,
              "mapairy": np.interp(q, cdf, lgm)}
        d = {k: np.log(L(v).mean() / Lr) for k, v in xs.items()}
        rows.append((d["warp"], d["lognormal"], d["mapairy"]))
        print(f"{tag:8s} bin {i+1:2d}: warp {d['warp']:+.3f} logn {d['lognormal']:+.3f} mapA {d['mapairy']:+.3f}")
    r = np.array(rows)
    print(f"{tag}: c={C:.3f}  sums warp {r[:,0].sum():+.3f} logn {r[:,1].sum():+.3f} mapA {r[:,2].sum():+.3f}")
