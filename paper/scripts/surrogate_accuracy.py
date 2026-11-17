"""
How well the surrogate reproduces the three quantities the likelihood needs, and how much its errors shift ln L.

Part one measures the interpolation error against the 150 held-out holodeck models of qlib.jsonl,
each simulated with 2000 realisations and stored as a ladder of quantiles of log10 h_c^2 per bin.
The three quantities are the median, the half-width between the 16th and 84th percentiles, and the
skew parameter c of equation (2.7), the last fitted to the stored ladder about its own median,
which is how the skew map is fitted.

Part two propagates those errors into the phase-transition scan. The 90th percentile of each is applied
coherently, with the same sign in every bin and at every scan point. What is reported is how far ln L moves at
fixed parameters inside the 2 sigma region of the scan. Part three compares ln L from the surrogate with ln L
from each held-out model's own quantiles.

    BINARYFOG_NG15_MODELS=<unpacked model files> python surrogate_accuracy.py
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from paths import FREESPEC_ROOT, INPUTS, SCANS
import binaryfog as bf                                                        # noqa: E402
import binaryfog.background as B                                              # noqa: E402
from fopt_model import spectrum_fopt                                          # noqa: E402
import holodeck_runs                                                          # noqa: E402

QUANT = np.array(holodeck_runs.QUANT)
I16, I50, I84 = 4, 6, 8
SCAN = "scan_holodeck_n50_ng15_cp_v2.csv"
T2SIGMA = 3.09                      # 2 sigma threshold in ln L for two scanned parameters (Wilks)


def warp_ladder(half, c):
    """Equation (2.7) relative to the median, at the stored probability levels."""
    z = np.maximum(norm.ppf(QUANT), -1.0 / (2.0 * c))    # the map is held flat below its turnover
    return half * (z + c * z ** 2)


def fitted_skew(ladder, half):
    """The single c that best describes one bin's stored quantile ladder about its median."""
    return float(optimize.minimize_scalar(
        lambda c: float(np.mean((warp_ladder(half, float(c)) - ladder) ** 2)),
        bounds=(0.02, 0.45), method="bounded").x)


def interpolation_error(bg):
    pref = 2 * np.pi ** 2 / (3 * B.H100_HZ ** 2) * bg.freqs ** 2
    dm, dw, dc = [], [], []
    for line in open(INPUTS / "library/qlib.jsonl"):
        e = json.loads(line)
        omega, half_hat, c_hat = bg.predict([e["params"][k] for k in bf.PARAM_NAMES])
        med_hat = np.log10(np.asarray(omega, float) / pref)
        q = np.asarray(e["quantiles"], float)
        med, half = q[I50], 0.5 * (q[I84] - q[I16])
        dm.append(med_hat - med)
        dw.append(half_hat / half - 1.0)
        dc.append(c_hat - np.array([fitted_skew(q[:, i] - med[i], half[i])
                                    for i in range(q.shape[1])]))
    return [np.abs(np.concatenate(x)) for x in (dm, dw, dc)]


class Shifted(bf.Background):
    """The surrogate with a coherent offset applied to each of its three outputs."""

    def __init__(self, freqs, d_median=0.0, d_width=0.0, d_skew=0.0):
        super().__init__(freqs)
        self.offsets = (d_median, d_width, d_skew)

    def predict(self, theta):
        omega, sigma, skew = super().predict(theta)
        dm, dw, dc = self.offsets
        return (np.asarray(omega) * 10.0 ** dm, sigma * (1.0 + dw),
                float(np.clip(skew + dc, 0.02, 0.45)))


def cost_in_lnL(data, table, d_median=0.0, d_width=0.0, d_skew=0.0):
    """How far ln L moves at every scan point inside the 2 sigma region, at fixed parameters."""
    like = bf.Likelihood(data, Shifted(data.freqs, d_median, d_width, d_skew), marginalise=True)
    base = bf.Likelihood(data, bf.Background(data.freqs), marginalise=True)
    keep = table[table.lnL >= table.lnL.max() - T2SIGMA]
    out = []
    for _, r in keep.iterrows():
        theta = [float(r[k]) for k in bf.PARAM_NAMES]
        np_spec = np.asarray(spectrum_fopt(data.freqs, r.alpha, r.betaH, r.Treh), float)
        out.append(like.ln_like(np_spec, theta) - base.ln_like(np_spec, theta))
    return np.abs(np.asarray(out))


def likelihood_error(bg, data):
    """ln L from the surrogate against ln L from each held-out model's own simulated ladder."""
    pref = 2 * np.pi ** 2 / (3 * B.H100_HZ ** 2) * bg.freqs ** 2
    like = bf.Likelihood(data, bg, marginalise=True)
    truth = bf.Likelihood(data, Fixed(data.freqs), marginalise=True)
    out = []
    for line in open(INPUTS / "library/qlib.jsonl"):
        e = json.loads(line)
        theta = [e["params"][k] for k in bf.PARAM_NAMES]
        q = np.asarray(e["quantiles"], float)
        med, half = q[I50], 0.5 * (q[I84] - q[I16])
        c = float(np.median([fitted_skew(q[:, i] - med[i], half[i])
                             for i in range(q.shape[1])]))
        truth.background.set(10.0 ** med * pref, half, c)
        out.append(like.ln_like(None, theta) - truth.ln_like(None, theta))
    return np.asarray(out)


class Fixed(bf.Background):
    """The surrogate replaced by values set by hand, so the library itself can drive the fit."""

    def set(self, omega, sigma, skew):
        self._fixed = (np.asarray(omega, float), np.asarray(sigma, float), float(skew))

    def predict(self, theta):
        return self._fixed


if __name__ == "__main__":
    bg = bf.Background()
    dm, dw, dc = interpolation_error(bg)
    for name, x, unit in (("median", dm, " in log10 h_c^2"), ("width", 100 * dw, " per cent"),
                          ("skew c", dc, "")):
        print(f"  {name:8s}: typical error {np.median(x):.3f}{unit}")
    data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT)
    table = pd.read_csv(SCANS / SCAN)
    for label, kw in (("median", dict(d_median=float(np.percentile(dm, 90)))),
                      ("width", dict(d_width=float(np.percentile(dw, 90)))),
                      ("skew", dict(d_skew=float(np.percentile(dc, 90))))):
        c = cost_in_lnL(data, table, **kw)
        print(f"  a coherent 90th-percentile error in the {label:6s} moves ln L by "
              f"{np.median(c):.3f} typically, over {len(c)} scan points inside the 2 sigma region")
    dl = likelihood_error(bf.Background(data.freqs), data)
    print(f"  ln L from the surrogate against ln L from each held-out model's own ladder: "
          f"16 to 84 per cent {np.percentile(dl, 16):+.2f} to {np.percentile(dl, 84):+.2f}")
