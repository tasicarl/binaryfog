"""Numbers quoted in the text with figure 3: at the best fit of the binaries alone (CURN free spectrum, NANOGrav
priors, bestfit_holodeck.npz), the width per bin, the ratio of mean to median of h_c^2 in bin 14, and in bin 14
the Kolmogorov-Smirnov and 1-Wasserstein distances of the warp, the log-normal and the map-Airy law to the 5000
realisations."""
import numpy as np
from scipy.stats import norm
from paths import SCANS
import binaryfog as bf
from mapairy import matched


def ks(a, b):
    a = np.sort(a)
    b = np.sort(b)
    x = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, x, "right") / a.size - np.searchsorted(b, x, "right") / b.size))


z = np.load(SCANS / "bestfit_holodeck.npz")
X = z["log10hc2_alone"]
c = float(np.ravel(bf.Background().predict(z["theta_alone"])[2])[0])
q = (np.arange(100000) + 0.5) / 100000
zq = norm.ppf(q)
half = 0.5 * (np.percentile(X, 84, axis=1) - np.percentile(X, 16, axis=1))
print("c", round(c, 3), "widths", np.round(half, 3).tolist())
print("bin 14 mean/median of hc2:", round(np.mean(10 ** X[13]) / 10 ** np.median(X[13]), 2))
i = 13
y = X[i] - np.median(X[i])
h = half[i]
zz = np.maximum(zq, -0.5 / c)
_, lgm, cdf = matched(h)
laws = {"warp": h * (zz + c * zz ** 2), "lognormal": h * zq, "mapairy": np.interp(q, cdf, lgm)}
qs = (np.arange(y.size) + 0.5) / y.size
for k, xs in laws.items():
    w1 = np.mean(np.abs(np.sort(y) - np.quantile(xs, qs)))
    print(f"bin {i+1} {k:9s} KS {ks(y, xs):.3f}  W1 {w1:.3f}")
