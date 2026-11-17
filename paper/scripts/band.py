"""The 1 sigma band in beta/H at fixed alpha, located between grid nodes rather than at one.

On a grid, the last node inside a region can only fall short of its edge, by up to one step. The edge
here is instead the point where the profile likelihood crosses the threshold, found by linear
interpolation in ln L against log10(beta/H) between the last node inside and the first node outside.
"""
import numpy as np
import pandas as pd

T1 = 1.15       # Wilks 1 sigma threshold in ln L for two scanned parameters


def profile_grid(path, value="lnL"):
    """log10 alpha, log10 beta/H and the profile ln L of a scan, as a grid."""
    d = pd.read_csv(path)
    p = d.pivot(index="alpha", columns="betaH", values=value)
    return np.log10(p.index.values), np.log10(p.columns.values), p.to_numpy(), d


def band_open(path, alpha=1e3, thr=T1, value="lnL"):
    """The band at the alpha of the scan closest to `alpha`: its two edges, and whether each edge is
    the end of the scanned range rather than a crossing."""
    a, b, L, _ = profile_grid(path, value)
    return edges(a, b, L, alpha, thr)


def edges(a, b, L, alpha, thr):
    """As band_open, for a grid L[alpha, beta/H] with log10 alpha = a and log10 beta/H = b."""
    row = L[int(np.argmin(np.abs(a - np.log10(alpha))))]
    level = np.nanmax(L) - thr
    ok = np.nan_to_num(row, nan=-np.inf) >= level
    if not ok.any():
        return np.nan, np.nan, (False, False)
    # The run of nodes inside the region that contains the best point of the row: the band is the part
    # connected to the best fit.
    i = j = int(np.argmax(np.where(ok, row, -np.inf)))
    while i > 0 and ok[i - 1]:
        i -= 1
    while j < len(row) - 1 and ok[j + 1]:
        j += 1

    def cross(k, nxt):
        if not 0 <= nxt < len(b) or not np.isfinite(row[nxt]):
            return b[k], True                      # the band reaches the end of the scan
        return b[k] + (level - row[k]) * (b[nxt] - b[k]) / (row[nxt] - row[k]), False

    lo, lo_open = cross(i, i - 1)
    hi, hi_open = cross(j, j + 1)
    return 10 ** lo, 10 ** hi, (lo_open, hi_open)
