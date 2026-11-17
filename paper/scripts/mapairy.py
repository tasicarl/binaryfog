"""
The map-Airy law for the distribution of h_c^2 over realisations (Ali-Haimoud 2026, arXiv:2604.19701).

For an effective source count N, the PDF of y = h_c^2 / mean(h_c^2) is

    P(y) = N^(1/3) * Pmap( N^(1/3) (y - 1) ),
    Pmap(x) = -2 exp(2x^3/3) [ x Ai(x^2) + Ai'(x^2) ]        (reflected map-Airy)
"""
import numpy as np
from scipy.special import airye


def map_airy_pdf(x):
    """Reflected map-Airy density (Ali-Haimoud 2026, arXiv:2604.19701)."""
    # with the exponentially scaled Airy functions, exp(2x^3/3) Ai(x^2) = airye(x^2) for x > 0, so the
    # loud tail stays finite where the unscaled form underflows to zero
    x = np.asarray(x, dtype=float)
    eai, eaip, _, _ = airye(x ** 2)
    val = -2.0 * np.exp((2 * x ** 3 - 2 * np.abs(x) ** 3) / 3) * (x * eai + eaip)
    return np.maximum(val, 0.0)


def pdf_y(y, N):
    """PDF of y = h_c^2 / mean, for effective source count N."""
    return N ** (1 / 3) * map_airy_pdf(N ** (1 / 3) * (np.asarray(y) - 1.0))


def _log10_cdf(N):
    """The map-Airy law of y on a grid, as (log10(y / median), cdf)."""
    # a logarithmic grid in y, so that small N, whose median lies far below the mean, stays resolved
    w = N ** (-1 / 3)
    yg = np.geomspace(1e-14, 1 + 80 * w, 80000)
    p = pdf_y(yg, N)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(yg))])
    c = c / c[-1]
    mid = float(np.interp(0.5, c, yg))
    return np.log10(yg / mid), c


def matched(half, tol=1e-4):
    """The map-Airy law matched to a half-width in log10 h_c^2, half the distance between its sixteenth and
    eighty-fourth percentiles: N by bisection in log10 N, since the width falls monotonically with N.
    Returns (N, log10(y / median), cdf)."""
    def width(lgn):
        lg, c = _log10_cdf(10.0 ** lgn)
        q16, q84 = np.interp([0.16, 0.84], c, lg)
        return 0.5 * (q84 - q16)
    lo, hi = -9.0, 9.0
    if not width(lo) > half > width(hi):
        # the family saturates at a finite half-width as N -> 0: wider distributions have no match
        raise ValueError(f"half-width {half:.3f} outside the map-Airy range")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if width(mid) > half else (lo, mid)
    N = 10.0 ** (0.5 * (lo + hi))
    lg, c = _log10_cdf(N)
    return N, lg, c
