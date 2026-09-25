"""
The supermassive black hole binary background, as a fast stand-in for a holodeck library.

A holodeck population model with parameters theta produces, in every PTA frequency bin, a
distribution of backgrounds rather than one spectrum: the binaries are a finite, random set of
sources, so each universe realises a different spectrum. This module reproduces that distribution
from four fitted maps, without calling holodeck:

    median   log10 h_c^2 of the median realisation, as a bend template plus a correction
    width    the 16 to 84 per cent half-width sigma of log10 h_c^2, in dex
    skew     the one number c of the warp  log10 h_c^2 = median + sigma (z + c z^2),  z normal

The bend template is
    h_c(f) = A (f/f_yr)^(-2/3) [1 + (f_bend/f)^kappa]^(-1/2),   kappa = 1.8 fixed,
with A and f_bend predicted from theta, and the maps are polynomials of degree 4 (median, width)
or 2 (skew) in theta, scaled to [-1, 1] over the prior box.

Parameters, in the order of PARAM_NAMES, with the uniform prior box of arXiv:2306.16220:
    gsmf_phi0_log10     normalisation of the galaxy stellar mass function, dimensionless
    gsmf_mchar0_log10   log10 of its characteristic mass in solar masses
    mmb_mamp_log10      log10 of the black-hole mass at a bulge mass of 1e11 solar masses
    mmb_scatter_dex     scatter of that relation, in dex
    hard_time           total binary lifetime tau_f, in Gyr
    hard_gamma_inner    inner hardening index nu_inner, dimensionless

The maps were fitted to a library of 2000 holodeck models with 2000 realisations each, built with
holodeck at commit ca79e282 (the classic phenomenological parameter space). Accuracy on held-out
models: 0.02 dex on the median, 5 per cent on the width, 0.02 on the skew.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

PARAM_NAMES = ["gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10",
               "mmb_scatter_dex", "hard_time", "hard_gamma_inner"]
PRIOR_BOX = {"gsmf_phi0_log10": (-3.5, -1.5), "gsmf_mchar0_log10": (10.5, 12.5),
             "mmb_mamp_log10": (7.6, 9.0), "mmb_scatter_dex": (0.0, 0.9),
             "hard_time": (0.1, 11.0), "hard_gamma_inner": (-1.5, 0.0)}

KAPPA = 1.8                       # fixed bend sharpness of the template
FREF_HZ = 5e-9                    # reference frequency of the bend map, Hz
FYR_HZ = 1.0 / 31557600.0         # 1/yr in Hz
H100_HZ = 100 / 3.0857e19         # 100 km/s/Mpc in Hz, the value the published fit used
_LO = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES])
_HI = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])


@lru_cache(maxsize=8)
def _exponents(degree):
    """Exponent matrix of the polynomial basis, in the order the maps were fitted in: the constant
    term, then for each degree the multisets of variables in the order of
    itertools.combinations_with_replacement. The order is part of the stored coefficients, so it
    must not be changed without refitting."""
    from itertools import combinations_with_replacement
    nv = len(PARAM_NAMES)
    rows = [np.zeros(nv, dtype=int)]
    for d in range(1, degree + 1):
        for c in combinations_with_replacement(range(nv), d):
            e = np.zeros(nv, dtype=int)
            for j in c:
                e[j] += 1
            rows.append(e)
    return np.array(rows, dtype=float)


def design(theta, degree):
    """Polynomial design matrix, theta scaled to [-1, 1] over the prior box."""
    theta = np.atleast_2d(np.asarray(theta, float))
    u = 2 * (theta - _LO) / (_HI - _LO) - 1
    return np.prod(u[:, None, :] ** _exponents(degree)[None, :, :], axis=2)


def bend_log10hc2(f_Hz, log10A, log10_fbend, kappa=KAPPA):
    """log10 h_c^2 of the bend template."""
    x = kappa * (log10_fbend - np.log10(f_Hz))
    log1p10 = np.where(x > 8, x, np.log10(1 + 10.0 ** np.clip(x, -300, 8)))
    return 2 * (log10A - (2 / 3) * np.log10(f_Hz / FYR_HZ) - 0.5 * log1p10)


def hc2_to_h2omega(f_Hz, log10hc2):
    """h^2 Omega_GW = 2 pi^2 / (3 H100^2) f^2 h_c^2."""
    return 2 * np.pi ** 2 / (3 * H100_HZ ** 2) * np.asarray(f_Hz) ** 2 * 10.0 ** log10hc2


class Background:
    """The binary background on one frequency grid.

    >>> bg = Background(freqs_Hz)
    >>> median_h2omega, sigma_dex, skew = bg.predict(theta)

    The maps were fitted on the NANOGrav 15 yr bins, f_i = i / 16.03 yr for i = 1 ... 14, which is
    the default grid. Other frequencies inside that range are interpolated by the template and are
    accurate to the same degree; outside it the maps are extrapolation and are refused, unless
    `extrapolate` is set, which allows a factor of two beyond each end and is meant for drawing a
    continuous curve rather than for a fit.
    """

    F_NG15 = np.arange(1, 15) / (16.03 * 31557600.0)
    param_names = PARAM_NAMES
    prior_box = PRIOR_BOX

    def __init__(self, freqs_Hz=None, extrapolate=False):
        self.freqs = np.asarray(self.F_NG15 if freqs_Hz is None else freqs_Hz, float)
        lo, hi = (0.5, 2.0) if extrapolate else (0.98, 1.02)
        if self.freqs.min() < lo * self.F_NG15[0] or self.freqs.max() > hi * self.F_NG15[-1]:
            raise ValueError("the fitted maps cover 1.98 to 27.7 nHz, the NANOGrav 15 yr band; "
                             "other bands need their own library (see build_library8.py). Pass "
                             "extrapolate=True to go a factor of two beyond each end for plotting.")
        self.extrapolated = bool(extrapolate)
        b = np.load(DATA / "bend_map_final.npz", allow_pickle=True)
        m = np.load(DATA / "median_merged.npz", allow_pickle=True)
        w = np.load(DATA / "width_map.npz", allow_pickle=True)
        s = np.load(DATA / "skew_map.npz", allow_pickle=True)
        self._W_bend = b["W"]
        self._W_med, self._deg_med = m["W"], int(m["degree"])
        self._W_wid, self._deg_wid = w["W"], int(w["degree"])
        self._W_skew, self._deg_skew = s["W"], int(s["degree"])
        self._wscale = float(np.load(DATA / "width_scale.npy")[0])
        if len(self.freqs) != len(self.F_NG15) or not np.allclose(self.freqs, self.F_NG15):
            self._on_ng15_grid = False
        else:
            self._on_ng15_grid = True

    def predict(self, theta):
        """theta: the six parameters. Returns (median h^2 Omega per bin, sigma in dex, skew c).

        The width and the skew are per bin quantities of the library grid; on a different grid
        they are taken from the nearest NANOGrav bin, which is why a dedicated library is needed
        for a different PTA.
        """
        th = np.atleast_2d(np.asarray(theta, float))
        o = design(th, 4) @ self._W_bend
        t = np.clip((o[:, 1] + 4 / 3) / KAPPA, 1e-6, 1 - 1e-6)
        log10_fbend = np.log10(FREF_HZ) + np.log10(t / (1 - t)) / KAPPA
        log10A = (0.5 * o[:, 0] + (2 / 3) * np.log10(FREF_HZ / FYR_HZ)
                  + 0.5 * np.log10(1 + 10.0 ** (KAPPA * (log10_fbend - np.log10(FREF_HZ)))))
        y = bend_log10hc2(self.freqs, log10A[0], log10_fbend[0])
        corr = (design(th, self._deg_med) @ self._W_med)[0]
        sigma = self._wscale * 10.0 ** (design(th, self._deg_wid) @ self._W_wid)[0]
        if not self._on_ng15_grid:
            # The median correction and the width are fitted per library bin. Off that grid they
            # are interpolated linearly in log frequency and held at the end values beyond it.
            # Taking the nearest bin instead makes both step from bin to bin, which shows up as a
            # visibly jagged curve as soon as the grid is finer than the bins.
            lg, lgb = np.log10(self.freqs), np.log10(self.F_NG15)
            corr = np.interp(lg, lgb, corr[: len(self.F_NG15)])
            sigma = np.interp(lg, lgb, sigma[: len(self.F_NG15)])
        else:
            corr = corr[: len(self.freqs)]
            sigma = sigma[: len(self.freqs)]
        skew = float(np.clip(design(th, self._deg_skew) @ self._W_skew, 0.02, 0.45)[0])
        return hc2_to_h2omega(self.freqs, y + corr), sigma, skew

    def on_frequencies(self, freqs_Hz):
        """The same background evaluated on another frequency grid, inside the fitted band.

        A prior_box set on the instance, for instance to restrict a scan to part of the library
        box, is carried over. A Likelihood rebuilds the background on the data's own frequencies,
        and dropping the restriction there would silently run a scan the caller did not ask for.
        """
        out = type(self)(freqs_Hz)
        if "prior_box" in self.__dict__:
            out.prior_box = dict(self.prior_box)
        return out

    def in_box(self, theta):
        """Is theta inside the box the maps were fitted over? Outside it they are extrapolation.

        This is the training box and not self.prior_box: narrowing the latter restricts a scan,
        which does not make the maps any less valid where they are still evaluated.
        """
        th = np.asarray(theta, float)
        return bool(np.all(th >= _LO) and np.all(th <= _HI))
