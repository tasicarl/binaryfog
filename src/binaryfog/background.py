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
or 2 (skew) in theta, scaled to [-1, 1] over the prior box. The bend map predicts the value and
the slope of log10 h_c^2 at 5 nHz. A bend confines the slope to (-4/3, kappa - 4/3); where the
polynomial overshoots that range, saturate() takes it smoothly back inside.

Parameters, in the order of PARAM_NAMES, with the uniform prior box of arXiv:2306.16220:
    gsmf_phi0_log10     normalisation of the galaxy stellar mass function, dimensionless
    gsmf_mchar0_log10   log10 of its characteristic mass in solar masses
    mmb_mamp_log10      log10 of the black-hole mass at a bulge mass of 1e11 solar masses
    mmb_scatter_dex     scatter of that relation, in dex
    hard_time           total binary lifetime tau_f, in Gyr
    hard_gamma_inner    inner hardening index nu_inner, dimensionless

The maps were fitted to a library of 2000 holodeck models with 2000 realisations each, built with
holodeck at commit ca79e282 (the classic phenomenological parameter space). Accuracy on 150 library
models held out of the fit, typical (worst tenth): 0.05 dex (0.15 dex) on the median, 4 (15) per cent
on the width, 0.05 (0.17) on the skew. In ln L of the NANOGrav free spectrum the surrogate differs from
each held-out model's own realisations by -1.2 to +1.3 (16 to 84 per cent); a contour or a preference
sees only the part of that error which varies across a scan. The IPTA DR2 maps (maps="ipta2"), fitted the same way to the same
models on the 30 IPTA bins, are accurate to 0.10 to 0.15 dex on the median in the 13 lowest bins and
to about 0.5 rms in ln L of the IPTA free spectrum; in the lowest bin, 1.05 nHz, the error of the median
reaches a few realisation widths for quiet populations.
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
T_EPS = 0.03                      # width of the smooth saturation of the bend parameter, see saturate()
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


def saturate(r, eps=T_EPS):
    """The bend parameter t = (slope + 4/3) / kappa, confined to (0, 1) by the bend, from the value r the
    slope map predicts: softplus on both sides, so that ln L stays smooth where the polynomial overshoots."""
    lo = eps * np.logaddexp(0.0, np.asarray(r, float) / eps)
    return np.clip(1.0 - eps * np.logaddexp(0.0, (1.0 - lo) / eps), 1e-12, 1 - 1e-12)


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


def _interp_logf(lg, lgb, v):
    """The per-bin quantities as a smooth function of log frequency.

    These are fitted one bin at a time, so from bin to bin they carry their own fitting noise of a
    few hundredths of a dex. Joining them with straight lines breaks the slope at every bin, and
    threading a curve exactly through them reproduces the noise as a visible wobble; both show up
    in a spectrum drawn on a grid finer than the bins. A low-order polynomial in log frequency is
    fitted through them instead, which is smooth by construction and extends past the band without
    any special case. Over the fourteen NANOGrav bins a cubic reproduces the median correction to
    at most 0.014 in log10 h_c^2, three per cent, against the 0.049 the surrogate itself is
    accurate to and the several tenths of a dex the measured posterior spans in every bin.

    This is only ever used off the fitted grid, that is for drawing. A likelihood runs on the
    measured bins, where the stored per-bin values are used unchanged.
    """
    deg = min(3, len(lgb) - 1)
    if deg < 1:
        return np.full_like(lg, float(v[0]), dtype=float)
    return np.polyval(np.polyfit(lgb, v, deg), lg)


class Background:
    """The binary background on one frequency grid.

    >>> bg = Background(freqs_Hz)
    >>> median_h2omega, sigma_dex, skew = bg.predict(theta)

    The default maps were fitted on the NANOGrav 15 yr bins, f_i = i / 16.03 yr for i = 1 ... 14.
    maps="ipta2" selects the maps fitted in the same way to the same 2000 holodeck models on the 30
    bins of IPTA DR2, 1.0496 nHz + n x 1.0858 nHz (data/ipta2/maps6.npz). Frequencies that are bins
    of the chosen grid take the per-bin values exactly; other frequencies inside its range are
    interpolated and are accurate to the same degree; outside it the maps are extrapolation and are
    refused, unless `extrapolate` is set, which allows a factor of two beyond each end and is meant
    for drawing a continuous curve rather than for a fit.
    """

    F_NG15 = np.arange(1, 15) / (16.03 * 31557600.0)
    param_names = PARAM_NAMES
    prior_box = PRIOR_BOX

    def __init__(self, freqs_Hz=None, extrapolate=False, maps="ng15"):
        self.maps = maps
        if maps == "ng15":
            b = np.load(DATA / "bend_map_final.npz", allow_pickle=True)
            m = np.load(DATA / "median_merged.npz", allow_pickle=True)
            w = np.load(DATA / "width_map.npz", allow_pickle=True)
            s = np.load(DATA / "skew_map.npz", allow_pickle=True)
            self._W_bend = b["W"]
            self._W_med, self._deg_med = m["W"], int(m["degree"])
            self._W_wid, self._deg_wid = w["W"], int(w["degree"])
            self._W_skew, self._deg_skew = s["W"], int(s["degree"])
            self._wscale = float(np.load(DATA / "width_scale.npy")[0])
            self._fgrid = self.F_NG15
        else:
            path = DATA / maps / "maps6.npz"
            if not path.is_file():
                raise FileNotFoundError(f"no maps for the grid {maps!r} at {path}")
            m = np.load(path, allow_pickle=True)
            self._W_bend = m["W_bend"]
            self._W_med, self._deg_med = m["W_med"], int(m["deg_med"])
            self._W_wid, self._deg_wid = m["W_wid"], int(m["deg_wid"])
            self._W_skew, self._deg_skew = m["W_skew"], int(m["deg_skew"])
            self._wscale = float(m["width_scale"])
            self._fgrid = np.asarray(m["freqs"], float)
        self.freqs = np.asarray(self._fgrid if freqs_Hz is None else freqs_Hz, float)
        lo, hi = (0.5, 2.0) if extrapolate else (0.98, 1.02)
        if self.freqs.min() < lo * self._fgrid[0] or self.freqs.max() > hi * self._fgrid[-1]:
            raise ValueError(f"the {maps} maps cover {self._fgrid[0]*1e9:.2f} to {self._fgrid[-1]*1e9:.1f} nHz; "
                             "other bands need their own library (see build_library6_ipta.py). Pass "
                             "extrapolate=True to go a factor of two beyond each end for plotting.")
        self.extrapolated = bool(extrapolate)
        # frequencies that are bins of the fitted grid (to the sixth digit, which is how far the data
        # products and the nominal grid differ) take the per-bin values exactly
        near = np.abs(np.log(self.freqs[:, None] / self._fgrid[None, :]))
        self._idx = np.argmin(near, axis=1) if np.all(near.min(axis=1) < 1e-4) else None

    def predict(self, theta):
        """theta: the six parameters. Returns (median h^2 Omega per bin, sigma in dex, skew c).

        The width and the skew are per bin quantities of the library grid; on a different grid
        they are taken from the nearest NANOGrav bin, which is why a dedicated library is needed
        for a different PTA.
        """
        th = np.atleast_2d(np.asarray(theta, float))
        o = design(th, 4) @ self._W_bend               # log10 h_c^2 and its slope at FREF
        t = saturate((o[:, 1] + 4 / 3) / KAPPA)
        log10_fbend = np.log10(FREF_HZ) + np.log10(t / (1 - t)) / KAPPA
        log10A = (0.5 * o[:, 0] + (2 / 3) * np.log10(FREF_HZ / FYR_HZ)
                  + 0.5 * np.log10(1 + 10.0 ** (KAPPA * (log10_fbend - np.log10(FREF_HZ)))))
        y = bend_log10hc2(self.freqs, log10A[0], log10_fbend[0])
        corr = (design(th, self._deg_med) @ self._W_med)[0]
        sigma = self._wscale * 10.0 ** (design(th, self._deg_wid) @ self._W_wid)[0]
        if self._idx is None:
            # The median correction and the width are fitted per library bin. Off that grid they
            # are interpolated linearly in log frequency, and continued with the slope of the end
            # pair beyond it. Taking the nearest bin instead makes both step from bin to bin, which
            # shows up as a visibly jagged curve as soon as the grid is finer than the bins; simply
            # holding the end value, which is what np.interp does, leaves the slope discontinuous
            # at the edge of the band, and that shows up as a kink in a drawn spectrum.
            lg, lgb = np.log10(self.freqs), np.log10(self._fgrid)
            corr = _interp_logf(lg, lgb, corr[: len(self._fgrid)])
            sigma = _interp_logf(lg, lgb, sigma[: len(self._fgrid)])
        else:
            corr = corr[self._idx]
            sigma = sigma[self._idx]
        skew = float(np.clip(design(th, self._deg_skew) @ self._W_skew, 0.02, 0.45)[0])
        return hc2_to_h2omega(self.freqs, y + corr), sigma, skew

    def on_frequencies(self, freqs_Hz):
        """The same background evaluated on another frequency grid, inside the fitted band.

        A prior_box set on the instance, for instance to restrict a scan to part of the library
        box, is carried over. A Likelihood rebuilds the background on the data's own frequencies,
        and dropping the restriction there would silently run a scan the caller did not ask for.
        """
        out = type(self)(freqs_Hz, maps=self.maps)
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
