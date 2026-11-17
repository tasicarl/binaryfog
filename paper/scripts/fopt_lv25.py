"""
Phase-transition spectrum in the strongly supercooled limit (section 5.1).

The template is the dissipative bulk-flow fit of Lewicki, Vaskonen and collaborators with the cosmic
expansion during the transition included, continued smoothly as f^3 below the horizon frequency, written
out from arXiv:2511.15687.

Those simulations assume a vacuum-dominated universe before the transition, alpha >> 1, with all the
released energy in the shells. The energy fraction is therefore K = kappa alpha/(1+alpha) with kappa = 1
and alpha -> infinity, K = 1, and the spectrum depends on beta/H and T_reh alone; the alpha argument is
accepted for a common call signature and ignored. The fits cover beta/H = 4 to about 64 and are
extrapolated outside that range.
"""
import numpy as np

from fopt_model import e_geffSM, s_geffSM

OMEGA_GAMMA_H2 = 2.473e-5     # photon energy density today, h^2 Omega_gamma
H_EFF_TODAY = 3.9309363       # entropy degrees of freedom today (Saikawa and Shirai, arXiv:1803.01038)
G_TODAY = 2                   # energy degrees of freedom of the photons today
G_DARK = 2                    # degrees of freedom added to the SM bath after reheating


def _shape(x, a, b, c):
    """Broken power law of eq. (4.1) of arXiv:2208.11697, equal to one at the peak x = f/f_p = 1."""
    return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c


def _flat(f, betaH, R, f_unit):
    """The fit with expansion, before the causality tail: amplitude eq. (23), peak eq. (22) of
    arXiv:2511.15687, dissipative shape exponents a = b = c = 2."""
    A = max(0.0, 0.06 * (1 + 0.8 * (1 - np.exp(np.sqrt(1 / betaH)))))
    f_p = f_unit * 0.7 * betaH * (1 + 1.8 * betaH ** (-1.2))
    return R * A / betaH ** 2 * _shape(f / f_p, 2.0, 2.0, 2.0)


def spectrum_lv25(f, alpha, betaH, Treh):
    """h^2 Omega_GW(f) today for beta/H and the reheating temperature Treh in GeV; alpha is ignored
    (alpha -> infinity, kappa = 1)."""
    f = np.asarray(f, float)
    g = e_geffSM(Treh, 1) + G_DARK
    h = s_geffSM(Treh, 1) + G_DARK
    # redshift of the amplitude, and the Hubble rate at reheating redshifted to today in Hz
    R = OMEGA_GAMMA_H2 * (H_EFF_TODAY / h) ** (4 / 3) * g / G_TODAY
    H0 = 1.65e-5 * (Treh / 100) * (g / 100) ** 0.5 * (100 / h) ** (1 / 3)
    f_unit = H0 / (2 * np.pi)
    out = _flat(f, betaH, R, f_unit)
    # causality tail, eq. (21): below f_H the spectrum steepens to f^3 without a kink; a is the
    # logarithmic slope of the fit at f_H
    f_H = f_unit * max(1e-3, 1.0 - 0.6 * betaH ** (-0.67))
    eps = 1e-3
    lo, hi = _flat(np.array([f_H * (1 - eps), f_H * (1 + eps)]), betaH, R, f_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = float(np.log(hi / lo) / np.log((1 + eps) / (1 - eps)))
    if not np.isfinite(a):
        a = 3.0
    return out * (1.0 + (f_H / f) ** 3) ** (-max(0.0, 3.0 - a) / 3.0)
