"""Sound-wave spectrum of a dark-sector phase transition, appendix A.1, as a function of the scanned
parameters (alpha, beta/H, T_reh)."""
from pathlib import Path

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

DATA_DIR = Path(__file__).resolve().parents[1] / "inputs"

OMEGA_GAMMA_H2 = 2.473e-5     # photon energy density today, h^2 Omega_gamma
H_TODAY = 3.9309363           # entropy degrees of freedom today
G_GAMMA = 2                   # energy degrees of freedom of the photons today
G_DARK = 2                    # light dark-sector degrees of freedom added to the SM bath
V_WALL, C_S = 1.0, 1 / np.sqrt(3)


def kappa_sw(alpha: float, vw: float, cs: float) -> float:
    """Sound-wave efficiency factor, fit formulas of Espinosa, Konstandin, No and Servant
    (arXiv:1004.4187)."""
    kA = vw ** (6 / 5) * 6.9 * alpha / (1.36 - 0.037 * np.sqrt(alpha) + alpha)
    kB = alpha ** (2 / 5) / (0.017 + (0.997 + alpha) ** (2 / 5))
    kC = np.sqrt(alpha) / (0.135 + np.sqrt(0.98 + alpha))
    kD = alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)
    dK = -0.9 * np.log(np.sqrt(alpha) / (1 + np.sqrt(alpha)))
    vJ = (cs + np.sqrt(alpha ** 2 + 2 * alpha / 3)) / (1 + alpha)
    lK = kC - kB - (vJ - cs) * dK

    if vw <= cs:
        num = cs ** (11 / 5) * kA * kB
        denom = (cs ** (11 / 5) - vw ** (11 / 5)) * kB + vw * cs ** (6 / 5) * kA
        return num / denom
    elif vw < vJ:
        return kB + (vw - cs) * dK + ((vw - cs) ** 3 / (vJ - cs) ** 3) * lK
    else:
        num = (vJ - 1) ** 3 * vJ ** (5 / 2) * vw ** (-5 / 2) * kC * kD
        denom = ((vJ - 1) ** 3 - (vw - 1) ** 3) * vJ ** (5 / 2) * kC + (vw - 1) ** 3 * kD
        return num / denom


# Standard Model energy and entropy degrees of freedom g(T), h(T): the table of Saikawa and Shirai (2018),
# inputs/geff_SM.dat (columns T in GeV, g, fit of g, h, ...), smoothed and spline-interpolated in log10 T.
_geff_sm_table = np.genfromtxt(DATA_DIR / "geff_SM.dat")
_Ie_geffSM_smooth = interpolate.CubicSpline(
    np.log10(_geff_sm_table[:, 0]), savgol_filter(_geff_sm_table[:, 1], 81, 3), bc_type="clamped")
_Is_geffSM_smooth = interpolate.CubicSpline(
    np.log10(_geff_sm_table[:, 0]), savgol_filter(_geff_sm_table[:, 3], 81, 3), bc_type="clamped")


def e_geffSM(TSM: float, CF: float) -> float:
    """Standard Model energy degrees of freedom at temperature TSM*CF in GeV."""
    return float(_Ie_geffSM_smooth(np.log10(TSM * CF)))


def s_geffSM(TSM: float, CF: float) -> float:
    """Standard Model entropy degrees of freedom at temperature TSM*CF in GeV."""
    return float(_Is_geffSM_smooth(np.log10(TSM * CF)))


def DBPL(f, f1, f2, Omega2, n1, n2, n3, a1, a2):
    """Doubly broken power law, normalised to Omega2 at f2 (eq. (2.8) of arXiv:2403.03723)."""
    S = (f / f1) ** n1 * (1 + (f / f1) ** a1) ** ((n2 - n1) / a1) * (1 + (f / f2) ** a2) ** ((n3 - n2) / a2)
    S_at_f2 = (f2 / f1) ** n1 * (1 + (f2 / f1) ** a1) ** ((n2 - n1) / a1) * 2 ** ((n3 - n2) / a2)
    return Omega2 * S / S_at_f2


def spectrum_fopt(f, alpha, betaH, Treh):
    """h^2 Omega_GW(f) today of the sound waves, for frequencies f in Hz, transition strength alpha,
    inverse duration beta/H and reheating temperature Treh in GeV."""
    kap_sw = kappa_sw(alpha, vw=V_WALL, cs=C_S)
    g = e_geffSM(Treh, 1) + G_DARK
    h = s_geffSM(Treh, 1) + G_DARK
    # mean bubble separation in Hubble units, at the percolation criterion 1 - P_f = 0.29
    RH = (8 * np.pi / 0.28957) ** (1 / 3) / betaH
    # redshift of the amplitude, and the Hubble rate at reheating redshifted to today in Hz
    Rh2 = OMEGA_GAMMA_H2 * (H_TODAY / h) ** (4. / 3.) * g / G_GAMMA
    H_p0_Hz = 1.65e-5 * (Treh / 100)
    H_p0_Hz *= (g / 100) ** (1 / 2)
    H_p0_Hz *= (100 / h) ** (1 / 3)
    # fraction of the released energy in bulk fluid motion, with the factor 0.6 of arXiv:2209.04369
    Ksw = 0.6 * kap_sw * (alpha / (1 + alpha))
    Delta_w = np.abs(V_WALL - C_S) / max(V_WALL, C_S)
    f1_sw_Hz = H_p0_Hz * 0.2 / RH
    f2_sw_Hz = H_p0_Hz * 0.5 / RH / Delta_w
    # suppression by the finite lifetime of the sound waves
    Htau_shell = RH / np.sqrt(0.6 * kap_sw * alpha / (1 + alpha) / (4 / 3))
    Htau_sw = min(1, Htau_shell)
    h2Omega_int_sw = Rh2 * 0.11 * Ksw ** 2 * Htau_sw * RH
    h2Omega_sw_peak_at_f2 = 1 / np.pi * (np.sqrt(2) + 2 * (f2_sw_Hz / f1_sw_Hz) /
                                          (1 + f2_sw_Hz ** 2 / f1_sw_Hz ** 2)) * h2Omega_int_sw
    return DBPL(f, f1_sw_Hz, f2_sw_Hz, h2Omega_sw_peak_at_f2, 3.0, 1.0, -3.0, 2, 4)
