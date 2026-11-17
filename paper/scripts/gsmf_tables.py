"""Published inputs of the survey-average priors of the binaryfog paper.

Every number is transcribed from the arXiv LaTeX source of the paper named in the comment, with the
table or equation it comes from.

Galaxy stellar mass functions (GSMF). All are total samples (star-forming plus quiescent), Chabrier
IMF, given as single or double Schechter functions per dex of stellar mass,
    Phi(log m) = ln10 * sum_i Phi_i * x^(alpha_i + 1) * exp(-x),   x = m / M*,
which is also holodeck's form. Phi is in Mpc^-3 dex^-1.
"""
import numpy as np

LN10 = np.log(10.0)


def dschech(logm, logMs, a1, logP1, a2=None, logP2=None):
    """log10 Phi per dex for a single (a2 None) or double Schechter function."""
    x = 10.0 ** (logm - logMs)
    out = 10.0 ** logP1 * x ** (a1 + 1)
    if a2 is not None:
        out = out + 10.0 ** logP2 * x ** (a2 + 1)
    return np.log10(LN10 * out * np.exp(-x))


# ---------------------------------------------------------------- the six data sets published since 2014
# Davidzon et al. 2017, COSMOS2015, arXiv:1701.02734, Table 1 (tab_schpar), total sample.
# Double Schechter for z <= 3. Phi_i in 1e-3 Mpc^-3 (h70 = 1). Eddington bias removed: the tabulated
# parameters are deconvolved with the Ilbert+13 kernel (Sect. 5). Chabrier IMF (Sect. 4.1).
# columns: zlo, zhi, logM*, alpha1, Phi1, alpha2, Phi2
D17 = [(0.2, 0.5, 10.78, -1.38, 1.187, -0.43, 1.92),
       (0.5, 0.8, 10.77, -1.36, 1.070, 0.03, 1.68),
       (0.8, 1.1, 10.56, -1.31, 1.428, 0.51, 2.19),
       (1.1, 1.5, 10.62, -1.28, 1.069, 0.29, 1.21),
       (1.5, 2.0, 10.51, -1.28, 0.969, 0.82, 0.64),
       (2.0, 2.5, 10.60, -1.57, 0.295, 0.07, 0.45),
       (2.5, 3.0, 10.59, -1.67, 0.228, -0.08, 0.21)]

# Weaver et al. 2023, COSMOS2020, arXiv:2212.02512, Table C.1 (fit_total), likelihood fit, median of
# the posterior. Phi_i in 1e-3 Mpc^-3 dex^-1. Single Schechter (alpha = -1.46) above z = 2.5. The
# fit convolves the Schechter function with the mass-error kernel, so the tabulated parameters are
# the intrinsic ones (Sect. 5.3). Chabrier IMF (end of Sect. 1).
C20 = [(0.2, 0.5, 10.89, -1.42, 0.73, -0.46, 1.09),
       (0.5, 0.8, 10.96, -1.39, 0.66, -0.61, 0.83),
       (0.8, 1.1, 11.02, -1.32, 0.84, -0.63, 0.66),
       (1.1, 1.5, 11.00, -1.33, 0.72, -0.51, 0.34),
       (1.5, 2.0, 10.86, -1.48, 0.29, -0.43, 0.64),
       (2.0, 2.5, 10.78, -1.46, 0.27, 0.07, 0.27),
       (2.5, 3.0, 10.97, -1.46, 0.24, None, None),
       (3.0, 3.5, 10.83, -1.46, 0.21, None, None)]

# McLeod et al. 2021, arXiv:2009.03176. Table 5: intrinsic double Schechter fits, Eddington bias
# deconvolved with sigma_M = 0.15 dex (Sects. 4.3-4.4). BC03 with Chabrier IMF (Sect. 3.4).
# columns: zlo, zhi, logM*, logPhi1, alpha1, logPhi2, alpha2
MCL_INT = [(0.25, 0.75, 10.64, -2.63, -0.25, -3.11, -1.49),
           (0.75, 1.25, 10.51, -2.67, 0.08, -3.07, -1.49),
           (1.25, 1.75, 10.54, -2.83, -0.07, -3.32, -1.60),
           (1.75, 2.25, 10.56, -3.05, -0.06, -3.51, -1.63),
           (2.25, 2.75, 10.55, -3.28, 0.02, -3.50, -1.66),
           (2.75, 3.75, 10.64, -4.08, 0.35, -3.74, -1.76)]

# Leja et al. 2020, arXiv:1910.04168, continuity model, Appendix B code: log Phi1, log Phi2 and
# log M* at z = 0.2, 1.6, 3.0, joined by the quadratic through the three anchors (their
# prescription); alpha1, alpha2 constant. Forward-modelled with the full mass posteriors (Sect. 4).
LEJA = dict(z=(0.2, 1.6, 3.0), logphi1=(-2.44, -3.08, -4.14), logphi2=(-2.89, -3.29, -3.51),
            logmstar=(10.79, 10.88, 10.84), alpha1=-0.28, alpha2=-1.48)

# Wright, Driver and Robotham 2018, arXiv:1808.01754, Table A3, regression of the two-component
# Schechter parameters, quadratic in z: X(z) = A0 + A1 z + A2 z^2. The Schechter function is
# convolved with sigma = 0.1 dex mass errors (Sect. 3). Chabrier IMF (end of Sect. 1).
WRIGHT = dict(M=(10.831, 0.153, -0.033), a1=(-0.579, 0.048, 0.022), a2=(-1.489, -0.087, 0.016),
              P1=(-2.312, -0.658, 0.016), P2=(-3.326, -0.158, -0.002))

# Driver et al. 2022, GAMA DR4, arXiv:2203.08539, Table 7, sample 'All', z < 0.1. The caption says to
# add 0.0807 dex to both log Phi for the SDSS renormalisation and the evolution to z = 0
# (log10(1.159 * 1.039), Sects. 6.1-6.2). MML fit with per-galaxy mass errors (Sect. 5). Chabrier
# (Sect. 3). Used at z = 0.
GAMA = dict(logM=10.745, logP1=-2.437 + 0.0807, logP2=-3.201 + 0.0807, a1=-0.466, a2=-1.530)

# ---------------------------------------------------------------- reference, not in the fit
# Tomczak et al. 2014, ZFOURGE, arXiv:1309.5972, Table 2 (double Schechter, total) and Table 3
# (single Schechter, total). Chabrier. Not corrected for Eddington bias.
TOMCZAK_D = [(0.20, 0.50, 10.78, -0.98, -2.54, -1.90, -4.29),
             (0.50, 0.75, 10.70, -0.39, -2.55, -1.53, -3.15),
             (0.75, 1.00, 10.66, -0.37, -2.56, -1.61, -3.39),
             (1.00, 1.25, 10.54, 0.30, -2.72, -1.45, -3.17),
             (1.25, 1.50, 10.61, -0.12, -2.78, -1.56, -3.43),
             (1.50, 2.00, 10.74, 0.04, -3.05, -1.49, -3.38),
             (2.00, 2.50, 10.69, 1.03, -3.80, -1.33, -3.26),
             (2.50, 3.00, 10.74, 1.62, -4.54, -1.57, -3.69)]
TOMCZAK_S = [(0.20, 0.50, 11.05, -1.35, -2.96), (0.50, 0.75, 11.00, -1.35, -2.93),
             (0.75, 1.00, 11.16, -1.38, -3.17), (1.00, 1.25, 11.09, -1.33, -3.19),
             (1.25, 1.50, 10.88, -1.29, -3.11), (1.50, 2.00, 11.03, -1.33, -3.28),
             (2.00, 2.50, 11.13, -1.43, -3.59), (2.50, 3.00, 11.35, -1.74, -4.36)]

# ---------------------------------------------------------------- black hole to bulge mass relations
# log10(M_BH/Msun) = a + b * log10(M_bulge / pivot), intrinsic scatter eps in dex.
BH_RELATIONS = {
    # Kormendy and Ho 2013, arXiv:1304.7762, Sect. 6.6.1, eq. 10: 0.49 (+0.06 -0.05) x 1e9,
    # slope 1.16 +- 0.08, intrinsic scatter 0.29 dex. Bulge masses: K-band M/L of Into and Portinari
    # 2013 (Kroupa IMF), raised by 0.1258 dex to match dynamical M/L at sigma_e = 250 km/s (text
    # before eq. 8).
    "Kormendy & Ho 2013": dict(a=np.log10(0.49e9), b=1.16, pivot=11.0, eps=0.29, to_chabrier=-0.1258),
    # McConnell and Ma 2013, arXiv:1211.2816, Table 2, 'Dynamical masses', 35 galaxies, MPFITEXY:
    # 8.46 +- 0.08, slope 1.05 +- 0.11, eps 0.34.
    "McConnell & Ma 2013": dict(a=8.46, b=1.05, pivot=11.0, eps=0.34, to_chabrier=-0.1258),
    # Saglia et al. 2016, arXiv:1601.00974, Table 11 (tab_1dim), M_BH-M_Bu, 'CorePowerEClass'
    # (71 ellipticals and classical bulges): log M_BH = 0.885 log M_Bu - 1.155, eps 0.424 +- 0.043.
    # Dynamical M/L.
    "Saglia et al. 2016": dict(a=-1.155 + 0.885 * 11.0, b=0.885, pivot=11.0, eps=0.424, to_chabrier=-0.1258),
    # Sahu, Graham and Davis 2019, arXiv:1903.04738, Table 5, 76 early-type galaxies,
    # BCES(M_BH|M_sph): slope 1.12 +- 0.08, intercept 8.43 +- 0.06 at 5e10 Msun, eps 0.40.
    # Spheroid masses from 3.6 micron, Upsilon = 0.6 (Meidt+14, Chabrier), Sect. 2.
    "Sahu et al. 2019": dict(a=8.43, b=1.12, pivot=np.log10(5e10), eps=0.40, to_chabrier=0.0),
}
