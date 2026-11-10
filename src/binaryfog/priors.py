"""
Prior sets for the six background parameters, as in table 1 of the paper.

    "flat"              uniform over the library box: what the data alone can do. None means the same.
    "nanograv"          the priors of NANOGrav's holodeck library (arXiv:2306.16220): Gaussians on the
                        normalisation and characteristic mass of the galaxy stellar mass function and on the
                        normalisation and scatter of the black-hole to bulge-mass relation. The mass-function
                        widths are four times those of the fits to Tomczak et al. (2014) they derive from.
    "tomczak_kormendy"  the centres and uncertainties of the fits these priors are based on: the single-Schechter
                        parameters of the total galaxy sample of Tomczak et al. (2014, table 3), fitted linearly in
                        redshift, and the black-hole to bulge-mass relation of Kormendy and Ho (2013, eq. 10).
    "survey_average"    the centre and spread of six mass-function measurements published since 2014, and the
                        black-hole to bulge-mass relation brought to the same (Chabrier) stellar-mass convention.

In every set the binary lifetime and the inner hardening index are flat over the box. Every set enforces the
box, outside which the fitted maps are extrapolation.

For the cross-check with the (A, gamma) power-law description of the binaries, ln_prior_agamma is NANOGrav's
two-dimensional Gaussian on its amplitude and slope (AGAMMA_MU, AGAMMA_COV), passed to a ProfileScan as
nuisance_prior rather than as astro_prior.
"""
import numpy as np

from .background import PARAM_NAMES, PRIOR_BOX

NANOGRAV_GAUSSIAN = {"gsmf_phi0_log10": (-2.56, 0.40), "gsmf_mchar0_log10": (10.90, 0.40),
                     "mmb_mamp_log10": (8.60, 0.20), "mmb_scatter_dex": (0.32, 0.15)}
# Mass function: linear-in-redshift fit to the single-Schechter parameters of the total sample in table 3 of
# Tomczak et al. 2014 (all eight redshift bins), intercepts at z = 0. Black hole to bulge mass: Kormendy and
# Ho 2013, eq. 10, normalisation 8.69 +- 0.05 and intrinsic scatter 0.28 dex; they quote no uncertainty on
# the scatter, so it takes 0.05.
TOMCZAK_KORMENDY_GAUSSIAN = {"gsmf_phi0_log10": (-2.61, 0.10), "gsmf_mchar0_log10": (10.96, 0.11),
                             "mmb_mamp_log10": (8.69, 0.05), "mmb_scatter_dex": (0.28, 0.05)}
# Mass function: psi0 and m_psi0 fitted to six Eddington-corrected determinations with Chabrier stellar masses
# (Davidzon et al. 2017, Weaver et al. 2023, McLeod et al. 2021, Leja et al. 2020, Wright et al. 2018, Driver
# et al. 2022), with the redshift slopes and the low-mass slope held at the library's values; centre the mean,
# width the spread between them. Black hole to bulge mass: Kormendy and Ho 2013 and Sahu et al. 2019 brought
# to Chabrier bulge masses, scatter from Sahu et al. 2019 and Saglia et al. 2016.
SURVEY_AVERAGE_GAUSSIAN = {"gsmf_phi0_log10": (-2.20, 0.15), "gsmf_mchar0_log10": (10.82, 0.11),
                           "mmb_mamp_log10": (8.80, 0.10), "mmb_scatter_dex": (0.40, 0.05)}
GAUSSIAN_SETS = {"nanograv": NANOGRAV_GAUSSIAN, "tomczak_kormendy": TOMCZAK_KORMENDY_GAUSSIAN,
                 "survey_average": SURVEY_AVERAGE_GAUSSIAN}
_LO = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES])
_HI = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])


# The two-dimensional Gaussian that NANOGrav put on the power-law background in their new-physics
# search, arXiv:2306.16219 eq. (A1): a fit of power laws to the 14 lowest bins of their GW-only
# holodeck library. Note that the reproduction kit of arXiv:2602.09092 carries an off-diagonal
# element of -0.03 where the paper and PTArcade have 10^-1 x (-0.026); the published value is used
# here.
AGAMMA_MU = np.array([-15.61492963, 4.70709637])
AGAMMA_COV = np.array([[0.27871359, -0.00263617], [-0.00263617, 0.12415383]])
_AGAMMA_ICOV = np.linalg.inv(AGAMMA_COV)
_AGAMMA_NORM = np.log(1 / (2 * np.pi * np.sqrt(np.linalg.det(AGAMMA_COV))))


def ln_prior_agamma(values):
    """Prior of the published power-law background, for use as a ProfileScan nuisance prior.

    values: a mapping carrying the amplitude and slope under either the plain names of power_law
    or the prefixed ones of smbhb_power_law. Other entries, the new-physics parameters, are
    ignored: they are profiled without a prior.
    """
    try:
        a = values["smbhb_log10_A"] if "smbhb_log10_A" in values else values["log10_A"]
        g = values["smbhb_gamma"] if "smbhb_gamma" in values else values["gamma"]
    except KeyError as exc:
        raise KeyError("the published background prior needs the amplitude and slope, as "
                       "log10_A and gamma or as smbhb_log10_A and smbhb_gamma") from exc
    d = np.array([a, g]) - AGAMMA_MU
    return float(-0.5 * d @ _AGAMMA_ICOV @ d + _AGAMMA_NORM)


def ln_prior(theta, which="nanograv", names=None, box=None):
    """Log prior of a background's parameters, up to a constant independent of theta.

    names and box default to the six parameters of the classic space; a background with more of
    them, for instance one that frees the outer hardening index and the break separation, passes
    its own. Parameters without an entry in the Gaussian set are flat inside the box.
    """
    names = names or PARAM_NAMES
    box = box or PRIOR_BOX
    lo = np.array([box[k][0] for k in names])
    hi = np.array([box[k][1] for k in names])
    th = np.asarray(theta, float)
    if np.any(th < lo) or np.any(th > hi):
        return -np.inf
    if which in (None, "flat"):
        return 0.0
    if which not in GAUSSIAN_SETS:
        raise ValueError(f"unknown prior set {which!r}; available: 'flat', "
                         f"{', '.join(repr(k) for k in GAUSSIAN_SETS)}, None")
    gauss = GAUSSIAN_SETS[which]
    lp = 0.0
    for name, v in zip(names, th):
        if name in gauss:
            mu, sd = gauss[name]
            lp += -0.5 * ((v - mu) / sd) ** 2
    return lp


def start_points(rng, n, which="nanograv", names=None, box=None):
    """Starting points for the maximisation: near the prior centres, spread over the box."""
    names = names or PARAM_NAMES
    box = box or PRIOR_BOX
    lo = np.array([box[k][0] for k in names])
    hi = np.array([box[k][1] for k in names])
    gauss = GAUSSIAN_SETS.get(which, NANOGRAV_GAUSSIAN)
    centres = np.array([gauss[k][0] if k in gauss
                        else 0.5 * (box[k][0] + box[k][1]) for k in names])
    scale = np.array([gauss[k][1] if k in gauss
                      else 0.25 * (box[k][1] - box[k][0]) for k in names])
    if which in (None, "flat"):
        scale = scale * 2.0
    out = centres + rng.normal(0.0, scale, size=(n, len(names)))
    return np.clip(out, lo, hi)
