"""
Prior sets for the six background parameters.

    "uniform"    flat over the library box: what the data alone can do, the bracket.
    "nanograv"   the astrophysical priors of arXiv:2306.16220: Gaussians on the galaxy stellar
                 mass function normalisation and characteristic mass and on the black-hole to
                 bulge-mass relation, flat on the binary lifetime and the inner hardening index.
                 Their widths are four times the standard deviation of a fit to the galaxy data of
                 Tomczak et al. (2014); the paper gives no reason for the factor.
    "unwidened"  the same Gaussians at the widths of the underlying fits: 0.1 on the mass-function
                 normalisation and characteristic mass (holodeck's linear-in-z fit to Tomczak et al. 2014,
                 table 3, without the factor four) and 0.02 dex on the scatter (Kormendy and Ho 2013
                 quote 0.29 +- 0.02 for M-sigma and 0.30 +- 0.01 for M-L_K, p. 566, and 0.28 without an
                 uncertainty for M-M_bulge, eq. 10);
                 the mass normalisation keeps 0.2.
    "fitwidths"  as "unwidened" with 0.05 on the mass normalisation as well (its uncertainty in
                 Kormendy and Ho 2013) and 0.05 on the scatter; "kh" the same centred on their 8.69.
    None         no prior term at all, for a pure profile likelihood over the background.

A third set, the present-day measurements with their own much smaller widths, needs a library built
on the double-Schechter mass function and the Illustris merger rate and is not available yet.

Every set enforces the box, outside which the fitted maps are extrapolation.
"""
import numpy as np

from .background import PARAM_NAMES, PRIOR_BOX

NANOGRAV_GAUSSIAN = {"gsmf_phi0_log10": (-2.56, 0.40), "gsmf_mchar0_log10": (10.90, 0.40),
                     "mmb_mamp_log10": (8.60, 0.20), "mmb_scatter_dex": (0.32, 0.15)}
UNWIDENED_GAUSSIAN = {"gsmf_phi0_log10": (-2.56, 0.10), "gsmf_mchar0_log10": (10.90, 0.10),
                      "mmb_mamp_log10": (8.60, 0.20), "mmb_scatter_dex": (0.32, 0.02)}
# all four at the widths of the underlying fits, the mass normalisation included (0.05 on the
# normalisation of Kormendy and Ho 2013), centred once at holodeck's 8.60 and once at their 8.69
# eps_mu takes 0.05 here: Kormendy and Ho quote no uncertainty on the scatter of the mass relation
# (0.02 is that of their K-band luminosity relation), and holodeck's strong-prior set uses 0.28 +- 0.05
FITWIDTHS_GAUSSIAN = dict(UNWIDENED_GAUSSIAN, mmb_mamp_log10=(8.60, 0.05), mmb_scatter_dex=(0.32, 0.05))
KH_GAUSSIAN = dict(UNWIDENED_GAUSSIAN, mmb_mamp_log10=(8.69, 0.05), mmb_scatter_dex=(0.32, 0.05))
# "literature": centres and widths taken from the sources NANOGrav cite, each from one source.
# Mass function: the linear-in-redshift fit of the single-Schechter parameters of Tomczak et al. 2014,
# table 3, total sample (all eight redshift bins; holodeck's own fit took two bins from the star-forming
# block), intercepts at z = 0. Black hole to bulge relation: Kormendy and Ho 2013, eq. 10, p. 571:
# normalisation 8.69 +- 0.05 and intrinsic scatter 0.28 dex, which they quote without an uncertainty, so
# 0.05 as in holodeck's strong-prior set. The slope stays at the library's 1.10 (Kormendy and Ho: 1.17).
LITERATURE_GAUSSIAN = {"gsmf_phi0_log10": (-2.61, 0.10), "gsmf_mchar0_log10": (10.96, 0.11),
                       "mmb_mamp_log10": (8.69, 0.05), "mmb_scatter_dex": (0.28, 0.05)}
GAUSSIAN_SETS = {"nanograv": NANOGRAV_GAUSSIAN, "unwidened": UNWIDENED_GAUSSIAN,
                 "fitwidths": FITWIDTHS_GAUSSIAN, "kh": KH_GAUSSIAN, "literature": LITERATURE_GAUSSIAN}
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
    if which in (None, "uniform", "none"):
        return 0.0
    if which not in GAUSSIAN_SETS:
        raise ValueError(f"unknown prior set {which!r}; available: 'uniform', "
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
    if which in (None, "uniform", "none"):
        scale = scale * 2.0
    out = centres + rng.normal(0.0, scale, size=(n, len(names)))
    return np.clip(out, lo, hi)
