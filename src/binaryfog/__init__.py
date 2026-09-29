"""
binaryfog: the supermassive black hole binary background as a nuisance in PTA searches.

A pulsar timing array measures one gravitational-wave background, and any new-physics source has
to share it with the binaries. The binaries are not a curve, though: they are a finite random set
of sources, so what a population model predicts in each frequency bin is a distribution. This
package supplies that distribution from a surrogate of holodeck, integrates over it inside the
likelihood, and profiles over the astrophysical parameters, so that a limit on a new-physics model
carries the astrophysical uncertainty rather than assuming it away.

    import numpy as np, binaryfog as bf

    data = bf.FreeSpectrum.load("ng15_hd_v2")            # the measured free spectrum
    like = bf.Likelihood(data, bf.Background())          # with the binary background

    def my_model(f_Hz, amplitude, tilt):                 # h^2 Omega_GW, dimensionless
        return amplitude * (f_Hz / 1e-8) ** tilt

    scan = bf.ProfileScan(like, my_model,
                          scan={"tilt": np.linspace(-2, 3, 40)},
                          nuisance={"amplitude": (1e-12, 1e-6, "log")})
    table = scan.run(jobs=8)
    inside = bf.region(table, 0.68, ndim=1)

Conventions: spectra are h^2 Omega_GW(f) with f in Hz; the six background parameters are those of
the classic phenomenological holodeck space, in the order of bf.PARAM_NAMES.
"""
from .background import PARAM_NAMES, PRIOR_BOX, Background, bend_log10hc2, hc2_to_h2omega
from .background8 import NANOGRAV_FIXED, PARAM_NAMES8, PRIOR_BOX8, BackgroundExtended
from .freespec import FreeSpectrum
from .likelihood import Likelihood, quantile_spectrum
from .models import (PTArcadeModel, Sum, broken_power_law, load_ptarcade, power_law,
                     smbhb_power_law)
from .priors import AGAMMA_COV, AGAMMA_MU, NANOGRAV_GAUSSIAN, ln_prior, ln_prior_agamma
from .profile import LEVELS_1D, LEVELS_2D, ProfileScan, preference, reference_lnL, region
from .significance import sigma_from_gain

__version__ = "0.1.0"
__all__ = ["Background", "BackgroundExtended", "FreeSpectrum", "Likelihood", "ProfileScan",
           "PARAM_NAMES", "PRIOR_BOX", "PARAM_NAMES8", "PRIOR_BOX8", "NANOGRAV_FIXED",
           "PTArcadeModel", "load_ptarcade", "power_law", "smbhb_power_law",
           "broken_power_law", "Sum",
           "NANOGRAV_GAUSSIAN", "ln_prior", "ln_prior_agamma", "AGAMMA_MU", "AGAMMA_COV",
           "region", "preference", "reference_lnL", "quantile_spectrum",
           "bend_log10hc2", "hc2_to_h2omega", "LEVELS_1D", "LEVELS_2D", "sigma_from_gain",
           "__version__"]
