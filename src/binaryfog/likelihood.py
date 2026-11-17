"""
The likelihood: a new-physics spectrum plus the binary background against the measured free spectrum.

Two forms are available.

Point estimate, the usual one, in which the binaries contribute one curve:
    ln L = sum_i ln p( rho_i(model) | data ).

Marginalised, which is what this package is for: the binary contribution B_i in bin i is not a
number but a distribution, because we observe one universe out of many,
    L = prod_i integral dB_i  p(B_i | theta)  p( rho_i(NP_i + B_i) | data ),
with NP_i the deterministic new-physics contribution and p(B_i | theta) the realisation
distribution predicted by the surrogate. The integral is done on a fixed Gauss-Hermite style grid
in the variable z of the warp
    log10 B_i = median_i + sigma_i (z + c z^2),   z ~ N(0, 1),
held flat below z = -1/(2c), where the warp turns over.

All spectra are h^2 Omega_GW(f), dimensionless.
"""
import numpy as np
from scipy.stats import norm

NQ = 101
_Z = np.linspace(-4.0, 4.0, NQ)
_W = np.exp(-0.5 * _Z ** 2)
_W = _W / _W.sum()


class Likelihood:
    """Bind a data product to a background model.

    >>> like = Likelihood(FreeSpectrum.load("ng15_hd_v2"), Background())
    >>> like.ln_like(np_spectrum, theta)          # marginalised over realisations
    >>> like.ln_like(np_spectrum)                 # new physics alone, no binaries
    """

    def __init__(self, data, background=None, marginalise=True):
        """marginalise=False collapses the background to its median spectrum instead of
        integrating over the realisations. It is not the right thing to quote, since we observe one
        universe and not the average of many, but comparing the two separates what the freedom in
        the astrophysical parameters does from what the scatter between universes does."""
        self.data = data
        self.marginalise = marginalise
        if background is not None:
            if len(background.freqs) != data.nbins:
                raise ValueError(f"background has {len(background.freqs)} bins, "
                                 f"data has {data.nbins}")
            # the data product carries its own frequencies, which differ from the nominal
            # i / 16.03 yr in the sixth digit; evaluate the background on the data's grid
            if not np.allclose(background.freqs, data.freqs, rtol=1e-12, atol=0):
                background = background.on_frequencies(data.freqs)
        self.background = background

    def ln_like(self, np_h2omega=None, theta=None):
        """np_h2omega: the new-physics spectrum per bin, or None for binaries only.
        theta: the six background parameters, or None for no binary contribution."""
        d = self.data
        pt = np.zeros(d.nbins) if np_h2omega is None else np.asarray(np_h2omega, float)
        if theta is None:
            return d.ln_like_point(pt)
        med, sigma, skew = self.background.predict(theta)
        if not self.marginalise:
            return d.ln_like_point(pt + np.asarray(med, float))
        return self.ln_like_from_distribution(pt, med, sigma, skew)

    def ln_like_from_distribution(self, np_h2omega, median_h2omega, sigma_dex, skew):
        """The same integral from a distribution given directly, for a library of your own:
        per bin the median background, the half-width in dex and the skew of the warp."""
        d = self.data
        pt = np.zeros(d.nbins) if np_h2omega is None else np.asarray(np_h2omega, float)
        total = 0.0
        c = max(float(skew), 1e-3)
        z = np.maximum(_Z, -0.5 / c)
        for i in range(d.nbins):
            s = max(float(sigma_dex[i]), 1e-4)
            b = 10.0 ** (np.log10(max(median_h2omega[i], 1e-300)) + s * (z + c * z ** 2))
            val = float(np.dot(_W, np.exp(d.ln_density(d.to_log10rho(pt[i] + b, i), i))))
            if val <= 0:
                return -np.inf
            total += np.log(val)
        return total

    def ln_like_binaries_only(self, theta):
        """The background alone, no new physics: the reference for any preference statement."""
        return self.ln_like(None, theta)


def quantile_spectrum(background, theta, q):
    """The q-quantile realisation of the background, for plotting bands.

    Uses the same warp as the likelihood, so the bands and the fit are consistent.
    """
    med, sigma, skew = background.predict(theta)
    c = max(float(skew), 1e-3)
    z = max(norm.ppf(q), -0.5 / c)
    return med * 10.0 ** (sigma * (z + c * z ** 2))
