"""
The measured free spectrum: what a PTA says about the power in each frequency bin.

A free-spectrum analysis drops the assumption of a power law and measures, in each Fourier bin i,
the posterior of rho_i, the square root of the timing-residual power in that bin, in seconds. The
posteriors are distributed as kernel-density estimates on a grid of log10 rho_i, the data products
that ceffyl and PTArcade read.

Conventions, following ceffyl's Ceffyl.ln_likelihood:
    below the grid the density is held at its lowest stored value, because there the signal is
    buried in the noise and the posterior is flat;
    above the grid the model is rejected.
ceffyl's constant log(d log10 rho) per bin is left out; it cancels in every likelihood ratio.

Available data sets (see fetch.py):
    ng15_cp_v1, ng15_hd_v1        NANOGrav 15 yr, 2024 release (Zenodo 10495907)
    ng15_cp_v2, ng15_hd_v2        the same after the PTMCMCSampler parallel-tempering fix
    ng15_hdall_v2                 Hellings-Downs modelled together with monopole, dipole and
                                  uncorrelated red noise: NANOGrav's conservative choice
    ng15_hd_dmgp_v1               Hellings-Downs with the dispersion-measure Gaussian process
    ipta2_cp_v1                   IPTA DR2, uncorrelated common red noise only
"cp" is the uncorrelated common red noise, "hd" the Hellings-Downs correlated process.
"""
from pathlib import Path

import numpy as np

from .background import H100_HZ

DEFAULT_DIR = Path.home() / ".binaryfog/freespec"


class FreeSpectrum:
    """One kernel-density data product, restricted to the bins used in the fit."""

    def __init__(self, path, nbins=14, name=None):
        path = Path(path)
        self.name = name or path.name
        self.path = path
        dens = np.load(path / "density.npy")
        self.grid = np.load(path / "log10rhogrid.npy")
        freqs = np.load(path / "freqs.npy")
        if nbins > len(freqs):
            raise ValueError(f"{self.name} has {len(freqs)} bins, {nbins} requested")
        self.nbins = nbins
        self.freqs = freqs[:nbins]
        self.density = dens[0, :nbins, :]
        self.tspan = 1.0 / freqs[0]

    @classmethod
    def load(cls, name, nbins=14, root=None):
        """Load by name from the cache directory, fetching it if it is not there yet."""
        root = Path(root) if root is not None else DEFAULT_DIR
        path = root / name
        if not (path / "density.npy").is_file():
            from .fetch import fetch
            fetch(name, root)
        return cls(path, nbins=nbins, name=name)

    def to_log10rho(self, h2omega, i):
        """Invert h^2 Omega = 8 pi^4 / H100^2 rho^2 f^5 T for bin i."""
        pref = 8 * np.pi ** 4 / H100_HZ ** 2 * self.freqs[i] ** 5 * self.tspan
        return 0.5 * np.log10(np.maximum(h2omega, 1e-300) / pref)

    def ln_density(self, log10rho, i):
        """log density of bin i at one or many values of log10 rho, with ceffyl's conventions."""
        return np.interp(log10rho, self.grid, self.density[i], right=-np.inf)

    def ln_like_point(self, h2omega):
        """Sum over bins of the log density at a deterministic model spectrum."""
        total = 0.0
        for i in range(self.nbins):
            v = float(self.ln_density(self.to_log10rho(h2omega[i], i), i))
            if not np.isfinite(v):
                return -np.inf
            total += v
        return total

    def __repr__(self):
        return (f"FreeSpectrum({self.name!r}, {self.nbins} bins, "
                f"{self.freqs[0]*1e9:.2f} to {self.freqs[-1]*1e9:.2f} nHz, "
                f"grid {self.grid[0]:.1f} to {self.grid[-1]:.1f})")
