"""
The background with the two hardening parameters that NANOGrav hold fixed set free.

NANOGrav fix the outer hardening index at nu_outer = +2.5 and the break separation of the
hardening rate at r_char = 100 pc. Neither is measured: the hardening rate is a phenomenological
double power law with no microphysical counterpart, so the choice is a modelling assumption, and
the honest thing is to measure what it does to a limit rather than to defend one value.

This background adds them to the six of the classic space,

    hard_gamma_outer   nu_outer,        uniform on [0, 2.5]
    log10_rchar_pc     log10(r_char/pc), uniform on [log10 2, log10 200]

and any of the eight can be held fixed, so that the same surrogate gives NANOGrav's setup, the
setup of holodeck's astrophysically motivated space, and everything in between.

Beware of one corner. With a steep outer index and a small break separation, the fixed-lifetime
normalisation of the hardening rate puts nearly the whole binary lifetime at kiloparsec
separations and makes the sub-parsec hardening so fast that the band empties: two of every five
models with nu_outer above 1.5 and r_char below 20 pc have no background left. Those models are in
the library and in the maps, so a fit that frees r_char can reach them and switch the binaries off
without touching a single galaxy parameter. That is a property of the model, not of the code, and
it is the reason to look at what freeing r_char does before deciding to free it.

Unlike the six-parameter maps, which go through a bend template, these are per-bin polynomials of
the continuum, of the median offset from it, and of the width, fitted with more weight on the
models loud enough to matter. See fit_maps8.py in the research repository.
"""
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from .background import DATA, hc2_to_h2omega

PARAM_NAMES8 = ["gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10", "mmb_scatter_dex",
                "hard_time", "hard_gamma_inner", "hard_gamma_outer", "log10_rchar_pc"]
PRIOR_BOX8 = {"gsmf_phi0_log10": (-3.5, -1.5), "gsmf_mchar0_log10": (10.5, 12.5),
              "mmb_mamp_log10": (7.6, 9.0), "mmb_scatter_dex": (0.0, 0.9),
              "hard_time": (0.1, 11.0), "hard_gamma_inner": (-1.5, 0.0),
              "hard_gamma_outer": (0.0, 2.5), "log10_rchar_pc": (np.log10(2.0), np.log10(200.0))}
# fallback only: the maps carry the frequencies they were fitted on, which come from the measured
# data products. The IPTA DR2 product is not a set of harmonics, so these nominal values are close
# but not exact, and the likelihood rebinds the background to the data's own grid anyway.
GRID_FREQS = {"ng15": np.arange(1, 15) / (16.03 * 31557600.0),
              "ipta2": 1.0496e-9 + np.arange(26) * 1.0858e-9}
NANOGRAV_FIXED = {"hard_gamma_outer": 2.5, "log10_rchar_pc": np.log10(100.0)}


@lru_cache(maxsize=8)
def _exponents(degree, nv=len(PARAM_NAMES8)):
    rows = [np.zeros(nv, dtype=int)]
    for d in range(1, degree + 1):
        for c in combinations_with_replacement(range(nv), d):
            e = np.zeros(nv, dtype=int)
            for j in c:
                e[j] += 1
            rows.append(e)
    return np.array(rows, dtype=float)


class BackgroundExtended:
    """
    >>> bg = BackgroundExtended(fixed=NANOGRAV_FIXED)      # NANOGrav's choice, six free parameters
    >>> bg = BackgroundExtended(fixed={"log10_rchar_pc": 1.0})   # r_char = 10 pc, nu_outer free
    >>> bg = BackgroundExtended()                          # everything free
    """

    def __init__(self, maps_path=None, grid="ng15", fixed=None, freqs_Hz=None):
        path = Path(maps_path) if maps_path is not None else DATA / "maps8.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"no eight-parameter maps at {path}. They come from the library of "
                "build_library8.py, fitted with fit_maps8.py; pass maps_path to use your own.")
        m = np.load(path, allow_pickle=True)
        self.degree = int(m["degree"])
        self.grid = grid
        self._cont = m[f"{grid}_continuum"]
        self._offset = m[f"{grid}_offset"]
        self._logw = m[f"{grid}_logwidth"]
        self._skew = m[f"{grid}_skew"]
        stored = m[f"{grid}_freqs"] if f"{grid}_freqs" in m.files else GRID_FREQS[grid]
        self.freqs = np.asarray(stored if freqs_Hz is None else freqs_Hz, float)
        if len(self.freqs) != self._cont.shape[1]:
            raise ValueError(f"the {grid} maps have {self._cont.shape[1]} bins, "
                             f"{len(self.freqs)} frequencies given")
        self.fixed = dict(fixed or {})
        for k in self.fixed:
            if k not in PARAM_NAMES8:
                raise ValueError(f"{k!r} is not one of {PARAM_NAMES8}")
        self.param_names = [k for k in PARAM_NAMES8 if k not in self.fixed]
        self.prior_box = {k: PRIOR_BOX8[k] for k in self.param_names}
        self._lo = np.array([PRIOR_BOX8[k][0] for k in PARAM_NAMES8])
        self._hi = np.array([PRIOR_BOX8[k][1] for k in PARAM_NAMES8])

    def _full(self, theta):
        """Insert the held parameters, in the order the maps were fitted in."""
        free = dict(zip(self.param_names, np.asarray(theta, float)))
        return np.array([free.get(k, self.fixed.get(k)) for k in PARAM_NAMES8], float)

    def _design(self, full):
        u = 2 * (full[None, :] - self._lo) / (self._hi - self._lo) - 1
        return np.prod(u[:, None, :] ** _exponents(self.degree)[None, :, :], axis=2)

    def predict(self, theta):
        """Returns (median h^2 Omega per bin, width in dex per bin, skew)."""
        x = self._design(self._full(theta))
        log10hc2 = (x @ self._cont)[0] + (x @ self._offset)[0]
        sigma = 10.0 ** (x @ self._logw)[0]
        skew = float(np.clip(x @ self._skew, 0.02, 0.45)[0])
        return hc2_to_h2omega(self.freqs, log10hc2), sigma, skew

    def on_frequencies(self, freqs_Hz):
        """The same maps evaluated on another frequency grid of the same length."""
        import copy
        freqs = np.asarray(freqs_Hz, float)
        if len(freqs) != self._cont.shape[1]:
            raise ValueError(f"the {self.grid} maps have {self._cont.shape[1]} bins, "
                             f"{len(freqs)} frequencies given")
        other = copy.copy(self)
        other.freqs = freqs
        return other

    def __repr__(self):
        held = ", ".join(f"{k}={v:.3g}" for k, v in self.fixed.items()) or "nothing"
        return (f"BackgroundExtended(grid={self.grid!r}, degree={self.degree}, "
                f"free {len(self.param_names)}, held {held})")
