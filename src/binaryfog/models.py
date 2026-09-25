"""
New-physics spectra.

A model is any callable whose first argument is the frequency array in Hz and which returns
h^2 Omega_GW, dimensionless. Two ready-made ones are here, and the loader below turns the model
files of the NANOGrav 15 yr new-physics search into the same form, so that the whole set of
sources studied there can be run through this likelihood without rewriting them.

The model files are a separate data release, "The NANOGrav 15-year search for new physics signals
- model files", doi:10.5281/zenodo.8092761, CC-BY-4.0. They import ptarcade.models_utils, so
PTArcade has to be installed to use them; nothing else in binaryfog needs it.

Note on priors: the parameters of a new-physics model are profiled inside their ranges without a
prior term, since constraining them is the point of the exercise. Only the astrophysical
parameters of the background carry a prior, because they are constrained from outside the PTA.
"""
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import numpy as np

from .background import FYR_HZ, H100_HZ


def power_law(f_Hz, log10_A, gamma):
    """The power law used for the binary background in the published analyses,

        h^2 Omega(f) = 2 pi^2 / (3 H100^2) A^2 (f/f_yr)^(5-gamma) f_yr^2,

    with A the characteristic-strain amplitude at f = 1/yr and gamma the slope of the
    timing-residual power spectrum. Binaries driven only by gravitational waves give gamma = 13/3.
    """
    f = np.asarray(f_Hz, float)
    return (2 * np.pi ** 2 / (3 * H100_HZ ** 2) * (10.0 ** log10_A) ** 2
            * (f / FYR_HZ) ** (5 - gamma) * FYR_HZ ** 2)


def smbhb_power_law(f_Hz, smbhb_log10_A, smbhb_gamma):
    """The same power law with unambiguous parameter names, for adding to a new-physics model.

    New-physics models have amplitudes of their own, and several of the NANOGrav ones call theirs
    log10_A, so a background added to them needs names that cannot clash.
    """
    return power_law(f_Hz, smbhb_log10_A, smbhb_gamma)


def broken_power_law(f_Hz, log10_amp, log10_fpeak, rise=3.0, fall=-4.0):
    """A generic peaked background: rising as f^rise, falling as f^fall, smooth at the peak."""
    x = np.asarray(f_Hz, float) / 10.0 ** log10_fpeak
    return 10.0 ** log10_amp * x ** rise / (1.0 + x ** (rise - fall))


class Sum:
    """Two spectra added, for instance a new-physics model on top of the published power law.

    The parameter names of the components must not clash; each keyword goes to the component that
    declares it. This is a class rather than a closure so that a scan can send it to worker
    processes.

    >>> total = Sum(model.spectrum, power_law)
    >>> total(freqs, log10_alpha=-1.0, ..., log10_A=-15.6, gamma=4.7)
    """

    def __init__(self, *components):
        self.components = components
        self.names, self.optional = [], []
        for c in components:
            pars = inspect.signature(c).parameters
            args = [a for a in pars if a not in ("f", "f_Hz")]
            self.names.append(args)
            # parameters with a default need not be supplied, for instance the fixed slopes of a
            # template; they keep their default unless the caller passes them
            self.optional.append({a for a in args
                                  if pars[a].default is not inspect.Parameter.empty})
        seen = [a for args in self.names for a in args]
        clash = {a for a in seen if seen.count(a) > 1}
        if clash:
            raise ValueError(f"the components share parameter names: {sorted(clash)}")

    def __call__(self, f_Hz, **kwargs):
        total = None
        for c, args, opt in zip(self.components, self.names, self.optional):
            use = {a: kwargs[a] for a in args if a in kwargs or a not in opt}
            part = np.asarray(c(f_Hz, **use), float)
            total = part if total is None else total + part
        return total


class PTArcadeModel:
    """One NANOGrav model file, ready for ProfileScan.

    >>> m = load_ptarcade("models_1.0.1/pt_sound.py")
    >>> m.name, list(m.ranges)
    >>> scan = ProfileScan(like, m.spectrum, scan={"log10_alpha": ...},
    ...                    nuisance=m.nuisance("log10_T_star", "a", "b", "c"))
    """

    def __init__(self, module):
        self.module = module
        self.name = getattr(module, "name", module.__name__)
        self.spectrum = module.spectrum
        self.smbhb = bool(getattr(module, "smbhb", False))
        self.ranges, self.prior_kind = {}, {}
        for key, par in module.parameters.items():
            kind, bounds = _bounds(par)
            self.ranges[key] = bounds
            self.prior_kind[key] = kind

    def nuisance(self, *names, scale=None):
        """The ProfileScan nuisance dictionary for these parameters, with their own ranges.

        scale: optional mapping name -> "log" or "linear". The model files already work in
        logarithms where that is natural, so the default is linear in the file's own variable.
        """
        scale = scale or {}
        names = names or [k for k in self.ranges]
        return {n: (self.ranges[n][0], self.ranges[n][1], scale.get(n, "linear")) for n in names}

    def __repr__(self):
        return f"PTArcadeModel({self.name!r}, parameters {list(self.ranges)})"


def _bounds(par):
    """Prior type and (lo, hi) of an enterprise parameter class, as the model files build them."""
    try:
        kwargs = par("x").prior.func_kwargs
    except Exception:                                     # noqa: BLE001 - fall back to the name
        kwargs = {k: float(v) for k, v in
                  re.findall(r"(\w+)=(-?[\d.eE+-]+)", getattr(par, "_typename", ""))}
    name = getattr(par, "_typename", "")
    if "pmin" in kwargs and "pmax" in kwargs:
        return ("uniform", (float(kwargs["pmin"]), float(kwargs["pmax"])))
    if "mu" in kwargs and "sigma" in kwargs:              # scanned inside five standard deviations
        mu, sd = float(kwargs["mu"]), float(kwargs["sigma"])
        return ("normal", (mu - 5 * sd, mu + 5 * sd))
    raise ValueError(f"cannot read the range of a parameter with prior {name!r}")


_LOADED = {}


def load_ptarcade(path):
    """Load a NANOGrav model file. Needs PTArcade installed, since the files import it.

    The same file always gives back the same object. That matters because a scan sends the
    spectrum function to worker processes: pickling a function needs its module to be importable
    under its own name, and a second, independent load of the same file would leave the name
    pointing at the first module and break the send.
    """
    path = Path(path).resolve()
    if path in _LOADED:
        return _LOADED[path]
    name = path.stem
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) != str(path):
        name = f"{path.stem}_{abs(hash(str(path))) % 10 ** 8}"      # two files, one stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "spectrum"):
        raise ValueError(f"{path.name} defines no spectrum function. Files that define 'signal' "
                         "instead describe a deterministic timing signal, for instance from "
                         "ultralight dark matter or from black-hole substructure, which is not a "
                         "stochastic background and cannot be used here")
    out = PTArcadeModel(module)
    _LOADED[path] = out
    return out
