"""
The numbers the paper quotes for the three further models of figure 6: how far each model is preferred, the best
fit with the binaryfog background, and the 1 sigma intervals with the (A, gamma) power law in the parameters
NANOGrav quotes.

The gain G is the maximum of the profile likelihood over the same fit with the new physics switched off (the
gain equation of section 3). The maximum is read from what profile_examples.py stored; the fit without new
physics is recomputed here, with the binaries represented either by the published power law under its Gaussian
prior or by the binaryfog background under the NANOGrav priors. The gain is converted to a significance with the
number of free parameters of the whole new-physics model, including those profiled over.

    BINARYFOG_NG15_MODELS=<model files> python example_preferences.py
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import DATA                                                     # noqa: E402
from profile_examples import AGAMMA_NUISANCE, FIGURE_JSON, interval          # noqa: E402

# free parameters of each NANOGrav model file, scanned and profiled together
NDOF = {"dw_ds": 4, "super": 2, "igw": 3}
# the profiles in the parameters NANOGrav quotes intervals for (inflation scanned in log10 r)
NANOGRAV_JSON = "ng15_examples_n80.json"
# best-fit parameters quoted, with the conversion applied for printing
BEST = {"super": [("P", "log10_P", lambda v: 10 ** v, ".1e")],
        "igw": [("n_t", "n_t", lambda v: v, ".2f"), ("T_rh / GeV", "log10_T_rh", lambda v: 10 ** v, ".2f")],
        "dw_ds": [("Delta N_eff", "log10_N_eff", lambda v: 10 ** v, ".2f"), ("b", "b", lambda v: v, ".2f"),
                  ("c", "c", lambda v: v, ".2f")]}


if __name__ == "__main__":
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    reference = {}
    for case in ("power law", "population"):
        like = bf.Likelihood(data, bf.Background() if case == "population" else None,
                             marginalise=(case == "population"))
        reference[case] = bf.reference_lnL(
            like,
            background_model=None if case == "population" else bf.smbhb_power_law,
            nuisance=None if case == "population" else dict(AGAMMA_NUISANCE),
            nuisance_prior=None if case == "population" else bf.ln_prior_agamma)
        print(f"  no new physics, {case:11s} ln L = {reference[case]:+.3f}")
    d = json.load(open(SCANS / FIGURE_JSON))
    for name, entry in d.items():
        print(f"{entry['label']} ({NDOF[name]} free parameters)")
        for case in ("power law", "population"):
            gain = float(np.max(entry[case]) - reference[case])
            print(f"    {case:11s} gain {gain:5.2f} in ln L, "
                  f"{bf.sigma_from_gain(gain, NDOF[name]):4.1f} standard deviations")
        print("    best fit with binaryfog: " + ", ".join(
            f"{label} = {f(entry['best'][key]):{fmt}}" for label, key, f, fmt in BEST[name]))
    print("1 sigma intervals with the power law (Delta ln L = 0.5, ends interpolated)")
    for name, entry in json.load(open(SCANS / NANOGRAV_JSON)).items():
        lo, hi, _ = interval(entry["grid"], entry["power law"], bf.LEVELS_1D[1])
        print(f"    {entry['label']:34s} {entry['parameter']:13s} {lo:+.2f} to {hi:+.2f}")
