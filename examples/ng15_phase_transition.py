"""
NANOGrav's own sound-wave phase-transition model, profiled with and without the binary background.

The model is pt_sound.py from the NANOGrav 15 yr new-physics model files: a first-order transition
of strength alpha at temperature T_star with mean bubble separation H_star R_star, and three
parameters a, b, c for the spectral shape. The scan is over alpha and H_star R_star; the
temperature and the shape parameters are profiled, and so are the six astrophysical parameters of
the binary background with the priors of arXiv:2306.16220.

    BINARYFOG_NG15_MODELS=<unpacked model files> python ng15_phase_transition.py
"""
import os
from pathlib import Path

import numpy as np

import binaryfog as bf

MODELS = Path(os.environ["BINARYFOG_NG15_MODELS"])
DATA = os.environ.get("BINARYFOG_FREESPEC_ROOT")
N = int(os.environ.get("NGRID", 24))
JOBS = int(os.environ.get("JOBS", 24))

if __name__ == "__main__":
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=DATA)
    model = bf.load_ptarcade(MODELS / "pt_sound.py")
    grid = {"log10_alpha": np.linspace(-2.0, 1.0, N), "log10_H_R": np.linspace(-3.0, 0.0, N)}

    for label, background in (("transition alone", None),
                              ("transition and binaries", bf.Background())):
        like = bf.Likelihood(data, background)
        scan = bf.ProfileScan(like, model.spectrum, scan=grid,
                              nuisance=model.nuisance("log10_T_star", "a", "b", "c"),
                              astro_prior="nanograv", restarts=3)
        table = scan.run(jobs=JOBS)
        table.to_csv(f"pt_sound_{'binaries' if background else 'alone'}.csv", index=False)
        best = table.loc[table["lnL"].idxmax()]
        inside = bf.region(table, 1, ndim=2)
        cell = ((grid["log10_alpha"][1] - grid["log10_alpha"][0])
                * (grid["log10_H_R"][1] - grid["log10_H_R"][0]))
        print(f"\n{label}:")
        print(f"  best fit: alpha = 1e{best['log10_alpha']:.2f}, H_star R_star = "
              f"1e{best['log10_H_R']:.2f}, T_star = 1e{best['log10_T_star']:.2f} GeV, "
              f"ln L = {best['lnL']:.2f}")
        print(f"  1 sigma region covers {inside.sum() * cell:.2f} of the "
              f"{len(table) * cell:.2f} square decades scanned")
        if background is not None:
            print(f"  gain over the binaries alone: {bf.preference(table, like):.2f} ln L units")
