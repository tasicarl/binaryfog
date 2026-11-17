"""
The preference for the phase transition at fixed binary lifetime tau_f and inner hardening index nu_inner (figure 8;
CURN, ng15_cp_v2, the three Gaussian prior sets).

In each cell of a 25 x 13 grid in (tau_f, nu_inner) the other four astrophysical parameters are profiled under their
prior (Nelder-Mead from the best 4 of 256 prior draws), once for the binaries alone and once with the transition. The
transition's (alpha, beta/H, T_reh) starts from candidates: the best point of every alpha column of the figure-5 scan
for that prior set, plus its 40 best points overall; in each cell the 8 candidates with the largest ln L at the
binaries-alone optimum or at the prior centres are profiled with the transition held fixed, and from the best two the
transition's three parameters are then profiled together with the four astrophysical ones, inside the box of the
figure-5 scans (alpha 0.01-1000, beta/H 0.7-300, T_reh 1 MeV-10 GeV). The gain
G = max lnL(transition + binaries) - max lnL(binaries) is converted with k = 3 (alpha, beta/H, T_reh).

    python hardening_map.py [jobs]      -> scans/hardening_map.npz
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                          # noqa: E402
from binaryfog import priors as bfp                                            # noqa: E402
from binaryfog.significance import sigma_from_gain                             # noqa: E402
from fopt_model import spectrum_fopt                                           # noqa: E402

DATA = "ng15_cp_v2"
TAU = np.round(np.geomspace(0.1, 11.0, 25), 4)
NU = np.round(np.linspace(-1.5, 0.0, 13), 4)
PANELS = [("nanograv", ""), ("tomczak_kormendy", "_tomczak_kormendy"), ("survey_average", "_survey_average")]   # prior set, scan file infix
NCAND, NTOP = 40, 8
NAMES = list(bfp.PARAM_NAMES)
I_TAU, I_NU = NAMES.index("hard_time"), NAMES.index("hard_gamma_inner")
FREE = [k for k in range(len(NAMES)) if k not in (I_TAU, I_NU)]
LIKE = bf.Likelihood(bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT), bf.Background())
F = LIKE.data.freqs


def full(x4, tau, nu):
    th = np.empty(len(NAMES)); th[FREE] = x4; th[I_TAU] = tau; th[I_NU] = nu
    return th


def lnpost(x4, tau, nu, pt, which):
    th = full(x4, tau, nu); lp = bfp.ln_prior(th, which)
    return -np.inf if not np.isfinite(lp) else LIKE.ln_like(pt, th) + lp


def profile(tau, nu, pt, which, starts):
    best = None
    for x0 in starts:
        r = minimize(lambda x: -lnpost(x, tau, nu, pt, which) if np.isfinite(lnpost(x, tau, nu, pt, which)) else 1e30,
                     x0, method="Nelder-Mead", options={"maxfev": 1500, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


# the box of the figure-5 scans, in log10 alpha, log10 beta/H, log10 T_reh/GeV
PT_BOX = np.array([[-2.0, 3.0], [np.log10(0.7), np.log10(300.0)], [-3.0, 1.0]])


def lnpost7(y, tau, nu, which):
    """The four profiled astrophysical parameters followed by log10 of alpha, beta/H and T_reh/GeV."""
    z = y[4:]
    if np.any(z < PT_BOX[:, 0]) or np.any(z > PT_BOX[:, 1]):
        return -np.inf
    return lnpost(y[:4], tau, nu, np.asarray(spectrum_fopt(F, *(10.0 ** z)), float), which)


def profile7(tau, nu, which, starts):
    def f(y):
        v = lnpost7(y, tau, nu, which)
        return -v if np.isfinite(v) else 1e30
    best = None
    for y0 in starts:
        r = minimize(f, y0, method="Nelder-Mead", options={"maxfev": 4000, "xatol": 1e-5, "fatol": 1e-5, "adaptive": True})
        # a restart from the end point, since Nelder-Mead in seven dimensions can stop on a collapsed simplex
        r = minimize(f, r.x, method="Nelder-Mead", options={"maxfev": 4000, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def cell(args):
    which, tau, nu, cands, params, seeds = args
    v = np.array([lnpost(x, tau, nu, None, which) for x in seeds])
    lbin, xb = profile(tau, nu, None, which, [seeds[i] for i in np.argsort(-v)[:4]])
    # with a transition the binaries are pushed towards the prior centres, so candidates are ranked, and profiled,
    # from there as well as from the binaries-alone optimum
    xc = np.array([bfp.GAUSSIAN_SETS[which][NAMES[k]][0] for k in FREE])
    pre = np.array([max(lnpost(xb, tau, nu, pt, which), lnpost(xc, tau, nu, pt, which)) for pt in cands])
    fixed = []
    for k in np.argsort(-pre)[:NTOP]:
        vv = np.array([lnpost(x, tau, nu, cands[k], which) for x in seeds[:64]])
        l, x = profile(tau, nu, cands[k], which, [xc, xb, seeds[int(np.argmax(vv))]])
        fixed.append((l, k, x))
    fixed.sort(key=lambda t: -t[0])
    # the transition's parameters profiled as well, from the best two candidates
    starts = [np.r_[x0, np.log10(params[k])] for _, k, x in fixed[:2] for x0 in (x, xc)]
    lpt, _ = profile7(tau, nu, which, starts)
    lpt = max(lpt, fixed[0][0])
    return which, tau, nu, lpt, lbin


def work_items(tag, infix, taus):
    """The cells of one prior set at the given lifetimes and every nu_inner, with their candidate transitions
    and astrophysical start points."""
    d = pd.read_csv(SCANS / f"scan_holodeck_n120_a-2to3{infix}_{DATA}.csv")
    rows = pd.concat([d.loc[d.groupby("alpha").lnL.idxmax()], d.nlargest(NCAND, "lnL")]).drop_duplicates(["alpha", "betaH"])
    par = rows[["alpha", "betaH", "Treh"]].to_numpy()
    cands = [np.asarray(spectrum_fopt(F, *p), float) for p in par]
    seeds = bfp.start_points(np.random.default_rng(20261110), 256, tag)[:, FREE]
    print(f"{tag}: {len(cands)} candidate transitions", flush=True)
    return [(tag, tau, nu, cands, par, seeds) for tau in taus for nu in NU]


if __name__ == "__main__":
    from multiprocessing import get_context
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    work = [w for tag, infix in PANELS for w in work_items(tag, infix, TAU)]
    with get_context("fork").Pool(jobs) as pool:
        res = pool.map(cell, work, chunksize=1)
    out = {"tau": TAU, "nu": NU}
    for tag, _ in PANELS:
        LPT = np.full((len(TAU), len(NU)), np.nan); LB = LPT.copy()
        for w_, tau, nu, lpt, lbin in res:
            if w_ == tag:
                i, j = int(np.where(TAU == tau)[0][0]), int(np.where(NU == nu)[0][0])
                LPT[i, j], LB[i, j] = lpt, lbin
        out[f"lnL_pt_{tag}"], out[f"lnL_bin_{tag}"] = LPT, LB
        out[f"sigma_{tag}"] = np.vectorize(lambda g: sigma_from_gain(g, 3))(LPT - LB)
        gfree = np.nanmax(LPT) - np.nanmax(LB)
        print(f"{tag}: sigma {np.nanmin(out[f'sigma_{tag}']):.2f} to {np.nanmax(out[f'sigma_{tag}']):.2f}; "
              f"max lnL with transition {np.nanmax(LPT):.3f}, binaries alone {np.nanmax(LB):.3f}, gain over the map "
              f"{gfree:.2f} ({sigma_from_gain(gfree, 3):.2f} sigma)", flush=True)
    np.savez(SCANS / "hardening_map.npz", **out)
    print(f"wrote hardening_map.npz; {time.time() - t0:.0f} s", flush=True)
