"""How much would the prior volume effect of Dai, Zaldarriaga and Ali-Haimoud (arXiv:2610.28739) move the
phase-transition preference? (section 5.3)

They find that NANOGrav's log-uniform priors on the intrinsic red noise of each pulsar pull the inferred
background amplitude up. Our free spectra inherit those priors. As a sensitivity test, the CURN free
spectrum (ng15_cp_v2) is moved down by delta = 0.1 and 0.2 in log10 rho in all 14 bins, and the gain
G = max ln L(phase transition + binaries) - max ln L(binaries), both with the log prior of the binary
parameters, is recomputed with the (A, gamma) power law and with binaryfog under the NANOGrav, the
Tomczak-Kormendy and the survey-average priors; Z from 1/2 delta + 1/2 chi2_3 (binaryfog.sigma_from_gain).

The maximisation is direct (Nelder-Mead over the binary parameters and log10 alpha, log10 beta/H,
log10 T_reh, in the box of run_scan_marginal.py), started at the stored best fit, along the best-fit ridge
and from prior draws. delta = 0 reproduces the preferences of table 2.

    python prior_volume_sensitivity.py          -> results/prior_volume_sensitivity.json
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

import numpy as np
from scipy import optimize

from paths import FREESPEC_ROOT, RESULTS, SCANS
import binaryfog as bf                                                        # noqa: E402
from binaryfog.priors import start_points                                     # noqa: E402
from fopt_model import spectrum_fopt                                          # noqa: E402

# log10 alpha, log10 beta/H, log10 T_reh/GeV: the box of the scans of run_scan_marginal.py
PT_LO = np.array([-2.0, np.log10(0.7), -3.0])
PT_HI = np.array([3.0, np.log10(300.0), 1.0])
DELTAS = (0.0, 0.1, 0.2)
PRIORS = ["power law", "nanograv", "tomczak_kormendy", "survey_average"]
# starting points in (alpha, beta/H) along the best-fit ridge, besides the stored best fit
RIDGE_STARTS = ((0.3, 0.8), (3.0, 6.0), (20.0, 20.0), (55.0, 46.0))


def shifted_data(delta):
    """The CURN free spectrum with the posterior of every bin moved down by delta in log10 rho."""
    data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT)
    orig = data.ln_density
    data.ln_density = lambda x, i: orig(np.asarray(x) + delta, i)
    return data


def pt_starts():
    """log10 (alpha, beta/H, T_reh) of the stored best fit, and the ridge points at its T_reh."""
    z = np.load(SCANS / "bestfit_holodeck.npz")
    a0, b0, t0 = z["pt"]
    return z, [np.log10([a0, b0, t0])] + [np.array([np.log10(a), np.log10(b), np.log10(t0)])
                                          for a, b in RIDGE_STARTS]


def maximise(neg, starts):
    best = -np.inf
    for x0 in starts:
        if not np.isfinite(neg(x0)):
            continue
        r = optimize.minimize(neg, x0, method="Nelder-Mead",
                              options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-6, "adaptive": True})
        if -r.fun > best:
            best, xbest = -r.fun, r.x
    return best, xbest


def gain_power_law(data):
    """The gain with the binaries as NANOGrav's (A, gamma) power law under its Gaussian prior."""
    like = bf.Likelihood(data, None, marginalise=False)
    lo = np.array([-18.0, 0.0]); hi = np.array([-13.0, 7.0])

    def lnpost(ag, pt=None):
        if np.any(ag < lo) or np.any(ag > hi):
            return -np.inf
        sb = bf.smbhb_power_law(data.freqs, *ag)
        npc = sb if pt is None else sb + spectrum_fopt(data.freqs, *(10.0 ** pt))
        return like.ln_like(npc) + bf.ln_prior_agamma({"smbhb_log10_A": ag[0], "smbhb_gamma": ag[1]})

    alone = max(-optimize.minimize(lambda x: -lnpost(x), x0, method="Nelder-Mead",
                                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7}).fun
                for x0 in ([-14.5, 3.5], [-14.2, 13 / 3], [-14.8, 4.0], [-15.0, 5.0]))

    def neg(x):
        if np.any(x[2:] < PT_LO) or np.any(x[2:] > PT_HI):
            return np.inf
        v = lnpost(x[:2], x[2:])
        return -v if np.isfinite(v) else np.inf

    _, sps = pt_starts()
    best, xbest = maximise(neg, [np.concatenate([ag, sp]) for sp in sps
                                 for ag in ([-14.6, 13 / 3], [-15.5, 13 / 3], [-14.3, 3.5], [-16.5, 13 / 3])])
    return alone, best, xbest[2:]


def gain_binaryfog(data, prior):
    """The gain with binaryfog under one of the Gaussian prior sets of table 1."""
    like = bf.Likelihood(data, bf.Background(), marginalise=True)
    names = list(like.background.param_names)
    box = dict(like.background.prior_box)
    lo = np.array([box[k][0] for k in names]); hi = np.array([box[k][1] for k in names])
    alone = bf.reference_lnL(like, astro_prior=prior, restarts=24)

    def neg(x):
        th, p = x[:6], x[6:]
        if np.any(th < lo) or np.any(th > hi) or np.any(p < PT_LO) or np.any(p > PT_HI):
            return np.inf
        v = like.ln_like(spectrum_fopt(data.freqs, *(10.0 ** p)), th) + bf.ln_prior(th, prior, names, box)
        return -v if np.isfinite(v) else np.inf

    z, sps = pt_starts()
    rng = np.random.default_rng(1)
    sts = [np.asarray(z["theta_with_pt"], float)] + list(start_points(rng, 6, prior, names, box))
    best, xbest = maximise(neg, [np.concatenate([np.clip(st, lo, hi), sp]) for sp in sps for st in sts])
    return alone, best, xbest[6:]


def job(args):
    delta, prior = args
    data = shifted_data(delta)
    alone, best, pt = gain_power_law(data) if prior == "power law" else gain_binaryfog(data, prior)
    G = best - alone
    return dict(delta=delta, prior=prior, lnL_alone=alone, lnL_pt=best, G=G,
                Z=bf.sigma_from_gain(G, 3), log10_pt=[float(v) for v in pt])


if __name__ == "__main__":
    jobs = [(d, p) for d in DELTAS for p in PRIORS]
    with mp.Pool(len(jobs)) as pool:
        out = pool.map(job, jobs)
    for r in out:
        a, b, _ = 10.0 ** np.array(r["log10_pt"])
        print(f"{'none' if r['delta'] == 0 else 'all bins':9s} delta {r['delta']:.1f}  {r['prior']:10s}  "
              f"G {r['G']:6.2f}  Z {r['Z']:.2f} sigma  (ln L alone {r['lnL_alone']:+.2f}, with PT {r['lnL_pt']:+.2f};"
              f" alpha {a:.3g}, beta/H {b:.3g})")
    (RESULTS / "prior_volume_sensitivity.json").write_text(json.dumps(out, indent=1))
