"""
Profile-likelihood scan over the phase-transition strength alpha and speed beta/H (section 5.1, table 2).

At every grid point the likelihood is maximised over the nuisance parameters:

  --mode fopt      T_reh only                        (phase transition alone)
  --mode agamma    T_reh, log10 A, gamma             (the (A, gamma) power law, with its Gaussian prior)
  --mode holodeck  T_reh and the six astrophysical parameters, with the binary background
                   marginalised over its realisations, eq. (3.2), and the astrophysical prior added

FREESPEC names the free spectrum, a directory of inputs/pta/freespec. PRIOR_SET selects the astrophysical
prior of the holodeck mode (table 1): nanograv (default), tomczak_kormendy, survey_average or flat.
PT_TEMPLATE=lv25 replaces the sound-wave spectrum of appendix A by the strongly supercooled template of
fopt_lv25.py, which does not depend on alpha and is therefore scanned along beta/H alone (--line).

    FREESPEC=ng15_cp_v2 python run_scan_marginal.py --mode holodeck -n 120 -j 28 --alpha-hi 3
    FREESPEC=ng15_cp_v2 python run_scan_marginal.py --mode agamma -n 50 -j 28
    FREESPEC=ng15_cp_v2 PT_TEMPLATE=lv25 python run_scan_marginal.py --mode holodeck --line -n 300 -j 28 --alpha-hi 3
"""
import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp

import numpy as np
from scipy import optimize

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from fopt_model import spectrum_fopt                                          # noqa: E402
from fopt_lv25 import spectrum_lv25                                           # noqa: E402

PT_TEMPLATE = os.environ.get("PT_TEMPLATE", "")
if PT_TEMPLATE not in ("", "lv25"):
    raise SystemExit(f"unknown PT_TEMPLATE {PT_TEMPLATE!r}")
spectrum = spectrum_lv25 if PT_TEMPLATE == "lv25" else spectrum_fopt

PRIOR_SET = os.environ.get("PRIOR_SET", "nanograv")
if PRIOR_SET not in ("flat", *bf.priors.GAUSSIAN_SETS):
    raise SystemExit(f"unknown PRIOR_SET {PRIOR_SET!r}")
if "FREESPEC" not in os.environ:
    raise SystemExit("set FREESPEC to one of the free spectra in inputs/pta/freespec")
FREESPEC = os.environ["FREESPEC"]
DATA = bf.FreeSpectrum.load(FREESPEC, root=FREESPEC_ROOT)
LIKE = bf.Likelihood(DATA, bf.Background())
FREQS = DATA.freqs

NRESTART_AG = 6       # simplex restarts per point, agamma mode
NRESTART_HD = 4       # at least this many in the holodeck mode
# T_reh in GeV from 1 MeV, the BBN bound, to 10 GeV (appendix A). The supercooled template is profiled
# on the points of a 500-point grid from 0.1 MeV that lie inside this range.
TREH_MIN = 1e-3
TREH_GRID = np.logspace(-4, 1, 500) if PT_TEMPLATE == "lv25" else np.logspace(-3, 1, 400)
TREH_GRID = TREH_GRID[TREH_GRID >= TREH_MIN]
TREH_MAX = float(TREH_GRID[-1])
NTOP_TREH = 5         # local refinements per point, see best_treh
# Every restart of the holodeck mode starts near the centres of the NANOGrav priors, whichever prior set
# is used.
CENTRES = np.array([mu for mu, _ in bf.NANOGRAV_GAUSSIAN.values()])


def ln_like(h2om):
    """Free-spectrum likelihood of a deterministic spectrum, no binaries."""
    return DATA.ln_like_point(h2om)


def ln_prior_ag(log10_A, gamma):
    return bf.ln_prior_agamma({"log10_A": log10_A, "gamma": gamma})


def profile_point(job):
    alpha, betaH, mode, seed = job
    rng = np.random.default_rng(seed)

    def best_treh(extra):
        """Maximise over T_reh on TREH_GRID, then refine with a bounded search between the neighbours of
        each of the NTOP_TREH best grid points."""
        vals = np.array([ln_like(spectrum(FREQS, alpha, betaH, t) + extra) for t in TREH_GRID])
        k = int(np.argmax(vals))
        best_t, best_v = float(TREH_GRID[k]), float(vals[k])
        for j in np.argsort(vals)[-NTOP_TREH:]:
            lo = TREH_GRID[max(int(j) - 1, 0)]
            hi = TREH_GRID[min(int(j) + 1, len(TREH_GRID) - 1)]
            if hi <= lo:
                continue
            r = optimize.minimize_scalar(
                lambda t: -ln_like(spectrum(FREQS, alpha, betaH, t) + extra),
                bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
            if -r.fun > best_v:
                best_t, best_v = float(r.x), float(-r.fun)
        return best_t, best_v

    if mode == "fopt":
        treh, best = best_treh(0.0)
        return dict(alpha=alpha, betaH=betaH, lnL=best, lnprior=0.0, lnL_data=best, Treh=treh)

    if mode == "agamma":
        def neg(x):
            if x[2] < TREH_MIN or x[2] > TREH_MAX:
                return 1e6
            v = ln_like(spectrum(FREQS, alpha, betaH, x[2]) + bf.power_law(FREQS, x[0], x[1])) \
                + ln_prior_ag(x[0], x[1])
            return -v if np.isfinite(v) else 1e6
        mu = bf.AGAMMA_MU
        treh0, _ = best_treh(bf.power_law(FREQS, mu[0], mu[1]))
        best = None
        for _ in range(NRESTART_AG):
            x0 = np.array([mu[0] + rng.normal(0, .2), mu[1] + rng.normal(0, .3),
                           treh0 * np.exp(rng.normal(0, .2))])
            r = optimize.minimize(neg, x0, method="Nelder-Mead",
                                  options={"maxiter": 1200, "fatol": 1e-7})
            if best is None or r.fun < best.fun:
                best = r
        # Re-profile T_reh on its grid at the converged background and restart the simplex from there
        # while that improves the fit.
        for _ in range(3):
            t, v = best_treh(bf.power_law(FREQS, best.x[0], best.x[1]))
            if v + ln_prior_ag(best.x[0], best.x[1]) <= -best.fun + 1e-9:
                break
            r = optimize.minimize(neg, np.array([best.x[0], best.x[1], t]),
                                  method="Nelder-Mead",
                                  options={"maxiter": 1200, "fatol": 1e-8})
            if r.fun >= best.fun:
                break
            best = r
        lp = ln_prior_ag(best.x[0], best.x[1])
        return dict(alpha=alpha, betaH=betaH, lnL=-best.fun, lnprior=lp,
                    lnL_data=-best.fun - lp, Treh=best.x[2],
                    logA=best.x[0], gamma=best.x[1])

    # holodeck: T_reh plus the six astrophysical parameters, binary part marginalised
    def neg(x):
        lp = bf.ln_prior(x[1:], PRIOR_SET)
        if not np.isfinite(lp) or x[0] < TREH_MIN or x[0] > TREH_MAX:
            return 1e6
        v = LIKE.ln_like(spectrum(FREQS, alpha, betaH, x[0]), x[1:]) + lp
        return -v if np.isfinite(v) else 1e6

    treh0, _ = best_treh(0.0)
    # The likelihood has separate maxima in T_reh (near 1 MeV and at GeV for slow transitions), and the
    # binaries can make either the better one. The restarts therefore cycle through the best T_reh of the
    # transition alone, its two best other local maxima in T_reh and the lower bound.
    vals0 = np.array([ln_like(spectrum(FREQS, alpha, betaH, t)) for t in TREH_GRID])
    peaks = [i for i in range(len(vals0))
             if vals0[i] >= vals0[max(i - 1, 0)] and vals0[i] >= vals0[min(i + 1, len(vals0) - 1)]]
    starts = [treh0] + [float(TREH_GRID[i]) for i in sorted(peaks, key=lambda i: -vals0[i])
                        if abs(np.log10(TREH_GRID[i] / treh0)) > 0.3][:2] + [TREH_MIN * 1.05]
    best = None
    for k in range(max(NRESTART_HD, len(starts))):
        th0 = np.concatenate([CENTRES + rng.normal(0, [.2, .2, .1, .07]),
                              [rng.uniform(0.2, 9.0), rng.uniform(-1.4, -0.1)]])
        t_start = starts[k % len(starts)]
        x0 = np.concatenate([[min(max(t_start * np.exp(rng.normal(0, .15)), TREH_MIN * 1.001),
                                  TREH_MAX * 0.999)], th0])
        r = optimize.minimize(neg, x0, method="Nelder-Mead",
                              options={"maxiter": 4000, "fatol": 1e-7, "xatol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    # the same re-profiling of T_reh as in the agamma mode
    for _ in range(3):
        lpp = bf.ln_prior(best.x[1:], PRIOR_SET)
        if not np.isfinite(lpp):
            break
        med, sig, sk = LIKE.background.predict(best.x[1:])
        vals = [LIKE.ln_like_from_distribution(spectrum(FREQS, alpha, betaH, t), med, sig, sk)
                for t in TREH_GRID]
        kk = int(np.argmax(vals))
        if vals[kk] + lpp <= -best.fun + 1e-9:
            break
        x0 = np.concatenate([[TREH_GRID[kk]], best.x[1:]])
        r = optimize.minimize(neg, x0, method="Nelder-Mead",
                              options={"maxiter": 4000, "fatol": 1e-8, "xatol": 1e-6})
        if r.fun >= best.fun:
            break
        best = r
    lp = bf.ln_prior(best.x[1:], PRIOR_SET)
    out = dict(alpha=alpha, betaH=betaH, lnL=-best.fun, lnprior=lp, lnL_data=-best.fun - lp,
               Treh=best.x[0])
    out.update({k: v for k, v in zip(bf.PARAM_NAMES, best.x[1:])})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fopt", "agamma", "holodeck"], required=True)
    ap.add_argument("-n", "--ngrid", type=int, default=50)
    ap.add_argument("-j", "--jobs", type=int, default=28)
    ap.add_argument("--alpha-hi", type=float, default=10.0,
                    help="log10 of the largest alpha; the smallest is 0.01")
    ap.add_argument("--line", action="store_true",
                    help="scan beta/H alone at alpha = 10^alpha-hi, with ngrid points")
    args = ap.parse_args()

    # alpha from 0.01, beta/H from 0.7 to 300, both logarithmic
    alpha_r = np.logspace(-2.0, args.alpha_hi, args.ngrid)
    betaH_r = np.logspace(np.log10(0.7), np.log10(300.0), args.ngrid)
    arange = "" if args.alpha_hi == 10.0 else f"_a-2to{args.alpha_hi:g}"
    if args.line:
        alpha_r = np.array([10.0 ** args.alpha_hi])
        arange = f"_line_a{args.alpha_hi:g}_b300"
    if PT_TEMPLATE:
        arange += f"_{PT_TEMPLATE}"
    if PRIOR_SET != "nanograv":
        arange += f"_{PRIOR_SET}"
    jobs = [(a, b, args.mode, i) for i, (a, b) in
            enumerate((a, b) for a in alpha_r for b in betaH_r)]
    out = SCANS / f"scan_{args.mode}_n{args.ngrid}{arange}_{FREESPEC}.csv"
    print(f"{args.mode}: {len(jobs)} grid points on {args.jobs} workers, prior set {PRIOR_SET}, "
          f"data {FREESPEC} -> {out.name}", flush=True)

    t0 = time.perf_counter()
    rows = []
    with mp.Pool(args.jobs) as pool:
        for n, r in enumerate(pool.imap(profile_point, jobs), 1):
            rows.append(r)
            if n % 200 == 0 or n == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {n}/{len(jobs)} | {el/60:.1f} min | eta {el/n*(len(jobs)-n)/60:.1f} min",
                      flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min, max lnL = "
          f"{max(r['lnL'] for r in rows):.3f}", flush=True)
