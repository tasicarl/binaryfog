"""
Repeat the fit of figure 9 of NANOGrav's 15-year new-physics paper with binaryfog, as an end-to-end
check of the likelihood (figure 4 of the paper).

That figure is the posterior of the six binary-evolution parameters of the phenomenological
holodeck library, fitted to the 15 yr Hellings-Downs-plus-monopole-plus-dipole-plus-common-process
free spectrum over its five lowest frequency bins, under two prior sets: flat over the library box,
and the observationally motivated Gaussians on the galaxy and black hole parameters.

Nothing about that fit involves new physics, so it exercises exactly the parts of binaryfog a
new-physics limit depends on: the surrogate, the marginalisation over realisations, the priors and
the free-spectrum likelihood. NANOGrav used Gaussian processes trained on the mean and standard
deviation of the same library; we use the warp of the paper. Agreement is therefore a
statement that the two independent pipelines put the astrophysics in the same place.

    python validate_ng15_fig9.py --data ng15_hdall_v2 --nbins 5
"""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402

PRIORS = ("flat", "nanograv")


def sampler_for(data, which, nwalk, steps, burn, seed):
    import emcee
    bg = bf.Background(data.freqs)
    like = bf.Likelihood(data, bg, marginalise=True)
    names = list(bf.PARAM_NAMES)
    lo = np.array([bg.prior_box[k][0] for k in names])
    hi = np.array([bg.prior_box[k][1] for k in names])

    def ln_post(th):
        lp = bf.ln_prior(th, which)
        if not np.isfinite(lp):
            return -np.inf
        ll = like.ln_like(None, th)
        return lp + ll if np.isfinite(ll) else -np.inf

    rng = np.random.default_rng(seed)
    # start from the prior: uniform over the box, and for the Gaussian set the Gaussian parameters
    # drawn from the Gaussian priors
    p0 = rng.uniform(lo, hi, size=(nwalk, len(names)))
    if which == "nanograv":
        for k, n in enumerate(names):
            if n in bf.priors.NANOGRAV_GAUSSIAN:
                mu, sd = bf.priors.NANOGRAV_GAUSSIAN[n]
                p0[:, k] = np.clip(rng.normal(mu, sd, nwalk), lo[k], hi[k])
    s = emcee.EnsembleSampler(nwalk, len(names), ln_post)
    s.run_mcmc(p0, steps, progress=False)
    chain = s.get_chain(discard=burn, thin=10, flat=True)
    acc = float(np.mean(s.acceptance_fraction))
    try:
        tau = float(np.max(s.get_autocorr_time(quiet=True)))
    except Exception:
        tau = float("nan")
    print(f"  {which:9s}: {len(chain)} samples, acceptance {acc:.2f}, "
          f"longest autocorrelation {tau:.0f} steps", flush=True)
    return chain, names, tau


def main(dataset, nbins, nwalk, steps, burn, seed, out):
    data = bf.FreeSpectrum.load(dataset, root=FREESPEC_ROOT, nbins=nbins)
    print(f"{dataset}, {data.nbins} bins up to {data.freqs[-1] * 1e9:.2f} nHz")
    res = {}
    for which in PRIORS:
        chain, names, tau = sampler_for(data, which, nwalk, steps, burn, seed)
        res[which] = chain
        res[f"tau_{which}"] = tau     # longest integrated autocorrelation time, in steps
        for k, n in enumerate(names):
            q = np.percentile(chain[:, k], [16, 50, 84])
            print(f"      {n:20s} {q[1]:+7.3f}  (+{q[2]-q[1]:.3f} -{q[1]-q[0]:.3f})")
    np.savez_compressed(out, names=np.array(names), **res)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="ng15_hdall_v2")
    ap.add_argument("--nbins", type=int, default=5)
    ap.add_argument("--walkers", type=int, default=64)
    # Defaults: the run behind figure 4
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--burn", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="ng15_fig9_binaryfog_erratum.npz", help="output file, written to scans/")
    a = ap.parse_args()
    main(a.data, a.nbins, a.walkers, a.steps, a.burn, a.seed, SCANS / a.out)
