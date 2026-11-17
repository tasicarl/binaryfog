"""
The numbers the paper quotes for the phase-transition fit under the NANOGrav priors (sections 5.1 and 5.4),
from the 120 x 120 scans of run_scan_marginal.py on the CURN free spectrum:

  1. the binaryfog best fit and the ridge of points within 1e-4 in ln L of the maximum;
  2. the gain G over the fit with no transition, split into the part that comes from describing the
     data better and the part that is the prior penalty the binaries pay when they have to account
     for the measurement on their own;
  3. where the astrophysical parameters sit, in units of their prior widths, in the two fits;
  4. under flat priors (the 50 x 50 scan), the fraction of the plane inside the 1 sigma region and the preference.

The fit with no transition is the best point of the weakest-transition column of the same scan,
where the transition contributes nothing to the spectrum in band.
"""
import pandas as pd

from paths import SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import TAG                                                      # noqa: E402

RIDGE = 1e-4

if __name__ == "__main__":
    for mode in ("agamma", "holodeck"):
        d = pd.read_csv(SCANS / f"scan_{mode}_n120_a-2to3{TAG}.csv")
        best = d.loc[d.lnL.idxmax()]
        weak = d[d.alpha == d.alpha.min()]
        alone = weak.loc[weak.lnL.idxmax()]
        print(f"{mode}:")
        print(f"    gain {best.lnL - alone.lnL:+.2f} = data {best.lnL_data - alone.lnL_data:+.2f} "
              f"+ prior {best.lnprior - alone.lnprior:+.2f}")
        print(f"    ln L {best.lnL:+.2f} with a transition, {alone.lnL:+.2f} for the binaries alone")
        if mode == "holodeck":
            ridge = d[d.lnL >= best.lnL - RIDGE]
            print(f"    best fit alpha {best.alpha:.3g}, beta/H {best.betaH:.3g}, T_reh {best.Treh * 1e3:.3g} MeV; "
                  f"{len(ridge)} points within {RIDGE:g} of the maximum, alpha {ridge.alpha.min():.3g} to "
                  f"{ridge.alpha.max():.3g}, beta/H {ridge.betaH.min():.3g} to {ridge.betaH.max():.3g}, "
                  f"T_reh {ridge.Treh.min() * 1e3:.3g} to {ridge.Treh.max() * 1e3:.3g} MeV")
            for k, (mu, sg) in bf.NANOGRAV_GAUSSIAN.items():
                print(f"      {k:20s} {(best[k]-mu)/sg:+5.2f} sigma with a transition, "
                      f"{(alone[k]-mu)/sg:+5.2f} sigma for the binaries alone")
            # the two hardening parameters carry no Gaussian, so they are reported as values
            for k in ("hard_time", "hard_gamma_inner"):
                print(f"      {k:20s} {best[k]:+6.3f} with a transition, "
                      f"{alone[k]:+6.3f} for the binaries alone (flat prior)")

    d = pd.read_csv(SCANS / f"scan_holodeck_n50_flat{TAG}.csv")
    weak = d[d.alpha == d.alpha.min()]
    gain = d.lnL.max() - weak.lnL.max()
    inside = (d.lnL >= d.lnL.max() - bf.LEVELS_2D[1]).mean()
    print(f"holodeck, flat priors: {100 * inside:.1f} per cent of the plane inside 1 sigma; "
          f"gain {gain:+.2f}, {bf.sigma_from_gain(gain, 3):.2f} sigma")
