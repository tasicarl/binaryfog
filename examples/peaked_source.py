"""
A self-contained example: a peaked new-physics source against the NANOGrav 15 yr free spectrum,
with and without the binary background.

The source is a broken power law, rising as f^3 like a causal cosmological background and falling
as f^-4 above its peak,
    h^2 Omega(f) = amp * (f/f_peak)^3 / (1 + (f/f_peak)^7)^(1)  (up to the shape below),
so that it has two parameters: the peak frequency and the amplitude. The scan profiles over the
amplitude and over the six astrophysical parameters of the background.

    python peaked_source.py            (a few minutes on 8 cores)
"""
import numpy as np

import binaryfog as bf


def peaked(f_Hz, log10_amp, log10_fpeak):
    x = np.asarray(f_Hz) / 10.0 ** log10_fpeak
    return 10.0 ** log10_amp * x ** 3 / (1.0 + x ** 7)


if __name__ == "__main__":
    data = bf.FreeSpectrum.load("ng15_hd_v2")
    fpeak = np.linspace(-8.5, -7.0, 16)

    for label, background in (("without the binaries", None), ("with the binaries", bf.Background())):
        like = bf.Likelihood(data, background)
        scan = bf.ProfileScan(like, peaked, scan={"log10_fpeak": fpeak},
                              nuisance={"log10_amp": (-12.0, -5.0, "linear")},
                              astro_prior="nanograv", restarts=3)
        table = scan.run(jobs=8)
        inside = bf.region(table, 2, ndim=1)
        lo, hi = table["log10_fpeak"][inside].min(), table["log10_fpeak"][inside].max()
        best = table.loc[table["lnL"].idxmax()]
        print(f"\n{label}:")
        print(f"  best fit: peak at {10 ** best['log10_fpeak'] * 1e9:.1f} nHz, "
              f"amplitude 1e{best['log10_amp']:.2f}, ln L = {best['lnL']:.2f}")
        print(f"  2 sigma interval of the peak frequency: {10 ** lo * 1e9:.1f} to {10 ** hi * 1e9:.1f} nHz")
        if background is not None:
            print(f"  gain over the binaries alone: {bf.preference(table, like):.2f} ln L units")
