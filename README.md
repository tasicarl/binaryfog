# binaryfog

The supermassive black hole binary background as a nuisance in pulsar timing array searches for
new physics.

A pulsar timing array measures one gravitational-wave background. Any new-physics source has to
share it with the inspiralling supermassive black hole binaries, so a limit on new physics is only
as good as the assumption made about them. The usual assumption is a power law with amplitude `A`
and slope `gamma` drawn from a fixed two-dimensional Gaussian. That hides two things: the
astrophysical parameters that produce the background are uncertain, and the background is not a
curve at all. The binaries are a finite random set of sources, so each universe realises a
different spectrum, and in the highest frequency bins that scatter reaches two fifths of the
measurement uncertainty.

`binaryfog` supplies the distribution of backgrounds predicted by a holodeck population model
without calling holodeck, integrates over it inside the likelihood, and profiles over the
astrophysical parameters, so that a limit carries the astrophysical uncertainty instead of assuming
it away.

> **Preliminary release.** Version 0.1.0 accompanies a paper in preparation. The interface may still
> change, and the accuracy of the surrogate is quantified under *What is inside* below: it is good
> enough for effects of a few units in `ln L`, but not by a wide margin.

## Install

```
pip install -e .            # numpy and scipy; pandas for tables, pytest for the tests
python -m binaryfog.fetch   # the measured free spectra, into ~/.binaryfog/freespec
```

`python -m binaryfog.fetch` downloads eight free-spectrum data products from their Zenodo records,
about 25 MB unpacked. All are CC-BY-4.0 and none of them is redistributed here.

| name | array, release and noise model |
|---|---|
| `ng15_cp_v1` | NANOGrav 15 yr, 2024 release, uncorrelated common process |
| `ng15_hd_v1` | the same, Hellings–Downs |
| `ng15_hdall_v1` | the same, Hellings–Downs inferred with a monopole, a dipole and a common process |
| `ng15_hd_dmgp_v1` | the same, dispersion measure as a Gaussian process rather than piecewise offsets |
| `ipta2_cp_v1` | IPTA DR2, uncorrelated common process |
| `ng15_cp_v2` | NANOGrav 15 yr, 2026 re-run, uncorrelated common process |
| `ng15_hd_v2` | the same, Hellings–Downs |
| `ng15_hdall_v2` | the same, Hellings–Downs with the other correlated components |

The 2026 products are the ones re-run after the parallel-tempering bug in `PTMCMCSampler`
(erratum, *Astrophys. J. Lett.* **1006** (2026) L67). That record does not carry a corrected
Gaussian-process product, so `ng15_hd_dmgp_v1` is the 2024 one. Each download is pinned to a
specific Zenodo version, so a later release cannot silently change what you get. Which product is used changes limits by more than the choice of
correlation pattern does.

## Quickstart

```python
import numpy as np, binaryfog as bf

data = bf.FreeSpectrum.load("ng15_cp_v2")          # NANOGrav 15 yr, corrected 2026 release
like = bf.Likelihood(data, bf.Background())        # with the binary background marginalised

def source(f_Hz, log10_amp, tilt):                 # your model: h^2 Omega_GW, dimensionless
    return 10.0 ** log10_amp * (f_Hz / 1e-8) ** tilt

scan = bf.ProfileScan(like, source,
                      scan={"tilt": np.linspace(-2.0, 3.0, 41)},
                      nuisance={"log10_amp": (-12.0, -6.0, "linear")},
                      astro_prior="nanograv")
table = scan.run(jobs=8)
allowed = table[bf.region(table, 1, ndim=1)]     # the 1 sigma interval
print(bf.preference(table, like), "ln L gained over the binaries alone")
```

Every spectrum is `h^2 Omega_GW(f)` with `f` in Hz. The model is any callable whose first argument
is the frequency array; its other arguments are either scanned, profiled as nuisance parameters, or
held fixed.

The model files of the NANOGrav 15 yr new-physics search run directly, so the sources studied there
need not be rewritten. All fifteen stochastic backgrounds load: phase transitions from sound waves
and from bubble collisions, domain walls, scalar-induced waves, stable, metastable and super
strings, and inflationary waves. The files that define a deterministic timing signal instead, from
ultralight dark matter or black-hole substructure, are not backgrounds and are refused.

```python
m = bf.load_ptarcade("models_1.0.1/pt_sound.py")      # needs PTArcade installed
scan = bf.ProfileScan(like, m.spectrum,
                      scan={"log10_alpha": np.linspace(-2, 1, 30)},
                      nuisance=m.nuisance("log10_T_star", "log10_H_R", "a", "b", "c"))
```

## Comparing treatments of the binaries

The point of the package is that the answer depends on how the binaries are treated, so it makes
the treatments easy to swap:

```python
alone      = bf.Likelihood(data)                                   # no background at all
published  = bf.Likelihood(data)                                   # their power law, their prior
scan_pub   = bf.ProfileScan(published, bf.Sum(model.spectrum, bf.power_law), scan=...,
                            nuisance={"log10_A": (-18, -13, "linear"), "gamma": (0, 7, "linear")},
                            nuisance_prior=bf.ln_prior_agamma, astro_prior=None)
population = bf.Likelihood(data, bf.Background())                  # this package's background
```

`bf.reference_lnL` gives the best fit without the new physics for any of them, which is what turns
a scan into a statement about whether the source is preferred.

## Inside someone else's sampler

The common case is a Markov chain over the parameters of a new-physics model, in which the six
astrophysical parameters are dealt with at every step. Two ways, with very different costs:

```python
like = bf.Likelihood(bf.FreeSpectrum.load("ng15_cp_v2"), bf.Background())

# (a) carry the astrophysics as six more sampled dimensions: ~3 ms per call
lnL = like.ln_like(my_spectrum(freqs, **my_params), theta_astro)

# (b) profile it out at each step: ~6 s per call, and it opens its own process pool,
#     so pass jobs=1 if the chain is already parallel
scan = bf.ProfileScan(like, my_spectrum, scan={k: np.array([v]) for k, v in my_params.items()},
                      astro_prior="nanograv", restarts=4, sweeps=0)
lnL = float(scan.run(jobs=1)["lnL"].iloc[0])
```

Notes for (b), measured on the dark sector phase transition of the paper, three model parameters
on top of the six astrophysical ones:

* One profiled point takes about six seconds, against three milliseconds for one likelihood call.
  Worth it only if a step of the surrounding chain already costs seconds.
* Without a grid there is no neighbour to start from, so set `sweeps=0`. The profile is still
  stable: four, eight and sixteen restarts agree to four decimals, and along a line in one
  parameter no point dips below its neighbours. A model with a larger nuisance space may need more
  restarts, and the check is the one just described.
* The optimiser is seeded per point, so evaluating the same parameters twice gives the same number.
  A sampler needs that.

## What is inside

* **The background** is a surrogate of holodeck's classic phenomenological model: from the six
  parameters `bf.PARAM_NAMES` it predicts the median spectrum, the 16 to 84 per cent half-width of
  `log10 h_c^2` in dex, and one skew number per model. It was fitted to 2000 holodeck models with
  2000 realisations each. On 150 models held out of the fit it reproduces the median to 0.05 dex,
  the half-width to 4 per cent and the skew to 0.05 for a typical model, and to 0.15 dex, 15 per
  cent and 0.17 for the worst tenth of them. Evaluating it takes 0.18 milliseconds against the
  180 seconds holodeck needs for the same thing.
  The library and the fit can be redone with the scripts in `training/`.
* **The realisation distribution** is neither log-normal nor Gaussian. `binaryfog` uses the warp
  `log10 h_c^2 = median + sigma (z + c z^2)` with `z` normal, which beats a log-normal by 74 per
  cent in mean quantile-ladder error.
* **The data** are the free-spectrum kernel densities that `ceffyl` and PTArcade read, with
  `ceffyl`'s conventions: the density is held flat below the grid and the model is rejected above
  it. The 2024 products carry a grid running far below the range the Markov chains visited, where
  the stored density is a numerical floor with isolated single-sample spikes that a profile
  likelihood can find; the 2026 products keep only the sampled range.
* **The priors** on the astrophysical parameters are the four sets of the paper, chosen with
  `astro_prior`: `"flat"` (the library box), `"nanograv"` (the Gaussians of arXiv:2306.16220),
  `"tomczak_kormendy"` (the centres and uncertainties of the fits those Gaussians derive from) and
  `"survey_average"` (the spread of mass-function measurements published since 2014). Limits should
  be quoted for more than one of them. For the cross-check with the `(A, gamma)` power law,
  `bf.ln_prior_agamma` is NANOGrav's Gaussian on its amplitude and slope, passed as `nuisance_prior`.

## Reproducing the paper

`paper/` holds the scripts that produce every figure and number of the paper, together with their inputs:
the holodeck library, the free-spectrum grids and the stored scan outputs. `paper/run_figures.sh` rebuilds the
figures and numbers from those inputs, `paper/run_scans.sh` reruns the scans; see `paper/README.md`.

## Tests

```
pip install -e ".[test]"
pytest                                                 # about 20 s
pip install -e ".[models]"                              # PTArcade, only to read the model files
BINARYFOG_NG15_MODELS=<unpacked model files> pytest    # adds the model-file tests
```

Tests whose inputs are missing skip themselves and say what is missing. The free-spectrum tests
need `python -m binaryfog.fetch` to have run. The model files are the ones released with the
NANOGrav 15 yr new-physics search ([Zenodo 8092761](https://doi.org/10.5281/zenodo.8092761)) and
reading them needs PTArcade, which is the `models` extra. Without it those tests
say so and skip rather than failing on an import error. One test file compares the package value for value against
the research scripts it was extracted from, and skips itself outside that repository.

## Limits of this version

* The six-parameter maps cover the NANOGrav 15 yr band, 1.98 to 27.7 nHz. Another array needs its
  own library. `training/` holds the scripts that build one from public holodeck and fit the maps
  to it (see `training/README.md`); `Background(maps="maps.npz")` then loads the result. Mind that a data product defines
  its own frequencies: the IPTA DR2 free spectrum is not a set of harmonics of its time span, its
  bins starting at 1.0496 nHz and spaced by 1.0858 nHz, and `Likelihood` therefore evaluates the
  background on the frequencies stored with the data.
* The realisation distribution is carried bin by bin. The correlations between bins are dropped,
  as they are in the analyses this package is compared against.
* Wilks' theorem is used for the interval thresholds (`LEVELS_1D`, `LEVELS_2D`). It does not hold
  where the new-physics amplitude is bounded below by zero and its other parameters stop meaning
  anything there. `sigma_from_gain` handles the boundary as a Chernoff mixture, which makes a
  significance *larger* than a plain chi-square would; the parameters being undefined rather than
  merely unconstrained pushes the other way and is not handled. See `significance.py` for both.
  The tail probability is quoted one-sided, Z = Phi^-1(1 - p) (Cowan et al. 2011), so that one
  parameter gives Z = sqrt(2 G); `sided="two"` gives the two-sided convention of Macias et al. 2018.
  A difference between two fits converted the same way is insensitive to this; an absolute
  significance is not.
* A profile over many parameters can have more than one local maximum at a grid point, and the
  neighbour sweeps cannot repair a point whose neighbours sit in the lower one too. For a phase
  transition with the six astrophysical parameters free, one point needed 24 restarts to find the
  higher maximum, which lies 0.5 above the lower in ln L. Rerun isolated low points with more
  restarts before reading a contour off a scan.

## Citing

Please cite the paper this package accompanies, which is in preparation, together with the software
itself (see `CITATION.cff`). `binaryfog` builds on [holodeck](https://github.com/nanograv/holodeck)
for the populations, on [`ceffyl`](https://github.com/astrolamb/ceffyl) for the shape of the data
products, on [PTArcade](https://github.com/andrea-mitridate/PTArcade) for the new-physics model
files, and on the NANOGrav 15 yr data set.

## Licence

MIT, see `LICENSE`.
