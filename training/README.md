# Training the maps

The maps in `binaryfog/data` are polynomial fits to a library of holodeck population models. The
two scripts here rebuild such a library from public holodeck and fit the maps to it, for example
for a new holodeck release or for the frequency bins of another pulsar timing array.

| script | input | output | time |
|---|---|---|---|
| `build_library.py` | parameter points, frequency grids | `library.jsonl` | about 7 core-minutes per point at 2000 realisations |
| `fit_maps.py` | `library.jsonl`, one grid | `maps.npz` | a few seconds |

## Install holodeck

The library needs holodeck **v1.7.1 or later from GitHub**. The newest version on PyPI
(`holodeck-gw` 1.6) does not contain the redshift evolution of the characteristic mass of the
galaxy stellar mass function (holodeck pull request #117, the erratum to the NANOGrav 15-year
astrophysics paper), and `build_library.py` refuses to run on it.

```
git clone https://github.com/nanograv/holodeck.git
cd holodeck && git checkout v1.7.1
pip install -r requirements.txt && pip install -e .
```

holodeck 1.7.1 declares Python 3.11 or later.

## Build a library

```
python build_library.py --grid ng15=ng15_hd_v2:14 --npoints 2000 --nreals 2000 --jobs 56 --out library.jsonl
```

Each `--grid name=source:nbins` is a frequency grid. `source` is the name of a data product that
`binaryfog.FreeSpectrum.load` knows, whose stored frequencies are taken as the bin centres, or a
`.npy` file of bin centres in Hz. Several grids can be given at once; holodeck then evolves each
population once for all of them. The parameter points are a Latin hypercube over the uniform prior
box of holodeck's `PS_Classic_Phenom_Uniform` parameter space (galaxy stellar mass function
normalisation and characteristic mass, black hole to bulge mass relation and its scatter, binary
lifetime and inner hardening index; the outer hardening index and the break separation at their
defaults). `--points library.jsonl` reuses the points of an existing library instead, for instance
to add the grid of another array. The run can be interrupted and resumed with the same command.

The first line of the output records the holodeck version and commit, the grids and the number of
realisations.

Binaries whose host galaxies merge so late that the binary has not merged by today get a negative
redshift from holodeck. They are removed: every cell of the mass, mass ratio and redshift grid with
such a corner contributes nothing. holodeck's own library code does the same since its pull request
#133, which is not part of any release yet; the script therefore does it itself and gives the same
result on every version from 1.7 on.

holodeck draws the Poisson realisations from a random generator that it seeds itself, so two runs
of the same point agree exactly in the continuum and the source count but only to within draw noise
in the quantiles.

## Fit the maps

```
python fit_maps.py library.jsonl --grid ng15 --out maps_ng15.npz --compare ng15
```

prints the accuracy of each map on a quarter of the points held out of its fit and, with
`--compare`, how far the new maps lie from shipped ones. The result is used as

```python
bg = binaryfog.Background(maps="maps_ng15.npz")
```

## Checks

- The shipped maps (`data/maps_ng15.npz`, `data/maps_ipta2.npz`) were fitted to a library of the same
  2000 points built with holodeck at commit 44fe3085 (pull request #117), with the binaries that have
  not merged by today removed. Their provenance string records this.
- `build_library.py` on public holodeck v1.7.1 reproduces that library to 7e-15 dex in the continuum
  and exactly in the effective source count, on both grids. The medians differ by the draw noise of
  2000 realisations: none of 87650 bins lies beyond four times the expected noise. The standard
  deviations agree to 4 per cent (16 to 84 per cent range).
- The training and held-out split is fixed by the index of a point in the design, so that the fit
  does not depend on the order in which the library was written.
