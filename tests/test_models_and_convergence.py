"""
The model adapter, the convergence of the profile maximisation, and a published-value check.

The NANOGrav model files are a separate data release; point BINARYFOG_NG15_MODELS at the unpacked
directory to run those tests. They also need PTArcade, which the files import, so run this file in
an environment that has it. The other test file needs holodeck instead, for the research scripts
it compares against, so the two files are usually run in different environments.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import binaryfog as bf                                                        # noqa: E402
from binaryfog.freespec import DEFAULT_DIR                                    # noqa: E402

MODELS = os.environ.get("BINARYFOG_NG15_MODELS")
# The model files import PTArcade themselves, so pointing BINARYFOG_NG15_MODELS at them is not
# enough: without it every one of these tests fails on an import error that has nothing to do with
# this package. Both conditions are checked, and the reason says which one is missing.
try:
    import ptarcade                                                          # noqa: F401
    HAVE_PTARCADE = True
except ImportError:
    HAVE_PTARCADE = False


def _data_dir():
    """Where the free spectra are: the download directory, or the research copy beside it."""
    research = Path(__file__).resolve().parents[2] / "smbhb/data/freespec"
    for d in (DEFAULT_DIR, research):
        if (d / "ng15_hd_v2/density.npy").is_file():
            return d
    return None


LOCAL_DATA = _data_dir()
needs_models = pytest.mark.skipif(
    not MODELS or not HAVE_PTARCADE,
    reason=("BINARYFOG_NG15_MODELS is not set" if not MODELS else
            "PTArcade is not installed, and the model files import it"))
needs_data = pytest.mark.skipif(LOCAL_DATA is None,
                                reason="no free spectrum; run `python -m binaryfog.fetch`")


def test_power_law_matches_the_published_normalisation():
    """At f = 1/yr the power law must give 2 pi^2 / (3 H100^2) A^2 f_yr^2."""
    f = np.array([bf.background.FYR_HZ])
    a = bf.power_law(f, -15.0, 13 / 3)[0]
    expected = 2 * np.pi ** 2 / (3 * bf.background.H100_HZ ** 2) * 1e-30 * bf.background.FYR_HZ ** 2
    assert a == pytest.approx(expected, rel=1e-12)


@needs_models
def test_every_stochastic_model_loads_and_evaluates():
    f = np.arange(1, 15) / (16.03 * 31557600.0)
    loaded, deterministic = [], []
    for path in sorted(Path(MODELS).glob("[!._]*.py")):
        try:
            m = bf.load_ptarcade(path)
        except ValueError:
            deterministic.append(path.stem)
            continue
        mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in m.ranges.items()}
        s = np.asarray(m.spectrum(f, **mid), float)
        assert s.shape == f.shape, m.name
        assert np.all(np.isfinite(s)) and np.all(s >= 0), m.name
        loaded.append(m.name)
    assert len(loaded) >= 15, loaded
    assert deterministic, "the ultralight dark matter files should be recognised as deterministic"


@needs_data
def test_profile_maximisation_is_converged():
    """The profiled maximum must not depend on how many restarts it is given: with ten free
    parameters an unconverged maximisation biases every contour."""
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=LOCAL_DATA)
    like = bf.Likelihood(data, bf.Background())

    def peaked(f_Hz, log10_amp, log10_fpeak, tilt):
        return bf.broken_power_law(f_Hz, log10_amp, log10_fpeak, rise=tilt)

    vals = []
    for restarts in (2, 6):
        scan = bf.ProfileScan(like, peaked, scan={"log10_fpeak": [-7.8]},
                              nuisance={"log10_amp": (-12.0, -5.0, "linear"),
                                        "tilt": (1.0, 5.0, "linear")},
                              astro_prior="nanograv", restarts=restarts, seed=3)
        vals.append(float(scan.run(jobs=1)["lnL"][0]))
    assert vals[0] == pytest.approx(vals[1], abs=0.02), vals


def test_sum_routes_parameters_and_rejects_clashes():
    import binaryfog as bf
    f = np.arange(1, 15) / (16.03 * 31557600.0)
    total = bf.Sum(bf.broken_power_law, bf.power_law)
    a = total(f, log10_amp=-9.0, log10_fpeak=-8.0, log10_A=-15.0, gamma=13 / 3)
    b = bf.broken_power_law(f, -9.0, -8.0) + bf.power_law(f, -15.0, 13 / 3)
    assert np.allclose(a, b, rtol=1e-12)
    with pytest.raises(ValueError):
        bf.Sum(bf.power_law, bf.power_law)


@needs_models
def test_loading_a_model_twice_gives_the_same_object():
    """A second load must not shadow the first, or sending the spectrum to a worker process
    fails with a pickling error that points nowhere useful."""
    import binaryfog as bf
    a = bf.load_ptarcade(Path(MODELS) / "pt_bubble.py")
    b = bf.load_ptarcade(Path(MODELS) / "pt_bubble.py")
    assert a.spectrum is b.spectrum
    import pickle
    assert pickle.loads(pickle.dumps(a.spectrum)) is a.spectrum


@needs_models
def test_the_background_can_be_added_to_every_model_without_a_name_clash():
    """Several NANOGrav models have an amplitude of their own called log10_A, so the background
    power law needs names that cannot collide with a model's."""
    import binaryfog as bf
    for path in sorted(Path(MODELS).glob("[!._]*.py")):
        try:
            m = bf.load_ptarcade(path)
        except ValueError:
            continue
        bf.Sum(m.spectrum, bf.smbhb_power_law)        # must not raise


def test_published_power_law_prior_peaks_at_the_published_centre():
    import binaryfog as bf
    at_centre = bf.ln_prior_agamma({"log10_A": bf.AGAMMA_MU[0], "gamma": bf.AGAMMA_MU[1]})
    off = bf.ln_prior_agamma({"log10_A": bf.AGAMMA_MU[0] - 1.0, "gamma": bf.AGAMMA_MU[1]})
    assert at_centre > off
    # the cost of a step is half the squared distance in the inverse-covariance metric, which for
    # a correlated Gaussian is not simply the step in units of the marginal standard deviation
    sd = np.sqrt(bf.AGAMMA_COV[0, 0])
    one_sigma = bf.ln_prior_agamma({"log10_A": bf.AGAMMA_MU[0] + sd, "gamma": bf.AGAMMA_MU[1]})
    expected = 0.5 * sd ** 2 * np.linalg.inv(bf.AGAMMA_COV)[0, 0]
    assert at_centre - one_sigma == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(0.5, abs=1e-3)


def test_prior_sets_are_those_of_the_paper():
    """The three Gaussian sets carry the centres and widths of table 1 of the paper, a step of one width
    in a Gaussian parameter costs 1/2 in ln prior, the flat set costs nothing inside the box, and a name
    that is not one of the paper's sets is refused."""
    from binaryfog.priors import GAUSSIAN_SETS, PARAM_NAMES, PRIOR_BOX, ln_prior
    table1 = {"nanograv": [(-2.56, 0.40), (10.90, 0.40), (8.60, 0.20), (0.32, 0.15)],
              "tomczak_kormendy": [(-2.61, 0.10), (10.96, 0.11), (8.69, 0.05), (0.28, 0.05)],
              "survey_average": [(-2.20, 0.15), (10.82, 0.11), (8.80, 0.10), (0.40, 0.05)]}
    assert set(GAUSSIAN_SETS) == set(table1)
    for which, rows in table1.items():
        assert [GAUSSIAN_SETS[which][k] for k in PARAM_NAMES[:4]] == rows
        centre = [mu for mu, _ in rows] + [0.5 * (PRIOR_BOX[k][0] + PRIOR_BOX[k][1]) for k in PARAM_NAMES[4:]]
        for idx, (_, sd) in enumerate(rows):
            off = list(centre)
            off[idx] += sd
            assert ln_prior(centre, which) - ln_prior(off, which) == pytest.approx(0.5, rel=1e-9)
        off = list(centre)
        off[4] += 1.0                                             # the binary lifetime is flat in every set
        assert ln_prior(off, which) == ln_prior(centre, which)
        assert ln_prior(centre, "flat") == 0.0 and ln_prior(centre, None) == 0.0
    for retired in ("uniform", "unwidened", "literature", "recent"):
        with pytest.raises(ValueError):
            ln_prior(centre, retired)


@needs_data
def test_nuisance_prior_pulls_the_power_law_to_its_centre():
    """With no data information a profiled power law must sit at its prior centre."""
    import binaryfog as bf
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=LOCAL_DATA)
    like = bf.Likelihood(data)
    flat = bf.ProfileScan(like, bf.power_law, scan={"gamma": [bf.AGAMMA_MU[1]]},
                          nuisance={"log10_A": (-18.0, -13.0, "linear")},
                          nuisance_prior=lambda v: bf.ln_prior_agamma(
                              {"log10_A": v["log10_A"], "gamma": bf.AGAMMA_MU[1]}),
                          astro_prior=None, restarts=3)
    table = flat.run(jobs=1)
    assert np.isfinite(float(table["lnL"][0]))
    assert -17.0 < float(table["log10_A"][0]) < -14.0


@needs_models
@needs_data
@pytest.mark.xfail(reason="the second grid point has two local maxima 0.5 apart in ln L and its neighbours sit in "
                          "the lower one, so the sweeps cannot repair it; see "
                          "test_a_profile_point_with_two_maxima_needs_many_restarts", strict=False)
def test_neighbour_sweeps_converge_a_many_parameter_profile():
    """With a dozen free parameters, restarts alone leave neighbouring grid points in different
    local maxima. Seeding each point from its best neighbour must remove that: a scan with few
    restarts and sweeps has to agree with a far more expensive one."""
    import binaryfog as bf
    m = bf.load_ptarcade(Path(MODELS) / "pt_bubble.py")
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=LOCAL_DATA)
    like = bf.Likelihood(data, bf.Background())
    par = list(m.ranges)[0]
    lo, hi = m.ranges[par]
    out = {}
    for restarts, sweeps in ((3, 2), (10, 2)):
        scan = bf.ProfileScan(like, m.spectrum,
                              scan={par: np.linspace(lo + 1e-6, hi - 1e-6, 5)},
                              nuisance=m.nuisance(*[k for k in m.ranges if k != par]),
                              astro_prior="nanograv", restarts=restarts, sweeps=sweeps, seed=5)
        out[restarts] = scan.run(jobs=4)["lnL"].to_numpy()
    assert np.max(np.abs(out[3] - out[10])) < 0.1, (out[3], out[10])


@needs_models
@needs_data
def test_a_profile_point_with_two_maxima_needs_many_restarts():
    """The second point of the five-point pt_bubble profile on the Hellings-Downs spectrum has two local
    maxima in the twelve profiled parameters, near ln L = -1.15 and -1.7. A single start can land in either,
    and the neighbours of that point sit in the lower one, so neither a few restarts nor the neighbour
    sweeps find the higher one reliably; 24 restarts do, for every seed."""
    import binaryfog as bf
    m = bf.load_ptarcade(Path(MODELS) / "pt_bubble.py")
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=LOCAL_DATA)
    like = bf.Likelihood(data, bf.Background())
    par = list(m.ranges)[0]
    lo, hi = m.ranges[par]
    point = np.linspace(lo + 1e-6, hi - 1e-6, 5)[1]
    nuisance = m.nuisance(*[k for k in m.ranges if k != par])

    def best(restarts, seed):
        scan = bf.ProfileScan(like, m.spectrum, scan={par: [point]}, nuisance=nuisance,
                              astro_prior="nanograv", restarts=restarts, sweeps=0, seed=seed)
        return float(scan.run(jobs=1)["lnL"].iloc[0])

    single = [best(1, seed) for seed in range(3)]
    assert max(single) - min(single) > 0.4, single              # single starts reach both maxima
    many = [best(24, seed) for seed in range(2)]
    assert max(many) - min(many) < 0.02, many                   # 24 restarts agree
    assert min(many) >= max(single) - 0.01, (many, single)      # on the higher maximum


@needs_models
@needs_data
def test_reproduces_the_published_cosmic_string_tension():
    """An end-to-end check against a published number from another pipeline.

    NANOGrav refit their stable-string model to the common-red-noise free spectrum with the
    standard noise model and report a maximum a posteriori tension of log10 G mu = -10.08
    (arXiv:2606.28554). Profiling the same model against the same data product here must land in
    the same place, up to the difference between a posterior mode and a profile maximum.
    """
    import binaryfog as bf
    m = bf.load_ptarcade(Path(MODELS) / "stable_n.py")
    data = bf.FreeSpectrum.load("ng15_cp_v1", root=LOCAL_DATA)
    like = bf.Likelihood(data)
    lo, hi = m.ranges["log10_Gmu"]
    scan = bf.ProfileScan(like, m.spectrum, scan={"log10_Gmu": np.linspace(lo, hi, 121)},
                          nuisance=None, astro_prior=None)
    table = scan.run(jobs=1)
    best = float(table.loc[table["lnL"].idxmax(), "log10_Gmu"])
    assert best == pytest.approx(-10.08, abs=0.15), best


@needs_data
def test_a_narrowed_prior_box_survives_the_rebuild_onto_the_data_frequencies():
    """Restricting a parameter's range must actually restrict the scan.

    A Likelihood rebuilds the background on the data's own frequencies, which used to construct a
    fresh object and silently discard a prior_box set on the instance. A systematic run that way
    reproduces the unrestricted result exactly, which looks like a null effect rather than a bug.
    """
    data = bf.FreeSpectrum.load("ng15_hd_v2", root=LOCAL_DATA, nbins=14)
    bg = bf.Background()
    assert not np.allclose(bg.freqs, data.freqs, rtol=1e-12, atol=0), \
        "the rebuild is what this test is about; if it no longer happens, the test is moot"
    bg.prior_box = dict(bg.prior_box, hard_time=(3.0, 11.0))
    like = bf.Likelihood(data, bg)
    assert like.background is not bg
    assert like.background.prior_box["hard_time"] == (3.0, 11.0)

    scan = bf.ProfileScan(like, lambda f, x: np.zeros(len(f)), scan={"x": np.array([0.0])},
                          astro_prior="nanograv")
    assert scan.theta_box["hard_time"] == (3.0, 11.0)
    row = scan.run(jobs=1).iloc[0]
    assert row["hard_time"] >= 3.0 - 1e-9

    # the default is untouched: the override lives on the instance, not on the class
    assert bf.Background().prior_box["hard_time"] == (0.1, 11.0)
