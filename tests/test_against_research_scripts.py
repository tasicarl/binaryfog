"""
The package must reproduce the research scripts it was extracted from, value for value.

Checked here: the surrogate prediction, the point likelihood, the marginalised likelihood, and the
best fit of the published phase-transition scan. Skipped automatically outside the research
repository, where smbhb/ and fig1_interface/ are not available.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SMBHB, KIT = REPO / "smbhb", REPO / "fig1_interface"
FREESPEC = SMBHB / "data/freespec"
AVAILABLE = SMBHB.is_dir() and KIT.is_dir() and FREESPEC.is_dir()
pytestmark = pytest.mark.skipif(not AVAILABLE,
                                reason="research scripts or their data are not available")

# the research paths go on sys.path only where they exist, so that importing this file outside the
# research repository skips rather than fails at collection
if AVAILABLE:
    sys.path.insert(0, str(SMBHB))
    sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import binaryfog as bf                                                        # noqa: E402

THETA = [-2.56, 10.90, 8.60, 0.32, 3.0, -1.0]
FOPT_BEST = (0.518, 8.306, 0.08980)          # alpha, beta/H, T_reh in GeV, from scan_fopt_n50.csv


@pytest.fixture(scope="module")
def research():
    import freespec_like
    import marginal_like
    import run_scan_marginal
    from fopt_model import spectrum_fopt
    return freespec_like, marginal_like, run_scan_marginal, spectrum_fopt


@pytest.fixture(scope="module")
def data():
    return bf.FreeSpectrum.load("ng15_cp_v1", root=FREESPEC)


def test_frequencies_match(research, data):
    freespec_like = research[0]
    assert np.allclose(data.freqs, freespec_like.FREQS)


def test_surrogate_matches(research, data):
    """The research code evaluates the surrogate on the frequencies stored in the data product,
    so the background has to be built on those, not on the nominal i / 16.03 yr."""
    run_scan_marginal = research[2]
    med_r, sig_r, skew_r = run_scan_marginal.binary_model(np.array(THETA)[None, :])
    med, sig, skew = bf.Background(data.freqs).predict(THETA)
    # compare in the logarithm: the spectra are of order 1e-10, where an absolute tolerance is blind
    assert np.allclose(np.log10(med), np.log10(med_r), rtol=0, atol=1e-12)
    assert np.allclose(sig, sig_r, rtol=1e-12)
    assert skew == pytest.approx(skew_r, rel=1e-12)


def test_point_likelihood_matches(research, data):
    freespec_like, _, _, spectrum_fopt = research
    pt = spectrum_fopt(data.freqs, *FOPT_BEST)
    assert data.ln_like_point(pt) == pytest.approx(freespec_like.ln_like(pt), rel=1e-12)


def test_marginal_likelihood_matches(research, data):
    _, marginal_like, run_scan_marginal, spectrum_fopt = research
    pt = spectrum_fopt(data.freqs, *FOPT_BEST)
    med, sig, skew = run_scan_marginal.binary_model(np.array(THETA)[None, :])
    expected = marginal_like.ln_like_marginal(pt, med, sig, skew=skew)
    got = bf.Likelihood(data, bf.Background()).ln_like(pt, THETA)
    assert got == pytest.approx(expected, rel=1e-12)


def test_profile_scan_reproduces_the_published_best_fit(research, data):
    """The transition alone, profiled over the reheating temperature, on the 2024 data:
    the published scan found ln L = -11.311."""
    spectrum_fopt = research[3]
    like = bf.Likelihood(data)                     # no binaries: the fopt mode of the scan
    scan = bf.ProfileScan(like, lambda f_Hz, alpha, betaH, Treh: spectrum_fopt(f_Hz, alpha, betaH, Treh),
                          scan={"alpha": [FOPT_BEST[0]], "betaH": [FOPT_BEST[1]]},
                          nuisance={"Treh": (1e-3, 1.0, "log")}, astro_prior=None, restarts=2)
    table = scan.run(jobs=1)
    assert float(table["lnL"].max()) == pytest.approx(-11.3108, abs=2e-3)


def test_binaries_only_and_preference(data):
    """The background alone must be finite, and adding a negligible new-physics spectrum must not
    change the likelihood."""
    like = bf.Likelihood(data, bf.Background())
    a = like.ln_like_binaries_only(THETA)
    b = like.ln_like(np.full(data.nbins, 1e-30), THETA)
    assert np.isfinite(a)
    assert a == pytest.approx(b, abs=1e-9)
