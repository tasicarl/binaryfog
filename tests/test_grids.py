"""The background on a subset of its fitted bins, and on the IPTA DR2 grid."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import binaryfog as bf                                                        # noqa: E402
from binaryfog.background import DATA                                          # noqa: E402

THETA = [-2.5, 11.0, 8.6, 0.3, 1.0, -0.5]


def test_a_subset_of_the_fitted_bins_takes_their_values_exactly():
    full = bf.Background().predict(THETA)
    part = bf.Background(bf.Background.F_NG15[:5]).predict(THETA)
    assert np.array_equal(part[0], full[0][:5]) and np.array_equal(part[1], full[1][:5])
    assert part[2] == full[2]


def test_on_the_fitted_bins_the_width_is_the_stored_per_bin_map():
    """Not the smooth curve through the bins that frequencies between them get."""
    from binaryfog.background import design
    for f in (None, bf.Background.F_NG15[:5]):
        bg = bf.Background(f)
        want = bg._wscale * 10.0 ** (design(THETA, bg._deg_wid) @ bg._W_wid)[0][: len(bg.freqs)]
        assert np.array_equal(bg.predict(THETA)[1], want)


def test_frequencies_between_the_bins_are_interpolated():
    f = np.sqrt(bf.Background.F_NG15[1:6] * bf.Background.F_NG15[2:7])
    med, sig, _ = bf.Background(f).predict(THETA)
    full = bf.Background().predict(THETA)
    assert np.all(med > np.minimum(full[0][1:6], full[0][2:7]) * 0.5)
    assert np.all(np.isfinite(sig))


@pytest.mark.skipif(not (DATA / "ipta2" / "maps6.npz").is_file(), reason="no IPTA DR2 maps")
def test_the_ipta_maps_cover_their_band_and_refuse_beyond():
    bg = bf.Background(maps="ipta2")
    assert len(bg.freqs) == 30 and bg.freqs[0] < 1.1e-9
    med, sig, skew = bg.predict(THETA)
    assert np.all(np.isfinite(med)) and np.all(sig > 0) and 0.02 <= skew <= 0.45
    sub = bf.Background(bg.freqs[:13], maps="ipta2").predict(THETA)
    assert np.array_equal(sub[0], med[:13])
    with pytest.raises(ValueError):
        bf.Background(np.array([0.5e-9]), maps="ipta2")
    with pytest.raises(ValueError):
        bf.Background(np.array([1.05e-9]))                 # below the NANOGrav maps
