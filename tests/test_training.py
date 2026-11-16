"""Maps written by training/fit_maps.py load through Background(maps=path)."""
import numpy as np
import pytest

from binaryfog import Background
from binaryfog.background import DATA, PARAM_NAMES, PRIOR_BOX


def test_maps_from_a_file_equal_the_named_maps():
    named = Background(maps="ipta2")
    from_file = Background(maps=str(DATA / "maps_ipta2.npz"))
    theta = [np.mean(PRIOR_BOX[k]) + 0.1 * np.diff(PRIOR_BOX[k])[0] for k in PARAM_NAMES]
    for a, b in zip(named.predict(theta), from_file.predict(theta)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(named.freqs, from_file.freqs)


def test_missing_map_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        Background(maps=str(tmp_path / "none.npz"))
