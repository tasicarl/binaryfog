"""Turning the gain of a fit into a number of standard deviations.

Wilks' theorem does not apply here as it stands. New physics usually enters through an amplitude
that cannot be negative, and when that amplitude vanishes the remaining parameters of the model do
nothing at all: a phase transition of zero strength has no duration and no reheating temperature.
Both conditions break the assumptions behind a plain chi-square.

The boundary is handled as in Chernoff's theorem, in the form of ref. [1]: under the null the
fitted amplitude falls on the wrong side of the boundary half the time, and the test statistic is
then exactly zero, so its null distribution is the mixture

    p(TS) = 1/2 delta(TS) + 1/2 chi2_k(TS)

with k the number of free parameters of the new-physics model. The tail probability of a measured
TS = 2 G is therefore half what a plain chi2_k gives, and the significance correspondingly larger.

What this does not cover is that the parameters other than the amplitude are not merely
unconstrained at the null but undefined, so the second branch is not exactly a chi2_k either. That
is the Davies problem, and it pushes the other way: the correct null distribution is the supremum
of a chi-square field over those parameters, whose tail is heavier. Settling both at once needs a
calibration by simulation. A difference between two fits converted the same way is insensitive to
the choice; an absolute significance is not.

Pass ``boundary=False`` for the plain chi2_k, which is what is usually quoted elsewhere.

[1] Macias et al., Nature Astron. 2 (2018) 387, arXiv:1611.06644, eqs. (2) and (3).
"""
from scipy import stats

__all__ = ["sigma_from_gain"]


def sigma_from_gain(gain, ndof, boundary=True):
    """Standard deviations from the gain in ln L of a model with `ndof` free parameters."""
    p = stats.chi2.sf(2.0 * max(float(gain), 0.0), int(ndof))
    if boundary:
        p *= 0.5
    return float(stats.norm.isf(max(p, 1e-300) / 2))
