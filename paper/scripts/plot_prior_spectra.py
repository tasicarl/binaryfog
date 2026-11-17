"""Figure 2: the prior-predictive spectrum h^2 Omega(f) under the NANOGrav, Tomczak-Kormendy and survey-average
priors, median and 1 sigma band including realisation scatter (the binaryfog surrogate with its warp), dashed the
1 sigma range at known tau_f and nu_inner, against the NANOGrav 15 yr CURN free spectrum.

    python plot_prior_spectra.py      -> plots/prior_spectra.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FREESPEC_ROOT, PLOTS                                    # noqa: E402
import binaryfog as bf                                                            # noqa: E402
from desy_colors import DESYorange, DESYdunkelblau, DESYaubergine, DESYdunkelrot  # noqa: E402
from paper_style import use_paper_style, WIDTH, frequency_axis, logo_in_corner      # noqa: E402

SETS = [("nanograv", "NANOGrav priors", DESYorange),
        ("tomczak_kormendy", "Tomczak–Kormendy priors", DESYdunkelblau),
        ("survey_average", "Survey-average priors", DESYaubergine)]


def pct(a, axis=0):
    return np.percentile(a, [16, 50, 84], axis=axis)


def prior_predictive(bg, key, ndraw, rng, box=None):
    """log10 h^2 Omega for ndraw draws of the prior set, each with one realisation of the background.
    One warp variable xi per draw is shared by all frequencies: the plotted percentiles are per-frequency
    marginals, which this leaves unchanged, while independent xi per bin make them jitter from bin to bin."""
    from binaryfog.priors import GAUSSIAN_SETS, PARAM_NAMES, PRIOR_BOX
    g = GAUSSIAN_SETS[key]
    box = {**PRIOR_BOX, **(box or {})}
    spec = []
    while len(spec) < ndraw:
        th = np.array([rng.normal(*g[k]) if k in g else rng.uniform(*box[k]) for k in PARAM_NAMES])
        if not all(PRIOR_BOX[k][0] <= v <= PRIOR_BOX[k][1] for k, v in zip(PARAM_NAMES, th)):
            continue
        med, sig, c = bg.predict(th)
        c = float(np.ravel(c)[0])
        xi = max(rng.standard_normal(), -0.5 / max(c, 1e-3))
        spec.append(np.log10(med) + np.asarray(sig) * (xi + c * xi ** 2))
    return np.array(spec)


def conditional_offsets(bg, key, rng, nhard=40, ndraw=2000):
    """The 16th and 84th percentiles minus the median at fixed tau_f and nu_inner, averaged over nhard points
    drawn from their uniform priors: the width the band would have if the hardening parameters were known."""
    from binaryfog.priors import PRIOR_BOX
    lo, hi = [], []
    for _ in range(nhard):
        t, n = rng.uniform(*PRIOR_BOX["hard_time"]), rng.uniform(*PRIOR_BOX["hard_gamma_inner"])
        q = pct(prior_predictive(bg, key, ndraw, rng, {"hard_time": (t, t), "hard_gamma_inner": (n, n)}))
        lo.append(q[0] - q[1]); hi.append(q[2] - q[1])
    return np.mean(lo, axis=0), np.mean(hi, axis=0)


def measured_median(data, i):
    """Median of the free-spectrum posterior in bin i, as log10 h^2 Omega."""
    dens = np.exp(data.density[i] - data.density[i].max())
    cdf = np.cumsum(dens) / dens.sum()
    log10rho = np.interp(0.5, cdf, data.grid)
    pref = 8 * np.pi ** 4 / bf.background.H100_HZ ** 2 * data.freqs[i] ** 5 * data.tspan
    return float(np.log10(pref) + 2 * log10rho)


def plot(ndraw=40000):
    from plot_other_models import violins
    data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT)
    measured = measured_median(data, 0)
    # drawn a little past the band at both ends, so that bands and lines leave the panel instead of
    # ending at the outermost bin; the axis is set inside that
    F = bf.Background.F_NG15
    fg = np.geomspace(F[0] / 1.35, F[-1] * 1.35, 140)
    bg = bf.Background(fg, extrapolate=True)
    f = fg * 1e9
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7), sharey=True)
    print(f"  measured median in the lowest bin ({data.freqs[0] * 1e9:.2f} nHz): log10 h2Omega {measured:.2f}")
    for ax, (key, lab, col) in zip(axes, SETS):
        violins(ax, data, DESYdunkelrot, label=(key == "nanograv"))
        q = pct(prior_predictive(bg, key, ndraw, rng))
        dlo, dhi = conditional_offsets(bg, key, rng)
        qr = np.array([q[1] + dlo, q[1], q[1] + dhi])
        ax.fill_between(f, 10 ** q[0], 10 ** q[2], color=col, alpha=0.25, lw=0,
                        label=r"$1\sigma$ range with realisation scatter" if key == "nanograv" else None)
        ax.plot(f, 10 ** qr[0], color=col, lw=0.9, ls="--",
                label=r"width at known $\tau_\mathrm{f}$, $\nu_\mathrm{inner}$" if key == "nanograv" else None)
        ax.plot(f, 10 ** qr[2], color=col, lw=0.9, ls="--")
        ax.plot(f, 10 ** q[1], color=col, lw=1.8)
        i2, i20 = np.argmin(abs(f - F[0] * 1e9)), np.argmin(abs(f - 20.0))
        print(f"  {key}: log10 h2Omega at {f[i2]:.2f} nHz, 16/50/84%: all {np.round(q[:, i2], 2)}, "
              f"at known tau_f, nu_inner {np.round(qr[:, i2], 2)}; at {f[i20]:.1f} nHz all {np.round(q[:, i20], 2)}, "
              f"at known tau_f, nu_inner {np.round(qr[:, i20], 2)}")
        print(f"  {key}: median below the measured median in the lowest bin by {measured - q[1, i2]:.2f} dex")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(F[0] * 1e9 / 1.25, F[-1] * 1e9 * 1.25)
        frequency_axis(ax.xaxis)   # 10 nHz major, 2, 5, 20 nHz labelled minor
        ax.set_ylim(2e-13, 1e-7)
        ax.set_title(lab, fontsize=9, color=col)
        ax.set_xlabel(r"$f$ / nHz")
    axes[0].set_ylabel(r"$h^2\Omega_\mathrm{GW}$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.17, top=0.80, wspace=0.06)
    logo_in_corner(fig, axes[-1], "lower right", margin=0.3)
    fig.savefig(PLOTS / "prior_spectra.pdf", dpi=200)
    print(f"wrote {PLOTS / 'prior_spectra.pdf'}")


if __name__ == "__main__":
    use_paper_style()
    plot()
