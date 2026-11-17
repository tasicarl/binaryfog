"""
Figure 1: what each parameter does to the predicted background.

Eight panels. Six show the median spectrum of the population model for five values of one parameter, equally
spaced over the range written in the panel, with the others at the centres of the NANOGrav priors; the ranges are
the 2 sigma ranges of the NANOGrav priors, cut at the edge of the library for m_psi0. Two show the (A, gamma) power
law for five values of one of its parameters over its prior centre +- 2 sigma. The baseline free spectrum is shown
for comparison. Also prints how much each parameter moves log10 h_c^2 at 2 nHz and the slope in the lowest five
bins (section 4.1).

    python plot_parameters.py      -> plots/parameters.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FREESPEC_ROOT, PLOTS
import binaryfog as bf                                                        # noqa: E402
from desy_colors import desy_ramp, DESYdunkelrot                              # noqa: E402
from paper_style import use_paper_style, WIDTH, frequency_axis
from baseline import DATA                                                     # noqa: E402


def measured_log10hc2(name, nbins):
    """The median of the measured spectrum per bin, as log10 h_c^2, from h_c^2 = 12 pi^2 rho^2 f^3 T; it sets the
    range of the vertical axis."""
    d = FREESPEC_ROOT / name
    grid = np.load(d / "log10rhogrid.npy")
    dens = np.load(d / "density.npy")[0]
    freqs = np.load(d / "freqs.npy")[:nbins]
    tspan = 1.0 / freqs[0]
    out = []
    for i in range(nbins):
        p = np.exp(dens[i] - dens[i].max())
        c = np.cumsum(p)
        rho = grid[np.searchsorted(c, 0.5 * c[-1])]
        out.append(np.log10(12 * np.pi ** 2) + 2 * rho + 3 * np.log10(freqs[i]) + np.log10(tspan))
    return np.array(out)


CENTRE = {"gsmf_phi0_log10": -2.56, "gsmf_mchar0_log10": 10.90, "mmb_mamp_log10": 8.60,
          "mmb_scatter_dex": 0.32, "hard_time": 3.0, "hard_gamma_inner": -1.0}
# each title names the quantity and its unit, on two lines so that it stays within its panel
LABELS = {"gsmf_phi0_log10": "GSMF normalisation\n" r"$\log_{10}(\psi_0/\mathrm{Mpc^{-3}})$",
          "gsmf_mchar0_log10": "GSMF turnover mass\n" r"$\log_{10}(m_{\psi,0}/M_\odot)$",
          "mmb_mamp_log10": r"$M_\bullet\!-\!M_\mathrm{bulge}$ amplitude" "\n"
                            r"$\log_{10}(\mu/M_\odot)$",
          "mmb_scatter_dex": "its scatter\n" r"$\epsilon_\mu$ in $\log_{10}(M_\bullet/M_\odot)$",
          "hard_time": "binary lifetime\n" r"$\tau_\mathrm{f}/\mathrm{Gyr}$",
          "hard_gamma_inner": "inner hardening index\n" r"$\nu_\mathrm{inner}$"}
# the 2 sigma ranges of the NANOGrav priors (centre +- 2 sigma of the Gaussians, the full range of the uniform
# priors); the range of the characteristic mass is cut at the lower edge of the library
RANGES = {"gsmf_phi0_log10": (-2.56 - 0.80, -2.56 + 0.80), "gsmf_mchar0_log10": (10.5, 10.90 + 0.80),
          "mmb_mamp_log10": (8.60 - 0.40, 8.60 + 0.40), "mmb_scatter_dex": (0.32 - 0.30, 0.32 + 0.30),
          "hard_time": (0.1, 11.0), "hard_gamma_inner": (-1.5, 0.0)}

if __name__ == "__main__":
    use_paper_style()
    matplotlib.rcParams.update({"lines.solid_capstyle": "round", "legend.handlelength": 1.5})
    fmeas = np.load(FREESPEC_ROOT / DATA / "freqs.npy")[:14]
    ref = measured_log10hc2(DATA, 14)
    # The models are broken power laws, so there is no reason for a drawn curve to stop where the
    # measurement does. They are evaluated a little past the band at both ends and the axis is set
    # inside that, so the curves leave the panel instead of ending at the outermost violin.
    f = np.geomspace(fmeas[0] / 1.8, fmeas[-1] * 1.8, 400)
    bg = bf.Background(f, extrapolate=True)
    H = 2 * np.pi ** 2 / (3 * bf.background.H100_HZ ** 2)
    # two more panels for the (A, gamma) power law, separated from the six population panels by a
    # gap and a vertical rule
    fig = plt.figure(figsize=(WIDTH, 3.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.16, 1], wspace=0.08, hspace=0.42)
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j], sharex=axes[0, 0] if (i or j) else None,
                                         sharey=axes[0, 0] if (i or j) else None)
    axpl = [fig.add_subplot(gs[i, 4], sharex=axes[0, 0], sharey=axes[0, 0]) for i in range(2)]
    ramp = desy_ramp(5)
    A0, G0 = float(bf.AGAMMA_MU[0]), float(bf.AGAMMA_MU[1])
    sA, sG = float(np.sqrt(bf.AGAMMA_COV[0, 0])), float(np.sqrt(bf.AGAMMA_COV[1, 1]))
    PL = [("power-law amplitude", r"\log_{10} A", A0 - 2 * sA, A0 + 2 * sA, "A"),
          ("power-law index", r"\gamma", G0 - 2 * sG, G0 + 2 * sG, "g")]
    for ax, (title, sym, lo, hi, which) in zip(axpl, PL):
        for j, v in enumerate(np.linspace(lo, hi, 5)):
            la, g = (v, G0) if which == "A" else (A0, v)
            ax.plot(f * 1e9, bf.power_law(f, la, g) / (H * f ** 2), color=ramp[j], lw=1.8)
        ax.set_title(f"{title}\n${sym}$", fontsize=8)          # symbol in the title as in the six others
        # the range (centre +- 2 sigma of the Gaussian prior on log10 A and gamma) in the lower left, as in the six others
        ax.text(0.06, 0.03, f"${lo:.1f}$ to ${hi:.1f}$", transform=ax.transAxes, fontsize=7,
                ha="left", va="bottom", zorder=5)
    for ax, (name, (lo, hi)) in zip(axes.flat, RANGES.items()):
        for j, v in enumerate(np.linspace(lo, hi, 5)):
            th = dict(CENTRE, **{name: v})
            med, sig, skew = bg.predict([th[k] for k in bf.PARAM_NAMES])
            ax.plot(f * 1e9, med / (H * f ** 2), color=ramp[j], lw=1.8)
        ax.set_title(LABELS[name], fontsize=8)
        # A legend of the five values would cover the curves and the measured spectrum in every panel.
        # The values are equally spaced between the two written here and the shade runs from light
        # to dark with the parameter, which the caption says, so the range is all the panel needs.
        fmt = "{:.1f}" if name == "hard_time" else "{:.2f}"
        # 0.06, not 0.03: the inward ticks of the left axis otherwise touch the first digit
        ax.text(0.06, 0.03, ("$" + fmt.format(lo) + "$ to $" + fmt.format(hi) + "$"),
                transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    lf = np.log10(data.freqs * 1e9)
    hw = 0.5 * np.min(np.diff(lf)) * 0.85
    for ax in list(axes.flat) + axpl:
        for i in range(data.nbins):
            dens = np.exp(data.density[i] - data.density[i].max())
            pref = (8 * np.pi ** 4 / bf.background.H100_HZ ** 2
                    * data.freqs[i] ** 5 * data.tspan)
            yv = pref * 10.0 ** (2 * data.grid) / (H * data.freqs[i] ** 2)
            # the stored grid has 10000 points per bin; thinning it and dropping the far tail
            # keeps the shape and keeps the figure a sensible size
            keep = np.flatnonzero(dens > 2e-2)[::8]
            w = hw * dens[keep] / dens[keep].max()
            ax.fill_betweenx(yv[keep], 10 ** (lf[i] - w), 10 ** (lf[i] + w),
                             color=DESYdunkelrot, alpha=0.45, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(fmeas[0] / 1.25 * 1e9, fmeas[-1] * 1.25 * 1e9)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
        frequency_axis(ax.xaxis)   # 10 nHz major, 2, 5, 20 nHz labelled minor
    for ax in list(axes[1]) + [axpl[1]]:
        ax.set_xlabel("frequency / nHz")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$h_\mathrm{c}^2$")
    for ax in list(axes[:, 1:].flat) + axpl:
        ax.tick_params(labelleft=False)
    for ax in list(axes[0]) + [axpl[0]]:
        ax.tick_params(which="both", labelbottom=False)
    # the lower limit leaves the range in the lower left of the power-law amplitude panel free of its faintest line
    axes[0, 0].set_ylim(10.0 ** (ref.min() - 4.4), 10.0 ** (ref.max() + 1.2))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.11)
    # the vertical rule between the population panels and the power-law panels
    x_rule = 0.5 * (axes[0, 2].get_position().x1 + axpl[0].get_position().x0)
    fig.add_artist(matplotlib.lines.Line2D([x_rule, x_rule], [0.06, 0.97], color="k", lw=1.2,
                                           transform=fig.transFigure))
    fig.savefig(PLOTS / "parameters.pdf", dpi=200)
    print("wrote plots/parameters.pdf")
    # how much each parameter moves the amplitude and the slope
    print(f"{'parameter':34s} {'range of log10 hc^2 at 2 nHz':>30s} {'range of slope, bins 1-5':>26s}")
    for name, (lo, hi) in RANGES.items():
        amps, slopes = [], []
        for v in np.linspace(lo, hi, 5):
            th = dict(CENTRE, **{name: v})
            med, _, _ = bg.predict([th[k] for k in bf.PARAM_NAMES])
            # the printed summary is quoted at the measured bins, as the text quotes it, so the
            # curve is read back onto them rather than onto the extended drawing grid
            y = np.interp(np.log10(fmeas), np.log10(f), np.log10(med / (H * f ** 2)))
            amps.append(y[0])
            slopes.append(np.polyfit(np.log10(fmeas[:5]), y[:5], 1)[0])
        print(f"{LABELS[name].splitlines()[0][:34]:34s} {max(amps)-min(amps):30.2f} "
              f"{max(slopes)-min(slopes):26.2f}")
