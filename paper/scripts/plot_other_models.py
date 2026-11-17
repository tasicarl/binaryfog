"""
Figure 6: three models of the NANOGrav 15 yr new-physics search with the binaries as the (A, gamma) power law and
as the binaryfog background.

Top row: the measured free spectrum and, at the lower end, the best fit and the upper end of the binaryfog
2 sigma interval of the scanned parameter, the binaries, the new physics and their sum, with the band holding
90% of the realisations of the binaries. Bottom row: the profile likelihood of the scanned parameter for the two
treatments of the binaries, with the 2 sigma intervals shaded; the printed lines give these intervals.

    python plot_other_models.py      -> plots/other_models.pdf
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from desy_colors import DESYcyan, DESYorange, DESYdunkelrot, DESYgruen, DESYlila, DESYmagenta
from paper_style import use_paper_style, WIDTH, frequency_axis, BF, logo_in_corner, LOGO_SMALL

from paths import FREESPEC_ROOT, PLOTS, SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import DATA                                                     # noqa: E402
from profile_examples import BANDS_JSON, FIGURE_JSON, interval               # noqa: E402

VIOLIN_LABEL = "NANOGrav 15 yr, 2026"

LEVELS = bf.LEVELS_1D               # thresholds in ln L at 1 and 2 sigma, one scanned parameter
# the domain wall model is unconstrained under binaryfog, so its shaded interval fills the whole
# panel; it goes last, where that reads as the end of a progression rather than as a mistake
ORDER = ["super", "igw", "dw_ds"]
# two lines each: on one line the names of neighbouring panels run into each other
AXIS = {"log10_T_star": "annihilation temperature\n" r"$T_*$ / GeV", "log10_Gmu": "string tension\n" r"$G\mu$",
        "log10_r": "tensor-to-scalar ratio\n" r"$r$", "n_t": "tensor spectral index\n" r"$n_\mathrm{t}$"}
# the plotted range of the string tension: the profile has left the panel well below 10^-9, and the
# rest of the prior range up to 10^-6 would only squeeze the interval into a corner
XLIM = {"super": (-13.8, -9.2)}   # inset from the decades, so that no tick label sits on a corner
# the lower end of the 2 sigma interval, the best fit and the upper end
SHADES = (DESYlila, DESYmagenta, DESYgruen)
LW = 1.4                             # line width of the totals
BAND_ALPHA = 0.16
LABEL = {"power law": r"$(A,\gamma)$ power law, Afzal et al.", "population": BF}


def two_lines(text, fits=19):
    """A panel title on one line if it fits, otherwise split where the two lines come out most even."""
    if len(text) <= fits or " " not in text:
        return text
    words = text.split()
    cuts = [(max(len(" ".join(words[:i])), len(" ".join(words[i:]))), i)
            for i in range(1, len(words))]
    i = min(cuts)[1]
    return " ".join(words[:i]) + "\n" + " ".join(words[i:])


def violins(ax, data, colour, label=True):
    lf = np.log10(data.freqs * 1e9)
    half = 0.5 * np.min(np.diff(lf)) * 0.85
    for i in range(data.nbins):
        dens = np.exp(data.density[i] - data.density[i].max())
        pref = 8 * np.pi ** 4 / bf.background.H100_HZ ** 2 * data.freqs[i] ** 5 * data.tspan
        y = pref * 10.0 ** (2 * data.grid)
        keep = np.flatnonzero(dens > 2e-2)[::8]
        w = half * dens[keep] / dens[keep].max()
        ax.fill_betweenx(y[keep], 10 ** (lf[i] - w), 10 ** (lf[i] + w), color=colour,
                         alpha=0.5, lw=0, label=VIOLIN_LABEL if (label and i == 0) else None)


def draw(out=PLOTS / "other_models.pdf"):
    use_paper_style()
    raw = json.load(open(SCANS / FIGURE_JSON))
    bands = json.load(open(SCANS / BANDS_JSON))
    d = {k: raw[k] for k in ORDER if k in raw}
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    fig, axes = plt.subplots(2, len(d), figsize=(WIDTH, 5.0),
                             gridspec_kw={"height_ratios": [1.0, 0.95]})
    for k, (key, e) in enumerate(d.items()):
        top, bot = axes[0, k], axes[1, k]
        f = np.array(e["freqs"]) * 1e9
        fmeas = np.array(e["freqs_measured"]) * 1e9
        violins(top, data, DESYdunkelrot, label=(k == 0))
        # the binaries of each of the three values with their band (example_bands.py)
        eb = bands[key]
        for c, lo_, hi_ in zip(SHADES, eb["backgrounds_lo"], eb["backgrounds_hi"]):
            top.fill_between(f, lo_, hi_, color=c, alpha=BAND_ALPHA, lw=0)
        for c, sp, bg in zip(SHADES, e["spectra"], eb["backgrounds_refit"]):
            sp, bg = np.array(sp), np.array(bg)
            top.plot(f, np.maximum(bg, 1e-30), color=c, lw=0.6 * LW, ls="--")
            top.plot(f, np.maximum(sp, 1e-30), color=c, lw=0.75 * LW, ls=":")
            top.plot(f, np.maximum(sp + bg, 1e-30), color=c, lw=LW)
        top.set_xscale("log")
        top.set_yscale("log")
        frequency_axis(top.xaxis)   # 10 nHz major, 2, 5, 20 nHz labelled minor
        top.set_xlim(fmeas[0] / 1.25, fmeas[-1] * 1.25)
        top.set_ylim(10 ** -11.5, 10 ** -6.8)
        top.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        top.set_xlabel(r"$f$ / nHz")
        # wrapped, because one line of this length is wider than a third of the text block
        top.set_title(two_lines(e["label"]), fontsize=8)
        if k == 0:
            top.set_ylabel(r"$h^2 \Omega_\mathrm{GW}$")
        x = np.array(e["grid"])
        # log10 parameters are drawn on a log axis; n_t (inflation scanned in its tilt) is linear
        linear = not e["parameter"].startswith("log10_")
        X = (lambda v: v) if linear else (lambda v: 10 ** np.asarray(v))
        for case, colour, style in (("power law", DESYcyan, "--"), ("population", DESYorange, "-")):
            y = np.array(e[case]); y = y - y.max()
            bot.plot(X(x), y, color=colour, ls=style, lw=2.0, label=LABEL[case])
            # only the 2 sigma band is shaded: four overlapping bands cannot be read. Both
            # thresholds are still drawn, so the 1 sigma interval can be read off the curve.
            lo, hi, edge = interval(x, y, LEVELS[2])
            bot.axvspan(X(lo), X(hi), color=colour, alpha=0.13, lw=0)
            print(f"{e['label']:34s} {case:11s} 2 sigma {lo:+.2f} to {hi:+.2f}  width {hi-lo:.2f} "
                  f"of {x[-1]-x[0]:.2f}{'  (at the edge)' if edge else ''}")
        # The three values drawn above are marked on this axis in their own shades rather than
        # named in a legend inside the panel above, where three lines of text would cover the curves
        # they label.
        for c, v in zip(SHADES, e["shown"]):
            bot.annotate("", xy=(X(v), 1.0), xycoords=bot.get_xaxis_transform(),
                         xytext=(0, 4), textcoords="offset points", annotation_clip=False,
                         arrowprops=dict(arrowstyle="-|>", lw=0, mutation_scale=9,
                                         facecolor=c, edgecolor=c, shrinkA=0, shrinkB=0))
        if not linear:
            bot.set_xscale("log")
        xl = XLIM.get(key, (x[0], x[-1]))
        bot.set_xlim(X(xl[0]), X(xl[1]))
        if key in XLIM:
            # every second decade: labels at every decade touch at this panel width
            bot.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-13, 1e-11]))
        # the same ticks in every panel, whatever height the panels end up with
        bot.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, -2, -4, -6]))
        bot.set_ylim(-6.0, 0.6)
        bot.set_xlabel(AXIS[e["parameter"]])
        for lvl in LEVELS:
            bot.axhline(-LEVELS[lvl], color="0.5", lw=0.8, ls=":")
        # The two Wilks levels are named once, on the right-hand axis of the last panel. Inside a
        # panel the labels would sit on the shaded intervals and on the profile curves; the levels are
        # values of the vertical axis, so that is where they belong. All three panels share the
        # same vertical range, so one pair of labels reads across the row.
        if k == len(d) - 1:
            sec = bot.secondary_yaxis("right")
            sec.set_yticks([-LEVELS[1], -LEVELS[2]])
            sec.set_yticklabels([r"$1\sigma$", r"$2\sigma$"], fontsize=7, color="0.4")
            sec.tick_params(axis="y", which="both", length=0, colors="0.4")
            sec.minorticks_off()
        if k == 0:
            bot.set_ylabel(r"profile $\Delta \ln L$")
    # Two legends, one per row, rather than one for the figure: the two profile colours belong to
    # the lower panels alone, and collecting them above the upper ones would send the reader looking for
    # curves of that colour there.
    # w_pad small, so the panels take the width rather than the margins between and beside them;
    # h_pad just large enough for the lower legend and the value markers between the rows
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.2, w_pad=0.4)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    top_items = [(hh, ll) for hh, ll in zip(h0, l0) if ll == VIOLIN_LABEL]
    top_items += [(Line2D([0], [0], color="0.35", lw=1.4, ls="-"), "total"),
                  (Line2D([0], [0], color="0.35", lw=1.2, ls="--"), "binaries"),
                  (Line2D([0], [0], color="0.35", lw=1.3, ls=":"), "new physics")]
    fig.legend([h for h, _ in top_items], [l for _, l in top_items], fontsize=7.5, frameon=False,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.7, columnspacing=1.2)

    hb, lb = axes[1, 0].get_legend_handles_labels()
    bot_items = list(zip(hb, lb))
    bot_items += [(Line2D([0], [0], color="0.45", lw=0, marker="v", ms=4.5),
                   "the three values drawn above")]
    # Just above the lower row rather than centred in the gap: the entries belong to those panels,
    # and a legend floating midway between the rows reads as belonging to neither.
    fig.legend([h for h, _ in bot_items], [l for _, l in bot_items], fontsize=7.5, frameon=False,
               ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, axes[1, 0].get_position().y1 + 0.008),
               handlelength=1.7, columnspacing=1.2)
    # the logo in the upper left panel rather than beside the legend, there only and small
    logo_in_corner(fig, axes[0, 0], "upper left", size=0.20, margin=0.15, path=LOGO_SMALL)
    # edge to edge: at \textwidth the y labels then sit at the margin of the text block
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()
