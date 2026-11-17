"""
Figure 7, plots/free_spectra.pdf: six versions of the NANOGrav 15 yr free spectrum, each fitted with the
binaries as the (A, gamma) power law and as the binaryfog background, from the 50 x 50 scans of
run_scan_marginal.py. Left, the 1 and 2 sigma bands in beta/H for a strong phase transition (alpha = 1000,
where the spectrum no longer depends on alpha); right, the preference for a phase transition, the gain G
converted with three degrees of freedom (binaryfog.sigma_from_gain). Prints both for every spectrum; "<"
marks a 1 sigma band that reaches the end of the scanned range.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from band import band_open
from desy_colors import DESYcyan, DESYorange, DESYdunkelrot
from paper_style import use_paper_style, WIDTH, BF, logo_in_corner

from paths import PLOTS, SCANS                                            # noqa: E402
import binaryfog as bf                                                    # noqa: E402
from baseline import DATA as DEFAULT                                      # noqa: E402
SETS = [("ng15_cp_v1_cut9", "2024, DM offsets\ncommon red noise"),
        ("ng15_hd_v1", "2024, DM offsets\nHellings\u2013Downs"),
        ("ng15_cp_v2", "2026, DM offsets\ncommon red noise"),
        ("ng15_hd_v2", "2026, DM offsets\nHellings\u2013Downs"),
        ("ng15_hdall_v2", "2026, DM offsets\nall four components"),
        # this one comes from NANOGrav's own 2023 record, not from the 2024 PTArcade one the two
        # other original products are taken from
        ("ng15_hd_dmgp_v1", "2023, DM Gaussian process\nHellings\u2013Downs")]

# Above this strength the transition is vacuum dominated and the spectrum stops depending on
# alpha, so the band in beta/H is the same at every alpha above it and one slice describes it all.
ALPHA_SLICE = 1e3


def row(tag, mode):
    """The 1 and 2 sigma bands in beta/H in the saturated regime, with the edges interpolated between
    grid nodes (band.py), and the preference, for one data set."""
    f = SCANS / f"scan_{mode}_n50_{tag}.csv"
    lo, hi, open_at = band_open(f, alpha=ALPHA_SLICE)
    lo2, hi2, open2 = band_open(f, alpha=ALPHA_SLICE, thr=3.09)     # and the 2 sigma band
    d = pd.read_csv(f)
    corner = d[d.alpha == d.alpha.min()]
    return (lo, hi), bf.sigma_from_gain(d.lnL.max() - corner.lnL.max(), 3), open_at, (lo2, hi2), open2


if __name__ == "__main__":
    use_paper_style()
    labels, vals = [], []
    for tag, label in SETS:
        pl, po = row(tag, "agamma"), row(tag, "holodeck")
        labels.append(label)
        vals.append((pl, po))
        flag = "".join("<" if pl[2][k] or po[2][k] else " " for k in (0, 1))
        print(f"{label.replace(chr(10), ', '):44s} 1sig {pl[0][0]:5.1f}-{pl[0][1]:5.1f} -> "
              f"{po[0][0]:5.1f}-{po[0][1]:5.1f} {flag}  2sig {pl[3][0]:5.1f}-{pl[3][1]:6.1f} -> "
              f"{po[3][0]:5.1f}-{po[3][1]:6.1f}  {pl[1]:4.1f} -> {po[1]:4.1f} sigma")
    y = np.arange(len(SETS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25 * (len(SETS) + 0.9) / 6.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    # left: the band itself, as a bar per treatment, so that both ends can be seen to move
    ax = axes[0]
    # Fixed limits, so that an open end has somewhere definite to point. The scan runs down to
    # beta/H = 0.7, but a transition that slow lasts a sizeable fraction of a Hubble time. The axis
    # stops at 1, as in figure 5, and no band is marked as continuing below it.
    XLIM = (1.0, 520.0)
    FLOOR = XLIM[0]
    for yy, (pl, po) in zip(y, vals):
        for row_, colour, off in ((pl, DESYcyan, +0.17), (po, DESYorange, -0.17)):
            # the 2 sigma band first and paler, so the 1 sigma one is drawn on top of it
            (glo, ghi), gopen = row_[3], row_[4]
            gblo = max(glo, FLOOR)
            gbhi = ghi / 1.7 if gopen[1] else ghi
            ax.plot([gblo, gbhi], [yy + off, yy + off], color=colour, lw=4.0, alpha=0.5,
                    solid_capstyle="butt", zorder=2)
            for start, target, is_open in ((gbhi, XLIM[1], gopen[1]),):
                if is_open:
                    ax.annotate("", xy=(target, yy + off), xytext=(start, yy + off),
                                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                                alpha=0.5, mutation_scale=11, shrinkA=0,
                                                shrinkB=0), zorder=2)
            (lo, hi), open_at = row_[0], row_[2]
            # An edge that is the end of the scanned range is not a measured edge: the band
            # continues past it. A flat end there would claim a bound we do not have, so the bar
            # stops short and a thinner arrow carries the last stretch off the axis.
            blo = max(lo, FLOOR)
            bhi = hi / 1.7 if open_at[1] else hi
            ax.plot([blo, bhi], [yy + off, yy + off], color=colour, lw=4.0,
                    solid_capstyle="butt", zorder=3,
                    label=(r"$(A,\gamma)$ power law, Afzal et al." if colour is DESYcyan
                           else BF) if yy == y[0] else None)
            for start, target, is_open in ((bhi, XLIM[1], open_at[1]),):
                if is_open:
                    ax.annotate("", xy=(target, yy + off), xytext=(start, yy + off),
                                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0,
                                                mutation_scale=11, shrinkA=0, shrinkB=0),
                                zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"band in $\beta/H$")
    ax = axes[1]
    for j, (yy, (pl, po)) in enumerate(zip(y, vals)):
        ax.plot([pl[1], po[1]], [yy, yy], color="0.75", lw=1.4, zorder=1)
        ax.scatter(pl[1], yy, s=90, color=DESYcyan, zorder=3, edgecolor="white", lw=0.8)
        ax.scatter(po[1], yy, s=44, color=DESYorange, marker="D", zorder=4,
                   edgecolor="white", lw=0.6)
    # room either side, so that a marker sits inside the frame rather than astride it
    sig = [v for row in vals for v in (row[0][1], row[1][1])]
    pad = 0.12 * (max(sig) - min(sig))
    ax.set_xlim(min(sig) - pad, max(sig) + pad)
    # two lines: at this panel width one line is wider than the panel and runs into the label of
    # the neighbouring one
    ax.set_xlabel("preference for\na transition $/\\ \\sigma$")
    for ax in axes:
        ax.set_axisbelow(True)
        # the category axis has no scale, so minor ticks on it carry nothing
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        if ax.get_xscale() != "log":
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune="lower"))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_ylim(y.min() - 0.6, y.max() + 0.6)
    # mark the product used for the main results
    yd = y[[t for t, _ in SETS].index(DEFAULT)]
    for ax in axes:
        ax.axhspan(yd - 0.45, yd + 0.45, color=DESYdunkelrot, alpha=0.07, zorder=0)
    # outside the frame, so that the label does not sit half on the shaded band and half off it
    axes[-1].annotate("used here", xy=(1.0, yd), xycoords=axes[-1].get_yaxis_transform(),
                     xytext=(5, 0), textcoords="offset points", fontsize=8,
                     color=DESYdunkelrot, va="center", ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    mid = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=len(labels),
               loc="upper center", bbox_to_anchor=(mid, 1.0), handlelength=1.5,
               handletextpad=0.6, columnspacing=2.2)
    logo_in_corner(fig, axes[1], "lower right", size=0.16, margin=0.3)    # small
    fig.savefig(PLOTS / "free_spectra.pdf", dpi=200)
    print("wrote plots/free_spectra.pdf")
