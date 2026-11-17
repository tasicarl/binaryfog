"""
Figure 4: binaryfog against figure 9 of NANOGrav's 15-year new-physics paper.

Our posterior over the six binary-evolution parameters, fitted to the same free spectrum over the
same five frequency bins under the same two prior sets, overlaid on the contours and marginals of
the replacement figure 9 of the erratum (inputs/pta/ng15_fig9_erratum_contours.json). Their
contours come from Gaussian processes trained on the mean and standard deviation of the holodeck
library; ours from the polynomial surrogate and the warp of this paper.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desy_colors import DESYcyan, DESYorange
from paper_style import use_paper_style, WIDTH, BF, logo_below_legend

from paths import INPUTS, PLOTS, SCANS                                    # noqa: E402
import binaryfog as bf                                                    # noqa: E402

CONTOURS = INPUTS / "pta/ng15_fig9_erratum_contours.json"
CHAINS = SCANS / "ng15_fig9_binaryfog_erratum.npz"
OUT = PLOTS / "nanograv_fit.pdf"
# NANOGrav drew their contours from an unsmoothed 34-bin histogram, and the flat-prior ones break
# into many small closed islands. Pieces spanning less than three of those bins in both directions
# are not drawn; the contours themselves are drawn as published.
MIN_ISLAND = 3 / 34
# the credible levels corner.py fills, quoted there as 1, 2 and 3 sigma in two dimensions
LEVELS = {1: 1 - np.exp(-0.5), 2: 1 - np.exp(-2.0), 3: 1 - np.exp(-4.5)}
# Every axis carries its symbol and its unit. The three parameters the model samples as logarithms
# are drawn on a logarithmic axis with 10^x tick labels rather than on a linear axis labelled with
# a logarithm; the chains, the contours and the published curves are all stored as the logarithm
# and are raised to the power of ten on the way in.
LOGAX = {"gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10"}
LABELS = {"gsmf_phi0_log10": r"$\psi_0 / \mathrm{Mpc}^{-3}$",
          "gsmf_mchar0_log10": r"$m_{\psi,0} / M_\odot$",
          "mmb_mamp_log10": r"$\mu / M_\odot$",
          "mmb_scatter_dex": r"$\epsilon_\mu$",
          "hard_time": r"$\tau_\mathrm{f}$ / Gyr", "hard_gamma_inner": r"$\nu_\mathrm{inner}$"}


def lin(par, v):
    """Values as the axis of `par` shows them: the quantity itself where the axis is logarithmic."""
    return 10.0 ** np.asarray(v, float) if par in LOGAX else np.asarray(v, float)
CASE = {"flat": ("uniform", DESYcyan), "nanograv": ("astro", DESYorange)}   # our chain: NANOGrav's contour


def on_panel_edge(x, y, rng_x, rng_y, flat=1e-3, span=0.01):
    """Points belonging to a segment that closes the contour along the boundary of the drawing.

    A filled band that reaches the edge of its panel is closed there by the figure it was read
    from, and that closing segment is not part of the posterior. Those segments are exactly
    axis-parallel and long, while a real contour is neither: it is a smooth curve, so a stretch
    over which one coordinate does not move at all, while the other crosses a per cent of its
    range, is a drawn border. Detecting them by that geometry rather than by their distance to the
    axis limit matters, because corner.py clips a band at the data range, which sits a per cent or
    two inside the limit, and a distance rule tuned to the limit misses them.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros(len(x), bool)
    for v, w, rv, rw in ((x, y, rng_x, rng_y), (y, x, rng_y, rng_x)):
        d_flat = np.abs(np.diff(v)) < flat * (rv[1] - rv[0])
        d_long = np.abs(np.diff(w)) > span * (rw[1] - rw[0])
        seg = d_flat & d_long
        out[:-1] |= seg
        out[1:] |= seg
    return out


def credible(x, y, rng_x, rng_y, level, bins=36, smooth=1.8):
    """Contour of the smallest region of a two-dimensional histogram holding `level`."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[rng_x, rng_y])
    H = gaussian_filter(H, smooth)
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat) / flat.sum()
    thr = flat[np.searchsorted(csum, level)]
    return (0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, thr)


if __name__ == "__main__":
    use_paper_style()
    pub = json.load(open(CONTOURS))
    ours = np.load(CHAINS, allow_pickle=True)
    names = list(ours["names"])
    P = pub["params"]
    assert names == P, (names, P)
    # the panel ranges of the published figure are the library prior box
    rng = {k: list(bf.background.PRIOR_BOX[k]) for k in P}

    n = len(P)
    fig, axes = plt.subplots(n, n, figsize=(WIDTH, 5.70))
    for ax in axes.flat:
        ax.set_visible(False)
        ax.tick_params(labelsize=6)
    for j in range(n):
        for i in range(n):
            if i > j:
                continue
            ax = axes[j, i]
            ax.set_visible(True)
            if i == j:                                   # marginal distribution
                from scipy.ndimage import gaussian_filter1d
                for case, (pubname, colour) in CASE.items():
                    h, e = np.histogram(ours[case][:, i], bins=40, range=rng[P[i]], density=True)
                    # the same smoothing as the two-dimensional panels, so that the marginal and
                    # the contours describe the same estimate of the posterior. The height of a
                    # marginal carries no information, so both are scaled to unit peak, which is
                    # what makes ours and theirs comparable at all.
                    hs = gaussian_filter1d(h, 1.8)
                    ax.plot(lin(P[i], 0.5 * (e[1:] + e[:-1])), hs / hs.max(), color=colour, lw=1.6)
                    for mg in pub.get("marginals", []):
                        if mg["case"] != pubname or mg["param"] != P[i]:
                            continue
                        ax.plot(lin(P[i], mg["x"]), mg["y"], color=colour, lw=0.5, ls="-",
                                alpha=0.9)
                ax.set_ylim(0.0, 1.12)
                ax.set_yticks([])
            else:
                for case, (pubname, colour) in CASE.items():
                    xs, ys = ours[case][:, i], ours[case][:, j]
                    for lvl in (1, 2, 3):
                        xc, yc, H, thr = credible(xs, ys, rng[P[i]], rng[P[j]], LEVELS[lvl])
                        ax.contour(lin(P[i], xc), lin(P[j], yc), H, levels=[thr], colors=[colour],
                                   linewidths=1.6, alpha=0.95)
                    for qq in pub["panels"]:
                        if qq["case"] != pubname or qq["x_param"] != P[i] or qq["y_param"] != P[j]:
                            continue
                        xv, yv = np.asarray(qq["x"]), np.asarray(qq["y"])
                        fx = (xv - rng[P[i]][0]) / (rng[P[i]][1] - rng[P[i]][0])
                        fy = (yv - rng[P[j]][0]) / (rng[P[j]][1] - rng[P[j]][0])
                        if max(np.ptp(fx), np.ptp(fy)) < MIN_ISLAND:
                            continue
                        xv = np.where(on_panel_edge(xv, yv, rng[P[i]], rng[P[j]]), np.nan, xv)
                        ax.plot(lin(P[i], xv), lin(P[j], yv), color=colour, lw=0.5, ls="-",
                                alpha=0.9)
                ax.set_ylim(*lin(P[j], rng[P[j]]))
                if P[j] in LOGAX:
                    ax.set_yscale("log")
            ax.set_xlim(*lin(P[i], rng[P[i]]))
            if P[i] in LOGAX:
                ax.set_xscale("log")
            # three ticks per axis
            if P[i] not in LOGAX:
                ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            if i != j and P[j] not in LOGAX:
                ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            if j == n - 1:
                ax.set_xlabel(LABELS[P[i]], fontsize=7)
            else:
                ax.set_xticklabels([])
            if i == 0 and j > 0:
                ax.set_ylabel(LABELS[P[j]], fontsize=7)
            elif i > 0:
                ax.set_yticklabels([])
    from matplotlib.lines import Line2D
    axes[0, n - 1].set_visible(True)
    axes[0, n - 1].axis("off")
    leg = axes[0, n - 1].legend(handles=[
        Line2D([0], [0], color=DESYcyan, lw=1.6), Line2D([0], [0], color=DESYorange, lw=1.6),
        Line2D([0], [0], color="0.35", lw=1.6), Line2D([0], [0], color="0.35", lw=0.5, ls="-")],
        labels=["flat priors", "NANOGrav priors", BF, "NANOGrav, erratum"],
        fontsize=8, frameon=False, loc="center")
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(wspace=0.14, hspace=0.14)
    # binaryfog logo directly below the legend, as wide as the legend
    logo_below_legend(fig, leg)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
