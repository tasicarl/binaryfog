"""
Figure 5, plots/phase_transition.pdf: the allowed region in the plane of transition strength and
inverse duration, with the binaries treated three ways, on the CURN free spectrum of the 2026
NANOGrav 15 yr release (baseline.DATA), from the 120 x 120 scans of run_scan_marginal.py and the
Delta N_eff grid of dneff_grid.py.

Left: the 1 and 2 sigma regions. Right: the best-fit spectra and the best fit of the binaries alone
with the band of 90 per cent of its realisations, against the measured free spectrum. Prints the
1 sigma band in beta/H at alpha = 1000 of each treatment, the 2 sigma band of binaryfog and the gain
G over the fit with no transition.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

from paths import FREESPEC_ROOT, PLOTS, SCANS
from fopt_model import spectrum_fopt                                          # noqa: E402
import binaryfog as bf                                                        # noqa: E402
from desy_colors import (DESYcyan, DESYorange, DESYpetrol, DESYdunkelrot,      # noqa: E402
                         DESYgruen)
from paper_style import use_paper_style, WIDTH, frequency_axis, BF, logo_in_corner, LOGO_SMALL      # noqa: E402
from band import edges                                                        # noqa: E402
from baseline import TAG, DATA                                                # noqa: E402

T1 = 1.15                         # Wilks 1 sigma threshold in ln L, two scanned parameters
T2 = 3.09                         # and the 2 sigma one (95.4%, not 95%)
# The profile is maximised over up to seven parameters at every grid point, so it carries the
# maximiser's own noise. For drawing it is smoothed by this much, in grid cells; the bands printed at
# the end are read off the unsmoothed grid.
SMOOTH = 0.8
# The contours are drawn on a mesh this many times finer than the scan, so that they do not follow the
# cell edges of the grid.
UPSAMPLE = 6
DNEFF_MAX = 0.22                  # BBN and CMB, Yeh et al. 2022
CASES = [("fopt", "phase transition alone", DESYpetrol, ":"),
         ("agamma", r"$+$ $(A,\gamma)$ power law, Afzal et al.", DESYcyan, "--"),
         ("holodeck", "$+$ " + BF, DESYorange, "-")]
# Decreasing line widths in the order the three are drawn, for the right panel. There the three
# best fits nearly coincide, because all three describe the same measurement, and at equal width
# the last one drawn hides the others. The contours of the left panel keep one width: they are
# nested rather than coincident, and the white outline each one carries to stay visible over the
# dark end of the colour map would eat into a wider curve drawn underneath it.
WIDTHS = (3.0, 2.1, 1.3)
GRID = "n120_a-2to3"              # 120 x 120 points, alpha from 0.01 to 1000


def load(mode):
    d = pd.read_csv(SCANS / f"scan_{mode}_{GRID}{TAG}.csv")
    p = d.pivot(index="alpha", columns="betaH", values="lnL")
    return np.log10(p.index.values), np.log10(p.columns.values), p.to_numpy(), d


# The maximum of each scan lies on a ridge towards weaker and slower transitions along which ln L changes
# by less than 1e-4, so the arg max can sit anywhere on it, also below the panel. The best fit marked on
# the left and drawn on the right is the maximum in the vacuum-dominated slab 10 <= alpha <= 100,
# beta/H >= 1, the same point in both panels.
MARKERS = {"fopt": ("s", 4.0), "agamma": ("D", 3.6), "holodeck": ("o", 4.6)}


def shown_best(d):
    q = d[(d.alpha >= 10) & (d.alpha <= 100) & (d.betaH >= 1)]
    return q.loc[q.lnL.idxmax()]


def connected(L, threshold):
    """Boolean mask of the part of {L >= threshold} that contains the maximum.

    A profile maximised over a dozen parameters leaves single grid points on the wrong side of a
    threshold while their neighbours are on the right side of it. That is the maximiser, not the
    likelihood, so the mask keeps the component containing the best fit and fills holes in it.
    """
    from scipy import ndimage
    Lf = np.nan_to_num(L, nan=-np.inf)
    mask = ndimage.binary_fill_holes(Lf >= threshold)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    return lab == lab[np.unravel_index(np.argmax(Lf), L.shape)]


def clean(Ls, mask, level, eps=1e-3):
    """The smoothed profile, changed only where the maximiser contradicts the connected region.

    Nodes inside the region that dip below the level are lifted just above it, and nodes outside it
    that reach the level are pushed just below it. Every other node keeps its value, so the edge is
    still found by interpolating a smooth field between nodes and does not follow the grid cells,
    as it would if the region were drawn from its indicator.
    """
    Lc = Ls.copy()
    Lc[mask & (Ls < level)] = level + eps
    Lc[~mask & (Ls >= level)] = level - eps
    return Lc


def fine(a, b, Z, factor=UPSAMPLE):
    """The field on a mesh `factor` times finer, by cubic interpolation in log alpha and log beta/H.

    Only the drawing uses this. A contour traced on the scan's own grid follows its cell edges, and
    along a boundary that runs nearly flat those edges read as a staircase that is a property of the
    grid and not of the likelihood.
    """
    from scipy.interpolate import RectBivariateSpline
    af = np.linspace(a[0], a[-1], (len(a) - 1) * factor + 1)
    bff = np.linspace(b[0], b[-1], (len(b) - 1) * factor + 1)
    return af, bff, RectBivariateSpline(a, b, Z, kx=3, ky=3, s=0)(af, bff)


def place_label(fig, ax, label, inside, pad_pt=1.0, hide_inside=True):
    """Move label to the position inside the region where it keeps the largest distance from every
    drawn contour line, with every point of its box (padded by pad_pt) inside the region and inside
    the axes. The test runs on the final layout, so it sees the label as printed. Says so when no such
    position exists, instead of leaving the label on a line silently."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    to_data_early = ax.transData.inverted()
    start = label.get_position()
    verts = []
    for c in ax.collections:
        if getattr(c, "filled", True):           # fills are areas, not lines
            continue
        for p in c.get_paths():
            if len(p.vertices):
                verts.append(c.get_transform().transform(p.vertices))
    for ln in ax.lines:
        verts.append(ln.get_transform().transform(ln.get_xydata()))
    verts = np.concatenate(verts) if verts else np.empty((0, 2))
    # lines hidden under the region (drawn below it) cannot touch the label; keep only visible ones
    if len(verts):
        verts = verts[~inside(to_data_early.transform(verts))] if hide_inside else verts
    axbb = ax.get_window_extent(r)
    to_data = ax.transData.inverted()
    best = None
    lx0, lx1 = np.log10(ax.get_xlim()); ly0, ly1 = np.log10(ax.get_ylim())
    for lx in np.linspace(lx0, lx1, 90):
        for ly in np.linspace(ly0, ly0 + 0.6 * (ly1 - ly0), 90):
            label.set_position((10 ** lx, 10 ** ly))
            bb = label.get_window_extent(r).padded(pad)
            if bb.x0 < axbb.x0 + pad or bb.x1 > axbb.x1 - pad or bb.y0 < axbb.y0 + pad or bb.y1 > axbb.y1 - pad:
                continue
            gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 25), np.linspace(bb.y0, bb.y1, 7))
            if not np.all(inside(to_data.transform(np.c_[gx.ravel(), gy.ravel()]))):
                continue
            dx = np.maximum(np.maximum(bb.x0 - verts[:, 0], verts[:, 0] - bb.x1), 0)
            dy = np.maximum(np.maximum(bb.y0 - verts[:, 1], verts[:, 1] - bb.y1), 0)
            clear = float(np.hypot(dx, dy).min()) if len(verts) else np.inf
            if clear > 0 and (best is None or clear > best[0]):
                best = (clear, lx, ly)
    if best is None:
        label.set_position(start)
        print("  warning: no position inside the region keeps the label clear of every contour line",
              flush=True)
        return False
    label.set_position((10 ** best[1], 10 ** best[2]))
    return True


if __name__ == "__main__":
    use_paper_style()
    # taller, because the legend has three rows in two columns
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(WIDTH, 2.75),
                                  gridspec_kw={"width_ratios": [1.0, 1.0]})

    # The population treatment is filled, its 1 sigma region darker than its 2 sigma one, which covers the
    # whole panel, so that only a tint can show it. The other two are drawn as outlines on top, and
    # their two levels are told apart by opacity rather than by line width.
    FILL = {T2: 0.16, T1: 0.42}
    LINE_ALPHA = {T1: 1.0, T2: 0.45}
    for mode, label, colour, style in reversed(CASES):
        a, b, L, d = load(mode)
        Ls = gaussian_filter(np.nan_to_num(L, nan=np.nanmin(L)), SMOOTH)
        mxs = np.nanmax(Ls)
        for thr in (T2, T1):
            mask = connected(Ls, mxs - thr)
            if mode != "holodeck" and mask.all():  # no boundary inside the panel to draw
                continue
            if mode == "holodeck" and mask.all():
                # the whole scan is inside: tint the whole panel rather than contour a field whose
                # interpolation dips below the level between nodes and leaves pinholes
                ax.axhspan(0.1, 1e4, color=colour, alpha=FILL[thr], lw=0, zorder=-10)
                continue
            Lc = clean(Ls, mask, mxs - thr)
            af, bff, Lf = fine(a, b, Lc)
            if mode == "holodeck":
                ax.contourf(10 ** af, 10 ** bff, Lf.T, levels=[mxs - thr, mxs + 1],
                            colors=[colour], alpha=FILL[thr], zorder=-10, antialiased=True)
            else:
                ax.contour(10 ** af, 10 ** bff, Lf.T, levels=[mxs - thr], colors=[colour],
                           linestyles=style, linewidths=1.2 if style == ":" else 1.8,
                           alpha=LINE_ALPHA[thr])

    # The gravitational waves themselves count as radiation: where they would add more than
    # Delta N_eff = 0.22 (Yeh et al. 2022) the point is excluded independently of the PTA fit. The values
    # come from dneff_grid.py, which integrates the spectrum at the T_reh the binaryfog profile chose at
    # each point.
    dn = pd.read_csv(SCANS / f"dneff_n120_a-2to3{TAG}.csv").pivot(index="alpha", columns="betaH",
                                                                values="DNeff_GW")
    an, bn = np.log10(dn.index.values), np.log10(dn.columns.values)
    af, bff, Nf = fine(an, bn, np.log10(dn.to_numpy()))
    # The excluded region is a white area of its own shape with a faint hatching on it.
    matplotlib.rcParams["hatch.color"] = "0.80"    # alpha in a hatch colour is ignored
    matplotlib.rcParams["hatch.linewidth"] = 0.6
    # Drawn above the contour lines as well as above the fills (zorder 2.1 and 2.2): the excluded
    # region shows nothing of what lies under it, lines included, which leaves the label room
    # fully inside it.
    ax.contourf(10 ** af, 10 ** bff, Nf.T, levels=[np.log10(DNEFF_MAX), 99], colors=["white"],
                zorder=2.1)
    ax.contourf(10 ** af, 10 ** bff, Nf.T, levels=[np.log10(DNEFF_MAX), 99], colors="none",
                hatches=["////"], zorder=2.2)
    # placed by place_label() once the layout is final, fully inside the region, on a white,
    # rounded, partly transparent box of its own
    dneff_label = ax.text(30, 1.3, r"$\Delta N_\mathrm{eff} > 0.22$", ha="center", va="center",
                          fontsize=5.5, color="0.25", zorder=3,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    from scipy.interpolate import RegularGridInterpolator
    _dn_at = RegularGridInterpolator((af, bff), Nf, bounds_error=False, fill_value=-99)
    # solid: matplotlib dashes a negative level by default, and log10(0.22) is negative, while
    # dashed already means the (A, gamma) power law in this panel
    ax.contour(10 ** af, 10 ** bff, Nf.T, levels=[np.log10(DNEFF_MAX)], colors=["0.45"],
               linewidths=0.8, linestyles="solid", zorder=2.3)

    # Above alpha of about 100 the transition is vacuum dominated and the spectrum stops
    # depending on alpha, so the region continues to the right of this panel unchanged; the
    # caption says so. It is scanned down to alpha of 0.01, a decade below the left edge, so that
    # the contours enter the panel rather than beginning inside it, and so that the no-transition
    # reference the preference is measured against is a genuinely negligible transition.
    ax.set_xscale("log")
    ax.set_yscale("log")
    # a little left of 10^-1, so that its label sits inside the panel instead of in the corner
    # where the lowest label of the other axis already is
    ax.set_xlim(0.07, 1e2)
    # three decades in a half-width panel: the decade labels touch at the default spacing
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    ax.tick_params(axis="x", labelsize=7)
    # The scan reaches beta/H = 0.7, but a transition that slow lasts a sizeable fraction of a
    # Hubble time. The axis stops at 1; the fill still covers every point of it.
    ax.set_ylim(1.0, 10 ** b.max())
    ax.set_xlabel(r"transition strength $\alpha$")
    ax.set_ylabel(r"transition speed $\beta/H$")
    # The point of the panel: the light tint reaches every corner, so with the population treated
    # as uncertain no part of the plane is excluded at 2 sigma.
    ax.annotate("$2\\sigma$-allowed parameter\nspace using " + BF,
                xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.5, color=DESYorange)
    for mode, _, colour, _ in CASES:
        bst = shown_best(load(mode)[3])
        mk, ms = MARKERS[mode]
        ax.plot(bst.alpha, bst.betaH, ls="none", marker=mk, ms=ms, color=colour, mec=colour, mew=0, zorder=4)
        print(f"  best fit shown, {mode}: alpha {bst.alpha:.3g}, beta/H {bst.betaH:.3g}, "
              f"T_reh {bst.Treh * 1e3:.3g} MeV, ln L {bst.lnL:.5f} (arg max of the scan {load(mode)[3].lnL.max():.5f})")

    # right panel: the best fits and the binary band
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    # Curves are drawn on a fine frequency grid rather than at the fourteen measured frequencies,
    # where they would be hidden behind the violins. The grid runs a little past the band at both
    # ends, where the surrogate is extrapolated (Background(extrapolate=True)); the fit itself uses
    # the measured bins only.
    f = np.geomspace(data.freqs[0] / 1.9, data.freqs[-1] * 1.9, 600)
    bg = bf.Background(f, extrapolate=True)
    for (mode, label, colour, style), wl in zip(CASES, WIDTHS):
        a, b, L, d = load(mode)
        best = shown_best(d)
        pt = np.asarray(spectrum_fopt(f, best.alpha, best.betaH, best.Treh), float)
        tot = pt.copy()
        if mode == "holodeck":
            th = [best[k] for k in bf.PARAM_NAMES]
            med, sig, skew = bg.predict(th)
            tot = pt + med
            # the binary component of this joint fit sits well below the data, because there the
            # transition carries the signal. The model that matters for the argument is a
            # different one: the best fit with no transition at all, taken from the weak-transition
            # corner of the same scan. That one has to account for the measurement on its own.
            corner = d[d.alpha == d.alpha.min()]
            alone = corner.loc[corner.lnL.idxmax()]
            tha = [alone[k] for k in bf.PARAM_NAMES]
            meda, _, _ = bg.predict(tha)
            lo = bf.quantile_spectrum(bg, tha, 0.05)
            hi = bf.quantile_spectrum(bg, tha, 0.95)
            axs.fill_between(f * 1e9, lo, hi, color=DESYgruen, alpha=0.20,
                             lw=0, label=r"$90\%$ of realisations")
            axs.plot(f * 1e9, meda, color=DESYgruen, lw=1.6, ls="-",
                     label="binaries alone, best fit")
        elif mode == "agamma":
            tot = pt + bf.power_law(f, best.logA, best.gamma)
        axs.plot(f * 1e9, tot, color=colour, ls=style, lw=wl, label=label)
    # the measured spectrum as violins of the stored posteriors themselves, not as error bars:
    # these posteriors are skewed and several are upper limits, so a symmetric bar misrepresents
    # them. Each violin is the kernel density estimate ceffyl distributes, drawn against the same
    # axis as the model curves and scaled to a fixed width in log-frequency.
    fd = data.freqs                       # the violins sit at the measured frequencies
    lf = np.log10(fd * 1e9)
    half = 0.5 * np.min(np.diff(lf)) * 0.85
    for i in range(len(fd)):
        dens = np.exp(data.density[i] - data.density[i].max())
        pref = 8 * np.pi ** 4 / bf.background.H100_HZ ** 2 * fd[i] ** 5 * data.tspan
        y = pref * 10.0 ** (2 * data.grid)
        keep = np.flatnonzero(dens > 2e-2)[::8]
        yy, dd = y[keep], dens[keep]
        w = half * dd / dd.max()
        axs.fill_betweenx(yy, 10 ** (lf[i] - w), 10 ** (lf[i] + w), color=DESYdunkelrot,
                          alpha=0.55, lw=0,
                          label="NANOGrav 15 yr, 2026" if i == 0 else None)
    # the curves are computed wider than the axis shows, so that they leave the panel at its edge
    # instead of stopping inside it
    axs.set_xlim(data.freqs[0] / 1.25 * 1e9, data.freqs[-1] * 1.25 * 1e9)
    axs.set_xscale("log")
    axs.set_yscale("log")
    frequency_axis(axs.xaxis)   # 10 nHz major, 2, 5, 20 nHz labelled minor
    axs.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    axs.set_xlabel(r"$f$ / nHz")
    axs.set_ylabel(r"$h^2 \Omega_\mathrm{GW}$")
    axs.set_ylim(1e-11, 1e-7)
    # The legend sits above both panels, where it covers neither the contours nor the violins.
    h, lab = axs.get_legend_handles_labels()
    from matplotlib.patches import Patch
    # the population entry shows both its fill on the left and its curve on the right
    # each line carries the symbol of its best fit in the left panel
    hl = [(Patch(color=c, alpha=FILL[T1], lw=0),
           Line2D([0], [0], color=c, ls=st, lw=1.3, marker=MARKERS[m][0], ms=MARKERS[m][1], mew=0))
          if m == "holodeck"
          else Line2D([0], [0], color=c, ls=st, lw=1.5 if st == ":" else 1.8,
                      marker=MARKERS[m][0], ms=MARKERS[m][1], mew=0) for m, _, c, st in CASES]
    ll = [lb for _, lb, _, _ in CASES]
    # The band belongs under the curve it brackets, and matplotlib fills a legend column by
    # column, so the order here is the order down the second column.
    order = ["binaries alone, best fit", r"$90\%$ of realisations", "NANOGrav 15 yr, 2026"]
    extra = sorted(((hh, bb) for hh, bb in zip(h, lab) if bb not in ll),
                   key=lambda t: order.index(t[1]) if t[1] in order else 99)
    leg = fig.legend(hl + [hh for hh, _ in extra], ll + [bb for _, bb in extra],
               fontsize=6.8, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    place_label(fig, ax, dneff_label, lambda xy: _dn_at(np.log10(xy)) > np.log10(DNEFF_MAX))
    # the logo in the upper left of the right panel, since the left one keeps its label there
    logo_in_corner(fig, axs, "upper left", size=0.26, margin=0.15, path=LOGO_SMALL)
    fig.savefig(PLOTS / "phase_transition.pdf", dpi=600)
    print("wrote plots/phase_transition.pdf")
    # the bands at alpha = 1000, read off the unsmoothed grid, and the gain G over the weakest
    # transition of the scan (the fit with no transition)
    for mode, label, _, _ in CASES:
        a, b, L, d = load(mode)
        mx = np.nanmax(L)
        lo, hi, _ = edges(a, b, L, 1e3, T1)
        line = f"  {label:26s} best lnL {mx:+.2f}, 1 sigma beta/H {lo:5.1f} to {hi:5.1f}"
        if mode == "holodeck":
            lo9, hi9, _ = edges(a, b, L, 1e3, T2)
            line += f", 2 sigma {lo9:5.1f} to {hi9:5.1f}"
        if mode != "fopt":
            line += f", gain over the weak corner {mx - np.nanmax(L[0]):5.2f}"
        print(line)
