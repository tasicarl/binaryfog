"""Figure 3: where the shape of the realisation law matters for ln L.

At the best fit of the binaries alone (CURN free spectrum, NANOGrav priors; bestfit_holodeck.npz, 5000 holodeck
realisations), in bins 1, 8 and 14. Top: the realisations and the three laws matched to their median and 16-84
half-width (warp with the surrogate's skew at this point, log-normal, map-Airy). Middle: the NANOGrav 15 yr
free-spectrum posterior of the bin (flat prior in log10 rho) at the binary amplitude, normalised to its maximum on
the axis. Bottom: the likelihood integrand, density times posterior, for the realisations (grey) and each law,
normalised to unit area under the realisations. The numbers are ln L(law) - ln L(realisations).

    python plot_realisations.py      -> plots/realisations.pdf
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from paths import FREESPEC_ROOT, PLOTS, SCANS
BINS = (0, 7, 13)        # 0-based: bins 1, 8, 14
OUT = "realisations.pdf"
LO, HI = -2.5, 5.5       # axis range in units of the width: the loud tail in full, the faint tail cut
NBIN = 31                # histogram bins across the axis: smooth for 5000 realisations

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import binaryfog as bf
    from binaryfog.background import hc2_to_h2omega
    from mapairy import matched
    from desy_colors import DESYaubergine, DESYorange, DESYdunkelblau, DESYgruen
    from paper_style import use_paper_style, WIDTH
    use_paper_style()

    z = np.load(SCANS / "bestfit_holodeck.npz", allow_pickle=True)
    X, th = z["log10hc2_alone"], z["theta_alone"]
    data = bf.FreeSpectrum.load("ng15_cp_v2", root=FREESPEC_ROOT)
    C = float(np.ravel(bf.Background().predict(th)[2])[0])
    f = data.freqs
    q = (np.arange(100000) + 0.5) / 100000

    # a low figure, the upper row the shortest, wide panels
    fig, axes2 = plt.subplots(3, 3, figsize=(WIDTH, 3.45), sharex="col",
                              gridspec_kw=dict(height_ratios=(0.7, 0.45, 0.62)))
    pending = []
    for ax, mid_ax, low, i in zip(axes2[0], axes2[1], axes2[2], BINS):
        y = X[i]
        med = np.median(y)
        half = 0.5 * (np.percentile(y, 84) - np.percentile(y, 16))
        v = y - med
        edges = np.linspace(LO * half, HI * half, NBIN + 1)
        bw = edges[1] - edges[0]
        u = np.linspace(LO * half, HI * half, 900)

        def binned(cdf):
            # the law averaged over a histogram bin, so that it is compared with the histogram like for like
            return (cdf(u + 0.5 * bw) - cdf(u - 0.5 * bw)) / bw

        def warp_cdf(x):
            disc = np.maximum(1.0 + 4.0 * C * x / half, 0.0)
            zz = (-1.0 + np.sqrt(disc)) / (2.0 * C)
            fl = -half / (4.0 * C)
            return np.where(x < fl, 0.0, norm.cdf(zz))

        _, lgq, cdfm = matched(half)
        zq = np.maximum(norm.ppf(q), -0.5 / C)
        laws = [("warp", warp_cdf, half * (zq + C * zq ** 2), DESYorange, "-", 1.8),
                ("log-normal", lambda x: norm.cdf(x, 0.0, half), half * norm.ppf(q), DESYaubergine, ":", 1.8),
                ("map-Airy", lambda x: np.interp(x, lgq, cdfm, left=0.0, right=1.0),
                 np.interp(q, cdfm, lgq), DESYdunkelblau, "-.", 1.5)]

        def L(x):
            h2 = hc2_to_h2omega(np.full(np.shape(x), f[i]), med + x)
            return np.exp(data.ln_density(data.to_log10rho(h2, i), i))

        # top: the distributions
        hv = np.histogram(v, bins=edges)[0] / (len(v) * bw)     # normalised to all realisations, not to the axis
        ax.stairs(hv, edges, fill=True, color="0.85", lw=0, label="5000 realisations")
        ax.stairs(hv, edges, color="0.55", lw=0.5)
        for lab, cdf, _, col, ls, lw in laws:
            ax.plot(u, binned(cdf), color=col, ls=ls, lw=lw,
                    label=(rf"warp, skew $c = {C:.2f}$" if lab == "warp" else lab))
        ax.set_xlim(LO * half, HI * half)
        ax.set_ylim(0, 1.12 * max(hv.max(), max(binned(c).max() for _, c, *_ in laws)))
        ax.set_yticks([])
        ax.set_title(rf"bin {i+1}, $f = {f[i]*1e9:.1f}$ nHz", fontsize=8)
        # middle: the NANOGrav posterior of the bin at the binary amplitude
        Lu = L(u)
        mid_ax.fill_between(u, Lu / Lu.max(), color=DESYgruen, alpha=0.25, lw=0)
        mid_ax.plot(u, Lu / Lu.max(), color=DESYgruen, lw=1.5)
        mid_ax.set_ylim(0, 1.15)
        mid_ax.set_yticks([])
        # bottom: density times posterior, normalised to unit area under the realisations
        Lr = float(np.mean(L(v)))
        hw, _ = np.histogram(v, bins=edges, weights=L(v))
        hw = hw / (len(v) * bw) / Lr
        low.stairs(hw, edges, fill=True, color="0.85", lw=0, zorder=1)
        low.stairs(hw, edges, color="0.55", lw=0.5, zorder=1)
        dls, peaks = [], []
        for lab, cdf, xs, col, ls, lw in laws:
            integ = binned(cdf) * Lu / Lr
            low.fill_between(u, integ, color=col, alpha=0.09, lw=0, zorder=2)
            low.plot(u, integ, color=col, ls=ls, lw=lw, zorder=3)
            peaks.append(integ)
            dls.append((lab, float(np.log(np.mean(L(xs)) / Lr)), col))   # exact, from the law's quantiles
        prof = np.max(np.vstack(peaks + [np.interp(u, 0.5 * (edges[1:] + edges[:-1]), hw)]), axis=0)
        top = 2.3 * prof.max()             # room above the curves for the labels
        low.set_ylim(0, top)
        low.set_yticks([])
        # the labels are placed below, once the layout is fixed
        pending.append((low, u, peaks, prof, dls, hw, edges))
        low.set_xlabel(r"$\log_{10} h_\mathrm{c}^2$ minus its median")
        print(f"  bin {i+1}: " + ", ".join(f"{lab} {dl:+.3f}" for lab, dl, _ in dls))
    axes2[0, 0].set_ylabel("probability\ndensity", fontsize=8)
    axes2[1, 0].set_ylabel("NANOGrav\nposterior", fontsize=8)
    axes2[2, 0].set_ylabel("density\n" r"$\times$ posterior", fontsize=8)
    h, lab = axes2[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.1)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.12, top=0.85, wspace=0.07, hspace=0.12)
    # labels placed after the layout is fixed: the three labels of a panel jointly at free spots (no curve, histogram
    # or other label under a box, no crossing lines) with the shortest total line to their own curves
    fig.canvas.draw()

    def label(lab, dl):
        # the quantity named once per panel, on the warp's number
        v = f"{dl:+.3f}" if abs(dl) >= 5e-4 else "0.000"
        return rf"$\Delta\ln L = {v}$" if lab == "warp" else rf"${v}$"
    rend = fig.canvas.get_renderer()
    for low, u, peaks, prof, dls, hw, edges in pending:
        top = low.get_ylim()[1]
        occ = np.maximum(prof, np.interp(u, 0.5 * (edges[1:] + edges[:-1]), hw))
        tr = low.transData
        x_lo, x_hi = low.get_xlim()

        def hits(bb, seg):
            return any(bb.contains(*pt) for pt in np.linspace(seg[0], seg[1], 60)[3:-1])

        def cross(s1, s2):
            (a, b_), (c, d) = s1, s2
            o = lambda p_, q_, r_: np.sign((q_[0] - p_[0]) * (r_[1] - p_[1]) - (q_[1] - p_[1]) * (r_[0] - p_[0]))
            return o(a, b_, c) != o(a, b_, d) and o(c, d, a) != o(c, d, b_)

        # candidates per label: a free spot for its box and a line to a point of its own curve
        cands = []
        for (lab, dl, col), integ in zip(dls, peaks):
            t = low.text(0, 0, label(lab, dl), fontsize=7.5, color=col, va="center")
            pts = tr.transform(np.column_stack([u, integ]))
            # anchors on the right flank of the curve, between a quarter and four fifths of its peak
            ok = np.nonzero((integ >= 0.25 * integ.max()) & (integ <= 0.8 * integ.max())
                            & (u > u[np.argmax(integ)]))[0][::6]
            c = []
            # a grid of spots in the upper part of the panel, the text left- or right-aligned at each
            for side, xf in ([("left", x) for x in np.arange(0.34, 0.80, 0.04)]
                             + [("right", x) for x in np.arange(0.22, 0.68, 0.04)]):
                for yf in np.linspace(0.86, 0.25, 22):
                    xt = x_lo + xf * (x_hi - x_lo)
                    t.set_ha(side)
                    t.set_position((xt, yf * top))
                    bb = t.get_window_extent(rend).expanded(1.35, 1.8)
                    frame = low.get_window_extent(rend)
                    if bb.x0 < frame.x0 + 2 or bb.x1 > frame.x1 - 2 or bb.y1 > frame.y1 - 2:
                        continue                                       # the label stays inside its panel
                    (x0, y0), (x1, y1) = tr.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                    under = occ[(u >= x0) & (u <= x1)]
                    if under.size and under.max() > y0 - 0.15 * top:     # clearance above the curves
                        continue
                    opts = []
                    for j in ok:
                        rel = 0.0 if pts[j, 0] < 0.5 * (bb.x0 + bb.x1) else 1.0
                        # from the bottom corner of the label, so the line never runs along the text
                        start = np.array([bb.x0 if rel == 0.0 else bb.x1, bb.y0])
                        opts.append((np.hypot(*(pts[j] - start)), side, xt, yf * top, j, bb, rel, (start, pts[j])))
                    c += sorted(opts, key=lambda r: r[0])[:3]     # the three nearest anchors per spot
            t.remove()
            cands.append(c)

        def compatible(a, b_):
            return not (a[5].overlaps(b_[5]) or hits(a[5], b_[7]) or hits(b_[5], a[7]) or cross(a[7], b_[7]))

        comp = {(m, n): np.array([[compatible(a, b_) for b_ in cands[n]] for a in cands[m]])
                for m, n in ((0, 1), (0, 2), (1, 2))}
        best = None
        for i0_, a in enumerate(cands[0]):
            for i1_ in np.nonzero(comp[0, 1][i0_])[0]:
                for i2_ in np.nonzero(comp[0, 2][i0_] & comp[1, 2][i1_])[0]:
                    cost = a[0] + cands[1][i1_][0] + cands[2][i2_][0]
                    if best is None or cost < best[0]:
                        best = (cost, (i0_, i1_, i2_))
        assert best is not None, f"no joint placement, candidates {[len(c) for c in cands]}"
        for (lab, dl, col), integ, c, k in zip(dls, peaks, cands, best[1]):
            _, side, xt, yt, j, bb, rel, seg = c[k]
            low.annotate(label(lab, dl), xy=(u[j], integ[j]), xytext=(xt, yt), fontsize=7.5,
                         ha=side, va="center", color=col,
                         arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=2, shrinkB=1, relpos=(rel, 0.0)))
    fig.savefig(PLOTS / OUT, dpi=200)
    print("wrote plots/" + OUT)
