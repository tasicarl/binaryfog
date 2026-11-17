"""
Figure 8: the preference for the phase transition at fixed binary lifetime tau_f and inner hardening index nu_inner
(hardening_map.py), for the three Gaussian prior sets on CURN: local significance from the gain with k = 3
(alpha, beta/H, T_reh), the other four astrophysical parameters profiled under their prior. Stars: the
binaries alone with both parameters free (the weakest-transition column of the figure-5 scan of each set).

    python plot_hardening.py      -> plots/hardening.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

from paths import PLOTS, SCANS                                            # noqa: E402
from paper_style import use_paper_style, WIDTH, logo_in_corner, LOGO_SMALL               # noqa: E402

PANELS = [("nanograv", "", "NANOGrav priors"), ("tomczak_kormendy", "_tomczak_kormendy", "Tomczak–Kormendy priors"),
          ("survey_average", "_survey_average", "survey-average priors")]
LEVELS = np.arange(1.5, 5.01, 0.5)
CMAP = "Blues"                                             # blue, as the logo
LABEL_COLOUR = {2: "k", 3: "w", 4: "w"}
LABEL_AT = {"nanograv": {2: (0.4, -1.0), 3: (6.0, -0.2), 4: (8.0, -1.25)},
            "tomczak_kormendy": {4: (0.5, -0.4)},
            "survey_average": {2: (0.5, -0.15), 3: (5.5, -0.2), 4: (7.5, -1.25)}}
STAR = "#1b2a49"                                           # the navy of the logo
CMAP_START = 0.12


def main():
    use_paper_style()
    d = np.load(SCANS / "hardening_map.npz")
    tau, nu = d["tau"], d["nu"]
    # one colour per 0.5 sigma band, the last above 5; the lightest band starts above white so that it is told
    # apart from the paper background and from the next band
    cmap = ListedColormap(plt.get_cmap(CMAP)(np.linspace(CMAP_START, 1.0, len(LEVELS))))
    norm = BoundaryNorm(LEVELS, cmap.N, extend="max")
    side = (WIDTH - 1.05) / 3.1
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, side + 0.95), sharey=True, layout="constrained")
    for ax, (tag, infix, title) in zip(axs, PANELS):
        # below the lowest level contourf leaves the area blank; it belongs to the lightest band
        sig = d[f"sigma_{tag}"]
        m = ax.contourf(tau, nu, np.maximum(sig, LEVELS[0]).T, levels=LEVELS, cmap=cmap, norm=norm, extend="max")
        # each line its own contour set, so that clabel finds the line of the label it places (with one set and a
        # subset of labelled levels, matplotlib looks among all lines and fails)
        for v in (2, 3, 4, 5):
            if not sig.min() < v < sig.max():
                continue
            cv = ax.contour(tau, nu, sig.T, levels=[v], colors="k", linewidths=0.6)
            if v in LABEL_AT[tag]:
                # 2, 3 and 4 sigma written on their lines, placed by hand at different heights since the 3 and 4 sigma
                # lines run close together at long lifetimes; black on light bands, white on dark ones
                for txt in ax.clabel(cv, fmt={v: rf"${v}\sigma$"}, inline=True, inline_spacing=1, fontsize=7,
                                     colors=LABEL_COLOUR[v], manual=[LABEL_AT[tag][v]]):
                    # upright: on the near-vertical lines matplotlib can turn the label upside down
                    txt.set(rotation=0, transform_rotates_text=False)   # horizontal
        s = pd.read_csv(SCANS / f"scan_holodeck_n120_a-2to3{infix}_ng15_cp_v2.csv")
        w = s[s.alpha == s.alpha.min()]
        b = w.loc[w.lnL.idxmax()]
        # a star on the edge tau_f = 0.1 Gyr is drawn slightly inside
        ax.plot(max(b.hard_time, 0.115), b.hard_gamma_inner, marker="*", ms=9, mfc=STAR, mec=STAR, mew=0.5,
                ls="none", zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(tau[0], tau[-1]); ax.set_ylim(nu[0], nu[-1])
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_box_aspect(1)
        ax.set_yticks([-1.5, -1.0, -0.5, 0.0])
        ax.set_xlabel(r"$\tau_{\rm f}$ / Gyr")
        ax.tick_params(axis="x", which="major", pad=7)
        ax.set_title(title, pad=6)
        print(f"{title:26s} sigma {np.nanmin(sig):.2f} to {np.nanmax(sig):.2f}; star at tau_f {b.hard_time:.2f} Gyr, "
              f"nu_inner {b.hard_gamma_inner:.2f}; map there "
              f"{sig[np.argmin(abs(tau - b.hard_time)), np.argmin(abs(nu - b.hard_gamma_inner))]:.2f} sigma")
    axs[0].set_ylabel(r"$\nu_{\rm inner}$")
    cb = fig.colorbar(m, ax=axs, shrink=0.85, pad=0.015, aspect=25, ticks=[2, 3, 4, 5])
    cb.set_label(r"local significance / $\sigma$")
    cb.add_lines(levels=[2, 3, 4, 5], colors=["k"] * 4, linewidths=[0.6] * 4)
    # fix the layout before the logo is placed in the first panel
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # the logo on the uniform band in the upper left of the first panel
    logo_in_corner(fig, axs[0], "upper left", size=0.26, margin=0.15, uniform=8, path=LOGO_SMALL)
    fig.savefig(PLOTS / "hardening.pdf")
    print("wrote plots/hardening.pdf")


if __name__ == "__main__":
    main()
