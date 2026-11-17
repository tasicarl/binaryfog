"""The survey-average priors of table 1 (appendix B) and figure 9.

Mass function: holodeck's single Schechter form with the redshift slopes and the low-mass slope fixed
at the library values (PSI_Z, MPSI_Z, alpha_holo). For each data set one least-squares fit of
(psi0, m_psi0) to all its redshift bins with z <= 3 over 10.5 <= log10(m*/Msun) <= 12. Residuals are in
log10 Phi, each weighted by sqrt(w / max w) with w = Phi_data * m^(5/3 * 1.10), the weight of a galaxy
in h_c^2 (h_c^2 ~ M_BH^(5/3), M_BH ~ m^1.10); the maximum is taken per redshift bin. Centre = mean over
the six data sets, width = their standard deviation (ddof = 1).

Black hole relation: each relation evaluated at a Chabrier bulge mass of 1e11 Msun.

Run:  python fit_survey_average_priors.py
Writes results/survey_average_priors.json and plots/survey_average_prior_fits.pdf.
"""
import json

import numpy as np
from scipy.optimize import least_squares

from paths import PLOTS, RESULTS  # noqa: E402
from binaryfog.priors import GAUSSIAN_SETS  # noqa: E402
import gsmf_tables as T  # noqa: E402

LOGM = np.arange(10.5, 12.0001, 0.05)
ZGRID = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0])   # for continuous models
PSI_Z, MPSI_Z = -0.60, 0.11
GW_EXP = 5 / 3 * 1.10           # h_c^2 ~ Phi m^(5/3 alpha_mu)
PRIORS = {name: dict(psi0=GAUSSIAN_SETS[key]["gsmf_phi0_log10"], m0=GAUSSIAN_SETS[key]["gsmf_mchar0_log10"])
          for name, key in (("NANOGrav", "nanograv"), ("Tomczak-Kormendy", "tomczak_kormendy"),
                            ("survey average", "survey_average"))}


def alpha_holo(z):
    return -1.21 - 0.03 * z


def holo(psi0, m0, z, logm=LOGM):
    return T.dschech(logm, m0 + MPSI_Z * z, alpha_holo(z), psi0 + PSI_Z * z)


# ------------------------------------------------------------------ published curves
def binned(rows, kind, logm=LOGM):
    out = []
    for r in rows:
        z = 0.5 * (r[0] + r[1])
        if kind == "lin":             # Phi in 1e-3, linear (Davidzon, COSMOS2020)
            c = (T.dschech(logm, r[2], r[3], np.log10(r[4] * 1e-3)) if r[5] is None else
                 T.dschech(logm, r[2], r[3], np.log10(r[4] * 1e-3), r[5], np.log10(r[6] * 1e-3)))
        elif kind == "mcl":           # McLeod: logM*, logPhi1, a1, logPhi2, a2
            c = T.dschech(logm, r[2], r[4], r[3], r[6], r[5])
        else:                         # Tomczak: logM*, a1, logPhi1[, a2, logPhi2]
            c = T.dschech(logm, *r[2:])
        out.append((z, c))
    return out


def quad(zs, ys, z):
    return np.polyval(np.polyfit(zs, ys, 2), z)


def leja(z, logm=LOGM):
    d = T.LEJA
    return T.dschech(logm, quad(d["z"], d["logmstar"], z), d["alpha1"], quad(d["z"], d["logphi1"], z),
                     d["alpha2"], quad(d["z"], d["logphi2"], z))


def wright(z, logm=LOGM):
    ev = lambda k: np.polyval(T.WRIGHT[k][::-1], z)
    return T.dschech(logm, ev("M"), ev("a1"), ev("P1"), ev("a2"), ev("P2"))


def gama(logm=LOGM):
    g = T.GAMA
    return T.dschech(logm, g["logM"], g["a1"], g["logP1"], g["a2"], g["logP2"])


def curve_at(name, z, logm):
    """The published log10 Phi of one data set on logm, in the redshift bin containing z (GAMA DR4: z = 0)."""
    tables = {"Davidzon+17": (T.D17, "lin"), "COSMOS2020": (T.C20, "lin"), "McLeod+21": (T.MCL_INT, "mcl")}
    if name in tables:
        rows, kind = tables[name]
        return next(binned([r], kind, logm)[0][1] for r in rows if r[0] < z <= r[1])
    if name == "Leja+20":
        return leja(z, logm)
    if name == "Wright+18":
        return wright(z, logm)
    return gama(logm)


def datasets(zmax=3.0):
    S = {"Davidzon+17": binned(T.D17, "lin"),
         "COSMOS2020": binned(T.C20, "lin"),
         "McLeod+21": binned(T.MCL_INT, "mcl"),
         "Leja+20": [(z, leja(z)) for z in ZGRID],
         "Wright+18": [(z, wright(z)) for z in ZGRID],
         "GAMA DR4": [(0.0, gama())]}
    return {k: [(z, c) for z, c in v if z <= zmax + 1e-9] for k, v in S.items()}


def reference_sets(zmax=3.0):
    S = {"Tomczak+14 T2": binned(T.TOMCZAK_D, "tom"), "Tomczak+14 T3": binned(T.TOMCZAK_S, "tom")}
    return {k: [(z, c) for z, c in v if z <= zmax + 1e-9] for k, v in S.items()}


# ------------------------------------------------------------------ fit
def weight(c):
    w = 10 ** c * 10 ** (LOGM * GW_EXP)
    return np.sqrt(w / w.max())


def fit(curves):
    """Least-squares (psi0, m_psi0) over all bins of one data set."""
    ws = [weight(c) for _, c in curves]

    def res(p):
        return np.concatenate([w * (holo(p[0], p[1], z) - c) for (z, c), w in zip(curves, ws)])
    return least_squares(res, [-2.6, 10.9], xtol=1e-12, ftol=1e-12, gtol=1e-12).x


def trapz(y, x):
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def log_moment(psi0, m0, z=0.5):
    """log10 n_GW, the integral of Phi (m/1e11 Msun)^1.83 dlog m over 10.5..12, of appendix B."""
    lm = np.arange(10.5, 12.00001, 0.005)
    c = holo(psi0, m0, z, logm=lm)
    return float(np.log10(trapz(10 ** c * 10 ** ((lm - 11) * GW_EXP), lm)))


def weight_peak(psi0, m0, z=0.5):
    """Where Phi m^1.83 peaks, and the fraction of its integral that comes from 10.5..12."""
    lm = np.arange(8.0, 13.00001, 0.001)
    w = 10 ** holo(psi0, m0, z, logm=lm) * 10 ** ((lm - 11) * GW_EXP)
    inside = (lm >= 10.5) & (lm <= 12.0)
    return float(lm[np.argmax(w)]), float(trapz(w[inside], lm[inside]) / trapz(w, lm))


def ensemble(fits):
    P = np.array([v[0] for v in fits.values()])
    M = np.array([v[1] for v in fits.values()])
    return dict(psi0=float(P.mean()), dpsi0=float(P.std(ddof=1)), m0=float(M.mean()), dm0=float(M.std(ddof=1)),
                corr=float(np.corrcoef(P, M)[0, 1]))


def main():
    R = {}
    S = datasets()
    base = {k: fit(v) for k, v in S.items()}
    R["per_survey"] = {k: dict(psi0=float(v[0]), m0=float(v[1]), logI05=log_moment(*v)) for k, v in base.items()}
    R["ensemble"] = ensemble(base)
    R["ensemble"]["logI05_mean"] = float(np.mean([log_moment(*v) for v in base.values()]))
    R["ensemble"]["logI05_std"] = float(np.std([log_moment(*v) for v in base.values()], ddof=1))
    R["reference"] = {}
    for k, v in reference_sets().items():
        f = fit(v)
        R["reference"][k] = dict(psi0=float(f[0]), m0=float(f[1]), logI05=log_moment(*f))
    R["logI05_prior_centres"] = {k: log_moment(p["psi0"][0], p["m0"][0]) for k, p in PRIORS.items()}
    R["weight_peak_prior_centres"] = {k: dict(zip(("log10_mstar", "fraction_10.5_12"),
                                                  weight_peak(p["psi0"][0], p["m0"][0])))
                                      for k, p in PRIORS.items()}
    # black hole relation
    bh = {}
    for k, d in T.BH_RELATIONS.items():
        native = d["a"] + d["b"] * (11.0 - d["pivot"])
        chab = d["a"] + d["b"] * (11.0 - d["pivot"] - d["to_chabrier"])
        bh[k] = dict(mu_native=float(native), mu_chabrier=float(chab), slope=d["b"], eps=d["eps"])
    R["bh"] = bh
    mus = [bh[k]["mu_chabrier"] for k in ("Kormendy & Ho 2013", "Sahu et al. 2019")]
    R["mu_survey_average"] = dict(mean_KH_Sahu=float(np.mean(mus)),
                          std_all_four=float(np.std([v["mu_chabrier"] for v in bh.values()], ddof=1)),
                          shift_KH=float(bh["Kormendy & Ho 2013"]["mu_chabrier"] - bh["Kormendy & Ho 2013"]["mu_native"]))
    json.dump(R, open(RESULTS / "survey_average_priors.json", "w"), indent=1)

    e = R["ensemble"]
    print(f"psi0 = {e['psi0']:.3f} +- {e['dpsi0']:.3f}   m_psi0 = {e['m0']:.3f} +- {e['dm0']:.3f}   corr {e['corr']:+.2f}")
    for k, v in R["per_survey"].items():
        print(f"  {k:12s} psi0 {v['psi0']:.3f}  m0 {v['m0']:.3f}  logI(0.5) {v['logI05']:.3f}")
    for k, v in R["reference"].items():
        print(f"  ref {k:14s} psi0 {v['psi0']:.3f}  m0 {v['m0']:.3f}  logI(0.5) {v['logI05']:.3f}")
    print("log I(z=0.5):", {k: round(v, 3) for k, v in R["logI05_prior_centres"].items()},
          f"ensemble {e['logI05_mean']:.3f} +- {e['logI05_std']:.3f}")
    for k, v in R["weight_peak_prior_centres"].items():
        print(f"  {k:16s} Phi m^1.83 at z = 0.5 peaks at log10(m*/Msun) = {v['log10_mstar']:.2f}, "
              f"{100 * v['fraction_10.5_12']:.0f}% of its integral from 10.5 to 12")
    print("black hole relations, mu at a bulge mass of 1e11 Msun:")
    for k, v in bh.items():
        print(f"  {k:22s} native {v['mu_native']:.3f}  Chabrier {v['mu_chabrier']:.3f}  slope {v['slope']}  eps {v['eps']}")
    print("mu:", R["mu_survey_average"])
    kh = T.BH_RELATIONS["Kormendy & Ho 2013"]
    print(f"Kormendy & Ho bulge masses: factor {10 ** -kh['to_chabrier']:.2f} above the stellar-population scale, "
          f"black hole masses {R['mu_survey_average']['shift_KH']:.3f} dex, "
          f"h2Omega factor {10 ** (5 / 3 * R['mu_survey_average']['shift_KH']):.2f}")
    paper_params(R, base)


# ------------------------------------------------------------------ figures
def paper_params(R, base):
    """The appendix figure on the survey-average priors. Left: what is fitted, the mass
    functions times the h_c^2 weight m^1.83 in the redshift bin containing z = 0.5, against holodeck's
    single Schechter form at the three prior centres. Right: the fitted (psi0, m_psi0) of each survey as a
    coloured symbol, the prior sets as ellipses in the colours of figure 2, and lines of constant
    h_c^2-weighted number density n_GW. Tomczak et al., not Eddington-corrected and not in the average, are
    fitted the same way to separate the effect of the fit from that of the newer data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    import paper_style as ps
    from desy_colors import (DESYorange, DESYdunkelblau, DESYaubergine, DESYcyan, DESYgruen, DESYrot,
                             DESYgelb, DESYhellbraun, DESYgrau)
    ps.use_paper_style()
    cols = {"Davidzon+17": DESYcyan, "COSMOS2020": DESYgruen, "McLeod+21": DESYrot, "Leja+20": DESYgelb,
            "Wright+18": DESYhellbraun, "GAMA DR4": DESYgrau}
    marks = {"Davidzon+17": "o", "COSMOS2020": "s", "McLeod+21": "^", "Leja+20": "D", "Wright+18": "v",
             "GAMA DR4": "P"}
    names = {"Davidzon+17": "Davidzon et al. (COSMOS2015)", "COSMOS2020": "Weaver et al. (COSMOS2020)",
             "McLeod+21": "McLeod et al.", "Leja+20": "Leja et al.", "Wright+18": "Wright et al.",
             "GAMA DR4": "Driver et al. (GAMA DR4, $z=0$)"}
    pstyle = {"NANOGrav": (DESYorange, "NANOGrav priors"),
              "Tomczak-Kormendy": (DESYdunkelblau, "Tomczak\u2013Kormendy priors"),
              "survey average": (DESYaubergine, "survey-average priors")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(ps.WIDTH, 0.62 * ps.WIDTH),
                                 gridspec_kw=dict(wspace=0.32, left=0.10, right=0.985, bottom=0.36, top=0.97))
    # (a) the fitted quantity in the redshift bin containing z = 0.5
    lm = np.arange(10.3, 12.2001, 0.01)
    wt = 10 ** ((lm - 11) * GW_EXP)
    for k in cols:
        a1.plot(lm, 10 ** curve_at(k, 0.5, lm) * wt * 1e3, color=cols[k], lw=1.3)
    for p_, d in PRIORS.items():
        col, _ = pstyle[p_]
        a1.plot(lm, 10 ** holo(d["psi0"][0], d["m0"][0], 0.5, logm=lm) * wt * 1e3, color=col, lw=2.0, ls="--")
    a1.axvspan(10.3, 10.5, color="0.92", lw=0)
    a1.axvspan(12.0, 12.2, color="0.92", lw=0)
    a1.set_xlim(10.3, 12.2)
    a1.set_ylim(0, 3.4)
    a1.set_xticks([10.5, 11.0, 11.5, 12.0])
    a1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    a1.set_xlabel(r"$\log_{10}(m_\star/M_\odot)$")
    a1.set_ylabel(r"$\Psi\,(m_\star/10^{11}M_\odot)^{1.83}\;/\;10^{-3}\,\mathrm{Mpc^{-3}\,dex^{-1}}$")
    # (b) the fitted parameters
    P, M = np.meshgrid(np.linspace(-3.6, -1.2, 120), np.linspace(9.8, 11.9, 120))
    I = np.vectorize(log_moment)(P, M)
    cs = a2.contour(P, M, I, levels=np.arange(-4.0, -1.9, 0.2), colors="0.82", linewidths=0.6, linestyles="solid", zorder=0)
    for p_, d in PRIORS.items():
        col, _ = pstyle[p_]
        for n, lw in ((1, 1.6), (2, 0.9)):
            a2.add_patch(Ellipse((d["psi0"][0], d["m0"][0]), 2 * n * d["psi0"][1], 2 * n * d["m0"][1],
                                 fill=False, ec=col, lw=lw))
    for k, v in base.items():
        # Tomczak et al. are left out of the figure and quoted in the text only
        a2.plot(v[0], v[1], marks[k], color=cols[k], mec=cols[k], ms=6, ls="none")
    # zoomed on the surveys: the 2 sigma ellipse of the NANOGrav priors leaves the panel (caption)
    a2.set_xlim(-3.1, -1.75)
    a2.set_ylim(10.35, 11.45)
    # the value of every line, written on it where it is farthest from the ellipses, the symbols and the
    # panel edges, and one of them with the symbol
    X0, X1, Y0, Y1 = -3.1, -1.75, 10.35, 11.45
    sx, sy = 1.0 / (X1 - X0), 0.62 / (Y1 - Y0)          # axes units, the panel is about 0.62 as high as wide
    t = np.linspace(0, 2 * np.pi, 400)
    # the symbols as discs of their printed size (6 pt, about 0.025 of the panel width)
    obst = [(v[0] + 0.025 / sx * np.cos(a_), v[1] + 0.025 / sy * np.sin(a_)) for v in base.values()
            for a_ in np.linspace(0, 2 * np.pi, 16)] + [(v[0], v[1]) for v in base.values()]
    for d in PRIORS.values():
        for n in (1, 2):
            obst += list(zip(d["psi0"][0] + n * d["psi0"][1] * np.cos(t), d["m0"][0] + n * d["m0"][1] * np.sin(t)))
    obst = np.array(obst)
    # every second line (0.4 dex); the first one with room carries the symbol. A label needs a rectangle
    # along its line free of ellipses, symbols, the inward ticks and the other labels.
    SHOWN, NAMED_ORDER = np.round(np.arange(-3.8, -2.1, 0.4), 1), (-2.2, -3.8, -2.6, -3.4)
    O = np.c_[(obst[:, 0] - X0) * sx, (obst[:, 1] - Y0) * sy]
    lines = {}
    for lev, segs in zip(cs.levels, cs.allsegs):
        lev = round(float(lev), 1)
        if lev in SHOWN and segs:
            pts = np.vstack(segs)
            pts = pts[(pts[:, 0] > X0) & (pts[:, 0] < X1) & (pts[:, 1] > Y0) & (pts[:, 1] < Y1)]
            if len(pts) >= 3:
                lines[lev] = pts
    placed = []

    def find(pts, half_len, half_h=0.014, tick=0.035):
        P = np.c_[(pts[:, 0] - X0) * sx, (pts[:, 1] - Y0) * sy]
        tang = np.gradient(P, axis=0); tang /= np.hypot(*tang.T)[:, None]
        Q = np.vstack([O] + ([np.array(placed)] if placed else []))
        best, kbest = -1.0, None
        for k in range(len(P)):
            c, u = P[k], tang[k]
            n = np.array([-u[1], u[0]])
            ends = np.array([c + half_len * u, c - half_len * u])
            if (ends[:, 0].min() < tick or ends[:, 0].max() > 1 - tick or ends[:, 1].min() < tick + half_h
                    or ends[:, 1].max() > 0.62 - tick - half_h):
                continue
            along, perp = (Q - c) @ u, (Q - c) @ n
            if ((np.abs(along) < half_len + 0.014) & (np.abs(perp) < half_h + 0.014)).any():
                continue
            clear = np.min(np.hypot(np.maximum(np.abs(along) - half_len, 0), np.maximum(np.abs(perp) - half_h, 0)))
            if clear > best:
                best, kbest = clear, k
        return kbest, P, tang

    pos, fmt, named = [], {}, None
    for lev in NAMED_ORDER:
        if lev in lines:
            k, P, tang = find(lines[lev], 0.13)
            if k is not None:
                named = lev
                break
    for lev in [named] + [l for l in lines if l != named] if named is not None else list(lines):
        k, P, tang = find(lines[lev], 0.13 if lev == named else 0.042)
        if k is None:
            print(f"  no free place for the label of {lev}")
            continue
        pos.append(tuple(lines[lev][k]))
        placed += [P[k] + s_ * (0.13 if lev == named else 0.042) * tang[k] for s_ in np.linspace(-1, 1, 7)]
        fmt[lev] = (rf"$n_\mathrm{{GW}} = 10^{{{lev:.1f}}}\,\mathrm{{Mpc^{{-3}}}}$" if lev == named
                    else rf"$10^{{{lev:.1f}}}$")
    fmt = {lv: fmt.get(round(float(lv), 1), "") for lv in cs.levels}
    labs = a2.clabel(cs, fmt=fmt, manual=pos, fontsize=5.5, inline=True, inline_spacing=2, colors="0.40")
    # the neighbouring lines are closer than a label is tall: a white box keeps them from under the text
    for lb in labs:
        lb.set_bbox(dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    a2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    a2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    a2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    a2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    a2.set_xlabel(r"$\psi_0 = \log_{10}(\Psi_0/\mathrm{Mpc^{-3}})$")
    a2.set_ylabel(r"$m_{\psi,0} = \log_{10}(M_\psi/M_\odot)$")
    # one legend for both panels, below them
    h = [Line2D([], [], color=cols[k], lw=1.3, marker=marks[k], mec=cols[k], ms=6) for k in cols]
    l = [names[k] for k in cols]
    h += [Line2D([], [], color=pstyle[p_][0], lw=2.0, ls="--") for p_ in PRIORS]
    l += [pstyle[p_][1] for p_ in PRIORS]
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.54, 0.0),
               handlelength=2.6, columnspacing=1.6)
    out = PLOTS / "survey_average_prior_fits.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
