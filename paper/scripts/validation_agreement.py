"""
How closely our posterior contours agree with the published ones, as a number rather than by eye.

Figure 4 of the paper overlays our posterior for the six astrophysical parameters on the contours
of figure 9 of NANOGrav's 15-year new-physics paper (replacement figure of the erratum). This
measures the separation between the two: for every point of their contour, the distance to the
nearest point of ours, with each axis scaled to the prior range of its parameter so that the
answer is dimensionless and the two axes of a panel count equally. The median over the points of a
panel, and then over panels, is what the text quotes.

Segments of their contour that run along the edge of a panel are dropped: a contour that leaves the
panel was closed along the edge by the figure it was read from, and those segments are not part of
the posterior.

    python validation_agreement.py

    --method corner           draw our contour the way corner.py drew theirs: 34 bins over the
                              sample range, its threshold rule, no smoothing
    --self-split              the reference is not the published figure but our own chain: the
                              contour of walkers 0-31 against that of walkers 32-63, the
                              sampling-noise baseline
    --per-panel               the separation of every panel, and by parameter
    --cases                   which prior sets: flat, nanograv or both
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paths import INPUTS, SCANS
import binaryfog as bf                                                        # noqa: E402
from plot_nanograv_fit import credible, LEVELS, CASE                          # noqa: E402

LEVELS_SHOWN = (1, 2)              # the levels figure 4 draws
# corner.py with 34 bins over the range of each sample, as in the vector source of the published
# figure (the step histograms on the diagonal and the grid the contour vertices sit on)
NANOGRAV_BINS = 34
NWALKERS = 64                      # walkers of the chain written by validate_ng15_fig9.py


def corner_grid(x, y, level, bins=NANOGRAV_BINS):
    """The density grid and threshold corner.py 2.2 contours: its hist2d, step for step.

    The histogram runs over the sample range, the threshold is the smallest bin value whose
    cumulative mass, from the top, does not exceed the level, and the grid is padded by two bins on
    every side, which is where corner.py's contours run past the data.
    """
    H, X, Y = np.histogram2d(x, y, bins=[np.linspace(x.min(), x.max(), bins + 1),
                                         np.linspace(y.min(), y.max(), bins + 1)])
    Hflat = np.sort(H.ravel())[::-1]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    below = Hflat[sm <= level]
    thr = below[-1] if len(below) else Hflat[0]
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1], H2[2:-2, -2] = H[:, 0], H[:, -1]
    H2[1, 2:-2], H2[-2, 2:-2] = H[0], H[-1]
    H2[1, 1], H2[1, -2], H2[-2, 1], H2[-2, -2] = H[0, 0], H[0, -1], H[-1, 0], H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])
    return X2, Y2, H2.T, thr


def our_contour(xs, ys, rng_x, rng_y, level, method="credible"):
    """The contour polyline of our samples, without drawing it.

    method "credible" is the estimate figure 4 draws; "corner" is corner.py's, and its line is cut
    at the panel edge as the published lines are.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = LEVELS[level]
    if method == "credible":
        xc, yc, H, thr = credible(xs, ys, rng_x, rng_y, p)
    else:
        xc, yc, H, thr = corner_grid(xs, ys, p)
    fig = plt.figure()
    cs = fig.gca().contour(xc, yc, H, levels=[thr])
    pts = [np.asarray(q.vertices) for q in cs.get_paths() if len(q.vertices) > 2]
    plt.close(fig)
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    if method == "corner" and len(pts):
        inside = ((pts[:, 0] >= rng_x[0]) & (pts[:, 0] <= rng_x[1])
                  & (pts[:, 1] >= rng_y[0]) & (pts[:, 1] <= rng_y[1]))
        pts = pts[inside]
    return pts


def theirs(panel, rng_x, rng_y):
    """Their contour, with the segments that ride a panel edge removed."""
    x, y = np.asarray(panel["x"], float), np.asarray(panel["y"], float)
    tol_x = 1e-3 * (rng_x[1] - rng_x[0])
    tol_y = 1e-3 * (rng_y[1] - rng_y[0])
    on_edge = ((np.abs(x - rng_x[0]) < tol_x) | (np.abs(x - rng_x[1]) < tol_x)
               | (np.abs(y - rng_y[0]) < tol_y) | (np.abs(y - rng_y[1]) < tol_y))
    return np.column_stack([x[~on_edge], y[~on_edge]])


def separations(ref_panels, chain, P, box, refname, method="credible"):
    """Median separation per panel, over all its polylines: a list of (level, x, y, points, median)."""
    from scipy.spatial import cKDTree
    cache, pooled = {}, {}
    for panel in ref_panels:
        if panel["case"] != refname:
            continue
        kx, ky = panel["x_param"], panel["y_param"]
        if kx not in P or ky not in P:
            continue
        lvl = panel.get("sigma")
        if lvl not in LEVELS_SHOWN:
            continue
        rx, ry = box[kx], box[ky]
        them = theirs(panel, rx, ry)
        if len(them) < 3:
            continue
        key = (kx, ky, lvl)
        if key not in cache:
            cache[key] = our_contour(chain[:, P.index(kx)], chain[:, P.index(ky)], rx, ry, lvl,
                                     method)
        us = cache[key]
        if len(us) == 0:
            continue
        # both contours in units of the prior range, so the two axes count equally
        scale = np.array([rx[1] - rx[0], ry[1] - ry[0]])
        lo = np.array([rx[0], ry[0]])
        u, t = (us - lo) / scale, (them - lo) / scale
        d, _ = cKDTree(u).query(t)
        pooled.setdefault(key, []).append(d)
    out = []
    for (kx, ky, lvl), ds in pooled.items():
        d = np.concatenate(ds)
        out.append((lvl, kx, ky, len(d), float(np.median(d))))
    return out


def summary(recs):
    """Median over the records of each level, and how many records there were."""
    return {lvl: (float(np.median([r[4] for r in recs if r[0] == lvl]))
                  if any(r[0] == lvl for r in recs) else np.nan,
                  sum(1 for r in recs if r[0] == lvl)) for lvl in LEVELS_SHOWN}


def report(label, recs):
    s = summary(recs)
    parts = [f"{lvl} sigma {100 * s[lvl][0]:.1f}% over {s[lvl][1]} panels" if s[lvl][1]
             else f"{lvl} sigma no panels" for lvl in LEVELS_SHOWN]
    print(f"  {label:22s} median separation, in units of the prior range: " + ", ".join(parts))
    return s


def per_panel(label, recs, P):
    """Every panel's separation, and the median over the panels involving each parameter."""
    print(f"  {label}, per panel (median separation, % of the prior range; points used):")
    for lvl in LEVELS_SHOWN:
        rs = sorted([r for r in recs if r[0] == lvl], key=lambda r: -r[4])
        for r in rs:
            print(f"      {lvl} sigma  {r[1]:18s} x {r[2]:18s} {100 * r[4]:6.1f}%  ({r[3]} points)")
        for p in P:
            v = [r[4] for r in rs if p in (r[1], r[2])]
            w = [r[4] for r in rs if p not in (r[1], r[2])]
            if v:
                print(f"      {lvl} sigma  panels with {p:18s}: median {100 * np.median(v):5.1f}%"
                      f" over {len(v)};  without it {100 * np.median(w):5.1f}% over {len(w)}")


def self_reference(chain, P, box, nwalk):
    """Our chain split by walkers: contours of the first half as the reference, second half to
    compare against. Returns (reference panels in the published format, the comparison chain)."""
    walker = np.arange(len(chain)) % nwalk
    a, b = chain[walker < nwalk // 2], chain[walker >= nwalk // 2]
    panels = []
    for i, kx in enumerate(P):
        for ky in P[i + 1:]:
            for lvl in LEVELS_SHOWN:
                pts = our_contour(a[:, P.index(kx)], a[:, P.index(ky)], box[kx], box[ky], lvl)
                if len(pts):
                    panels.append({"case": "self", "x_param": kx, "y_param": ky,
                                   "x": pts[:, 0], "y": pts[:, 1], "sigma": lvl})
    return panels, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contours", default=str(INPUTS / "pta/ng15_fig9_erratum_contours.json"))
    ap.add_argument("--chains", default=str(SCANS / "ng15_fig9_binaryfog_erratum.npz"))
    ap.add_argument("--method", choices=("credible", "corner"), default="credible")
    ap.add_argument("--per-panel", action="store_true")
    ap.add_argument("--self-split", action="store_true")
    ap.add_argument("--cases", default="flat,nanograv")
    a = ap.parse_args()

    ours = np.load(a.chains, allow_pickle=True)
    P = list(ours["names"])
    box = {k: list(bf.background.PRIOR_BOX[k]) for k in P}
    if not a.self_split:
        pub = json.load(open(a.contours))
        print(f"  {Path(a.contours).name} against {Path(a.chains).name}; method {a.method}")
        for case in a.cases.split(","):
            recs = separations(pub["panels"], ours[case], P, box, CASE[case][0], a.method)
            label = "flat priors" if case == "flat" else "observational priors"
            report(label, recs)
            if a.per_panel:
                per_panel(label, recs, P)
    else:
        print(f"  self-split of {Path(a.chains).name}: walkers 0-{NWALKERS // 2 - 1} against "
              f"walkers {NWALKERS // 2}-{NWALKERS - 1}, method {a.method}")
        for case in a.cases.split(","):
            label = "flat priors" if case == "flat" else "observational priors"
            ref, chain = self_reference(ours[case], P, box, NWALKERS)
            recs = separations(ref, chain, P, box, "self", a.method)
            if a.per_panel:
                per_panel(label, recs, P)
            s = summary(recs)
            print(f"  {label:22s} median separation between our halves: "
                  + ", ".join(f"{lvl} sigma {100 * s[lvl][0]:.1f}%" for lvl in LEVELS_SHOWN))
