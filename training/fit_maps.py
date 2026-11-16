"""
Fit the four maps of binaryfog.Background to a library from build_library.py, on one of its frequency grids.

    median   log10 h_c^2 of the median realisation, as a bend template plus a correction per bin:
             bend        log10 h_c^2 and its slope at 5 nHz of a bend with kappa = 1.8, fitted to each point's
                         continuum, as a polynomial of degree 4 in the six parameters; the slope is taken into the
                         range a bend allows by saturate()
             correction  the median minus the bend, per bin, degree 4
    width    log10 of the standard deviation of log10 h_c^2 over the realisations, per bin, degree 4; the conversion
             to the 16 to 84 per cent half-width is the median ratio of the two over the library
    skew     per point the c of the warp  median + sigma (z + c z^2)  fitted to the quantile ladder, then a
             polynomial of the degree (1 to 3) with the smallest held-out scatter
The bend, the correction and the width use only points whose continuum at about 2 nHz lies within 3 (in
log10 h_c^2) of the measured signal level: far quieter populations cannot matter in a fit to the data and would
otherwise dominate the polynomials. Each fit keeps a quarter of the points (the skew 30 per cent) aside and
reports its accuracy on them.

    python fit_maps.py library.jsonl --grid ng15 --out maps_ng15.npz
    python fit_maps.py library.jsonl --grid ng15 --out maps_ng15.npz --compare ng15    # against shipped maps

The output loads as  binaryfog.Background(maps="maps_ng15.npz").
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from binaryfog.background import (FREF_HZ, FYR_HZ, KAPPA, PARAM_NAMES, Background, bend_log10hc2, design,
                                  saturate)

QUANT = [0.005, 0.02, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.98, 0.995]
C0_DATA = 2 * np.log10(7.5e-15)          # measured signal level, log10 h_c^2 at about 2 nHz
F_SELECT = 1.9768e-9                      # frequency at which that level is compared, Hz (lowest NANOGrav bin)
Q = np.array(QUANT)
Z = norm.ppf(Q)
I16, I50, I84 = [QUANT.index(q) for q in (0.16, 0.5, 0.84)]


def load(path, grid):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    head = next((r["header"] for r in rows if "header" in r), {})
    if head.get("quantiles", QUANT) != QUANT:
        raise ValueError("the library stores other quantiles than this script expects")
    rows = [r for r in rows if "params" in r and "error" not in r and grid in r
            and np.all(np.isfinite(r[grid]["continuum_log10hc2"]))]
    if head:
        # build_library.py writes the points in the order its workers finish; the index of a point in
        # the design fixes the training and held-out split, so the fit does not depend on the scheduling
        rows.sort(key=lambda r: r["i"])
    if not rows:
        raise ValueError(f"no usable points with grid {grid!r} in {path}")
    P = np.array([[r["params"][k] for k in PARAM_NAMES] for r in rows])
    g = lambda key: np.array([r[grid][key] for r in rows], float)
    QU = np.array([r[grid]["quantiles"] for r in rows], float)                 # (points, quantiles, bins)
    return head, P, g("continuum_log10hc2"), g("median"), g("std"), QU, np.array(rows[0][grid]["freqs"])


def fit_bend(f, y):
    """(log10 A, log10 f_bend) of the bend template closest to one continuum, and the rms of the residual."""
    r = least_squares(lambda p: bend_log10hc2(f, p[0], p[1], KAPPA) - y, [-14.5, np.log10(3e-9)],
                      bounds=([-24, -12], [-8, -6]), xtol=1e-12, ftol=1e-12)
    return r.x, float(np.sqrt((r.fun ** 2).mean()))


def bend_spec(W, P, freqs):
    """log10 h_c^2 of the bend the map W predicts, as Background.predict computes it."""
    o = design(P, 4) @ W
    t = saturate((o[:, 1] + 4 / 3) / KAPPA)
    lgfb = np.log10(FREF_HZ) + np.log10(t / (1 - t)) / KAPPA
    a = 0.5 * o[:, 0] + (2 / 3) * np.log10(FREF_HZ / FYR_HZ) + 0.5 * np.log10(1 + 10.0 ** (KAPPA * (lgfb - np.log10(FREF_HZ))))
    return np.array([bend_log10hc2(freqs, ai, bi, KAPPA) for ai, bi in zip(a, lgfb)])


def split(n, frac=0.25, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    nt = int(frac * n)
    return idx[nt:], idx[:nt]


def fit(path, grid):
    head, P, Y, MED, SIG, QU, freqs = load(path, grid)
    print(f"{path}, grid {grid}: {len(P)} points, {len(freqs)} bins, {freqs[0]*1e9:.3f} to {freqs[-1]*1e9:.3f} nHz")
    out = {"freqs": freqs, "param_names": np.array(PARAM_NAMES),
           "provenance": np.array(json.dumps({"library": str(path), **head}))}
    i5 = int(np.argmin(np.abs(np.log(freqs / F_SELECT))))
    loud = Y[:, i5] > C0_DATA - 3
    # bend: value and slope at FREF of the best bend through each continuum
    par, rms = np.array([fit_bend(freqs, y) for y in Y], dtype=object).T
    par, rms = np.array(list(par)), np.array(rms, float)
    good = loud & (rms < 0.05)
    u = KAPPA * (par[:, 1] - np.log10(FREF_HZ))
    t = 1 / (1 + 10.0 ** (-u))
    y5 = np.array([bend_log10hc2(np.array([FREF_HZ]), a, b, KAPPA)[0] for a, b in par])
    OBS = np.column_stack([y5, -4 / 3 + KAPPA * t])
    tr, te = split(good.sum())
    Pg, Og, Yg = P[good], OBS[good], Y[good]
    W, *_ = np.linalg.lstsq(design(Pg[tr], 4), Og[tr], rcond=None)
    e = bend_spec(W, Pg[te], freqs) - Yg[te]
    print(f"  bend: {good.sum()} points, held-out rms {np.sqrt((e**2).mean()):.4f} dex in log10 h_c^2 (continuum)")
    out["W_bend"] = W
    # median correction
    ok = np.isfinite(MED).all(axis=1) & loud
    D = MED[ok] - bend_spec(W, P[ok], freqs)
    tr, te = split(ok.sum())
    Wm, *_ = np.linalg.lstsq(design(P[ok][tr], 4), D[tr], rcond=None)
    e = design(P[ok][te], 4) @ Wm - D[te]
    print(f"  median: {ok.sum()} points, held-out residual {e.std():.4f} dex, bias {e.mean():+.4f} dex, "
          f"90th percentile |error| {np.percentile(np.abs(e), 90):.4f} dex")
    out["W_med"], out["deg_med"] = Wm, 4
    # width
    rel = np.isfinite(SIG).all(axis=1) & (SIG > 0).all(axis=1) & loud
    tr, te = split(rel.sum())
    Ww, *_ = np.linalg.lstsq(design(P[rel][tr], 4), np.log10(SIG[rel][tr]), rcond=None)
    e = design(P[rel][te], 4) @ Ww - np.log10(SIG[rel][te])
    print(f"  width: {rel.sum()} points, held-out {e.std():.4f} dex in log10 sigma ({100*(10**e.std()-1):.0f} per cent)")
    half = 0.5 * (QU[rel][:, I84, :] - QU[rel][:, I16, :])
    ratio = (half / SIG[rel]).ravel()
    wscale = float(np.median(ratio[np.isfinite(ratio)]))
    print(f"  half-width / standard deviation: median {wscale:.4f}")
    out["W_wid"], out["deg_wid"], out["width_scale"] = Ww, 4, wscale
    # skew: c per point from the quantile ladder, then a polynomial
    cp = np.full(len(P), np.nan)
    for k in range(len(P)):
        s = SIG[k]
        m = np.isfinite(s) & (s > 0)
        if not m.any():
            continue
        M = (QU[k][:, m] - QU[k][I50, m]).T                     # (bins, quantiles)
        S = s[m][:, None]
        cp[k] = ((M - S * Z) * (S * Z ** 2)).sum() / ((S * Z ** 2) ** 2).sum()
    gpts = np.where(np.isfinite(cp))[0]
    tr, te = split(len(gpts), 0.3, 1)
    tr, te = gpts[tr], gpts[te]
    best = None
    for deg in (1, 2, 3):
        Wc, *_ = np.linalg.lstsq(design(P[tr], deg), cp[tr], rcond=None)
        r = design(P[te], deg) @ Wc - cp[te]
        if best is None or r.std() < best[2]:
            best = (Wc, deg, r.std())
    print(f"  skew: per-point c median {np.nanmedian(cp):.3f}; map degree {best[1]}, held-out scatter {best[2]:.4f} "
          f"(spread over the library {np.nanstd(cp):.4f})")
    out["W_skew"], out["deg_skew"] = best[0], best[1]
    return out, P[loud]


def compare(new_path, ref_maps, P):
    """Median, width and skew of two sets of maps on the points P, on the reference's grid."""
    ref, new = Background(maps=ref_maps), Background(maps=str(new_path))
    if not np.allclose(ref.freqs, new.freqs, rtol=1e-4, atol=0):
        new = new.on_frequencies(ref.freqs)
    dm, rs, dc = [], [], []
    for th in P:
        m0, s0, c0 = ref.predict(th)
        m1, s1, c1 = new.predict(th)
        dm.append(np.log10(m1 / m0)); rs.append(s1 / s0); dc.append(c1 - c0)
    dm, rs, dc = np.array(dm), np.array(rs), np.array(dc)
    print(f"  against the maps {ref_maps!r}, on {len(P)} library points:")
    print(f"    median: rms {np.sqrt((dm**2).mean()):.4f} dex, 90th percentile |difference| "
          f"{np.percentile(np.abs(dm), 90):.4f} dex, largest {np.abs(dm).max():.4f} dex")
    print(f"    width: ratio median {np.median(rs):.4f}, 16 to 84 per cent {np.percentile(rs,16):.3f} to "
          f"{np.percentile(rs,84):.3f}")
    print(f"    skew: rms difference {np.sqrt((dc**2).mean()):.4f}")
    return dm, rs, dc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library")
    ap.add_argument("--grid", required=True, help="name of the grid in the library, as given to build_library.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="maps to compare with: a name shipped with binaryfog (ng15, ipta2) or a file")
    args = ap.parse_args()
    out, P_loud = fit(args.library, args.grid)
    np.savez(args.out, **out)
    print(f"  wrote {args.out}")
    if args.compare:
        compare(args.out, args.compare, P_loud)


if __name__ == "__main__":
    main()
