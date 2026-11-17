"""
Held-out error of the median map against the number of library models it is fitted to.

The median map is the bend template plus a polynomial correction in the six parameters
(binaryfog.background). The correction is refitted on random subsets of the library, at degrees
2, 3 and 4, and its error is measured on the 150 held-out models of inputs/library/qlib.jsonl,
which no fit sees.
"""
import json

import numpy as np

from paths import INPUTS
from holodeck_runs import QUANT
from binaryfog.background import (PARAM_NAMES, KAPPA, FREF_HZ, FYR_HZ, DATA, design, saturate,
                                  bend_log10hc2)

C0_DATA = 2 * np.log10(7.5e-15)       # measured log10 h_c^2 in the lowest bin; models 3 dex below are left out
IMED = int(np.argmin(abs(np.array(QUANT) - 0.5)))
_MAPS = np.load(DATA / "maps_ng15.npz")
W_BEND, FREQS = _MAPS["W_bend"], _MAPS["freqs"]


def read_jsonl(name):
    for line in (INPUTS / "library" / name).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                yield r


def load_library():
    """Parameters, continuum spectra and realisation medians of the 2000 library models."""
    cont, real = {}, {}
    for cf, rf in (("cont_n400_a.jsonl", "lib6d_n400_r2000_f14.jsonl"),
                   ("cont_n1600_b.jsonl", "lib6d_n1600_r2000_f14_b.jsonl")):
        for r in read_jsonl(cf):
            if np.all(np.isfinite(r["continuum_log10hc2"])):
                cont[(cf, r["i"])] = r
        for r in read_jsonl(rf):
            real[(cf, r["i"])] = r
    keys = sorted(set(cont) & set(real))
    P = np.array([[cont[k]["params"][n] for n in PARAM_NAMES] for k in keys])
    Y = np.array([cont[k]["continuum_log10hc2"] for k in keys])
    MED = np.array([real[k]["median"] for k in keys])
    return P, Y, MED


def load_heldout():
    """Parameters and median log10 h_c^2 per bin of the 150 held-out models."""
    P, M = [], []
    for r in read_jsonl("qlib.jsonl"):
        q = np.array(r["quantiles"])
        if np.all(np.isfinite(q)):
            P.append([r["params"][n] for n in PARAM_NAMES])
            M.append(q[IMED])
    return np.array(P), np.array(M)


def bend(theta):
    """The bend template of the median map, without the polynomial correction."""
    o = design(np.atleast_2d(theta), 4) @ W_BEND
    t = saturate((o[:, 1] + 4 / 3) / KAPPA)
    lgfb = np.log10(FREF_HZ) + np.log10(t / (1 - t)) / KAPPA
    a = (0.5 * o[:, 0] + (2 / 3) * np.log10(FREF_HZ / FYR_HZ)
         + 0.5 * np.log10(1 + 10.0 ** (KAPPA * (lgfb - np.log10(FREF_HZ)))))
    return bend_log10hc2(FREQS, a[0], lgfb[0])


if __name__ == "__main__":
    P, Y, MED = load_library()
    ok = np.isfinite(MED).all(axis=1) & np.isfinite(Y).all(axis=1) & (Y[:, 0] > C0_DATA - 3)
    Pg = P[ok]
    D = MED[ok] - np.array([bend(p) for p in Pg])
    Pq, Mq = load_heldout()
    baseq = np.array([bend(p) for p in Pq])

    rng = np.random.default_rng(0)
    print(f"training pool {len(Pg)} models; held-out set {len(Pq)} models, {Mq.shape[1]} bins each\n")
    print(f"  {'train':>6} {'deg 2':>9} {'deg 3':>9} {'deg 4':>9}   (held-out scatter, dex)")
    for n in (100, 200, 400, 800, 1200, 1600, len(Pg)):
        errs = []
        for deg in (2, 3, 4):
            r = []
            for rep in range(5 if n < len(Pg) else 1):
                idx = rng.permutation(len(Pg))[:n]
                A = design(Pg[idx], deg)
                if A.shape[0] <= A.shape[1]:
                    r.append(np.nan)
                    continue
                W, *_ = np.linalg.lstsq(A, D[idx], rcond=None)
                r.append(np.nanstd(design(Pq, deg) @ W + baseq - Mq))
            errs.append(np.nanmean(r))
        print(f"  {n:6d} " + " ".join(f"{e:9.4f}" for e in errs))
