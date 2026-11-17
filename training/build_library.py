"""
Build the holodeck library that the binaryfog maps are fitted to.

Each parameter point of holodeck's classic phenomenological model (parameter space PS_Classic_Phenom_Uniform: galaxy
stellar mass function normalisation psi_0 and characteristic mass m_psi,0, black hole to bulge mass relation mu and
its scatter epsilon_mu, binary lifetime tau_f and inner hardening index nu_inner; the outer index and the break
separation at holodeck's defaults) is run once, on one or more frequency grids. Per point and grid the output holds
the quantiles, median and standard deviation of log10 h_c^2 over the Poisson realisations of the binary population,
the continuum (the expectation value, no Poisson draws) and the effective number of sources per bin. fit_maps.py
reads exactly these.

Runs on public holodeck without modifications (tested with v1.7.1, commit 7a5cb8ae). Binaries that have not merged by
today (holodeck returns a negative redshift for them) are removed here, by zeroing every cell of the (total mass,
mass ratio, redshift) grid with such a corner, which is what holodeck's own library code does on its main branch
since PR #133 (not yet in a release).

A frequency grid is a measured data product, whose stored frequencies are the bin centres (the edges are the
centres plus and minus half the spacing, which sets how many binaries fall in a bin and so how much the
realisations scatter):
    --grid ng15=ng15_hd_v2:14 --grid ipta2=ipta2_cp_v1:30
or a file of bin centres in Hz:
    --grid mygrid=/path/to/freqs.npy:20

    python build_library.py --grid ng15=ng15_hd_v2:14 --npoints 2000 --nreals 2000 --jobs 56 --out lib.jsonl
    python build_library.py --grid ipta2=ipta2_cp_v1:30 --points lib.jsonl --out lib_ipta2.jsonl   # same points

One point takes about 7 min on one core at 2000 realisations; 2000 points therefore about 4 h on 56 cores.
Resumable: points already in --out are skipped.
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

import numpy as np
from scipy.stats import qmc

PARAM_NAMES = ["gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10",
               "mmb_scatter_dex", "hard_time", "hard_gamma_inner"]
QUANT = [0.005, 0.02, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.98, 0.995]
SPACE = "PS_Classic_Phenom_Uniform"
_space = None


def get_space():
    global _space
    if _space is None:
        import holodeck.librarian  # noqa: F401
        import holodeck as holo
        _space = holo.librarian.param_spaces_dict[SPACE]()
    return _space


def prior_box():
    """(low, high) of each parameter, read from holodeck's parameter space."""
    space = get_space()
    box = {}
    for dist in space._parameters:
        if dist.name in PARAM_NAMES:
            box[dist.name] = (float(dist._lo), float(dist._hi))
    missing = set(PARAM_NAMES) - set(box)
    if missing:
        raise RuntimeError(f"{SPACE} has no uniform parameter(s) {sorted(missing)}")
    return box


def grid_freqs(spec, root=None):
    """Bin centres and edges in Hz of one grid specification 'source:nbins'."""
    source, nbins = spec.rsplit(":", 1)
    nbins = int(nbins)
    if source.endswith(".npy"):
        f = np.load(source)[:nbins]
    else:
        from binaryfog import FreeSpectrum
        f = FreeSpectrum.load(source, nbins=nbins, root=root).freqs
    f = np.asarray(f, float)
    if len(f) < nbins:
        raise ValueError(f"{source} has {len(f)} frequencies, {nbins} requested")
    step = float(np.median(np.diff(f))) if len(f) > 1 else float(f[0])
    edges = np.concatenate([f - 0.5 * step, [f[-1] + 0.5 * step]])
    return f, edges


def summary(x):
    """x: log10 h_c^2 over realisations, shape (bins, realisations)."""
    return {"quantiles": np.quantile(x, QUANT, axis=1).tolist(),
            "median": np.median(x, axis=1).tolist(), "std": x.std(axis=1).tolist()}


def merged_cells(redz):
    """Cells of the (mass, ratio, redshift) grid all of whose corners hold binaries that have merged by today.

    redz has shape (M, Q, Z, F) on the grid edges; the result has shape (M-1, Q-1, Z-1, F)."""
    ok = redz >= 0.0
    for ax in range(3):
        a = [slice(None)] * 4
        b = [slice(None)] * 4
        a[ax], b[ax] = slice(1, None), slice(None, -1)
        ok = ok[tuple(a)] & ok[tuple(b)]
    return ok


def spectra(sam, hard, grids, nreals, keep_realisations=False):
    """Realisations, continuum and source count of one model on each grid.

    grids maps a name to (bin centres, bin edges) of observed gravitational-wave frequencies in Hz. Per grid the
    result holds the summary of log10 h_c^2 over the realisations and, with keep_realisations, all of them as
    "log10hc2", an array (bins, realisations)."""
    import holodeck as holo
    import holodeck.cyutils  # noqa: F401
    from holodeck import cosmo
    from holodeck.sams import sam_cyutils

    names = list(grids)
    raw = np.concatenate([grids[g][0] for g in names]) / 2.0          # orbital frequencies
    order = np.argsort(raw)                                            # holodeck needs them sorted
    pos = np.empty(len(raw), int)
    pos[order] = np.arange(len(raw))
    redz, dnum = sam_cyutils.dynamic_binary_number_at_fobs(raw[order], sam, hard, cosmo)
    out, k = {}, 0
    for g in names:
        cents, edges_f = grids[g]
        sl = pos[k:k + len(cents)]
        k += len(cents)
        edges = [sam.mtot, sam.mrat, sam.redz, edges_f / 2.0]
        rz = np.ascontiguousarray(redz[..., sl])
        number = sam_cyutils.integrate_differential_number_3dx1d(edges, np.ascontiguousarray(dnum[..., sl]))
        ok = merged_cells(rz)
        hc2 = holo.gravwaves.char_strain_sq_from_bin_edges_redz(edges, np.where(rz >= 0.0, rz, 1.0))
        hc2 = np.ascontiguousarray(np.where(ok, hc2, 0.0))
        if not np.all(np.isfinite(hc2)):
            raise FloatingPointError("non-finite strain in a cell of merged binaries")
        number = np.ascontiguousarray(np.where(ok, number, 0.0))
        gwb = np.sqrt(holo.cyutils.sam_poisson_gwb(number, hc2, nreals))   # (bins, realisations)
        s1 = (number * hc2).sum(axis=(0, 1, 2))
        s2 = (number * hc2 ** 2).sum(axis=(0, 1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            n_src = np.where(s2 > 0, s1 ** 2 / s2, 0.0)
            lg = 2 * np.log10(np.asarray(gwb))
            rec = summary(lg)
            rec["continuum_log10hc2"] = np.log10(s1).tolist()
        rec["n_src"] = [float(v) for v in n_src]
        rec["freqs"] = cents.tolist()
        if keep_realisations:
            rec["log10hc2"] = lg
        out[g] = rec
    return out


def realisations(theta, nreals, cents=None, edges=None):
    """log10 h_c^2 of nreals realisations of the model theta (the six parameters in the order of PARAM_NAMES),
    shape (bins, realisations). Without a grid: holodeck's default, the 14 harmonics of its 16.03 yr span."""
    if cents is None:
        import holodeck as holo
        from holodeck.constants import YR
        from holodeck.librarian import DEF_PTA_DUR
        cents, edges = holo.utils.pta_freqs(dur=DEF_PTA_DUR * YR, num=14)
    sam, hard = get_space().model_for_params(dict(zip(PARAM_NAMES, map(float, theta))))
    grids = {"g": (np.asarray(cents, float), np.asarray(edges, float))}
    return spectra(sam, hard, grids, nreals, keep_realisations=True)["g"]["log10hc2"]


def one_point(job):
    idx, values, grids, nreals = job
    params = dict(zip(PARAM_NAMES, [float(v) for v in values]))
    t0 = time.perf_counter()
    try:
        sam, hard = get_space().model_for_params(params)
        rec = {"i": idx, "params": params, **spectra(sam, hard, grids, nreals)}
    except Exception as exc:                                           # noqa: BLE001
        rec = {"i": idx, "params": params, "error": f"{type(exc).__name__}: {exc}"}
    rec["seconds"] = time.perf_counter() - t0
    return rec


def check_version():
    """holodeck 1.6 and earlier (the newest on PyPI as of November 2026) lack the redshift evolution of the
    characteristic mass of the stellar mass function (holodeck PR #117, the erratum to the NANOGrav 15 yr
    astrophysics paper); libraries built with them are wrong."""
    import holodeck as holo
    v = tuple(int(x) for x in str(holo.__version__).split(".")[:2] if x.isdigit())
    if v < (1, 7):
        raise RuntimeError(f"holodeck {holo.__version__} at {holo.__file__}: releases before 1.7 lack the fix of "
                           "PR #117; install v1.7.1 or later from https://github.com/nanograv/holodeck (README.md here)")


def provenance():
    import holodeck as holo
    root = Path(holo.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"holodeck_version": getattr(holo, "__version__", None), "holodeck_commit": commit,
            "holodeck_modified": dirty, "parameter_space": SPACE, "python": platform.python_version(),
            "numpy": np.__version__}


def read_points(path):
    """(index, parameters) of every point of an existing library, with the index it carries there."""
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "params" in r and "error" not in r:
                pts.append((int(r["i"]), [float(r["params"][k]) for k in PARAM_NAMES]))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="append", required=True, help="name=source:nbins, see above")
    ap.add_argument("--npoints", type=int, default=2000, help="Latin hypercube points over the prior box")
    ap.add_argument("--seed", type=int, default=20260817, help="seed of the Latin hypercube")
    ap.add_argument("--points", help="take the parameter points (and their indices) from this library instead")
    ap.add_argument("--nreals", type=int, default=2000, help="Poisson realisations per point")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="only the first n points (timing, tests)")
    ap.add_argument("--freespec-root", help="cache directory of the data products (default ~/.binaryfog/freespec)")
    ap.add_argument("--out", default="library.jsonl")
    args = ap.parse_args()

    check_version()
    grids = {}
    for spec in args.grid:
        name, src = spec.split("=", 1)
        grids[name] = grid_freqs(src, args.freespec_root)
    if args.points:
        pts = read_points(args.points)
    else:
        box = prior_box()
        lo = np.array([box[k][0] for k in PARAM_NAMES])
        hi = np.array([box[k][1] for k in PARAM_NAMES])
        sample = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=args.seed).random(args.npoints)
        pts = list(enumerate((lo + sample * (hi - lo)).tolist()))
    if args.limit:
        pts = pts[:args.limit]

    out = Path(args.out)
    done, have_head = set(), False
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                have_head = have_head or "header" in r
                if "i" in r and "error" not in r:
                    done.add(r["i"])
        print(f"resuming: {len(done)} points already in {out}", flush=True)
    head = {"header": {**provenance(), "grids": {g: args.grid[n] for n, g in enumerate(grids)},
                       "nreals": args.nreals, "quantiles": QUANT, "param_names": PARAM_NAMES,
                       "points": args.points or f"latin hypercube, seed {args.seed}, {args.npoints} points"}}
    if head["header"]["holodeck_modified"]:
        print("warning: the holodeck checkout has local modifications", flush=True)
    jobs = [(i, v, grids, args.nreals) for i, v in pts if i not in done]
    print(f"{len(jobs)} of {len(pts)} points, grids {', '.join(args.grid)}, {args.nreals} realisations, "
          f"{args.jobs} workers, holodeck {head['header']['holodeck_version']} -> {out}", flush=True)
    t0 = time.perf_counter()
    with open(out, "a", buffering=1) as fh, mp.Pool(args.jobs) as pool:
        if not have_head:
            fh.write(json.dumps(head) + "\n")
        for n, r in enumerate(pool.imap_unordered(one_point, jobs), 1):
            fh.write(json.dumps(r) + "\n")
            if n % 20 == 0 or n == len(jobs) or "error" in r:
                el = time.perf_counter() - t0
                print(f"  {n}/{len(jobs)} | {el / 60:.1f} min | eta {el / n * (len(jobs) - n) / 60:.1f} min | last "
                      f"{r['seconds']:.0f} s{' ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
