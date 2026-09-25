"""
Profile likelihood over a grid of the parameters you want to constrain.

At every grid point the likelihood is maximised over everything else: the nuisance parameters of
the new-physics model, and the six background parameters with their prior as a penalty. What comes
back is a table with the maximum and the parameters that achieved it, which is what the contours
and the limits are drawn from.

    scan = ProfileScan(like, spectrum, scan={"alpha": ..., "betaH": ...},
                       nuisance={"Treh": (1e-3, 1.0, "log")}, astro_prior="nanograv")
    table = scan.run(jobs=28)

Thresholds for the regions, from Wilks' theorem: 1.15 and 3.09 in ln L for 68 and 95 per cent with
two scanned parameters, 0.5 and 1.92 with one. At the edge of the parameter space, for instance
where the new-physics amplitude goes to zero and its other parameters stop meaning anything, those
thresholds are approximate.
"""
import itertools
import os
from functools import partial

import numpy as np
from scipy import optimize

from .background import PARAM_NAMES, PRIOR_BOX
from .priors import ln_prior, start_points

LEVELS_2D = {0.68: 1.15, 0.95: 3.09}
LEVELS_1D = {0.68: 0.5, 0.95: 1.92}


class ProfileScan:
    """scan: name -> array of values. nuisance: name -> (lo, hi, "log" or "linear")."""

    def __init__(self, like, spectrum, scan, nuisance=None, fixed=None,
                 astro_prior="nanograv", restarts=4, seed=0, nuisance_prior=None, sweeps=2):
        self.like = like
        self.spectrum = spectrum
        self.scan = {k: np.asarray(v, float) for k, v in scan.items()}
        self.nuisance = dict(nuisance or {})
        self.fixed = dict(fixed or {})
        self.astro_prior = astro_prior
        # a prior on the new-physics or background nuisance parameters, for instance the published
        # Gaussian on the power-law background; None leaves them flat inside their ranges, which is
        # what a profile likelihood on parameters you want to constrain should do
        self.nuisance_prior = nuisance_prior
        self.restarts = restarts
        # how many times to sweep the grid seeding each point from its best neighbour
        self.sweeps = sweeps
        self.seed = seed
        self.with_background = like.background is not None
        bg = like.background
        self.theta_names = list(bg.param_names) if self.with_background else []
        self.theta_box = dict(bg.prior_box) if self.with_background else {}

    def _spec(self, values):
        """The model spectrum, or None where the model cannot be evaluated.

        Models are other people's code: tabulated ones raise at the edge of their tables, and some
        raise for parameter combinations that are physically empty. A scan should not die because
        one grid point does; those points get no likelihood, and run() reports how many there were.
        """
        try:
            out = np.asarray(self.spectrum(self.like.data.freqs, **values, **self.fixed), float)
        except Exception:                                  # noqa: BLE001 - any model may raise
            return None
        return out if np.all(np.isfinite(out)) else None

    # Everything is optimised in a unit cube, so that parameters with ranges of very different
    # size, a reheating temperature over decades and a spectral index of order one, present the
    # same scale to a simplex method. Outside the cube the objective is a large constant.

    @property
    def _bounds(self):
        """(lo, hi) per optimised coordinate, nuisance parameters first, in their own variable."""
        b = []
        for name, (lo, hi, kind) in self.nuisance.items():
            b.append((np.log10(lo), np.log10(hi)) if kind == "log" else (lo, hi))
        if self.with_background:
            b.extend((self.theta_box[k][0], self.theta_box[k][1]) for k in self.theta_names)
        return np.array(b, float).reshape(-1, 2)

    def _to_cube(self, x):
        b = self._bounds
        return (x - b[:, 0]) / (b[:, 1] - b[:, 0])

    def _from_cube(self, u):
        b = self._bounds
        return b[:, 0] + np.asarray(u) * (b[:, 1] - b[:, 0])

    def _unpack(self, x):
        """x holds the nuisance parameters, in their own variable, then the background ones."""
        vals, k = {}, 0
        for name, (lo, hi, kind) in self.nuisance.items():
            vals[name] = 10.0 ** x[k] if kind == "log" else x[k]
            k += 1
        theta = np.asarray(x[k:]) if self.with_background else None
        return vals, theta

    def _neg(self, u, point, theta_fixed=None):
        """Objective in the unit cube. With theta_fixed the background parameters are held."""
        if np.any(np.asarray(u) < -1e-9) or np.any(np.asarray(u) > 1 + 1e-9):
            return 1e6
        if theta_fixed is not None:
            nn = len(self.nuisance)
            full = np.concatenate([np.asarray(u), self._to_cube(
                np.concatenate([np.zeros(nn), theta_fixed]))[nn:]])
        else:
            full = np.asarray(u)
        vals, theta = self._unpack(self._from_cube(full))
        spec = self._spec({**point, **vals})
        if spec is None:
            return 1e6
        lp = 0.0
        if self.nuisance_prior is not None and vals:
            lp += self.nuisance_prior(vals)
            if not np.isfinite(lp):
                return 1e6
        if theta is not None:
            lp = ln_prior(theta, self.astro_prior, self.theta_names, self.theta_box)
            if not np.isfinite(lp):
                return 1e6
        v = self.like.ln_like(spec, theta) + lp
        return -v if np.isfinite(v) else 1e6

    def _x0(self, rng):
        x = []
        for name, (lo, hi, kind) in self.nuisance.items():
            x.append(rng.uniform(np.log10(lo), np.log10(hi)) if kind == "log"
                     else rng.uniform(lo, hi))
        if self.with_background:
            x.extend(start_points(rng, 1, self.astro_prior, self.theta_names, self.theta_box)[0])
        return np.array(x)

    def _one_point(self, job):
        idx, point = job
        rng = np.random.default_rng(self.seed + idx)
        nn = len(self.nuisance)
        if not nn and not self.with_background:
            # nothing left to profile: the model has one parameter and it is the scanned one
            out = dict(point)
            spec = self._spec(point)
            out["lnL"] = -np.inf if spec is None else self.like.ln_like(spec)
            out["lnprior"] = 0.0
            out["lnL_data"] = out["lnL"]
            return out
        best_u, best_f = None, np.inf
        for _ in range(self.restarts):
            x0 = self._x0(rng)
            if nn == 1:
                # a coarse sweep over a single nuisance parameter, as for the reheating
                # temperature: the likelihood along it has narrow maxima that a simplex misses
                name, (lo, hi, kind) = next(iter(self.nuisance.items()))
                grid = (np.logspace(np.log10(lo), np.log10(hi), 300) if kind == "log"
                        else np.linspace(lo, hi, 300))
                theta0 = x0[1:] if self.with_background else None
                vals = []
                for g in grid:
                    spec = self._spec({**point, name: g})
                    vals.append(-np.inf if spec is None
                                else self.like.ln_like(spec, theta0))
                g = grid[int(np.argmax(vals))]
                x0[0] = np.log10(g) if kind == "log" else g
            u0 = np.clip(self._to_cube(x0), 0.0, 1.0)
            if nn and self.with_background:
                # stage one: the new-physics parameters alone, with the background held, which is
                # a small problem and lands the simplex near the right amplitude
                r = optimize.minimize(self._neg, u0[:nn], args=(point, x0[nn:]),
                                      method="Nelder-Mead",
                                      options={"maxiter": 1500, "fatol": 1e-6, "xatol": 1e-4})
                u0[:nn] = np.clip(r.x, 0.0, 1.0)
            u, f = u0, self._neg(u0, point)
            for _polish in range(4):          # restart the simplex until it stops improving
                r = optimize.minimize(self._neg, u, args=(point,), method="Nelder-Mead",
                                      options={"maxiter": 4000, "fatol": 1e-8, "xatol": 1e-6})
                if r.fun > f - 1e-6:
                    u, f = (r.x, r.fun) if r.fun < f else (u, f)
                    break
                u, f = r.x, r.fun
            if f < best_f:
                best_u, best_f = np.clip(u, 0.0, 1.0), f
        return self._row(point, best_u, best_f)

    def _row(self, point, u, f):
        best = optimize.OptimizeResult(x=self._from_cube(u), fun=f)
        vals, theta = self._unpack(best.x)
        lp = (ln_prior(theta, self.astro_prior, self.theta_names, self.theta_box)
              if theta is not None else 0.0)
        if self.nuisance_prior is not None and vals:
            lp += self.nuisance_prior(vals)
        out = dict(point)
        out.update(vals)
        out["lnL"] = -best.fun
        out["lnprior"] = lp
        out["lnL_data"] = -best.fun - lp
        if theta is not None:
            out.update(dict(zip(self.theta_names, theta)))
        out["_u"] = np.asarray(u, float).tolist()
        return out

    def _from_seed(self, job):
        """Re-optimise one grid point starting from a solution found elsewhere on the grid.

        Neighbouring grid points have neighbouring optima, so the best fit of a neighbour is a far
        better starting point than a random draw. This is what keeps a profile smooth when the
        model has many parameters and the simplex would otherwise stop in different local maxima at
        neighbouring points.
        """
        idx, point, seed_u, current = job
        r = optimize.minimize(self._neg, np.clip(seed_u, 0.0, 1.0), args=(point,),
                              method="Nelder-Mead",
                              options={"maxiter": 4000, "fatol": 1e-8, "xatol": 1e-6})
        if r.fun < current - 1e-9:
            return self._row(point, np.clip(r.x, 0.0, 1.0), r.fun)
        return None

    def _neighbour_sweeps(self, rows, points, names, jobs):
        """Seed every grid point from the best of its neighbours, repeatedly, until nothing
        improves. Cheap next to the restarts, and it is what removes the ragged profiles that a
        many-parameter model produces."""
        if self.sweeps <= 0 or len(rows) < 2:
            return rows
        shape = tuple(len(self.scan[n]) for n in names)
        for sweep in range(self.sweeps):
            grid = np.array([r["lnL"] for r in rows]).reshape(shape)
            best_neighbour = {}
            for flat in range(len(rows)):
                pos = np.unravel_index(flat, shape)
                cands = []
                for axis in range(len(shape)):
                    for step in (-1, 1):
                        q = list(pos)
                        q[axis] += step
                        if 0 <= q[axis] < shape[axis]:
                            cands.append(int(np.ravel_multi_index(tuple(q), shape)))
                if not cands:
                    continue
                top = max(cands, key=lambda c: grid.flat[c])
                if grid.flat[top] > grid.flat[flat] and rows[top].get("_u") is not None:
                    best_neighbour[flat] = rows[top]["_u"]
            if not best_neighbour:
                break
            work = [(i, points[i], np.array(u), -rows[i]["lnL"])
                    for i, u in best_neighbour.items()]
            if jobs > 1:
                import multiprocessing as mp
                with mp.Pool(jobs) as pool:
                    out = list(pool.imap(partial(_call_seed, self), work, chunksize=4))
            else:
                out = [self._from_seed(w) for w in work]
            improved = 0
            for (i, *_), new in zip(work, out):
                if new is not None:
                    rows[i] = new
                    improved += 1
            if improved == 0:
                break
        return rows

    def run(self, jobs=1, progress=True):
        names = list(self.scan)
        points = [dict(zip(names, v)) for v in itertools.product(*(self.scan[n] for n in names))]
        jobs_list = list(enumerate(points))
        if jobs > 1:
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(v, "1")
            import multiprocessing as mp
            with mp.Pool(jobs) as pool:
                rows = list(pool.imap(partial(_call, self), jobs_list, chunksize=4))
        else:
            rows = [self._one_point(j) for j in jobs_list]
        rows = self._neighbour_sweeps(rows, points, names, jobs)
        for r in rows:
            r.pop("_u", None)
        bad = sum(1 for r in rows if not np.isfinite(r["lnL"]))
        if bad:
            print(f"  note: {bad} of {len(rows)} grid points have no likelihood, because the "
                  "model could not be evaluated there or predicted nothing measurable")
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows


def _call(scan, job):                      # module-level, so that the pool can pickle it
    return scan._one_point(job)


def _call_seed(scan, job):
    return scan._from_seed(job)


def region(table, level=0.68, ndim=2):
    """Boolean mask of the grid points inside a profile-likelihood region."""
    thr = (LEVELS_2D if ndim == 2 else LEVELS_1D)[level]
    return table["lnL"] >= table["lnL"].max() - thr


def reference_lnL(like, background_model=None, nuisance=None, nuisance_prior=None,
                  astro_prior="nanograv", restarts=6):
    """The best fit without any new physics.

    With a background attached to the likelihood this maximises over its parameters under their
    prior. Without one, background_model is the spectrum that stands in for the binaries, for
    instance the published power law, with its own nuisance parameters and prior.
    """
    if background_model is None:
        if like.background is None:
            return like.ln_like(None)
        return preference_reference(like, astro_prior, restarts)
    scan = ProfileScan(like, background_model, scan={}, nuisance=nuisance,
                       nuisance_prior=nuisance_prior, astro_prior=astro_prior, restarts=restarts)
    return float(scan.run(jobs=1)["lnL"].max())


def preference_reference(like, astro_prior="nanograv", restarts=12):
    """The background alone, maximised over its parameters under their prior."""
    from .priors import start_points
    names = list(like.background.param_names)
    box = dict(like.background.prior_box)
    rng = np.random.default_rng(0)
    best = -np.inf
    for th in start_points(rng, restarts, astro_prior, names, box):
        r = optimize.minimize(
            lambda t: -(like.ln_like(None, t) + ln_prior(t, astro_prior, names, box)), th,
            method="Nelder-Mead", options={"maxiter": 4000})
        best = max(best, -r.fun)
    return float(best)


def preference(table, like, theta_grid=None):
    """How much the best fit gains over the binaries alone, in ln L units.

    Without new physics the model is the background by itself, maximised over its parameters; the
    gain is the number to quote for "is the new physics preferred", and p = exp(-gain) with two
    scanned parameters turns it into a significance.
    """
    from .priors import start_points
    names = list(like.background.param_names)
    box = dict(like.background.prior_box)
    which = "nanograv" if theta_grid is None else theta_grid
    rng = np.random.default_rng(0)
    best = -np.inf
    for th in start_points(rng, 12, which, names, box):
        r = optimize.minimize(
            lambda t: -(like.ln_like(None, t) + ln_prior(t, which, names, box)), th,
            method="Nelder-Mead", options={"maxiter": 4000})
        best = max(best, -r.fun)
    return float(table["lnL"].max() - best)
