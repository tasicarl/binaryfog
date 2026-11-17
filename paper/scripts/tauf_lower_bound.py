"""
The preference for the phase transition when the binary lifetime is bounded from below, tau_f >= tau_min
(section 5.4), from the cells of the map of figure 8 (scans/hardening_map.npz) together with the further cells of
tauf_fine_cells.py (scans/tauf_fine_cells.json, NANOGrav and survey-average priors).

For each prior set and each lifetime tau_min of the cells, the gain is the maximum of ln L with the transition over
the cells with tau_f >= tau_min minus the same maximum for the binaries alone, converted to a significance with
k = 3. Between those lifetimes the significance is interpolated linearly in log tau_min.

    python tauf_lower_bound.py
"""
import json

import numpy as np

from paths import SCANS
import binaryfog as bf                                                        # noqa: E402

PANELS = [("nanograv", "NANOGrav"), ("tomczak_kormendy", "Tomczak-Kormendy"), ("survey_average", "survey-average")]


def best_per_lifetime(d, cells, tag):
    """Lifetimes and, at each, the best ln L over nu_inner with the transition and for the binaries alone."""
    best = {round(float(t), 4): (np.nanmax(d[f"lnL_pt_{tag}"][i]), np.nanmax(d[f"lnL_bin_{tag}"][i]))
            for i, t in enumerate(d["tau"])}
    for c in cells:
        if c["prior"] == tag:
            t = round(c["tau"], 4)
            pt, alone = best.get(t, (-np.inf, -np.inf))
            best[t] = (max(pt, c["lnL_pt"]), max(alone, c["lnL_bin"]))
    tau = np.array(sorted(best))
    return tau, np.array([best[t][0] for t in tau]), np.array([best[t][1] for t in tau])


def significance_above(pt, alone):
    """The significance with tau_f >= tau_min, for every lifetime tau_min of the cells."""
    return np.array([bf.sigma_from_gain(pt[i:].max() - alone[i:].max(), 3) for i in range(len(pt))])


def crossing(log_tau, sig, level):
    """The tau_min at which the significance first reaches level, or None if it never does or starts above."""
    if sig[0] >= level or sig[-1] < level:
        return None
    i = int(np.flatnonzero(sig >= level)[0])
    return 10 ** np.interp(level, sig[i - 1:i + 1], log_tau[i - 1:i + 1])


if __name__ == "__main__":
    d = np.load(SCANS / "hardening_map.npz")
    cells = json.loads((SCANS / "tauf_fine_cells.json").read_text())
    for tag, name in PANELS:
        tau, pt, alone = best_per_lifetime(d, cells, tag)
        log_tau = np.log10(tau)
        sig = significance_above(pt, alone)
        at = {t: float(np.interp(np.log10(t), log_tau, sig)) for t in (1.0, 3.0, 6.0)}
        print(f"{name:17s} priors, tau_f >= 1, 3, 6 Gyr: "
              + ", ".join(f"{at[t]:.2f} sigma" for t in at)
              + f" (tau_f free: {sig[0]:.2f} sigma)")
        for level in (2, 3):
            t = crossing(log_tau, sig, level)
            print(f"    {level} sigma " + (f"from tau_min = {t:.2f} Gyr" if t is not None else
                                         f"{'everywhere' if sig[0] >= level else 'nowhere'} on the map"))
