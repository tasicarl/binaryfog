"""The preference for a strong (supercooled) phase transition, section 5.1: the template of fopt_lv25.py,
scanned along beta/H with T_reh profiled, on the CURN free spectrum, with the binaries as the (A, gamma)
power law and as binaryfog under the NANOGrav priors.

G = max ln L(phase transition + binaries) - max ln L(binaries alone, same prior), with k = 2 (beta/H, T_reh).
The 1 sigma interval is where Delta ln L <= 0.5 (one scanned parameter), restricted to beta/H >= 4, the
slowest phase transition simulated for the template. The binaries-alone references are those of
scans/prior_table_refs.json (build_prior_table.py).

    python lv25_preferences.py      -> results/lv25_preferences.json
"""
import json

import numpy as np
import pandas as pd

from paths import RESULTS, SCANS
import binaryfog as bf                                                        # noqa: E402

DATA = "ng15_cp_v2"
REFS = json.loads((SCANS / "prior_table_refs.json").read_text())[DATA]
FILES = {"power law": f"scan_agamma_n300_line_a3_b300_lv25_{DATA}.csv",
         "nanograv": f"scan_holodeck_n300_line_a3_b300_lv25_{DATA}.csv"}
BFLOOR = 4.0


def interval(b, dl, thr):
    ok = (dl <= thr) & (b >= BFLOOR)
    return (float(b[ok].min()), float(b[ok].max())) if ok.any() else (np.nan, np.nan)


out = {}
for t, f in FILES.items():
    tab = pd.read_csv(SCANS / f).sort_values("betaH")
    b, l = tab.betaH.values, tab.lnL.values
    G = float(l.max() - REFS[t])
    r = out[f"{DATA}|{t}"] = {"G": G, "Z": float(bf.sigma_from_gain(G, 2)),
                              "1sigma": interval(b, l.max() - l, 0.5)}
    print(f"{DATA:14s} {t:10s} G {G:5.2f}  {r['Z']:.2f} sigma  1sigma beta/H {r['1sigma'][0]:5.1f}-{r['1sigma'][1]:5.1f}")
(RESULTS / "lv25_preferences.json").write_text(json.dumps(out, indent=1))
