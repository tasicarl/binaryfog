"""The Tomczak-Kormendy numbers of section 5.4: the split of the gain as in pt_decomposition.py, on the
120 x 120 scan under the Tomczak-Kormendy priors, and the binaries-alone fit.

    python tk_decomposition.py scan_holodeck_n120_a-2to3_tomczak_kormendy_ng15_cp_v2.csv
"""
import sys
from pathlib import Path

import pandas as pd

from paths import SCANS
import binaryfog as bf                                                        # noqa: E402

for f in sys.argv[1:]:
    d = pd.read_csv(SCANS / Path(f).name)
    best = d.loc[d.lnL.idxmax()]
    weak = d[d.alpha == d.alpha.min()]
    alone = weak.loc[weak.lnL.idxmax()]
    print(f"{Path(f).name}\n  best lnL {best.lnL:+.2f}, binaries alone lnL {alone.lnL:+.2f}; gain {best.lnL-alone.lnL:+.2f} = data "
          f"{best.lnL_data-alone.lnL_data:+.2f} + prior {best.lnprior-alone.lnprior:+.2f}")
    print("  pulls of the binaries-alone fit:",
          {k: round((alone[k]-m)/s, 2) for k, (m, s) in bf.TOMCZAK_KORMENDY_GAUSSIAN.items()},
          "tau_f", round(alone.hard_time, 3), "nu", round(alone.hard_gamma_inner, 3))
