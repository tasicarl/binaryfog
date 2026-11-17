"""
The binary backgrounds figure 6 draws at its three values of each scanned parameter (lower end, best fit, upper
end of the 2 sigma interval): FIGURE_JSON keeps the astrophysical parameters only at the best fit, so each value
is profiled again (population case) to recover them. Writes the median spectrum and the band holding 90% of the
realisations to BANDS_JSON.

    BINARYFOG_NG15_MODELS=<model files> python example_bands.py
"""
import json
import os
from pathlib import Path

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import DATA                                                     # noqa: E402
from profile_examples import BANDS_JSON, FIGURE_JSON, example_scan           # noqa: E402


def main():
    models = Path(os.environ["BINARYFOG_NG15_MODELS"])
    raw = json.loads((SCANS / FIGURE_JSON).read_text())
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    out = {}
    for key, e in raw.items():
        model = bf.load_ptarcade(models / f"{key}.py")
        param = e["parameter"]
        fdraw = np.asarray(e["freqs"], float)
        bgf = bf.Background(fdraw, extrapolate=True)
        lo, hi, med = [], [], []
        for v in e["shown"]:
            row = example_scan(data, model, param, np.array([v]), "population",
                               restarts=8, sweeps=2).run(jobs=1).iloc[0]
            th = [float(row[k]) for k in bf.PARAM_NAMES]
            med.append(list(map(float, np.asarray(bgf.predict(th)[0], float))))
            lo.append(list(map(float, bf.quantile_spectrum(bgf, th, 0.05))))
            hi.append(list(map(float, bf.quantile_spectrum(bgf, th, 0.95))))
            print(f"  {key:6s} {param} = {v:+.3f} profiled", flush=True)
        out[key] = {"backgrounds_lo": lo, "backgrounds_hi": hi, "backgrounds_refit": med}
    (SCANS / BANDS_JSON).write_text(json.dumps(out))
    print(f"wrote {BANDS_JSON}")


if __name__ == "__main__":
    main()
