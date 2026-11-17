"""
The three values figure 6 draws for each model (lower end, best fit, upper end of the binaryfog interval at the
2 sigma threshold of one scanned parameter, Delta ln L = 2.00), read again from the profile stored in
FIGURE_JSON after repair_examples.py. Where an end moves, that point is profiled again for its spectrum.

    BINARYFOG_NG15_MODELS=<model files> python update_examples_shown.py
"""
import json
import os
from pathlib import Path

import numpy as np

from paths import FREESPEC_ROOT, SCANS
import binaryfog as bf                                                        # noqa: E402
from baseline import DATA                                                     # noqa: E402
from profile_examples import FIGURE_JSON, example_scan                        # noqa: E402

SOURCE = SCANS / FIGURE_JSON


def main():
    models = Path(os.environ["BINARYFOG_NG15_MODELS"])
    raw = json.loads(SOURCE.read_text())
    data = bf.FreeSpectrum.load(DATA, root=FREESPEC_ROOT)
    for key, e in raw.items():
        grid, y = np.asarray(e["grid"], float), np.asarray(e["population"], float)
        ok = y >= y.max() - bf.LEVELS_1D[2]
        new = [float(grid[ok].min()), e["shown"][1], float(grid[ok].max())]
        print(f"  {key:6s} shown {np.round(e['shown'], 3).tolist()} -> {np.round(new, 3).tolist()}")
        if np.allclose(new, e["shown"]):
            continue
        model = bf.load_ptarcade(models / f"{key}.py")
        param = e["parameter"]
        fdraw = np.asarray(e["freqs"], float)
        for k in (0, 2):
            if np.isclose(new[k], e["shown"][k]):
                continue
            row = example_scan(data, model, param, np.array([new[k]]), "population",
                               restarts=8, sweeps=2).run(jobs=1).iloc[0]
            kw = {q: float(row[q]) for q in model.ranges}
            e["spectra"][k] = list(map(float, np.asarray(model.spectrum(fdraw, **kw), float)))
        e["shown"] = new
    SOURCE.write_text(json.dumps(raw, indent=1))
    print(f"wrote {SOURCE.name}")


if __name__ == "__main__":
    main()
