"""
Table 2: the preference for each of the four new-physics models on each of three NANOGrav 15 yr free spectra,
with the binaries treated as

    power law         the (A, gamma) power law of NANOGrav's new-physics search, under its Gaussian prior
    nanograv          binaryfog under the NANOGrav priors
    tomczak_kormendy  binaryfog under the Tomczak-Kormendy priors
    survey_average    binaryfog under the survey-average priors

G = max ln L(model + binaries) - max ln L(binaries alone), both under the same prior; Z from the one-sided
boundary mixture 1/2 delta + 1/2 chi2_k, k = number of free parameters of the model (phase transition 3,
strings 2, inflation 3, domain walls 4).

Inputs:
  phase transition  scans/scan_agamma_* (power law), scans/scan_holodeck_*[_<prior set>]_*: 120 x 120 grids,
                    except 50 x 50 on HD+MP+DP+CURN and for the two narrower prior sets on HD
  further models    scans/<examples json>["power law" / "population"] and scans/examples_<m>_<prior set>_n80<SUF>.csv
  binaries alone    bf.reference_lnL (24 restarts for binaryfog), stored in scans/prior_table_refs.json and
                    computed only for a free spectrum missing there

    python build_prior_table.py          prints the table, writes results/prior_table.json and results/prior_table.tex
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

from paths import FREESPEC_ROOT, RESULTS, SCANS                               # noqa: E402
import binaryfog as bf                                                        # noqa: E402
from profile_examples import AGAMMA_NUISANCE                                  # noqa: E402

DATASETS = ["ng15_cp_v2", "ng15_hd_v2", "ng15_hdall_v2"]
TREAT = ["power law", "nanograv", "tomczak_kormendy", "survey_average"]
MODELS = [("pt", 3), ("super", 2), ("igw", 3), ("dw_ds", 4)]
CACHE = SCANS / "prior_table_refs.json"
# grid tag of the phase-transition scans: power law and NANOGrav priors, then the two narrower prior sets
PT_GRID = {"ng15_cp_v2": ("n120_a-2to3", "n120_a-2to3"), "ng15_hd_v2": ("n120_a-2to3", "n50"),
           "ng15_hdall_v2": ("n50", "n50")}


def suf(d):
    return "" if d == "ng15_cp_v2" else f"_{d}"


def examples_json(d):
    return "ng15_examples_n80.json" if d == "ng15_cp_v2" else f"{d}_examples_n80.json"


def references(d, cache):
    if d in cache:
        return cache[d]
    data = bf.FreeSpectrum.load(d, root=FREESPEC_ROOT)
    pl = bf.Likelihood(data, None, marginalise=False)
    pop = bf.Likelihood(data, bf.Background(), marginalise=True)
    out = {"power law": bf.reference_lnL(pl, background_model=bf.smbhb_power_law,
                                         nuisance=dict(AGAMMA_NUISANCE), nuisance_prior=bf.ln_prior_agamma)}
    for t in TREAT[1:]:
        out[t] = bf.reference_lnL(pop, astro_prior=t, restarts=24)
    cache[d] = out
    CACHE.write_text(json.dumps(cache, indent=1))
    return out


def pt_max(d):
    """Maximum of the phase-transition profile likelihood per treatment."""
    n, n_sets = PT_GRID[d]
    files = {"power law": f"scan_agamma_{n}_{d}.csv",
             "nanograv": f"scan_holodeck_{n}_{d}.csv",
             "tomczak_kormendy": f"scan_holodeck_{n_sets}_tomczak_kormendy_{d}.csv",
             "survey_average": f"scan_holodeck_{n_sets}_survey_average_{d}.csv"}
    return {t: float(pd.read_csv(SCANS / f).lnL.max()) for t, f in files.items()}


def model_max(d, m):
    e = json.loads((SCANS / examples_json(d)).read_text())[m]
    out = {"power law": float(np.nanmax(e["power law"])), "nanograv": float(np.nanmax(e["population"]))}
    for t in ("tomczak_kormendy", "survey_average"):
        out[t] = float(pd.read_csv(SCANS / f"examples_{m}_{t}_n80{suf(d)}.csv").lnL.max())
    return out


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    res = {}
    for d in DATASETS:
        ref = references(d, cache)
        print(f"{d}: binaries alone ln L " + ", ".join(f"{t} {ref[t]:+.3f}" for t in TREAT))
        res[d] = {}
        for m, k in MODELS:
            mx = pt_max(d) if m == "pt" else model_max(d, m)
            res[d][m] = {}
            for t in TREAT:
                G = mx[t] - ref[t]
                res[d][m][t] = {"max": mx[t], "ref": ref[t], "G": G, "Z": float(bf.sigma_from_gain(G, k))}
    (RESULTS / "prior_table.json").write_text(json.dumps(res, indent=1))

    label = {"pt": "phase transition", "super": "cosmic superstrings", "igw": "inflation",
             "dw_ds": "domain walls"}
    dlab = {"ng15_cp_v2": "CURN", "ng15_hd_v2": "HD", "ng15_hdall_v2": "HD+MP+DP+CURN"}
    for d in DATASETS:
        print(f"== {d}")
        for m, _ in MODELS:
            print(f"  {label[m]:20s}" + "".join(
                f"  {t}: G {res[d][m][t]['G']:5.2f} Z {res[d][m][t]['Z']:4.2f}" for t in TREAT))

    # the body of table 2, row by row as in the paper
    lines = []
    for m, k in MODELS:
        for i, d in enumerate(DATASETS):
            first = f"{label[m]} ($k = {k}$)" if i == 0 else ""
            lines.append(f"{first} & {dlab[d]} & " + " & ".join(f"${res[d][m][t]['Z']:.1f}\\sigma$" for t in TREAT)
                         + r" \\")
        lines.append(r"\midrule" if m != MODELS[-1][0] else r"\bottomrule")
    (RESULTS / "prior_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote results/prior_table.json, results/prior_table.tex")


if __name__ == "__main__":
    main()
