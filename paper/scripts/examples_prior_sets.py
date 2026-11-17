"""
The three further models of figure 6 (cosmic superstrings, domain walls decaying into a dark sector, inflation)
on top of the binaries, profiled as in profile_examples.py but under one of the two narrower prior sets of
table 1, tomczak_kormendy or survey_average, for table 2.

    BINARYFOG_NG15_MODELS=<model files> python examples_prior_sets.py --prior-set tomczak_kormendy --data ng15_cp_v2
Writes scans/examples_<model>_<prior set>_n<ngrid>[_<data>].csv; results on ng15_cp_v2 carry no data tag.
Existing output files are kept.
"""
import argparse
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from paths import FREESPEC_ROOT, SCANS                                        # noqa: E402
import binaryfog as bf                                                        # noqa: E402
from profile_examples import example_scan, scan_values                        # noqa: E402

EXAMPLES = [("super", "log10_Gmu"), ("dw_ds", "log10_T_star"), ("igw", "log10_r")]


def main(dataset, prior, ngrid, jobs, restarts):
    models = Path(os.environ["BINARYFOG_NG15_MODELS"])
    data = bf.FreeSpectrum.load(dataset, root=FREESPEC_ROOT)
    dtag = "" if dataset == "ng15_cp_v2" else f"_{dataset}"
    for fname, param in EXAMPLES:
        out = SCANS / f"examples_{fname}_{prior}_n{ngrid}{dtag}.csv"
        if out.is_file():
            continue
        model = bf.load_ptarcade(models / f"{fname}.py")
        lo, hi = model.ranges[param]
        table = example_scan(data, model, param, scan_values(lo, hi, ngrid), "population",
                             astro_prior=prior, restarts=restarts, sweeps=2).run(jobs=jobs)
        table.to_csv(out, index=False)
        print(f"  {fname:6s} {prior} max ln L {table['lnL'].max():+.3f} -> {out.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior-set", required=True, choices=("tomczak_kormendy", "survey_average"))
    ap.add_argument("--data", default="ng15_cp_v2")
    ap.add_argument("--ngrid", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 28)))
    ap.add_argument("--restarts", type=int, default=8)
    a = ap.parse_args()
    main(a.data, a.prior_set, a.ngrid, a.jobs, a.restarts)
