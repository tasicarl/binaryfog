"""Directories of the reproduction kit. Importing this module also makes the binaryfog package of
this repository importable, so the scripts run without installing it."""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
INPUTS = ROOT / "inputs"      # holodeck libraries, surrogate maps, free spectra: what the scans start from
SCANS = ROOT / "scans"        # outputs of run_scans.sh
RESULTS = ROOT / "results"    # tables and numbers written by run_figures.sh
PLOTS = ROOT / "plots"        # figures written by run_figures.sh
FREESPEC_ROOT = INPUTS / "pta" / "freespec"

for _d in (SCANS, RESULTS, PLOTS):
    _d.mkdir(exist_ok=True)
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
