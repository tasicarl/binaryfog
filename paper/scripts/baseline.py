"""The free spectrum the main results of the paper are computed on, in one place.

The baseline is the common uncorrelated red noise spectrum ng15_cp_v2 of the 2026 NANOGrav 15 yr release.
Results on the other free spectra carry the name of that spectrum in their file names.
"""
import os

DATA = "ng15_cp_v2"
TAG = f"_{DATA}"                                   # suffix of run_scan_marginal.py's file names
# the profiles of the further models (figure 6); EXAMPLES_JSON=<file in scans/> selects another one
EXAMPLES_JSON = os.environ.get("EXAMPLES_JSON", "ng15_examples_n80.json")
BANDS_JSON = EXAMPLES_JSON.replace(".json", "_bands.json")
