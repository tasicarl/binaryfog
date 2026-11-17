"""holodeck itself at one point of the population model: log10 h_c^2 in the 14 lowest NANOGrav 15-year
frequency bins for each realisation, in the library space PS_Classic_Phenom_Uniform of NANOGrav's
astrophysics analysis. Needs holodeck (see ../../training/README.md). The six parameters and their box are
spelled out here, so that this module runs in the holodeck environment without binaryfog."""
import numpy as np

PARAM_NAMES = ["gsmf_phi0_log10", "gsmf_mchar0_log10", "mmb_mamp_log10", "mmb_scatter_dex",
               "hard_time", "hard_gamma_inner"]
PRIOR_BOX = {"gsmf_phi0_log10": (-3.5, -1.5), "gsmf_mchar0_log10": (10.5, 12.5),
             "mmb_mamp_log10": (7.6, 9.0), "mmb_scatter_dex": (0.0, 0.9),
             "hard_time": (0.1, 11.0), "hard_gamma_inner": (-1.5, 0.0)}
# the quantile levels of log10 h_c^2 stored per bin in inputs/library/qlib.jsonl (build_quantile_library.py)
QUANT = [0.005, 0.02, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.98, 0.995]


def realisations(theta, nreals=2000):
    """log10 h_c^2, shape (14, nreals), at the parameter point theta (in the order of PARAM_NAMES)."""
    import holodeck as holo
    import holodeck.librarian  # noqa: F401
    from holodeck.librarian import DEF_PTA_DUR, DEF_NUM_LOUDEST
    space = holo.librarian.param_spaces_dict["PS_Classic_Phenom_Uniform"]()
    sam, hard = space.model_for_params(dict(zip(PARAM_NAMES, map(float, theta))))
    out = holo.librarian.lib_tools.run_model(sam, hard, pta_dur=DEF_PTA_DUR, nfreqs=14, nreals=nreals,
                                             nloudest=DEF_NUM_LOUDEST, gwb_flag=True, singles_flag=False,
                                             details_flag=False, params_flag=False, log=None)
    return 2 * np.log10(np.asarray(out["gwb"]))
