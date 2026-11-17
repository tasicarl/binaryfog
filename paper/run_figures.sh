#!/bin/bash
# The figures and numbers of the paper from the stored scan outputs in scans/ (about ten minutes).
# Figures go to plots/, tables and numbers to results/ and the printed output of each step to logs/<step>.log.
# The scans themselves are produced by run_scans.sh.
#   PY          python with binaryfog's dependencies, matplotlib and pandas (default: python3)
#   BINARYFOG_NG15_MODELS  directory of the NANOGrav 15-year model files (PTArcade), for the further models
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p logs
run() { local name=$1; shift; "$@" > logs/$name.log 2>&1; echo "$name exit $?"; }

# The surrogate, its priors and its accuracy (sections 2 and 4, appendix B)
run prior_sets            $PY scripts/plot_prior_spectra.py                    # figure 2
run survey_average_priors $PY scripts/fit_survey_average_priors.py             # figure 9, survey-average priors of table 1
run prior_table           $PY scripts/build_prior_table.py                     # table 2
run anatomy               $PY scripts/plot_parameters.py                       # figure 1
run realisations          $PY scripts/plot_realisations.py                     # figure 3
run realisations_numbers  $PY scripts/realisations_numbers.py                  # numbers quoted with figure 3
run law_dlnL              $PY scripts/law_dlnL_allbins.py                      # comparison of the three parametrisations
run floor                 $PY scripts/floor_matters.py                         # floor of the warp
run accuracy              $PY scripts/surrogate_accuracy.py                    # accuracy on held-out models
run learning_curve        $PY scripts/learning_curve.py                        # accuracy against library size
run surrogate_vs_library  $PY scripts/surrogate_vs_library.py analyse          # surrogate against the library along the scan
# Validation against the NANOGrav analysis (section 4)
run validation            bash scripts/validation_numbers.sh ng15_fig9_binaryfog_erratum.npz
run validation_corner     $PY scripts/validation_agreement.py --method corner --per-panel --cases flat
run validation_plot       $PY scripts/plot_nanograv_fit.py                     # figure 4
# New-physics fits (section 5)
run pt                    $PY scripts/plot_phase_transition.py                 # figure 5
run pt_decomposition      $PY scripts/pt_decomposition.py                      # gain split into data and prior
run tk_decomposition      $PY scripts/tk_decomposition.py scan_holodeck_n120_a-2to3_tomczak_kormendy_ng15_cp_v2.csv
run datasets              $PY scripts/plot_free_spectra.py                     # figure 7
run lv25                  $PY scripts/lv25_preferences.py                      # strongly supercooled template
run pvs                   $PY scripts/prior_volume_sensitivity.py              # prior-volume test
run example_prefs         $PY scripts/example_preferences.py                   # numbers of section 5.2
run examples              $PY scripts/plot_other_models.py                     # figure 6
run hardening_plot        $PY scripts/plot_hardening.py                        # figure 8
run tauf_lower_bound      $PY scripts/tauf_lower_bound.py                      # preference against a lower bound on tau_f
echo ALL DONE
