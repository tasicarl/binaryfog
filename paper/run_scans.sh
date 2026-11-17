#!/bin/bash
# The profile-likelihood scans behind the paper, written to scans/. These are the expensive steps: of order a day
# on a 28-core node in total, dominated by the 120 x 120 phase-transition scans. run_figures.sh turns their outputs
# into figures and numbers. Each block can be run on its own.
#   PY          python with binaryfog's dependencies and pandas (default: python3)
#   PY_HOLODECK python with holodeck installed (default: $PY)
#   BINARYFOG_NG15_MODELS  directory of the NANOGrav 15-year model files (PTArcade), for the further models
#   JOBS        number of worker processes (default: 28)
# PRIOR_SET selects the Gaussian astrophysical priors of table 1 for run_scan_marginal.py and repair_scan.py:
# nanograv (default), tomczak_kormendy, survey_average or flat.
set -eu
cd "$(dirname "$0")"
PY=${PY:-python3}; HO=${PY_HOLODECK:-$PY}; J=${JOBS:-28}
export JOBS=$J PYTHONWARNINGS=ignore
mkdir -p logs
S=scripts

# 1. holodeck itself at the best fits (figure 3)
$HO $S/bestfit_holodeck_run.py

# 2. Validation chains against the NANOGrav analysis (figure 4)
OMP_NUM_THREADS=1 $PY $S/validate_ng15_fig9.py --data ng15_hdall_v2 --nbins 5 --walkers 64 --steps 30000 --burn 5000 \
  --seed 1 --out ng15_fig9_binaryfog_erratum.npz

# 3. Phase transition on the CURN free spectrum, 120 x 120 (figures 5 and 8, table 2)
for s in nanograv tomczak_kormendy survey_average; do
  infix=$([ $s = nanograv ] && echo "" || echo "_$s")
  PRIOR_SET=$s FREESPEC=ng15_cp_v2 $PY $S/run_scan_marginal.py --mode holodeck -n 120 -j $J --alpha-hi 3
  PRIOR_SET=$s FREESPEC=ng15_cp_v2 $PY $S/repair_scan.py scan_holodeck_n120_a-2to3${infix}_ng15_cp_v2.csv 0.05
done
FREESPEC=ng15_hd_v2 $PY $S/run_scan_marginal.py --mode holodeck -n 120 -j $J --alpha-hi 3
FREESPEC=ng15_hd_v2 $PY $S/repair_scan.py scan_holodeck_n120_a-2to3_ng15_hd_v2.csv 0.05
# the phase transition alone, and on top of the (A, gamma) power law
for m in fopt agamma; do FREESPEC=ng15_cp_v2 $PY $S/run_scan_marginal.py --mode $m -n 120 -j $J --alpha-hi 3; done
FREESPEC=ng15_hd_v2 $PY $S/run_scan_marginal.py --mode agamma -n 120 -j $J --alpha-hi 3
# flat priors, 50 x 50, with two repair passes (section 5.4)
PRIOR_SET=flat FREESPEC=ng15_cp_v2 $PY $S/run_scan_marginal.py --mode holodeck -n 50 -j $J
for pass in 1 2; do PRIOR_SET=flat FREESPEC=ng15_cp_v2 $PY $S/repair_scan.py scan_holodeck_n50_flat_ng15_cp_v2.csv 0.05; done
$PY $S/dneff_grid.py

# 4. Phase transition on the other free spectra, 50 x 50 (figure 7, table 2)
for tag in ng15_cp_v2 ng15_cp_v1_cut9 ng15_hd_v1 ng15_hd_v2 ng15_hdall_v2 ng15_hd_dmgp_v1; do
  FREESPEC=$tag $PY $S/run_scan_marginal.py --mode holodeck -n 50 -j $J
  FREESPEC=$tag $PY $S/repair_scan.py scan_holodeck_n50_${tag}.csv 0.05
  FREESPEC=$tag $PY $S/run_scan_marginal.py --mode agamma -n 50 -j $J
done
for s in tomczak_kormendy survey_average; do for d in ng15_hd_v2 ng15_hdall_v2; do
  PRIOR_SET=$s FREESPEC=$d $PY $S/run_scan_marginal.py --mode holodeck -n 50 -j $J
  PRIOR_SET=$s FREESPEC=$d $PY $S/repair_scan.py scan_holodeck_n50_${s}_${d}.csv 0.05
done; done

# 5. Strongly supercooled template (fopt_lv25.py) along a line of the plane (section 5)
for m in holodeck agamma; do
  FREESPEC=ng15_cp_v2 PT_TEMPLATE=lv25 $PY $S/run_scan_marginal.py --mode $m --line -n 300 -j $J --alpha-hi 3 > logs/${m}_lv25_ng15_cp_v2.log
  FREESPEC=ng15_cp_v2 PT_TEMPLATE=lv25 $PY $S/repair_scan.py scan_${m}_n300_line_a3_b300_lv25_ng15_cp_v2.csv
done

# 6. Cosmic superstrings, inflation and domain walls (figure 6, table 2)
$PY $S/profile_examples.py --data ng15_cp_v2 --ngrid 80 --jobs $J --out ng15_examples_n80.json
IGW_PARAM=n_t ONLY=igw $PY $S/profile_examples.py --data ng15_cp_v2 --ngrid 80 --jobs $J --out ng15_examples_n80_igwnt_only.json
$PY -c "
import json; a = json.load(open('scans/ng15_examples_n80.json'))
a['igw'] = json.load(open('scans/ng15_examples_n80_igwnt_only.json'))['igw']
open('scans/ng15_examples_n80_nt.json', 'w').write(json.dumps(a, indent=1))"
for E in ng15_examples_n80.json ng15_examples_n80_nt.json; do $PY $S/repair_examples.py $E; done
$PY $S/update_examples_shown.py
$PY $S/example_bands.py
for d in ng15_cp_v2 ng15_hd_v2 ng15_hdall_v2; do
  [ $d = ng15_cp_v2 ] || $PY $S/profile_examples.py --data $d --ngrid 80 --jobs $J --out ${d}_examples_n80.json
  for s in tomczak_kormendy survey_average; do
    $PY $S/examples_prior_sets.py --prior-set $s --data $d --ngrid 80 --jobs $J
  done
done

# 7. Binary lifetime: preference at fixed tau_f and nu_inner (figure 8), and further lifetimes that resolve where the
#    preference with tau_f bounded from below crosses 2 and 3 sigma
$PY $S/hardening_map.py $J
$PY $S/tauf_fine_cells.py $J \
  1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,3.5,3.75,4.0,4.25,4.5,4.75,5.0,6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75,10.0 \
  nanograv,survey_average tauf_fine_cells.json

# 8. Surrogate against the holodeck library along the phase-transition scan
$PY $S/surrogate_vs_library.py scan -n 50 -j $J
echo ALL DONE
