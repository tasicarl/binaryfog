#!/bin/bash
# The numbers of the validation against NANOGrav's posteriors (section 4, figure 4), for one chain file of scans/: $1
PY=${PY:-python3}
cd "$(dirname "$0")"
CH=../scans/$1
J=../inputs/pta/ng15_fig9_erratum_contours.json
C="--contours $J --chains $CH"
echo "== published vs ours";            $PY validation_agreement.py $C 2>&1 | grep "median separation"
echo "== ours drawn like corner.py";    $PY validation_agreement.py $C --method corner --cases flat 2>&1 | grep "median separation"
echo "== two halves of our chain";      $PY validation_agreement.py $C --self-split --cases flat 2>&1 | grep "median separation"
echo "== medians against the published marginals"
$PY - "$CH" "$J" <<'PYEOF'
import json, sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True); names = list(d["names"])
pub = json.load(open(sys.argv[2]))


def published_median(m):
    """Median of a published marginal, a step histogram given as the outline (x, y) of its bars."""
    x, y = np.asarray(m["x"], float), np.asarray(m["y"], float)
    bars = [(x[i], x[i + 1], y[i]) for i in range(len(x) - 1) if x[i + 1] > x[i] and y[i + 1] == y[i]]
    lo, hi, h = (np.array(v) for v in zip(*bars))
    cdf = np.concatenate([[0.0], np.cumsum(h * (hi - lo))])
    return float(np.interp(0.5, cdf / cdf[-1], np.concatenate([lo, hi[-1:]])))


for case, pubname in (("flat", "uniform"), ("nanograv", "astro")):
    s = d[case]
    shift = {}
    for m in pub["marginals"]:
        if m["case"] != pubname:
            continue
        i = names.index(m["param"])
        med, ref = np.median(s[:, i]), published_median(m)
        shift[m["param"]] = (med, ref, (med - ref) / s[:, i].std())
    worst = max(shift, key=lambda k: abs(shift[k][2]))
    print(f"{case}: largest median offset {abs(shift[worst][2]):.2f} posterior standard deviations ({worst})")
    if case != "flat":
        continue
    med, ref, z = shift["hard_gamma_inner"]
    print(f"{case}: nu_inner median {med:.2f}, published {ref:.2f}, shift {z:+.2f} posterior standard deviations")
PYEOF
