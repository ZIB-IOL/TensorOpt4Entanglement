#!/usr/bin/env bash
# EXPERIMENT 3 of 3 -- DDPS vs DDPS+ ablation of the sBB oracle.
#
# DDPS+ is not "folded into" the LMO: it IS the LMO's relaxation --
# initRelaxationNode and initRelaxationThreshold call the same
# strengthenRelaxation. This experiment re-runs the three affected algorithms
# with the oracle restricted to the plain DDPS outer approximation, so the
# contribution of the McCormick families is measured rather than argued.
#
# Only the DDPS variants are run here; their DDPS+ counterparts come from
# exp_main.sh, so the two experiments do not overlap.
#
#   RLT_DDPS  vs RLT     root lower bound
#   D_DDPS    vs D       cutting plane
#   LDL_DDPS  vs LDL     iterative refinement
#
# Produces tab.ddps3 / ddps4 / ddps5.  Results -> results/ddps/
#
# Usage:
#   bash scripts/exp_ddps_ablation.sh          # m = 3 and 4
#   M="3 4 5" bash scripts/exp_ddps_ablation.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"
set_part ddps

M="${M:-3 4}"
failed=0
for m in $M; do
    run_table "ddps_m${m}" "$m" RLT_DDPS D_DDPS LDL_DDPS || failed=$((failed+1))
done
[[ $failed -eq 0 ]] || exit 1
echo
echo "analysis:  python3 scripts/make_tables.py --table ddps3"
