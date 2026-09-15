#!/usr/bin/env bash
# DDPS vs DDPS+ ablation (referee: "is DDPS+ folded into the LMO of CP? do you
# have results on how much it helps IR compared to DPS?").
#
# DDPS+ *is* the relaxation the sBB oracle uses -- initRelaxationNode and
# initRelaxationThreshold call the same strengthenRelaxation. This script runs
# each of the three affected algorithms twice, once with the oracle restricted
# to the plain DDPS outer approximation and once with the McCormick families
# added, so the difference is measured rather than argued:
#
#   RLT  vs RLT_DDPS    the root lower bound
#   D    vs D_DDPS      cutting plane
#   LDL  vs LDL_DDPS    iterative refinement
#
# Defaults to m=3 and m=4; pass -m 5 for the expensive case.
#
# Usage:
#   bash scripts/table_ddps.sh                 # m = 3 and 4
#   ABLATION_M="3" bash scripts/table_ddps.sh  # just m = 3
#   bash scripts/table_ddps.sh --dry-run
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"

ABLATION_M="${ABLATION_M:-3 4}"
failed=0
for m in $ABLATION_M; do
    run_table "ddps${m}" "$m" RLT RLT_DDPS D D_DDPS LDL LDL_DDPS || failed=$((failed+1))
done
[[ $failed -eq 0 ]] || exit 1
echo "generate the comparison with:"
for m in $ABLATION_M; do echo "  python3 scripts/make_tables.py --table ddps${m}"; done
