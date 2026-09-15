#!/usr/bin/env bash
# Bound-convergence trajectories (referee: "plotting bound convergence over
# iterations").
#
# Runs a few representative instances with trajectory recording on, producing
# one CSV per (instance, algorithm) under --trace-dir:
#
#   <state>_<algo>.cp.csv      ub_relx / lb_relx / b_lower per CP iteration
#   <state>_<algo>.ladmm.csv   penalty, objective and coupling residual per
#                              LADMM iteration
#
# By default one instance per subsystem count (the GHZ family, whose threshold
# 1 - 1/(1 + 2^(m-1)) is known analytically, so the plots can show the true
# value as a reference line) and the three algorithms that iterate.
#
# Usage:
#   bash scripts/experiment_convergence.sh              # paper time limits
#   bash scripts/experiment_convergence.sh -t 300       # quick version
#   STATES="state_133.jl state_13.jl" bash scripts/experiment_convergence.sh
#   bash scripts/experiment_convergence.sh --help
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"

# GHZ_3, GHZ_4, GHZ_5 -- one per size, analytic threshold known
STATES="${STATES:-state_033.jl state_0.jl state_10.jl}"
ALGOS="${ALGOS:-D LDL LD1}"

check_env
echo "=============================================================="
echo " Bound-convergence trajectories"
echo " states:     $STATES"
echo " algorithms: $ALGOS   (CP, IR, LADMM)"
echo " traces:     $TRACE_DIR"
echo "=============================================================="

failed=0
for s in $STATES; do
    for a in $ALGOS; do
        run_job "$s" "$a" || failed=$((failed+1))
    done
done

echo "--------------------------------------------------------------"
if [[ $failed -gt 0 ]]; then
    echo "$failed job(s) FAILED"; exit 1
fi
echo "done. Summarise with:"
echo "  python3 scripts/summarize_traces.py --trace-dir $TRACE_DIR"
