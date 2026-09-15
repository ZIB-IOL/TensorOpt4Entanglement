#!/usr/bin/env bash
# EXPERIMENT 1 of 3 -- the main benchmark.
#
# Every instance x the six algorithms compared in the paper, at the paper's
# per-size time limits (1/2/3 h for m = 3/4/5).
#
#   A    Alt-SDP     LD1  LADMM      D    CP
#   LDL  IR          PPT  DPS        RLT  DDPS+
#
# This single run is the raw material for five of the paper's tables. It does
# not overlap with the other two experiments.
#
#   tab.m3 / tab.m4 / tab.m5   final bounds
#   tab.m5CP                   averaged over the CP trajectories recorded here
#   memory / size tables       diagnostics recorded here
#   performance profiles       computed from these results
#   convergence plots          from the trajectories recorded here
#
# Results -> results/main/{,traces/,logs/}
#
# Usage:
#   bash scripts/exp_main.sh                 # all sizes
#   M="3" bash scripts/exp_main.sh           # one size
#   bash scripts/exp_main.sh --dry-run
#   bash scripts/exp_main.sh --force         # redo everything
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"
set_part main

M="${M:-3 4 5}"
failed=0
for m in $M; do
    run_table "main_m${m}" "$m" Alt-SDP LADMM CP IR DPS DDPS+ || failed=$((failed+1))
done
[[ $failed -eq 0 ]] || exit 1
echo
echo "analysis:"
echo "  python3 scripts/make_tables.py --table m3   (also m4, m5, m5cp, mem3.., size3..)"
echo "  python3 scripts/performance_profile.py"
echo "  python3 scripts/summarize_traces.py"
