#!/usr/bin/env bash
# EXPERIMENT 2 of 3 -- LADMM factorisation-size sweep.
#
# LADMM at r = 400, 500, 600, 700, 800, 900 on the m = 5 instances, testing how
# sensitive the heuristic is to the low-rank approximation. Disjoint from
# exp_main.sh: those are the LDR* algorithm codes, which exp_main does not run.
#
# Produces tab.m5low.  Results -> results/lowrank/
#
# Usage: as exp_main.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"
set_part lowrank

run_table "lowrank_m5" 5 LDR0 LDR1 LDR2 LDR3 LDR4 LDR5 || exit 1
echo
echo "analysis:  python3 scripts/make_tables.py --table m5low"
