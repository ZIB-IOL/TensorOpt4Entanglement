#!/usr/bin/env bash
# Average results on gap-closing CP iterations, m=5 (tab.m5CP)
#
# Rows: CP (D) and IR (LDL). Unlike the other tables this one averages over the
# CP iterations that run *after* the algorithm enters its gap-closing phase, so
# it is built from the per-iteration traces (results/traces/*.csv) rather than
# from the final values in results/.
#
# Those traces are written by any run of D or LDL, so if you have already run
# scripts/table_m5.sh this script has nothing left to do.
#
# Usage: see scripts/table_m5.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run_table "m5_gapclosing" 5 D LDL
missing=0
while read -r s; do
  for a in D LDL; do
    [[ -s "$TRACE_DIR/${s}_${a}.csv" ]] || { echo "  WARNING: no trace for $s $a" >&2; missing=$((missing+1)); }
  done
done < <(instances_with_m 5)
[[ $missing -eq 0 ]] || echo "NOTE: $missing trace file(s) missing; tab.m5CP will be incomplete." >&2
