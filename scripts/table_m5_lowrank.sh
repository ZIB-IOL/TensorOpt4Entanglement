#!/usr/bin/env bash
# Results of low-rank approximations for LADMM, m=5 (tab.m5low)
#
# Rows: LADMM_400 … LADMM_900, i.e. the factorisation size r swept over
# 400/500/600/700/800/900 via the LDR0…LDR5 algorithm codes.
#
# Usage: see scripts/table_m5.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"
run_table "m5_lowrank" 5 LDR0 LDR1 LDR2 LDR3 LDR4 LDR5
