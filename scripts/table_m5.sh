#!/usr/bin/env bash
# Experimental results for m=5 (tab.m5)
#
# Rows: Alt-SDP (A), LADMM (LD1), CP (D), IR (LDL), DPS (PPT), DDPS+ (RLT).
# The PDGR row in the paper comes from an external implementation
# (liu2025unified / FrankWolfe.jl) and is not produced here.
#
# Usage:
#   bash scripts/table_m5.sh              # run locally, skipping finished jobs
#   DRY_RUN=1 bash scripts/table_m5.sh    # list the jobs only
#   FORCE=1   bash scripts/table_m5.sh    # re-run everything
#   USE_SLURM=1 bash scripts/table_m5.sh  # emit a job list for run.slurm
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_args "$@"
run_table "m5" 5 A LD1 D LDL PPT RLT
