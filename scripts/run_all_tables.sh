#!/usr/bin/env bash
# Reproduce every experimental table in the paper, in increasing cost order.
#
#   tab.m3      scripts/table_m3.sh             3 states x 6 algos, 1 h limit
#   tab.m4      scripts/table_m4.sh             4 states x 6 algos, 2 h limit
#   tab.m5      scripts/table_m5.sh             4 states x 6 algos, 3 h limit
#   tab.m5low   scripts/table_m5_lowrank.sh     4 states x 6 ranks, 3 h limit
#   tab.m5CP    scripts/table_m5_gapclosing.sh  reuses the m5 CP/IR traces
#
# WARNING: run end to end this is on the order of 200 CPU-hours. Use
# USE_SLURM=1 on a cluster, or TIME_LIMIT=<seconds> for a quick smoke test.
#
# Usage:
#   bash scripts/run_all_tables.sh                     # everything, sequentially
#   bash scripts/run_all_tables.sh m3 m5               # only these tables
#   bash scripts/run_all_tables.sh --dry-run           # list all jobs
#   bash scripts/run_all_tables.sh --force             # redo every experiment
#   bash scripts/run_all_tables.sh m3 -t 60            # fast smoke test
#   bash scripts/run_all_tables.sh --slurm             # emit job lists
#
# Finished jobs are skipped, so an interrupted run can simply be restarted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_TABLES=(m3 m4 m5 m5_lowrank m5_gapclosing)

# split bare table names from flags; flags are forwarded to every table script
TABLES=(); FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $(basename "$0") [tables...] [options]"
            echo "  tables: ${ALL_TABLES[*]}   (default: all)"
            echo "  options are forwarded to each table script:"
            bash "$HERE/table_m3.sh" --help | sed -n '3,$p'
            exit 0 ;;
        -*) FLAGS+=("$1")
            # these flags take a value
            case "$1" in -t|--time-limit|--results-dir|--trace-dir|--log-dir|--julia)
                FLAGS+=("$2"); shift ;; esac
            shift ;;
        *)  TABLES+=("$1"); shift ;;
    esac
done
[[ ${#TABLES[@]} -eq 0 ]] && TABLES=("${ALL_TABLES[@]}")

started=$(date +%s)
declare -a failed=()
for t in "${TABLES[@]}"; do
    script="$HERE/table_${t}.sh"
    if [[ ! -f "$script" ]]; then
        echo "ERROR: unknown table '$t' (known: ${ALL_TABLES[*]})" >&2
        failed+=("$t"); continue
    fi
    echo; echo "### $t ###"
    bash "$script" "${FLAGS[@]+"${FLAGS[@]}"}" || failed+=("$t")
done

echo
echo "=============================================================="
echo " total wall time: $(( ($(date +%s) - started) / 60 )) min"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo " FAILED tables: ${failed[*]}"
    echo "=============================================================="
    exit 1
fi
echo " all requested tables complete"
echo
echo " generate the LaTeX with:"
echo "   python3 scripts/make_tables.py --table all"
echo "=============================================================="
