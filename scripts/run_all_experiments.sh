#!/usr/bin/env bash
# Run every experiment, in increasing cost order. The three parts are disjoint:
# no instance/algorithm pair is executed twice.
#
#   1  exp_main.sh            all instances x 6 algorithms      ~210 CPU-h
#   2  exp_lowrank.sh         m=5 x LADMM r = 400..900          ~72 CPU-h
#   3  exp_ddps_ablation.sh   m=3,4 x 3 DDPS-only variants      ~44 CPU-h
#
# Each writes into its own results/<part>/ directory. Finished jobs are
# skipped, so an interrupted run can be restarted.
#
# Usage:
#   bash scripts/run_all_experiments.sh                  # everything
#   bash scripts/run_all_experiments.sh main ddps        # a subset
#   bash scripts/run_all_experiments.sh --dry-run
#   bash scripts/run_all_experiments.sh --force
#   bash scripts/run_all_experiments.sh main -t 60       # smoke test
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL=(main lowrank ddps_ablation)
PARTS=(); FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $(basename "$0") [parts...] [options]"
            echo "  parts: ${ALL[*]}   (default: all)"
            bash "$HERE/exp_main.sh" --help | sed -n '3,$p'
            exit 0 ;;
        -*) FLAGS+=("$1")
            case "$1" in -t|--time-limit|--results-dir|--trace-dir|--log-dir|--julia)
                FLAGS+=("$2"); shift ;; esac
            shift ;;
        *)  PARTS+=("$1"); shift ;;
    esac
done
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=("${ALL[@]}")

started=$(date +%s); failed=()
for p in "${PARTS[@]}"; do
    script="$HERE/exp_${p}.sh"
    [[ -f "$script" ]] || { echo "ERROR: unknown part '$p' (known: ${ALL[*]})" >&2; failed+=("$p"); continue; }
    echo; echo "########## $p ##########"
    bash "$script" "${FLAGS[@]+"${FLAGS[@]}"}" || failed+=("$p")
done

echo
echo "=============================================================="
echo " wall time: $(( ($(date +%s) - started) / 60 )) min"
if [[ ${#failed[@]} -gt 0 ]]; then echo " FAILED: ${failed[*]}"; echo "======"; exit 1; fi
cat <<'MSG'
 all experiments complete

 analysis:
   python3 scripts/make_tables.py --table all        # every LaTeX table
   python3 scripts/make_tables.py --manifest         # raw file behind each cell
   python3 scripts/performance_profile.py            # performance profiles
   python3 scripts/summarize_traces.py --out plots/  # convergence trajectories
==============================================================
MSG
