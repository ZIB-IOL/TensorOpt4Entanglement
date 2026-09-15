#!/bin/bash
# Single entry point for the experiments.
#
# The experiment design lives in scripts/exp_*.sh -- which instances, which
# algorithms, which time limits. This script only decides WHERE they run, so
# the cluster and a local machine execute exactly the same thing.
#
#   bash runjobs.sh                    submit all experiments to Slurm
#   bash runjobs.sh main               just one experiment
#   bash runjobs.sh --local            run here instead, sequentially
#   bash runjobs.sh --force            redo everything, ignoring existing results
#   bash runjobs.sh --dry-run          show what would happen, do nothing
#   bash runjobs.sh --local main -t 60 forward options to the experiments
#
# Cluster path:
#   runjobs.sh -> scripts/exp_*.sh --slurm -> job_list_<part>.txt -> run.slurm
#
# Each part writes to results/<part>/, so the three never collide.
#
# By default a job whose result file already exists is skipped, in every mode,
# so re-running this script queues only what is still missing -- that is how an
# interrupted or partially failed run is resumed. --force ignores existing
# results and redoes everything.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# One stamp for this invocation: every job list it writes carries it, so a
# resubmission cannot disturb the lists a pending array is still reading.
export JOBLIST_STAMP="$(date +%Y%m%d-%H%M%S)-$$"

# the available parts are whatever experiment scripts exist -- no second list
ALL=()
for f in scripts/exp_*.sh; do
    [[ -e "$f" ]] || continue
    p="${f##*/exp_}"; ALL+=("${p%.sh}")
done

MODE=slurm
PARTS=(); FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)      MODE=local; shift ;;
        --check)      MODE=check; shift ;;
        --dry-run|-n) MODE=dry; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [parts...] [--local|--dry-run|--check] [options]"
            echo "  parts: ${ALL[*]}   (default: all)"
            bash scripts/exp_main.sh --help | sed -n '3,$p'
            exit 0 ;;
        -*) FLAGS+=("$1")
            case "$1" in -t|--time-limit|-s|--state|-a|--algo|--results-dir|--trace-dir|--log-dir|--julia)
                FLAGS+=("$2"); shift ;; esac
            shift ;;
        *)  PARTS+=("$1"); shift ;;
    esac
done
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=("${ALL[@]}")

if [[ "$MODE" == "check" ]]; then
    # Generate this invocation's lists, then audit coverage against *only*
    # those: joblists/ accumulates one set per submission, so globbing all of
    # them would count stale jobs as dispatched.
    for part in "${PARTS[@]}"; do
        # stderr is kept: if an experiment cannot even build its list (no Mosek
        # licence, a bad --algo) the coverage report would otherwise blame the
        # tables for a failure that happened here.
        if ! USE_SLURM=1 bash "scripts/exp_${part}.sh" "${FLAGS[@]+"${FLAGS[@]}"}" >/dev/null; then
            echo "ERROR: could not generate the job list for '$part'; coverage below is meaningless" >&2
            exit 1
        fi
    done
    python3 scripts/check_coverage.py --job-lists "joblists/*-${JOBLIST_STAMP}.txt"
    exit $?
fi

if [[ "$MODE" == "slurm" ]]; then
    export LC_ALL=C
    export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-.julia_depot}"
    export MOSEKHOME="${MOSEKHOME:-/software/mosek/10.2}"
    export MOSEKLM_LICENSE_FILE="${MOSEKLM_LICENSE_FILE:-27007@solice01.zib.de}"
    echo "instantiating the project..."
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || exit 1
fi

started=$(date +%s); failed=()
for part in "${PARTS[@]}"; do
    script="scripts/exp_${part}.sh"
    [[ -f "$script" ]] || { echo "unknown part: $part (known: ${ALL[*]})" >&2; failed+=("$part"); continue; }
    echo; echo "########## $part ##########"
    case "$MODE" in
        local) bash "$script" "${FLAGS[@]+"${FLAGS[@]}"}" || failed+=("$part") ;;
        dry)   bash "$script" --dry-run "${FLAGS[@]+"${FLAGS[@]}"}" || failed+=("$part") ;;
        slurm) USE_SLURM=1 bash "$script" "${FLAGS[@]+"${FLAGS[@]}"}" || failed+=("$part") ;;
    esac
done

if [[ "$MODE" == "slurm" ]]; then
    echo
    total=0
    for list in joblists/*-"$JOBLIST_STAMP".txt; do
        [[ -f "$list" ]] || continue
        n=$(wc -l < "$list")
        if [[ $n -eq 0 ]]; then
            echo "nothing to submit for $(basename "$list"): all results already present"
            rm -f "$list"
            continue
        fi
        total=$((total + n))
        echo "submitting $list ($n jobs)"
        JOB_LIST="$list" sbatch --array=1-"$n" run.slurm || failed+=("$list")
    done
    if [[ $total -eq 0 ]]; then
        echo "nothing submitted: every experiment already has results (use --force to redo)"
    else
        echo "total jobs submitted: $total"
    fi
fi

echo
echo "wall time: $(( ($(date +%s) - started) / 60 )) min"
if [[ ${#failed[@]} -gt 0 ]]; then echo "FAILED: ${failed[*]}"; exit 1; fi
cat <<'MSG'
analysis:
  python3 scripts/make_tables.py --table all --out tables/
  python3 scripts/make_tables.py --manifest --out tables/
  python3 scripts/performance_profile.py --out plots/
  python3 scripts/summarize_traces.py --out plots/
MSG
