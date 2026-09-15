#!/bin/bash
# Cluster entry point: generate the job lists and submit them to Slurm.
#
# The experiment definitions live in scripts/exp_*.sh -- which instances, which
# algorithms, which time limits -- and this script only turns them into Slurm
# array jobs. There is no second copy of the experiment design here, so the
# cluster runs exactly what a local run would.
#
#   bash runjobs.sh                      # all three experiments
#   bash runjobs.sh main                 # just one
#   bash runjobs.sh --dry-run            # generate lists, submit nothing
#
# Each part's results go to results/<part>/, so the three never collide.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export LC_ALL=C
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-.julia_depot}"
export MOSEKHOME="${MOSEKHOME:-/software/mosek/10.2}"
export MOSEKLM_LICENSE_FILE="${MOSEKLM_LICENSE_FILE:-27007@solice01.zib.de}"

SUBMIT=1
PARTS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) SUBMIT=0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) PARTS+=("$arg") ;;
    esac
done
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=(main lowrank ddps_ablation)

rm -f job_list_*.txt        # never submit a list left over from a previous run

echo "instantiating the project..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || exit 1

total=0
for part in "${PARTS[@]}"; do
    script="scripts/exp_${part}.sh"
    [[ -f "$script" ]] || { echo "unknown part: $part" >&2; exit 2; }
    echo
    echo "### $part ###"
    # --slurm makes the experiment script emit its job list instead of running
    USE_SLURM=1 FORCE=1 bash "$script" || exit 1
done

echo
for list in job_list_*.txt; do
    [[ -f "$list" ]] || continue
    n=$(wc -l < "$list")
    total=$((total + n))
    if [[ $SUBMIT -eq 1 ]]; then
        echo "submitting $list ($n jobs)"
        JOB_LIST="$list" sbatch --array=1-"$n" run.slurm
    else
        echo "would submit: JOB_LIST=$list sbatch --array=1-$n run.slurm"
    fi
done
echo
echo "total jobs: $total"
[[ $SUBMIT -eq 1 ]] || echo "(dry run: nothing submitted)"
