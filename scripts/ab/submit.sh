#!/bin/bash
# Run every cell in cells.txt on BOTH checkouts and collect the output.
#
#   bash scripts/ab/submit.sh                 submit to Slurm
#   bash scripts/ab/submit.sh --local         run here, sequentially
#   bash scripts/ab/submit.sh --cells f.txt   a different cell list
#   bash scripts/ab/submit.sh -t 600          override the time limit
#
# Both sides of a pair get identical resources and, on Slurm, the same node
# constraint -- otherwise a difference could be the hardware rather than the
# code, which is the question being asked.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
LEGACY="${AB_LEGACY:-$REPO/../legacy-ab}"
CELLS="$REPO/scripts/ab/cells.txt"
MODE=slurm; TL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)  MODE=local; shift ;;
        --cells)  CELLS="$2"; shift 2 ;;
        -t)       TL_OVERRIDE="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$LEGACY" ]]; then
    echo "legacy checkout missing: $LEGACY" >&2
    echo "run: bash scripts/ab/setup.sh" >&2
    exit 1
fi
export JULIA_BIN="${JULIA_BIN:-julia}"
mkdir -p ab

stamp=$(date +%Y%m%d-%H%M%S)
n=0
while read -r instance cur leg tl; do
    [[ -z "${instance:-}" || "$instance" == \#* ]] && continue
    [[ -n "$TL_OVERRIDE" ]] && tl="$TL_OVERRIDE"
    cell="${instance%.jl}_${cur}"
    for side in current legacy; do
        if [[ "$side" == legacy ]]; then tree="$LEGACY"; algo="$leg"; else tree="$REPO"; algo="$cur"; fi
        out="$REPO/ab/$stamp/$cell/$side"
        mkdir -p "$out"
        env_args=(AB_TREE="$tree" AB_ENTRY="$side" AB_INSTANCE="$instance" AB_ALGO="$algo" AB_TL="$tl" AB_OUT="$out")
        if [[ "$MODE" == local ]]; then
            echo "== $cell / $side (local)"
            env "${env_args[@]}" bash scripts/ab/run.slurm
        else
            echo "== $cell / $side -> sbatch"
            env "${env_args[@]}" sbatch --export=ALL --job-name="ab-$cell-$side" \
                scripts/ab/run.slurm
        fi
        n=$((n+1))
    done
done < "$CELLS"

echo
echo "$n run(s) dispatched; output under ab/$stamp/"
echo "compare with:  python3 scripts/ab/compare.py ab/$stamp"
