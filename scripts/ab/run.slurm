#!/bin/bash
#SBATCH --job-name=ab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5222
#SBATCH --output=ab/slurm-%A_%a.out
#
# One side of one comparison. Driven by submit.sh, which passes everything in
# the environment so both sides are launched identically:
#
#   AB_TREE      checkout to run in
#   AB_ENTRY     legacy | current   (legacy has no CLI of its own)
#   AB_INSTANCE  AB_ALGO  AB_TL
#   AB_OUT       directory for the log, result and trace
set -uo pipefail
: "${JULIA_BIN:=julia}"
mkdir -p "$AB_OUT"

echo "side=$AB_ENTRY tree=$AB_TREE node=$(hostname) started $(date -Is)"
echo "instance=$AB_INSTANCE algo=$AB_ALGO timelimit=$AB_TL"
# Record the node precisely: if two sides land on different CPUs the comparison
# measures hardware, which is the thing being tested for.
grep -m1 'model name' /proc/cpuinfo || true

cd "$AB_TREE" || exit 2
# Trajectories: this branch writes CSV traces, legacy has none, so its
# per-iteration numbers are recovered from stdout by compare.py. Both sides'
# stdout is therefore kept in full.
export EXACTENT_RESULTS_DIR="$AB_OUT"
export EXACTENT_TRACE="$AB_OUT/trace"

# srun only inside an allocation, so submit.sh --local runs the same script
launch=()
[[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1 && launch=(srun)

if [[ "$AB_ENTRY" == "legacy" ]]; then
    "${launch[@]}" $JULIA_BIN --project=. src/main.jl \
         -s "$AB_INSTANCE" -a "$AB_ALGO" -t "$AB_TL" > "$AB_OUT/run.log" 2>&1
else
    "${launch[@]}" $JULIA_BIN --project=. scripts/run_experiment.jl \
         -s "$AB_INSTANCE" -a "$AB_ALGO" -t "$AB_TL" > "$AB_OUT/run.log" 2>&1
fi
rc=$?
echo "finished $(date -Is) rc=$rc"
# Legacy writes into its own results/ tree; bring it where compare.py looks.
[[ "$AB_ENTRY" == "legacy" ]] && \
    cp -f "$AB_TREE/results/${AB_INSTANCE}_${AB_ALGO}" "$AB_OUT/" 2>/dev/null
exit $rc
