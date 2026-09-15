#!/bin/bash
#SBATCH --job-name=sdto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5222 # the paper reports Gold 5122
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#
# Runs one line of a job list produced by scripts/exp_*.sh --slurm.
#
# Works both ways:
#   sbatch --array=1-N run.slurm     (array job; uses SLURM_ARRAY_TASK_ID)
#   sbatch run.slurm 7               (single line, as runjobs.sh used to do)
#
# JOB_LIST selects the list and is required. runjobs.sh writes one timestamped
# list per part per invocation under joblists/, so a resubmission never
# rewrites a list that a pending array is still reading:
#   JOB_LIST=joblists/main-20250115-120000.txt sbatch --array=1-66 run.slurm
#
# Line format, produced by lib.sh:
#   <instance> <algorithm> <timelimit> [<results_dir>]
# The Julia binary is deliberately NOT in the list: it is a property of the
# machine, not of the experiment, so set JULIA_BIN here or in the environment.
# The optional 4th field routes results to that experiment's directory.

LINENUM="${SLURM_ARRAY_TASK_ID:-$1}"
JOB_LIST="${JOB_LIST:-}"

if [[ -z "$LINENUM" ]]; then
    echo "no line selected: pass --array=1-N or a line number" >&2
    exit 2
fi
if [[ -z "$JOB_LIST" ]]; then
    echo "JOB_LIST is not set; runjobs.sh sets it when submitting." >&2
    echo "To run one line by hand: JOB_LIST=joblists/main-<stamp>.txt sbatch run.slurm 7" >&2
    exit 2
fi
if [[ ! -f "$JOB_LIST" ]]; then
    echo "job list not found: $JOB_LIST (generate it with runjobs.sh --dry-run)" >&2
    exit 2
fi

PARAM=$(sed -n "${LINENUM}p" "$JOB_LIST")
[[ -n "$PARAM" ]] || { echo "line $LINENUM not present in $JOB_LIST" >&2; exit 2; }
read -r instance algorithm timelimit resultsdir <<< "$PARAM"
juliabin="${JULIA_BIN:-julia}"

mkdir -p outputs
if [[ -n "$resultsdir" ]]; then
    export EXACTENT_RESULTS_DIR="$resultsdir"
    export EXACTENT_TRACE="$resultsdir/traces/${instance}_${algorithm}"
    mkdir -p "$resultsdir/traces"
fi

echo "$juliabin --project=. scripts/run_experiment.jl -s $instance -a $algorithm -t $timelimit"
# unquoted: JULIA_BIN may be a juliaup selector ("julia +1.11.6"), which has to split
srun $juliabin --project=. scripts/run_experiment.jl \
     -s "$instance" -a "$algorithm" -t "$timelimit" \
     > "outputs/output_${instance}_${algorithm}.out" 2>&1
