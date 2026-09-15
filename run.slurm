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
# JOB_LIST selects the list (default job_list.txt), so the three experiments
# can each have their own without colliding:
#   JOB_LIST=job_list_main.txt sbatch --array=1-66 run.slurm
#
# Line format, produced by lib.sh:
#   <instance> <algorithm> <timelimit> [<results_dir>]
# The Julia binary is deliberately NOT in the list: it is a property of the
# machine, not of the experiment, so set JULIA_BIN here or in the environment.
# The optional 4th field routes results to that experiment's directory.

LINENUM="${SLURM_ARRAY_TASK_ID:-$1}"
JOB_LIST="${JOB_LIST:-job_list.txt}"

if [[ -z "$LINENUM" ]]; then
    echo "no line selected: pass --array=1-N or a line number" >&2
    exit 2
fi
if [[ ! -f "$JOB_LIST" ]]; then
    echo "job list not found: $JOB_LIST (generate it with scripts/exp_*.sh --slurm)" >&2
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
srun "$juliabin" --project=. scripts/run_experiment.jl \
     -s "$instance" -a "$algorithm" -t "$timelimit" \
     > "outputs/output_${instance}_${algorithm}.out" 2>&1
