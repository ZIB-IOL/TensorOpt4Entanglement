#!/bin/bash
#SBATCH --job-name=sdto
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 24G, not 10G: DPS at m=3 uses level 6 and peaks at 12.9 GiB -- the smallest
# physical instance builds the largest SDP here, and at 10G Slurm killed those
# three jobs with no result file and nothing in the log. Alt-SDP at m=5 peaks
# at 8.4 GiB, which left under 2 GiB of headroom for every other job.
#SBATCH --mem=24G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5222 # the paper reports Gold 5122
# Slurm's own output, kept rather than discarded: a job killed for exceeding
# --mem or --time is reported HERE by slurmstepd and nowhere else. The Julia
# log beside the results only has what the process itself printed, so with
# these pointed at /dev/null an OOM kill left no trace anywhere and the cell
# simply looked as though it had never run. stderr merges into this file.
# %A is the array job id, %a the task index.
#SBATCH --output=outputs/slurm-%A_%a.out
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

# Context first, so a truncated or killed job still says what it was and where.
echo "job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname)"
echo "instance=$instance algorithm=$algorithm timelimit=$timelimit"
echo "mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK:-?} started $(date -Is)"

# Log beside the results, exactly where a --local run puts it, so a job's
# output is in one predictable place however it was started. outputs/ is the
# fallback for a job list that carries no results directory.
logfile="outputs/output_${instance}_${algorithm}.out"
if [[ -n "$resultsdir" ]]; then
    export EXACTENT_RESULTS_DIR="$resultsdir"
    export EXACTENT_TRACE="$resultsdir/traces/${instance}_${algorithm}"
    mkdir -p "$resultsdir/traces" "$resultsdir/logs"
    logfile="$resultsdir/logs/${instance}_${algorithm}.log"
else
    mkdir -p outputs
fi

# --compiled-modules=existing: use the caches runjobs.sh warmed, but never write
# new ones. Without it every job in an array recompiles the shared depot at the
# same time and they serialise on each other's pidfiles rather than solving.
# A cache that is missing or stale is then compiled in memory: slower for that
# job, but no contention and no half-written cache files.
echo "$juliabin --project=. scripts/run_experiment.jl -s $instance -a $algorithm -t $timelimit"
# unquoted: JULIA_BIN may be a juliaup selector ("julia +1.11.6"), which has to split
srun $juliabin --compiled-modules=existing --project=. scripts/run_experiment.jl \
     -s "$instance" -a "$algorithm" -t "$timelimit" \
     > "$logfile" 2>&1
rc=$?
echo "finished $(date -Is) rc=$rc; julia output in $logfile"
exit $rc
