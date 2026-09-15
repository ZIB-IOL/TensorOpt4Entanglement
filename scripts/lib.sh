#!/usr/bin/env bash
# Shared helpers for the per-table reproduction scripts.
#
# Every table script sources this file, then calls `run_table` with the
# subsystem count, the algorithm list and a table name.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${BENCHMARK_DIR:-$REPO_ROOT/benchmark}"

# Each experiment writes into its own directory, so the parts never overwrite
# one another and every raw file is attributable to the run that produced it:
#
#   results/main/     the main benchmark      (exp_main.sh)
#   results/lowrank/  the LADMM rank sweep    (exp_lowrank.sh)
#   results/ddps/     the DDPS ablation       (exp_ddps_ablation.sh)
#
# results/ itself still holds the flat files published with the paper.
# `set_part <name>` is called by each experiment script before running.
PART="${PART:-}"
declare -A _JOBLIST_STARTED
set_part() {
    PART="$1"
    RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$PART}"
    TRACE_DIR="${TRACE_DIR:-$RESULTS_DIR/traces}"
    LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs}"
}
RESULTS_DIR="${RESULTS_DIR:-}"
TRACE_DIR="${TRACE_DIR:-}"
LOG_DIR="${LOG_DIR:-}"

# -t -1 lets the code pick the paper's per-size limit (1/2/3 h for m=3/4/5).
TIME_LIMIT="${TIME_LIMIT:--1}"

DRY_RUN="${DRY_RUN:-0}"     # 1 = print the jobs, run nothing
FORCE="${FORCE:-0}"         # 1 = re-run even if a result file exists
USE_SLURM="${USE_SLURM:-0}" # 1 = emit a job list for run.slurm instead of running

# --- argument parsing -----------------------------------------------------
# Every table script accepts the same flags; the environment variables above
# remain available and the flags simply override them.
usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

  -f, --force            re-run every job, even if a result file exists
  -n, --dry-run          list the jobs that would run, run nothing
  -t, --time-limit SEC   per-job time limit (-1 = the paper's limit for this m)
      --slurm            write a job list for run.slurm instead of running
      --results-dir DIR  where result files go        (default results/)
      --trace-dir DIR    where trajectory CSVs go     (default results/traces/)
      --log-dir DIR      where per-job logs go        (default results/logs/)
      --julia CMD        julia command to use         (default: julia +1.11.6)
  -h, --help             this message

Equivalent environment variables: FORCE, DRY_RUN, TIME_LIMIT, USE_SLURM,
RESULTS_DIR, TRACE_DIR, LOG_DIR, JULIA_BIN.
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--force)       FORCE=1; shift ;;
            -n|--dry-run)     DRY_RUN=1; shift ;;
            -t|--time-limit)  TIME_LIMIT="$2"; shift 2 ;;
            --slurm)          USE_SLURM=1; shift ;;
            --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
            --trace-dir)      TRACE_DIR="$2"; shift 2 ;;
            --log-dir)        LOG_DIR="$2"; shift 2 ;;
            --julia)          JULIA_BIN="$2"; shift 2 ;;
            -h|--help)        usage; exit 0 ;;
            *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        esac
    done
}

# --- julia ----------------------------------------------------------------
pick_julia() {
    if [[ -n "${JULIA_BIN:-}" ]]; then echo "$JULIA_BIN"; return; fi
    # the project pins Julia 1.11.x; prefer it when juliaup can provide it
    if command -v juliaup >/dev/null 2>&1 && juliaup status 2>/dev/null | grep -q "1.11.6"; then
        echo "julia +1.11.6"; return
    fi
    echo "julia"
}


check_env() {
    JULIA="$(pick_julia)"
    if ! command -v ${JULIA%% *} >/dev/null 2>&1; then
        echo "ERROR: julia not found on PATH (set JULIA_BIN)." >&2; exit 1
    fi
    local v; v="$($JULIA --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
    case "$v" in
        1.11.*) ;;
        *) echo "WARNING: using Julia $v; this project is pinned to 1.11.x (README)." >&2 ;;
    esac
    if [[ -z "${MOSEKLM_LICENSE_FILE:-}" && ! -f "$HOME/mosek/mosek.lic" ]]; then
        echo "ERROR: no Mosek licence found." >&2
        echo "       set MOSEKLM_LICENSE_FILE=/path/to/mosek.lic, or place it at ~/mosek/mosek.lic" >&2
        exit 1
    fi
    mkdir -p "$RESULTS_DIR" "$TRACE_DIR" "$LOG_DIR"
}

# --- instance selection ---------------------------------------------------
# Instances are chosen by their declared subsystem count, not by filename, so
# adding a benchmark automatically joins the right table.
instances_with_m() {
    local m="$1" f
    for f in "$BENCHMARK_DIR"/*.jl; do
        [[ -f "$f" ]] || continue
        if grep -qE "^N[[:space:]]*=[[:space:]]*$m[[:space:]]*$" "$f"; then
            basename "$f"
        fi
    done | sort
}

state_name() { grep -m1 '^name' "$BENCHMARK_DIR/$1" | sed 's/.*=[[:space:]]*//; s/"//g'; }

# --- one job --------------------------------------------------------------
run_job() {
    local state="$1" algo="$2"
    local out="$RESULTS_DIR/${state}_${algo}"
    if [[ -f "$out" && "$FORCE" != "1" ]]; then
        echo "  skip   $state $algo (result exists; FORCE=1 to redo)"
        return 0
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  would run: $JULIA --project=. scripts/run_experiment.jl -s $state -a $algo -t $TIME_LIMIT"
        return 0
    fi
    local trace="$TRACE_DIR/${state}_${algo}"
    rm -f "$trace".*.csv
    echo "  run    $state $algo"
    local start; start=$(date +%s)
    ( cd "$REPO_ROOT" && EXACTENT_TRACE="$trace" EXACTENT_RESULTS_DIR="$RESULTS_DIR" \
        $JULIA --project=. scripts/run_experiment.jl -s "$state" -a "$algo" -t "$TIME_LIMIT" \
        > "$LOG_DIR/${state}_${algo}.log" 2>&1 )
    local rc=$? dur=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then
        echo "  FAIL   $state $algo (exit $rc, ${dur}s) -- see $LOG_DIR/${state}_${algo}.log" >&2
        return 1
    fi
    echo "  ok     $state $algo (${dur}s)"
}

# --- one table ------------------------------------------------------------
# run_table <table-name> <m> <algo...>
run_table() {
    local table="$1" m="$2"; shift 2
    local algos=("$@")
    check_env
    local states; mapfile -t states < <(instances_with_m "$m")
    if [[ ${#states[@]} -eq 0 ]]; then
        echo "ERROR: no benchmark instances with N = $m in $BENCHMARK_DIR" >&2; exit 1
    fi

    echo "=============================================================="
    echo " Table: $table   (m = $m)"
    echo " states:     ${#states[@]}  -> $(for s in "${states[@]}"; do printf '%s ' "$(state_name "$s")"; done)"
    echo " algorithms: ${algos[*]}"
    echo " time limit: $TIME_LIMIT  (-1 = paper default for this m)"
    echo " results:    $RESULTS_DIR"
    echo "=============================================================="

    if [[ "$USE_SLURM" == "1" ]]; then
        # One list per experiment part, in the format run.slurm parses. The
        # fifth field routes results to this part's directory.
        local joblist="$REPO_ROOT/job_list_${PART:-$table}.txt"
        # An experiment calls run_table once per subsystem count, so truncate
        # only on the first call in this process and append afterwards --
        # otherwise m=4 would overwrite the m=3 jobs.
        if [[ -z "${_JOBLIST_STARTED[$joblist]:-}" ]]; then
            : > "$joblist"
            _JOBLIST_STARTED[$joblist]=1
        fi
        for s in "${states[@]}"; do for a in "${algos[@]}"; do
            echo "$s $a $TIME_LIMIT $RESULTS_DIR" >> "$joblist"
        done; done
        local n; n=$(wc -l < "$joblist")
        echo "wrote $joblist ($n jobs)"
        echo "submit with:"
        echo "  JOB_LIST=$(basename "$joblist") sbatch --array=1-$n run.slurm"
        return 0
    fi

    local failed=0
    for s in "${states[@]}"; do
        for a in "${algos[@]}"; do
            run_job "$s" "$a" || failed=$((failed+1))
        done
    done
    echo "--------------------------------------------------------------"
    if [[ $failed -gt 0 ]]; then
        echo "Table $table: $failed job(s) FAILED"; return 1
    fi
    echo "Table $table: all jobs complete"
}
