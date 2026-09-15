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
#   bash runjobs.sh --env              check the site settings below resolve
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

# ---------------------------------------------------------------------------
# Site settings -- edit these for your machine.
#
# Everything the runs need is set here and nowhere else. An explicit value
# from the environment always wins, so you can override one without editing:
#     MOSEKBINDIR=/other/path bash runjobs.sh
#
# Check that they resolve on this machine with:  bash runjobs.sh --env
# ---------------------------------------------------------------------------

export LC_ALL=C

# Julia executable. On a cluster this is usually just "julia" from the module.
export JULIA_BIN="${JULIA_BIN:-julia}"

# MOSEK solver library. Mosek.jl reads MOSEKBINDIR at BUILD time and bakes the
# path into deps.jl, so it must point at the bin directory of a MOSEK whose
# major.minor matches the Mosek.jl version (10.2 for this project's Manifest).
MOSEKBINDIR_DEFAULT="/software/mosek/10.2/tools/platform/linux64x86/bin"
# an explicit value always wins; the default only applies where it exists,
# since pointing Mosek.jl at a missing directory is worse than saying nothing.
# ./.mosek_bin, if scripts/fix_execstack.py made one, wins over the site copy:
# it is the same libraries with a marking the loader refuses cleared.
if [[ -n "${MOSEKBINDIR:-}" ]];          then export MOSEKBINDIR
elif [[ -d "$PWD/.mosek_bin" ]];         then export MOSEKBINDIR="$PWD/.mosek_bin"
elif [[ -d "$MOSEKBINDIR_DEFAULT" ]];    then export MOSEKBINDIR="$MOSEKBINDIR_DEFAULT"
fi
# libmosek64 links libtbb and friends, which MOSEK ships in this same directory.
# A site install usually carries no RPATH, so dlopen finds them only here --
# without this the build succeeds and `using Mosek` then fails to load.
if [[ -n "${MOSEKBINDIR:-}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$MOSEKBINDIR:"*) ;;
        *) export LD_LIBRARY_PATH="$MOSEKBINDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

# MOSEK licence: a file path, or port@host for a floating licence server.
# ~/mosek/mosek.lic wins over the default, so a laptop needs no edit here.
MOSEKLM_LICENSE_FILE_DEFAULT="27007@solice01.zib.de"
if [[ -n "${MOSEKLM_LICENSE_FILE:-}" ]];  then export MOSEKLM_LICENSE_FILE
elif [[ ! -f "$HOME/mosek/mosek.lic" ]];  then export MOSEKLM_LICENSE_FILE="$MOSEKLM_LICENSE_FILE_DEFAULT"
fi

# Julia package depot. A project-local depot keeps packages on the same
# filesystem as the repo and isolates the run from a shared ~/.julia; it is
# opt-in, so `mkdir .julia_depot` once to use it and "" to force ~/.julia.
# A relative path is made absolute below: it would otherwise resolve against
# each job's working directory rather than the repo.
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$([[ -d "$PWD/.julia_depot" ]] && echo "$PWD/.julia_depot")}"
case "$JULIA_DEPOT_PATH" in
    "")   unset JULIA_DEPOT_PATH ;;                    # use ~/.julia
    /*|*:*) export JULIA_DEPOT_PATH ;;                 # already absolute
    *)    export JULIA_DEPOT_PATH="$PWD/$JULIA_DEPOT_PATH" ;;
esac

# Kept for site scripts that read it; Mosek.jl itself does not.
MOSEKHOME_DEFAULT="/software/mosek/10.2"
if [[ -n "${MOSEKHOME:-}" ]];        then export MOSEKHOME
elif [[ -d "$MOSEKHOME_DEFAULT" ]];  then export MOSEKHOME="$MOSEKHOME_DEFAULT"
fi

# ---------------------------------------------------------------------------

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
        --env)        MODE=env; shift ;;
        --dry-run|-n) MODE=dry; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [parts...] [--local|--dry-run|--check|--env] [options]"
            echo "  parts: ${ALL[*]}   (default: all)"
            bash scripts/exp_main.sh --help | sed -n '3,$p'
            exit 0 ;;
        -*) FLAGS+=("$1")
            case "$1" in -t|--time-limit|-s|--state|-a|--algo|-m|--size|--results-dir|--trace-dir|--log-dir|--julia)
                FLAGS+=("$2"); shift ;; esac
            shift ;;
        *)  PARTS+=("$1"); shift ;;
    esac
done
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=("${ALL[@]}")

# --env validates the environment these settings actually produce, which is
# not the same as the caller's ambient one.
if [[ "$MODE" == "env" ]]; then
    exec bash scripts/check_env.sh
fi

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

# Only a real submission needs the packages resolved up front; --dry-run and
# --check must not pay for a full instantiate/precompile.
if [[ "$MODE" == "slurm" ]]; then
    echo "instantiating the project..."
    "$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || exit 1
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
