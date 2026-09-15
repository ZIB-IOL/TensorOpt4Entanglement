#!/usr/bin/env bash
# Check that this machine can actually run the experiments.
#
#   bash scripts/check_env.sh
#
# Reports every problem it finds rather than stopping at the first, and exits
# non-zero if anything would prevent a run. Safe to run anywhere: it solves one
# tiny LP but starts no experiment and writes nothing outside a temp file.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ok=0; bad=0; warn=0
pass() { printf "  \033[32mok\033[0m    %s\n" "$1"; ok=$((ok+1)); }
fail() { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; bad=$((bad+1)); }
note() { printf "  \033[33mwarn\033[0m  %s\n" "$1"; warn=$((warn+1)); }

echo "== host =="
printf "  %s, %s cores, %s GiB RAM\n" "$(hostname)" "$(nproc)" "$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
printf "  repo: %s\n" "$PWD"
printf "  commit: %s\n" "$(git rev-parse --short HEAD 2>/dev/null || echo '(not a git checkout)')"

echo "== julia =="
JULIA_BIN="${JULIA_BIN:-julia}"
if ! command -v "${JULIA_BIN%% *}" >/dev/null 2>&1; then
    fail "julia not on PATH (set JULIA_BIN)"
else
    v=$($JULIA_BIN --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    case "$v" in
        1.11.*) pass "julia $v" ;;
        "")     fail "could not determine the julia version" ;;
        *)      note "julia $v; the project is pinned to 1.11.x (README). 1.12+ changes package resolution" ;;
    esac
fi

echo "== project =="
[[ -f Project.toml ]] && pass "Project.toml present" || fail "Project.toml missing - wrong directory?"
n=$(ls benchmark/*.jl 2>/dev/null | wc -l)
[[ $n -gt 0 ]] && pass "$n benchmark instances" || fail "no benchmark instances in benchmark/"
if [[ -n "${JULIA_DEPOT_PATH:-}" ]]; then
    d="${JULIA_DEPOT_PATH%%:*}"
    mkdir -p "$d" 2>/dev/null && [[ -w "$d" ]] && pass "depot writable: $d" || fail "depot not writable: $d"
fi
for d in results joblists outputs; do
    mkdir -p "$d" 2>/dev/null && [[ -w "$d" ]] && pass "writable: $d/" || fail "not writable: $d/"
done
avail=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
[[ -n "$avail" && "$avail" -ge 5 ]] && pass "${avail} GiB free" || note "only ${avail:-?} GiB free"

echo "== mosek =="
if [[ -z "${MOSEKLM_LICENSE_FILE:-}" && ! -f "$HOME/mosek/mosek.lic" ]]; then
    fail "no licence: set MOSEKLM_LICENSE_FILE (a file, or port@host) or place ~/mosek/mosek.lic"
else
    src="${MOSEKLM_LICENSE_FILE:-$HOME/mosek/mosek.lic}"
    case "$src" in
        *@*) pass "licence server: $src" ;;
        *)   [[ -r "$src" ]] && pass "licence file: $src" || fail "licence file unreadable: $src" ;;
    esac
fi

echo "== a real solve =="
if command -v "${JULIA_BIN%% *}" >/dev/null 2>&1; then
    # `using` must be a top-level statement of its own: a macro like @variable
    # inside the same block is resolved before the using has run.
    probe=$($JULIA_BIN --project=. -e '
        using JuMP, MosekTools
        function probe()
            m = Model(Mosek.Optimizer); set_silent(m)
            @variable(m, x >= 1.5); @objective(m, Min, x); optimize!(m)
            return termination_status(m) == OPTIMAL && abs(objective_value(m) - 1.5) < 1e-9
        end
        try
            println(probe() ? "SOLVE_OK" : "SOLVE_WRONG")
        catch e
            println("SOLVE_FAIL: ", first(sprint(showerror, e), 200))
        end' 2>&1)
    out=$(echo "$probe" | grep -E '^SOLVE_' | tail -1)
    [[ -n "$out" ]] || out="SOLVE_FAIL: $(echo "$probe" | grep -E 'ERROR|error' | head -1)"
    case "$out" in
        SOLVE_OK)   pass "Mosek solved a test LP" ;;
        SOLVE_FAIL*) fail "${out#SOLVE_FAIL: }" ;;
        SOLVE_WRONG) fail "Mosek ran but returned the wrong answer" ;;
        *)          fail "could not test the solver: $out" ;;
    esac
fi

echo "== slurm =="
if command -v sbatch >/dev/null 2>&1; then
    pass "sbatch present"
    part=$(grep -oP '^#SBATCH --partition=\K\S+' run.slurm)
    cons=$(grep -oP '^#SBATCH --constraint=\K\S+' run.slurm)
    if command -v sinfo >/dev/null 2>&1; then
        sinfo -h -p "$part" >/dev/null 2>&1 && [[ -n "$(sinfo -h -p "$part" 2>/dev/null)" ]] \
            && pass "partition '$part' exists" || fail "partition '$part' not found (edit run.slurm)"
        if [[ -n "$(sinfo -h -p "$part" -o '%f' 2>/dev/null | tr ',' '\n' | grep -Fx "$cons")" ]]; then
            pass "constraint '$cons' offered by '$part'"
        else
            note "no node in '$part' advertises feature '$cons'; jobs may queue forever"
        fi
    fi
    lim=$(scontrol show config 2>/dev/null | grep -oP 'MaxArraySize\s*=\s*\K[0-9]+')
    [[ -z "$lim" ]] || { [[ $lim -ge 66 ]] && pass "MaxArraySize $lim (need 66)" || fail "MaxArraySize $lim < 66"; }
else
    note "no sbatch here - use 'bash runjobs.sh --local' (this is not a cluster node)"
fi

echo
echo "  $ok ok, $warn warning(s), $bad problem(s)"
[[ $bad -eq 0 ]] && echo "  environment looks ready" || echo "  fix the problems above before submitting"
exit $(( bad > 0 ))
