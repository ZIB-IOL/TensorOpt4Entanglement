#!/usr/bin/env bash
# Check that this machine can actually run the experiments.
#
#   bash scripts/check_env.sh     inspect the current shell's environment
#   bash runjobs.sh --env         inspect the one the jobs will actually get
#
# The second applies runjobs.sh's Site settings block first, so prefer it when
# you are about to submit; this script on its own sees only your shell.
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
        *)      note "julia $v; Manifest.toml is resolved for 1.11.x, and instantiating"
                printf "        under %s re-resolves it to different package versions\n" "$v"
                if command -v juliaup >/dev/null 2>&1; then
                    printf "        fix: juliaup add 1.11.6 && export JULIA_BIN='julia +1.11.6'\n"
                else
                    printf "        fix: unpack an official 1.11.6 tarball and set JULIA_BIN to it\n"
                    printf "        (no juliaup here, so the 'julia +1.11.6' selector will NOT work)\n"
                fi ;;
    esac
    # a "+version" selector only works through juliaup's shim; a plain binary
    # takes it as a filename, which is a confusing way to find out
    case "$JULIA_BIN" in
        *\ +*) command -v juliaup >/dev/null 2>&1 \
                   || fail "JULIA_BIN uses a '+version' selector but juliaup is not installed;
        a plain julia binary reads it as a filename (SystemError: opening file \"+...\")" ;;
    esac
fi

echo "== project =="
[[ -f Project.toml ]] && pass "Project.toml present" || fail "Project.toml missing - wrong directory?"
n=$(ls benchmark/*.jl 2>/dev/null | wc -l)
[[ $n -gt 0 ]] && pass "$n benchmark instances" || fail "no benchmark instances in benchmark/"
# which depot will actually be used, and does it hold the packages?
if [[ -n "${JULIA_DEPOT_PATH:-}" ]]; then
    d="${JULIA_DEPOT_PATH%%:*}"
    case "$JULIA_DEPOT_PATH" in
        /*) : ;;
        *)  note "JULIA_DEPOT_PATH is relative ('$JULIA_DEPOT_PATH'); it resolves against each job's working directory" ;;
    esac
    case "$JULIA_DEPOT_PATH" in
        *:) note "JULIA_DEPOT_PATH ends in ':', so the default depots are appended and ~/.julia is still used" ;;
    esac
elif [[ -d "$PWD/.julia_depot" ]]; then
    d="$PWD/.julia_depot"
    pass "project-local depot in use: .julia_depot/"
else
    d="$HOME/.julia"
    note "using the default depot $d (create .julia_depot/ here for a project-local one)"
fi
if [[ -d "$d" ]]; then
    [[ -w "$d" ]] && pass "depot writable: $d" || fail "depot not writable: $d"
else
    parent="$(dirname "$d")"
    [[ -w "$parent" ]] && pass "depot $d will be created on first use" \
                       || fail "cannot create depot $d ($parent not writable)"
fi
DEPOT="$d"          # $d is reused below; keep the depot path for later messages
if [[ -d "$d/packages/Mosek" ]]; then
    pass "Mosek package present in the depot"
else
    note "no Mosek package in $d yet - the first run will install it"
fi
for d in results joblists outputs; do
    if [[ -d "$d" ]]; then
        [[ -w "$d" ]] && pass "writable: $d/" || fail "not writable: $d/"
    else
        [[ -w . ]] && pass "$d/ will be created on first use" || fail "cannot create $d/"
    fi
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

# We never set MOSEKBINDIR -- Mosek.jl fetches its own solver. One left over in
# the environment overrides that, and the build fails outright when it does not
# match Mosek.jl's version.
if [[ -n "${MOSEKBINDIR:-}" ]]; then
    want=$(grep -A5 '^\[\[deps.Mosek\]\]$' Manifest.toml 2>/dev/null \
           | grep '^version' | head -1 | cut -d'"' -f2 | cut -d. -f1,2)
    got=$("$MOSEKBINDIR/mosek" 2>/dev/null | grep -oE 'MOSEK Version [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
    if [[ -n "$want" && "$got" != "$want" ]]; then
        fail "MOSEKBINDIR is set to ${MOSEKBINDIR} (MOSEK ${got:-unreadable}), but Mosek.jl needs $want;
        its build will reject it. This project sets no MOSEKBINDIR -- run: unset MOSEKBINDIR"
    else
        note "MOSEKBINDIR is set ($MOSEKBINDIR); Mosek.jl will use it instead of its own copy"
    fi
fi

# A MOSEK path in a shell profile comes back on the next login and breaks the
# build again, long after the variable was unset in this shell.
for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
    [[ -f "$rc" ]] || continue
    if grep -qE '^[^#]*\b(MOSEKBINDIR|MOSEKHOME)=' "$rc" 2>/dev/null; then
        note "$rc sets MOSEKBINDIR/MOSEKHOME; it will return next login and Mosek.jl's
        build rejects any MOSEK that is not its own version - remove it there"
    fi
done

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
        SOLVE_FAIL*)
            msg="${out#SOLVE_FAIL: }"
            fail "$msg"
            if [[ "$msg" == *"not properly installed"* ]]; then
                printf "        Mosek.jl is installed but never built: Pkg.build writes\n"
                printf "        deps/deps.jl, and nothing loads without it. Run:\n"
                printf "          julia --project=. -e 'using Pkg; Pkg.build(\"Mosek\"; verbose=true)'\n"
                printf "        the build DOWNLOADS MOSEK, so this machine needs outbound\n"
                printf "        internet; the reason it failed is in its build.log:\n"
                printf "          %s/scratchspaces/*/*/build.log\n" "${DEPOT:-$HOME/.julia}"
            elif [[ "$msg" == *libmosek* || "$msg" == *"Unable to load"* ]]; then
                printf "        Mosek.jl cannot load the solver it built against. Re-fetch it:\n"
                printf "          julia --project=. -e 'using Pkg; Pkg.build(\"Mosek\"); Pkg.precompile()'\n"
                printf "        (this downloads MOSEK, so the machine needs outbound internet)\n"
            fi ;;
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
