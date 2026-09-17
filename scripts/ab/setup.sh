#!/bin/bash
# Prepare a legacy checkout that differs from this branch ONLY in source code.
#
#   bash scripts/ab/setup.sh [worktree-dir]
#
# The comparison is meaningless unless both trees solve with the same solver
# and the same JuMP, so this copies THIS branch's Manifest.toml into the legacy
# worktree and instantiates it there against the same depot. Legacy pins
# Mosek 10.2 and an older JuMP of its own; left alone it would differ in the
# solver as well as the code, and a divergence could not be attributed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
DEST="${1:-$REPO/../legacy-ab}"
: "${JULIA_BIN:=julia}"

if [[ -d "$DEST/.git" || -f "$DEST/.git" ]]; then
    echo "reusing existing worktree: $DEST"
else
    git worktree add "$DEST" origin/legacy || exit 1
fi

# Legacy lists Test as a direct dependency and this branch keeps it in
# [extras]; everything else matches, so the manifest transfers cleanly.
cp "$REPO/Manifest.toml" "$DEST/Manifest.toml"
python3 - "$DEST/Project.toml" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
# take the solver pins out of legacy's [compat] so the copied manifest stands
s = re.sub(r'^Mosek\s*=\s*"10\.2\.0"\s*$', 'Mosek = "11"', s, flags=re.M)
open(p, "w").write(s)
PY

echo "instantiating the legacy worktree against the shared depot..."
( cd "$DEST" && $JULIA_BIN --project=. -e '
    using Pkg
    Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()' ) || exit 1

echo
echo "== versions that will be compared =="
for tree in "$REPO" "$DEST"; do
    printf '%-14s ' "$(basename "$tree")"
    ( cd "$tree" && $JULIA_BIN --project=. -e '
        using Pkg
        v = Dict(e.name => string(e.version) for (_, e) in Pkg.dependencies()
                 if e.name in ("JuMP","MathOptInterface","Mosek","MosekTools"))
        println(join(["$k $(v[k])" for k in sort(collect(keys(v)))], "  "))' 2>/dev/null )
done
echo
echo "if those two lines differ, STOP -- the comparison would confound code with solver."
