#!/usr/bin/env python3
"""Generate the LaTeX bodies of the paper's tables from saved results.

This is a dispatcher: each table family lives in its own module under
`scripts/tables/`, and this script routes a table name to its owner.

    scripts/tables/main.py        m3, m4, m5     main results
    scripts/tables/lowrank.py     m5low          LADMM rank sweep
    scripts/tables/gapclosing.py  m5cp           CP gap-closing averages
    scripts/tables/ddps.py        ddps3/4/5      DDPS vs DDPS+ ablation
    scripts/tables/memory.py      mem3/4/5       per-level memory
    scripts/tables/size.py        size3/4/5      relaxation size and file size

Usage
    python3 scripts/make_tables.py --table all
    python3 scripts/make_tables.py --table m3 --out tables/
    python3 scripts/make_tables.py --manifest      # raw file behind every cell
    python3 scripts/make_tables.py --list          # tables and their experiments

Results are searched in results/main, results/lowrank, results/ddps, then
results/ itself (the flat files published with the paper), first hit winning,
so a fresh run shadows the published one without deleting it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tables
from tables.common import (ROOT, RESULT_DIRS, TRACE_DIRS, PROVENANCE, Context,
                           load_instances, display_name)


def build(name, ctx):
    """Dispatch one table to the module that owns it."""
    mod, spec = tables.REGISTRY[name]
    if not ctx.states(spec["m"]):
        print(f"WARNING: no instances with N = {spec['m']}", file=sys.stderr)
    return mod.build(name, spec, ctx)


def emit_manifest(wanted, ctx):
    print("# Raw result files behind each table.")
    print("# Regenerate a table with: python3 scripts/make_tables.py --table <name>")
    missing_total = 0
    for name in wanted:
        mod, spec = tables.REGISTRY[name]
        PROVENANCE.clear()
        build(name, ctx)                       # populates PROVENANCE as a side effect
        cells = [(s, a) for s in ctx.states(spec["m"]) for a, _, _ in spec["rows"]]
        found = dict(PROVENANCE)
        missing = [c for c in cells
                   if c not in found and (c[0], c[1] + " [trace]") not in found]
        missing_total += len(missing)
        print(f"\n## table {name}  ({len(cells) - len(missing)}/{len(cells)} cells)"
              f"  [{mod.__name__.split('.')[-1]}, from {spec['experiment']}]")
        for (state, algo), path in sorted(found.items()):
            print(f"  {ctx.name(state):<12} {algo:<14} {path}")
        for state, algo in missing:
            print(f"  {ctx.name(state):<12} {algo:<14} MISSING -> run scripts/{spec['experiment']}")
    if missing_total:
        print(f"\n# {missing_total} cell(s) have no raw file; "
              f"run the experiment named beside each one", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="all", choices=sorted(tables.REGISTRY) + ["all"])
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--results-dir", action="append",
                    help="raw-result directory; repeatable, searched in order "
                         f"(default: {', '.join(os.path.relpath(d, ROOT) for d in RESULT_DIRS)})")
    ap.add_argument("--trace-dir", action="append", help="trajectory directory; repeatable")
    ap.add_argument("--out", help="directory to write <table>.tex into (default: stdout)")
    ap.add_argument("--manifest", action="store_true",
                    help="list the raw file behind every cell instead of emitting LaTeX")
    ap.add_argument("--list", action="store_true",
                    help="list the tables, their module and the experiment that fills them")
    args = ap.parse_args()

    if args.list:
        print(f"{'table':<8}{'module':<13}{'experiment':<26}subsystems")
        for name in sorted(tables.REGISTRY):
            mod, spec = tables.REGISTRY[name]
            print(f"{name:<8}{mod.__name__.split('.')[-1]:<13}"
                  f"{spec['experiment']:<26}m = {spec['m']}")
        return

    instances = load_instances(args.benchmark_dir)
    if not instances:
        sys.exit(f"no benchmark instances found in {args.benchmark_dir}")
    ctx = Context(instances, args.results_dir or RESULT_DIRS, args.trace_dir or TRACE_DIRS)

    wanted = sorted(tables.REGISTRY) if args.table == "all" else [args.table]

    if args.manifest:
        emit_manifest(wanted, ctx)
        return

    for name in wanted:
        body = build(name, ctx)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, f"{name}.tex")
            open(path, "w").write(body + "\n")
            print(f"wrote {path}")
        else:
            print(f"% ---------- table {name} ----------")
            print(body)
            print()


if __name__ == "__main__":
    main()
