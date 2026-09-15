#!/usr/bin/env python3
"""Check that the generated job lists cover every table cell.

    bash runjobs.sh --check
    python3 scripts/check_coverage.py

For each table, compares the cells it needs against the (instance, algorithm)
pairs the job lists will dispatch, plus anything already on disk from an
earlier run. Exits non-zero if any cell would be left without data, so a
missing experiment is caught before the cluster time is spent rather than
after.
"""
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables
from tables.common import ROOT, RESULT_DIRS, Context, load_instances


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--job-lists", default=os.path.join(ROOT, "joblists", "*.txt"))
    args = ap.parse_args()

    ctx = Context(load_instances(args.benchmark_dir))

    dispatched = set()
    lists = sorted(glob.glob(args.job_lists))
    for f in lists:
        for line in open(f):
            p = line.split()
            if len(p) >= 2:
                dispatched.add((p[0], p[1]))
    if not lists:
        print("no job lists found; generate them with: bash runjobs.sh --dry-run",
              file=sys.stderr)

    print(f"job lists: {len(lists)} file(s), {len(dispatched)} (instance, algorithm) pairs\n")
    print(f"{'table':<8}{'needed':>7}{'dispatch':>10}{'on disk':>9}{'MISSING':>9}  missing algorithms")
    print("-" * 78)
    missing_total = 0
    for name in sorted(tables.REGISTRY):
        _, spec = tables.REGISTRY[name]
        cells = [(s, a) for s in ctx.states(spec["m"]) for a, _, _ in spec["rows"]]
        disp = [c for c in cells if c in dispatched]
        disk = [c for c in cells if c not in dispatched and
                any(os.path.isfile(os.path.join(d, f"{c[0]}_{c[1]}")) for d in ctx.result_dirs)]
        miss = [c for c in cells if c not in dispatched and c not in disk]
        missing_total += len(miss)
        gap = ", ".join(sorted({a for _, a in miss}))
        print(f"{name:<8}{len(cells):>7}{len(disp):>10}{len(disk):>9}{len(miss):>9}  {gap}")
    print("-" * 78)
    if missing_total:
        print(f"{missing_total} cell(s) would have no data.", file=sys.stderr)
        return 1
    print("every table cell is covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
