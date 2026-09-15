#!/usr/bin/env python3
"""Performance profiles over the benchmark (Dolan-More).

    python3 scripts/performance_profile.py                    # text profile
    python3 scripts/performance_profile.py --out plots/        # pgfplots .dat
    python3 scripts/performance_profile.py --metric time

For each metric a solver's performance ratio on an instance is its value
divided by the best value any solver achieved on that instance, and
rho_s(tau) is the fraction of instances where solver s is within a factor tau
of the best. A curve that rises earlier and higher is better.

Metrics
    ub      upper bound ub_relx        smaller is better
    lb      lower bound lb_relx        larger is better (ratios use best/value)
    gap     ub_relx - lb_relx          smaller is better
    time    wall-clock seconds         smaller is better
    mem     peak RSS (MiB)             smaller is better

Solvers with no value on an instance (a dash in the paper's tables) are treated
as failures there, which is what a profile is meant to show.

No matplotlib dependency: --out writes .dat files for \\addplot table.
"""
import argparse, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import (ROOT, RESULT_DIRS, MAIN_ROWS, load_instances,
                         load_result, display_name)

# metric -> (field extractor, larger_is_better)
METRICS = {
    "ub":   (lambda r: r["glbub"] if r["glbub"] != 0.0 and math.isfinite(r["glbub"]) else None, False),
    "lb":   (lambda r: r["glblb"] if math.isfinite(r["glblb"]) else None, True),
    "gap":  (lambda r: (r["glbub"] - r["glblb"])
                       if r["glbub"] != 0.0 and math.isfinite(r["glbub"]) and math.isfinite(r["glblb"])
                       else None, False),
    "time": (lambda r: r["time"] if r["time"] > 0 else None, False),
    "mem":  (lambda r: r.get("mem_total_peak_rss_mib") or r.get("peak_rss_mib"), False),
}


def ratios(instances, solvers, metric, results_dirs, msizes):
    """ratio[solver][instance]; math.inf marks a failure."""
    get, larger_better = METRICS[metric]
    states = [s for s, (_, m) in instances.items() if m in msizes]
    table = {s: {} for s, _, _ in solvers}
    for state in states:
        vals = {}
        for algo, _, _ in solvers:
            r = load_result(results_dirs, state, algo)
            vals[algo] = None if r is None else get(r)
        good = [v for v in vals.values() if v is not None]
        if not good:
            continue
        best = max(good) if larger_better else min(good)
        for algo, v in vals.items():
            if v is None:
                table[algo][state] = math.inf
            elif best == 0:
                table[algo][state] = 1.0 if v == 0 else math.inf
            else:
                table[algo][state] = (best / v) if larger_better else (v / best)
                if table[algo][state] < 1:      # guard against tiny numeric drift
                    table[algo][state] = 1.0
    return table


def profile(rs, taus):
    n = len(rs)
    return [sum(1 for r in rs.values() if r <= t) / n if n else 0.0 for t in taus]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="all", choices=list(METRICS) + ["all"])
    ap.add_argument("--m", default="3,4,5", help="subsystem counts to include")
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--results-dir", action="append")
    ap.add_argument("--out", help="write <metric>.dat for pgfplots")
    args = ap.parse_args()
    results_dirs = args.results_dir or RESULT_DIRS
    msizes = {int(x) for x in args.m.split(",")}

    instances = load_instances(args.benchmark_dir)
    solvers = MAIN_ROWS
    taus = [1.0 + 0.05 * i for i in range(0, 81)]        # 1.0 .. 5.0

    for metric in (list(METRICS) if args.metric == "all" else [args.metric]):
        table = ratios(instances, solvers, metric, results_dirs, msizes)
        ninst = len(next(iter(table.values()))) if table else 0
        if ninst == 0:
            print(f"metric {metric}: no data", file=sys.stderr)
            continue
        print(f"\n== performance profile: {metric}  "
              f"(m in {sorted(msizes)}, {ninst} instances) ==")
        print(f"{'solver':<12}{'rho(1)':>8}{'rho(1.5)':>10}{'rho(2)':>8}{'rho(5)':>8}  "
              f"{'solved':>7}")
        for algo, label, _ in solvers:
            rs = table[algo]
            p = profile(rs, [1.0, 1.5, 2.0, 5.0])
            solved = sum(1 for r in rs.values() if math.isfinite(r))
            print(f"{label:<12}{p[0]:>8.2f}{p[1]:>10.2f}{p[2]:>8.2f}{p[3]:>8.2f}  "
                  f"{solved:>4}/{len(rs)}")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, f"profile_{metric}.dat")
            with open(path, "w") as fh:
                fh.write("# tau " + " ".join(l.replace(" ", "") for _, l, _ in solvers) + "\n")
                cols = [profile(table[a], taus) for a, _, _ in solvers]
                for i, t in enumerate(taus):
                    fh.write(f"{t:.3f} " + " ".join(f"{c[i]:.4f}" for c in cols) + "\n")
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
