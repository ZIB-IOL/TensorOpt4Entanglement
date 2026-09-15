#!/usr/bin/env python3
"""Turn trajectory CSVs into pgfplots-ready data and a text summary.

    python3 scripts/summarize_traces.py                      # summary table
    python3 scripts/summarize_traces.py --out plots/         # .dat per series

matplotlib is not a dependency, so this emits whitespace-separated .dat files
that `\\addplot table` reads directly, rather than rendering images.

For GHZ instances the analytic white-noise threshold 1 - 1/(1 + 2^(m-1)) is
reported alongside, so a convergence plot can show the true value.
"""
import argparse, csv, glob, math, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ghz_threshold(m):
    """Analytic white-noise threshold of the m-party GHZ state."""
    return 1 - 1 / (1 + 2 ** (m - 1))


def instance_info(benchmark_dir):
    out = {}
    for fn in sorted(os.listdir(benchmark_dir)):
        if not fn.endswith(".jl"):
            continue
        text = open(os.path.join(benchmark_dir, fn)).read()
        n = re.search(r"^N\s*=\s*(\d+)\s*$", text, re.M)
        name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
        if n and name:
            out[fn] = (name.group(1), int(n.group(1)))
    return out


def read_trace(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rec = {}
            for k, v in r.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            rows.append(rec)
    return rows


def finite(xs):
    return [x for x in xs if isinstance(x, float) and math.isfinite(x)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", default=os.path.join(ROOT, "results", "traces"))
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--out", help="write one .dat per series into this directory")
    args = ap.parse_args()

    info = instance_info(args.benchmark_dir)
    files = sorted(glob.glob(os.path.join(args.trace_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no trace files in {args.trace_dir}; "
                         f"run scripts/experiment_convergence.sh first")

    print(f"{'instance':<14}{'algo':<10}{'kind':<7}{'iters':>6}  summary")
    print("-" * 86)
    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(.+\.jl)_(.+)\.(cp|ladmm)\.csv$", base)
        if not m:
            continue
        state, algo, kind = m.groups()
        rows = read_trace(path)
        if not rows:
            continue
        name, nsubs = info.get(state, (state, 0))

        if kind == "cp":
            ub = finite([r["ub_relx"] for r in rows])
            lb = finite([r["lb_relx"] for r in rows])
            nlast = sum(1 for r in rows if str(r.get("is_last")).lower() == "true")
            nob = sum(1 for r in rows if not (isinstance(r["b_lower"], float)
                                              and math.isfinite(r["b_lower"])))
            extra = (f"ub {ub[0]:.5f}->{ub[-1]:.5f}" if ub else "ub n/a")
            extra += (f"  lb ->{lb[-1]:.5f}" if lb else "  lb none")
            # how often the oracle stopped early without producing a bound
            extra += f"  no-bound iters {nob}/{len(rows)}  gap-closing {nlast}"
        else:
            res = finite([r["residual"] for r in rows])
            extra = (f"residual {res[0]:.3e}->{res[-1]:.3e}" if res else "residual n/a")
            z = finite([r["z"] for r in rows])
            if z:
                extra += f"  z ->{z[-1]:.5f}"
        if name.startswith("GHZ") and nsubs:
            extra += f"  [GHZ analytic {ghz_threshold(nsubs):.5f}]"
        print(f"{name:<14}{algo:<10}{kind:<7}{len(rows):>6}  {extra}")

        if args.out:
            os.makedirs(args.out, exist_ok=True)
            dat = os.path.join(args.out, f"{name}_{algo}_{kind}.dat")
            with open(dat, "w") as fh:
                cols = list(rows[0].keys())
                fh.write("# " + " ".join(cols) + "\n")
                for r in rows:
                    vals = []
                    for c in cols:
                        v = r[c]
                        # pgfplots skips non-finite points rather than drawing them
                        vals.append("nan" if isinstance(v, float) and not math.isfinite(v) else str(v))
                    fh.write(" ".join(vals) + "\n")
    if args.out:
        print(f"\nwrote .dat files to {args.out}/ (use with \\addplot table)")


if __name__ == "__main__":
    main()
