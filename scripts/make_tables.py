#!/usr/bin/env python3
"""Generate the LaTeX bodies of the paper's experimental tables from results/.

    python3 scripts/make_tables.py --table all
    python3 scripts/make_tables.py --table m3 --out tables/

Tables
    m3, m4, m5      main results, one block per state
    m5low           low-rank LADMM sweep (r = 400..900)
    m5cp            averages over gap-closing CP iterations (needs traces)

The PDGR rows in the paper come from an external implementation
(liu2025unified / FrankWolfe.jl) and are not generated here; they are emitted
as a commented placeholder so the block can be pasted in directly.
"""
import argparse, math, os, re, sys, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# algorithm code -> (display name, emphasise?)  emphasis marks methods
# developed in this paper, matching the existing tables.
MAIN_ROWS = [("A", "Alt-SDP", False), ("LD1", "LADMM", True), ("D", "CP", True),
             ("LDL", "IR", True), ("PPT", "DPS", False), ("RLT", "DDPS+", True)]
LOWRANK_ROWS = [(f"LDR{i}", f"LADMM\\_{r}", True)
                for i, r in enumerate((400, 500, 600, 700, 800, 900))]
GAPCLOSING_ROWS = [("D", "CP", True), ("LDL", "IR", True)]

TABLES = {
    "m3":    dict(m=3, rows=MAIN_ROWS,       kind="main"),
    "m4":    dict(m=4, rows=MAIN_ROWS,       kind="main"),
    "m5":    dict(m=5, rows=MAIN_ROWS,       kind="main"),
    "m5low": dict(m=5, rows=LOWRANK_ROWS,    kind="main"),
    "m5cp":  dict(m=5, rows=GAPCLOSING_ROWS, kind="gapclosing"),
}


def load_instances(benchmark_dir):
    """Map benchmark filename -> (display name, subsystem count)."""
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


def load_result(results_dir, state, algo):
    path = os.path.join(results_dir, f"{state}_{algo}")
    if not os.path.isfile(path):
        return None
    rec = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            rec[k.strip()] = v.strip()
    try:
        return dict(glbub=float(rec["glbub"]), glblb=float(rec["glblb"]),
                    approxub=float(rec["approxub"]), approxfeas=float(rec["approxfeas"]),
                    time=float(rec["time"]))
    except (KeyError, ValueError):
        return None


def load_trace_means(trace_dir, state, algo):
    """Mean ub_relx / lb_relx / b_lower over the gap-closing CP iterations."""
    path = os.path.join(trace_dir, f"{state}_{algo}.cp.csv")
    if not os.path.isfile(path):
        return None
    ub, lb, b = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if str(row.get("is_last", "")).strip().lower() != "true":
                continue
            try:
                u, l, bb = float(row["ub_relx"]), float(row["lb_relx"]), float(row["b_lower"])
            except (KeyError, ValueError):
                continue
            # a round where the oracle returned no usable bound carries -Inf;
            # averaging it would swallow the whole column
            if not (math.isfinite(u) and math.isfinite(l) and math.isfinite(bb)):
                continue
            ub.append(u); lb.append(l); b.append(bb)
    if not ub:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    return dict(ub=mean(ub), lb=mean(lb), b=mean(b), n=len(ub))


def fmt(x, dash_when=None):
    if x is None:
        return "N/A"
    if dash_when is not None and dash_when(x):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.5f}"


def escape(name):
    return name.replace("_", r"\_")


def display_name(name):
    """Benchmark names carry a trailing local dimension for GHZ (GHZ_5_2);
    the paper writes those as GHZ_5. Dicke_5_1 keeps its excitation count."""
    m = re.match(r"^(GHZ)_(\d+)_2$", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name


def bold(value_str, is_best):
    return f"\\textbf{{{value_str}}}" if is_best and value_str != "-" else value_str


def main_block(states, instances, rows, results_dir):
    out = []
    for state in states:
        disp = escape(display_name(instances[state][0]))
        # the paper bolds the best upper and lower bound within each state block
        recs = {a: load_result(results_dir, state, a) for a, _, _ in rows}
        # compare at the printed precision, so ties that look equal are both bold
        rnd = lambda v: round(v, 5)
        ubs = [rnd(r["glbub"]) for r in recs.values()
               if r and r["glbub"] != 0.0 and math.isfinite(r["glbub"])]
        lbs = [rnd(r["glblb"]) for r in recs.values() if r and math.isfinite(r["glblb"])]
        best_ub = min(ubs) if ubs else None      # upper bound: smaller is tighter
        best_lb = max(lbs) if lbs else None      # lower bound: larger is tighter
        out.append("\\midrule")
        out.append(f"\\multirow{{{len(rows)}}}{{*}}{{{disp}}}")
        for algo, label, emph in rows:
            shown = f"\\emph{{{label}}}" if emph else label
            r = recs[algo]
            if r is None:
                out.append(f" & {shown} & N/A & N/A & N/A & N/A & N/A \\\\")
                continue
            # a zero upper bound means the algorithm reports no upper bound at all
            ub = bold(fmt(r["glbub"], dash_when=lambda v: v == 0.0),
                      best_ub is not None and rnd(r["glbub"]) == best_ub)
            lb = bold(fmt(r["glblb"]),
                      best_lb is not None and math.isfinite(r["glblb"]) and rnd(r["glblb"]) == best_lb)
            # the heuristic bound is only meaningful alongside its residual
            aub = "-" if r["approxfeas"] == 0.0 else fmt(r["approxub"])
            afe = "-" if r["approxfeas"] == 0.0 else fmt(r["approxfeas"])
            out.append(f" & {shown} & {ub} & {lb} & {aub} & {afe} & {int(r['time'])} \\\\")
        if rows is MAIN_ROWS:
            out.append("% & PDGR & <external: liu2025unified / FrankWolfe.jl> \\\\")
    out.append("\\bottomrule")
    return "\n".join(out)


def gapclosing_block(states, instances, rows, trace_dir):
    out, missing = [], 0
    for state in states:
        disp = escape(display_name(instances[state][0]))
        out.append("\\midrule")
        out.append(f"\\multirow{{{len(rows)}}}{{*}}{{{disp}}}")
        for algo, label, emph in rows:
            shown = f"\\emph{{{label}}}" if emph else label
            t = load_trace_means(trace_dir, state, algo)
            if t is None:
                out.append(f" & {shown} & N/A & N/A & N/A \\\\")
                missing += 1
                continue
            out.append(f" & {shown} & {t['ub']:.5f} & {t['lb']:.5f} & {t['b']:.5f} \\\\")
    out.append("\\bottomrule")
    if missing:
        print(f"WARNING: {missing} row(s) have no gap-closing trace; "
              f"run scripts/table_m5_gapclosing.sh", file=sys.stderr)
    return "\n".join(out)


def build(table, args, instances):
    spec = TABLES[table]
    # sorted by display name for determinism; the paper's blocks are in an
    # arbitrary historical order, so compare by state name, not by position.
    states = sorted((s for s, (_, m) in instances.items() if m == spec["m"]),
                    key=lambda s: display_name(instances[s][0]))
    if not states:
        print(f"WARNING: no instances with N = {spec['m']}", file=sys.stderr)
    if spec["kind"] == "gapclosing":
        return gapclosing_block(states, instances, spec["rows"], args.trace_dir)
    return main_block(states, instances, spec["rows"], args.results_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="all", choices=list(TABLES) + ["all"])
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--trace-dir", default=os.path.join(ROOT, "results", "traces"))
    ap.add_argument("--out", help="directory to write <table>.tex into (default: stdout)")
    args = ap.parse_args()

    instances = load_instances(args.benchmark_dir)
    if not instances:
        sys.exit(f"no benchmark instances found in {args.benchmark_dir}")

    wanted = list(TABLES) if args.table == "all" else [args.table]
    for t in wanted:
        body = build(t, args, instances)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, f"{t}.tex")
            open(path, "w").write(body + "\n")
            print(f"wrote {path}")
        else:
            print(f"% ---------- table {t} ----------")
            print(body)
            print()


if __name__ == "__main__":
    main()
