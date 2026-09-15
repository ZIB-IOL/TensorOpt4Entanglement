"""Shared loading, formatting and provenance for the table generators."""
import math
import os
import re
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Experiments write into their own directory; results/ itself still holds the
# flat files published with the paper. Searched in order, first hit wins.
RESULT_DIRS = [os.path.join(ROOT, "results", p) for p in ("main", "lowrank", "ddps")]
RESULT_DIRS.append(os.path.join(ROOT, "results"))
TRACE_DIRS = [os.path.join(d, "traces") for d in RESULT_DIRS]

# every cell resolved in this process, for --manifest
PROVENANCE = {}


class Context:
    """Search paths and benchmark metadata shared by all generators."""

    def __init__(self, instances, result_dirs=None, trace_dirs=None):
        self.instances = instances
        self.result_dirs = result_dirs or RESULT_DIRS
        self.trace_dirs = trace_dirs or TRACE_DIRS

    def states(self, m):
        """Instances with `m` subsystems, ordered by display name.

        The paper's blocks are in an arbitrary historical order, so compare by
        state name rather than by position.
        """
        return sorted((s for s, (_, mm) in self.instances.items() if mm == m),
                      key=lambda s: display_name(self.instances[s][0]))

    def name(self, state):
        return display_name(self.instances[state][0])


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


def load_result(results_dirs, state, algo):
    """First matching raw file across the search path, or None."""
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    path = next((os.path.join(d, f"{state}_{algo}")
                 for d in results_dirs
                 if os.path.isfile(os.path.join(d, f"{state}_{algo}"))), None)
    if path is None:
        return None
    PROVENANCE[(state, algo)] = os.path.relpath(path, ROOT)
    rec = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            rec[k.strip()] = v.strip()
    try:
        out = dict(glbub=float(rec["glbub"]), glblb=float(rec["glblb"]),
                   approxub=float(rec["approxub"]), approxfeas=float(rec["approxfeas"]),
                   time=float(rec["time"]))
    except (KeyError, ValueError):
        return None
    # diagnostics are absent from results produced before they were added
    for k, cast in (("relax_nvars", int), ("relax_ncons", int), ("relax_nnz", int),
                    ("relax_cbf_bytes", int), ("peak_rss_mib", float),
                    ("mem_total_alloc_gib", float), ("mem_cp_alloc_gib", float),
                    ("mem_lmo_alloc_gib", float), ("mem_ladmm_alloc_gib", float),
                    ("mem_lmo_calls", int), ("mem_total_peak_rss_mib", float),
                    ("mem_lmo_model_nnz", int)):
        try:
            out[k] = cast(float(rec[k]))
        except (KeyError, ValueError):
            out[k] = None
    return out


def load_trace_means(trace_dirs, state, algo):
    """Mean ub_relx / lb_relx / b_lower over the gap-closing CP iterations."""
    if isinstance(trace_dirs, str):
        trace_dirs = [trace_dirs]
    path = next((os.path.join(d, f"{state}_{algo}.cp.csv")
                 for d in trace_dirs
                 if os.path.isfile(os.path.join(d, f"{state}_{algo}.cp.csv"))), None)
    if path is None:
        return None
    PROVENANCE[(state, algo + " [trace]")] = os.path.relpath(path, ROOT)
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


# ---- formatting -----------------------------------------------------------

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


def label_of(algo, text, emph):
    return f"\\emph{{{text}}}" if emph else text


def bounds_block(ctx, m, rows, pdgr_placeholder=False):
    """The bounds table shared by the results, rank-sweep and ablation tables.

    Columns: ub_relx, lb_relx, ub_heur, feas_heur, time. The best upper and
    lower bound in each state block are bolded, compared at the printed
    precision so ties that look equal are both bold -- which is how the paper
    breaks them.
    """
    out = []
    for state in ctx.states(m):
        recs = {a: load_result(ctx.result_dirs, state, a) for a, _, _ in rows}
        rnd = lambda v: round(v, 5)
        ubs = [rnd(r["glbub"]) for r in recs.values()
               if r and r["glbub"] != 0.0 and math.isfinite(r["glbub"])]
        lbs = [rnd(r["glblb"]) for r in recs.values() if r and math.isfinite(r["glblb"])]
        best_ub = min(ubs) if ubs else None      # upper bound: smaller is tighter
        best_lb = max(lbs) if lbs else None      # lower bound: larger is tighter

        out.append("\\midrule")
        out.append(f"\\multirow{{{len(rows)}}}{{*}}{{{escape(ctx.name(state))}}}")
        for algo, text, emph in rows:
            shown = label_of(algo, text, emph)
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
        if pdgr_placeholder:
            out.append("% & PDGR & <external: liu2025unified / FrankWolfe.jl> \\\\")
    out.append("\\bottomrule")
    return "\n".join(out)
