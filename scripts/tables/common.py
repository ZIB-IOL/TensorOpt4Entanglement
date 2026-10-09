"""Shared loading, formatting and provenance for the table generators."""
import math
import os
import re
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Experiments write into their own directory; results/ itself still holds the
# flat files published with the paper. Searched in order, first hit wins.
RESULT_DIRS = [os.path.join(ROOT, "results", p) for p in ("main", "lowrank", "ddps", "pdgr")]
RESULT_DIRS.append(os.path.join(ROOT, "results"))
TRACE_DIRS = [os.path.join(d, "traces") for d in RESULT_DIRS]

# every cell resolved in this process, for --manifest
PROVENANCE = {}

# Filenames of the runs published with the paper carry the codes used before
# the algorithm names were aligned with it. Kept in sync with LEGACY_ALIASES in
# src/Drivers.jl (inverted: canonical -> the old code to also look for).
LEGACY_CODE = {
    "Alt-SDP": "A", "LADMM": "LD1", "CP": "D", "IR": "LDL",
    "DPS": "PPT", "DDPS+": "RLT",
    "IR-nolazy": "LD", "IR-clear": "LD0", "Alt-SDP+CP": "AD", "DualALM": "LDual",
    "DDPS": "RLT_DDPS", "CP-DDPS": "D_DDPS", "IR-DDPS": "LDL_DDPS",
    **{f"LADMM_{r}": f"LDR{i}" for i, r in enumerate((400, 500, 600, 700, 800, 900))},
}


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
    names = [algo]
    if algo in LEGACY_CODE:
        names.append(LEGACY_CODE[algo])          # a run published before the rename
    path = next((os.path.join(d, f"{state}_{n}")
                 for d in results_dirs for n in names
                 if os.path.isfile(os.path.join(d, f"{state}_{n}"))), None)
    if path is None:
        return None
    PROVENANCE[(state, algo)] = os.path.relpath(path, ROOT)
    rec = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            rec[k.strip()] = v.strip()
    # Field names are the paper's symbols; runs published before the rename
    # used the older names, so accept either.
    def field(*names):
        for n in names:
            if n in rec:
                return float(rec[n])
        raise KeyError(names[0])
    try:
        out = dict(glbub=field("ub_relx", "glbub"),
                   glblb=field("lb_relx", "glblb"),
                   approxub=field("ub_heur", "approxub"),
                   approxfeas=field("feas_heur", "approxfeas"),
                   time=field("time"))
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


def last_run(rows):
    """Infer the final run from a reset to the opening state-pool size."""
    if not rows:
        return rows
    opening = rows[0].get("n_states")
    if opening is None:
        return rows
    starts = [0] + [i for i, r in enumerate(rows)
                    if i and r.get("iter", "1").strip() in ("0", "0.0")
                    and r.get("n_states") == opening]
    return rows[starts[-1]:]


def load_trace_means(trace_dirs, state, algo):
    """Mean recorded bounds over the final run's gap-closing phase.

    The lower bound retains earlier improvements. The current LMO correction
    need not belong to the recorded upper bound, so it is not averaged alongside
    these bounds or used to reconstruct them.
    """
    if isinstance(trace_dirs, str):
        trace_dirs = [trace_dirs]
    names = [algo] + ([LEGACY_CODE[algo]] if algo in LEGACY_CODE else [])
    path = next((os.path.join(d, f"{state}_{n}.cp.csv")
                 for d in trace_dirs for n in names
                 if os.path.isfile(os.path.join(d, f"{state}_{n}.cp.csv"))), None)
    if path is None:
        return None
    PROVENANCE[(state, algo + " [trace]")] = os.path.relpath(path, ROOT)
    ub, lb = [], []
    with open(path) as fh:
        for row in last_run(list(csv.DictReader(fh))):
            if str(row.get("is_last", "")).strip().lower() != "true":
                continue
            try:
                u, l = float(row["ub_relx"]), float(row["lb_relx"])
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(u) and math.isfinite(l)):
                continue
            ub.append(u); lb.append(l)
    if not ub:
        return None
    mean = lambda xs: sum(xs) / len(xs)
    return dict(ub=mean(ub), lb=mean(lb), n=len(ub))


# ---- formatting -----------------------------------------------------------

def fmt(x, dash_when=None):
    if x is None:
        return "N/A"
    if dash_when is not None and dash_when(x):
        return "-"
    if not math.isfinite(x):
        return "-"
    return f"{x:.5f}"


def fmt_residual(x):
    """Keep small positive residuals visible at two significant digits."""
    if x is None or not math.isfinite(x):
        return "-"
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return "$" + mantissa + rf"\times10^{{{int(exponent)}}}$"


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
    # An explicit underline: ulem is no longer loaded, so \emph is italics again
    # and the experiment tables ask for the rule they actually want.
    return f"\\underline{{{text}}}" if emph else text


PDGR_CSV = os.path.join(ROOT, "data", "pdgr.csv")
EXACT_CSV = os.path.join(ROOT, "data", "exact_thresholds.csv")


def load_exact():
    """Known exact thresholds by display name: (value, bibtex key)."""
    out = {}
    if os.path.isfile(EXACT_CSV):
        with open(EXACT_CSV) as fh:
            for r in csv.DictReader(l for l in fh if not l.startswith("#")):
                out[r["instance"]] = (float(r["value"]), r["cite"])
    return out


def state_cell(name, exact=None):
    """State label, with the known exact threshold and its source underneath."""
    if exact is None:
        return escape(name)
    value, cite = exact
    return (f"\\shortstack[l]{{{escape(name)}\\\\ \\scriptsize"
            f"$\\opt={value:.5f}$~\\cite{{{cite}}}" "}")


def load_pdgr(result_dirs=None):
    """Local PDGR runs by display name, with the published CSV as fallback.

    A local run replaces the entire row, including missing one-sided bounds,
    so certificates from different configurations are never silently mixed.
    """
    out = {}
    if os.path.isfile(PDGR_CSV):
        with open(PDGR_CSV) as fh:
            for r in csv.DictReader(l for l in fh if not l.startswith("#")):
                num = lambda k: float(r[k]) if r[k].strip() else None
                out[r["instance"]] = dict(ub=num("ub_relx"), lb=num("lb_relx"),
                                          time=num("time"))
    for state, (name, _) in load_instances(os.path.join(ROOT, "benchmark")).items():
        rec = load_result(RESULT_DIRS if result_dirs is None else result_dirs, state, "PDGR")
        if rec is not None:
            finite = lambda value: value if math.isfinite(value) else None
            out[display_name(name)] = dict(ub=finite(rec["glbub"]),
                                          lb=finite(rec["glblb"]), time=rec["time"])
        elif display_name(name) in out:
            PROVENANCE[(state, "PDGR")] = os.path.relpath(PDGR_CSV, ROOT)
    return out


def bounds_block(ctx, m, rows, pdgr_placeholder=False, known=False):
    """The bounds table shared by the results, rank-sweep and ablation tables.

    Columns: ub_relx, lb_relx, ub_heur, feas_heur, time. The best upper and
    lower bound in each state block are bolded, compared at the printed
    precision so ties that look equal are both bold -- which is how the paper
    breaks them.
    """
    out = []
    pdgr = load_pdgr(ctx.result_dirs) if pdgr_placeholder else {}
    exact = load_exact() if known else {}
    for state in ctx.states(m):
        recs = {a: load_result(ctx.result_dirs, state, a) for a, _, _ in rows}
        rnd = lambda v: round(v, 5)
        ubs = [rnd(r["glbub"]) for r in recs.values()
               if r and r["glbub"] != 0.0 and math.isfinite(r["glbub"])]
        lbs = [rnd(r["glblb"]) for r in recs.values() if r and math.isfinite(r["glblb"])]
        pd = pdgr.get(display_name(ctx.name(state)))
        if pd:
            # Both local PDGR runs and the published fallback compete for boldface.
            if pd["ub"] is not None: ubs.append(rnd(pd["ub"]))
            if pd["lb"] is not None: lbs.append(rnd(pd["lb"]))
        best_ub = min(ubs) if ubs else None      # upper bound: smaller is tighter
        best_lb = max(lbs) if lbs else None      # lower bound: larger is tighter

        out.append("\\midrule")
        nrow = len(rows) + (1 if pd else 0)
        cell = state_cell(ctx.name(state), exact.get(display_name(ctx.name(state))))
        out.append(f"\\multirow{{{nrow}}}{{*}}{{{cell}}}")
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
            afe = "-" if r["approxfeas"] == 0.0 else fmt_residual(r["approxfeas"])
            out.append(f" & {shown} & {ub} & {lb} & {aub} & {afe} & {int(r['time'])} \\\\")
            if pd and algo == "Alt-SDP":
                pu = bold(fmt(pd["ub"]) if pd["ub"] is not None else "-",
                          best_ub is not None and pd["ub"] is not None
                          and rnd(pd["ub"]) == best_ub)
                pl = bold(fmt(pd["lb"]) if pd["lb"] is not None else "-",
                          best_lb is not None and pd["lb"] is not None
                          and rnd(pd["lb"]) == best_lb)
                pt = "-" if pd["time"] is None else str(int(pd["time"]))
                out.append(f" & PDGR & {pu} & {pl} & - & - & {pt} \\\\")
    out.append("\\bottomrule")
    return "\n".join(out)


# --------------------------------------------------------------------------
BOUNDS_SPEC = "{l|l|ccccc}"
BOUNDS_HEAD = (r"    \textbf{State} & \textbf{Algorithm} & $\ub_{\relx}$ & "
               r"$\lb_{\relx}$ & $\ub_{\heur}$ & $\feas_{\heur}$ & "
               r"\textbf{Time (s)} \\")


def wrap_table(body, spec, head, caption, label, toprule=r"\toprule", pre=None):
    """Wrap a generated body in the table environment the paper uses, so the
    whole float is generated rather than the rows alone.

    `pre` is emitted inside the float, before the tabular, for a table that
    needs a local layout tweak (tighter column separation, say); being inside
    the environment keeps the change from leaking into the other tables.
    """
    return "\n".join([
        "% generated by scripts/make_tables.py -- do not edit by hand",
        r"\begin{table}[!htbp]", r"\centering"] + ([pre] if pre else []) + [
        r"\begin{tabular}" + spec, r"\toprule", head, toprule,
        body,
        r"\end{tabular}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\end{table}", ""])
