"""Gap-closing CP iteration averages (tab.m5CP).

Unlike the other tables this one averages over the CP iterations that run
*after* the algorithm enters its gap-closing phase, which the result files do
not record. It is therefore built from the per-iteration trajectories
(`<state>_<algo>.cp.csv`) written whenever EXACTENT_TRACE is set.

Columns: average recorded ub_relx and lb_relx. The latter retains earlier
improvements and need not equal the current upper bound plus the current LMO
correction, so the correction is not included in this table.

Produced by: scripts/exp_main.sh (it records the trajectories)
"""
import sys

from .common import escape, label_of, load_trace_means, wrap_table

ROWS = [("CP", "CP", True), ("IR", "IR", True)]

SPEC = "{l|lcc}"
HEAD = (r"    \textbf{State} & \textbf{Algorithm} & $\ub_{\relx}$ & "
        r"$\lb_{\relx}$ \\")
CAPTION = r"\lid{Average bounds during gap-closing CP iterations for $m=5$.}"

TABLES = {"m5cp": dict(m=5, rows=ROWS, experiment="exp_main.sh",
                       label="tab.m5CP", caption=CAPTION)}


def build(name, spec, ctx):
    out, missing = [], 0
    for state in ctx.states(spec["m"]):
        out.append("\\midrule")
        out.append(f"\\multirow{{{len(spec['rows'])}}}{{*}}{{{escape(ctx.name(state))}}}")
        for algo, text, emph in spec["rows"]:
            shown = label_of(algo, text, emph)
            t = load_trace_means(ctx.trace_dirs, state, algo)
            if t is None:
                out.append(f" & {shown} & - & - \\\\")
                missing += 1
                continue
            out.append(f" & {shown} & {t['ub']:.5f} & {t['lb']:.5f} \\\\")
    out.append("\\bottomrule")
    if missing:
        print(f"NOTE: {missing} row(s) have no usable gap-closing iterations; "
              f"their averages are shown as dashes", file=sys.stderr)
    return wrap_table("\n".join(out), SPEC, HEAD, spec["caption"], spec["label"],
                      toprule=r"\midrule")
