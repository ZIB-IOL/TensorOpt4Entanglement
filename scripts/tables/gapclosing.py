"""Gap-closing CP iteration averages (tab.m5CP).

Unlike the other tables this one averages over the CP iterations that run
*after* the algorithm enters its gap-closing phase, which the result files do
not record. It is therefore built from the per-iteration trajectories
(`<state>_<algo>.cp.csv`) written whenever EXACTENT_TRACE is set.

Columns: ub_relx, lb_relx, b_lower, where b_lower is the sBB oracle's lower
bound for that round, so lb_relx = ub_relx + b_lower.

Produced by: scripts/exp_main.sh (it records the trajectories)
"""
import sys

from .common import escape, label_of, load_trace_means, wrap_table

ROWS = [("CP", "CP", True), ("IR", "IR", True)]

SPEC = "{l|lccc}"
HEAD = (r"    \textbf{State} & \textbf{Algorithm} & $\ub_{\relx}$ & "
        r"$\lb_{\relx}$ & $\underbar{b}$   \\")
CAPTION = (r"Average results on gap-closing CP iterations for $m=5$. Italics as "
           r"in \Cref{tab.m3}.")

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
                out.append(f" & {shown} & N/A & N/A & N/A \\\\")
                missing += 1
                continue
            out.append(f" & {shown} & {t['ub']:.5f} & {t['lb']:.5f} & {t['b']:.5f} \\\\")
    out.append("\\bottomrule")
    if missing:
        print(f"WARNING: {missing} row(s) have no gap-closing trace; "
              f"run scripts/exp_main.sh (it records the CP trajectories)", file=sys.stderr)
    return wrap_table("\n".join(out), SPEC, HEAD, spec["caption"], spec["label"],
                      toprule=r"\midrule")
