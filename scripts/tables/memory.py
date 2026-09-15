"""Per-level memory (tab.mem3/4/5).

Answers "estimate the memory requirements". Memory is attributed to each
algorithmic level rather than only the process total. The phases NEST --
:total contains :cp, which contains the :lmo calls it makes -- so the figures
are inclusive, and :lmo is reported separately (with its call count) so its
share is visible.

Columns: total, CP, LMO (calls), LADMM allocation in GiB, then peak RSS in MiB.

Produced by: scripts/exp_main.sh
"""
import sys

from .common import escape, label_of, load_result
from .main import ROWS

TABLES = {f"mem{m}": dict(m=m, rows=ROWS, experiment="exp_main.sh") for m in (3, 4, 5)}


def build(name, spec, ctx):
    out, missing = [], 0
    g = lambda r, k: "-" if r.get(k) is None else f"{r[k]:.2f}"
    for state in ctx.states(spec["m"]):
        out.append("\\midrule")
        out.append(f"\\multirow{{{len(spec['rows'])}}}{{*}}{{{escape(ctx.name(state))}}}")
        for algo, text, emph in spec["rows"]:
            shown = label_of(algo, text, emph)
            r = load_result(ctx.result_dirs, state, algo)
            if r is None or r.get("mem_total_alloc_gib") is None:
                out.append(f" & {shown} & N/A & N/A & N/A & N/A & N/A \\\\")
                missing += 1
                continue
            calls = "-" if r.get("mem_lmo_calls") in (None, 0) else str(r["mem_lmo_calls"])
            rss = "-" if r.get("mem_total_peak_rss_mib") is None else f"{r['mem_total_peak_rss_mib']:.0f}"
            out.append(f" & {shown} & {g(r,'mem_total_alloc_gib')} & {g(r,'mem_cp_alloc_gib')} & "
                       f"{g(r,'mem_lmo_alloc_gib')} ({calls}) & {g(r,'mem_ladmm_alloc_gib')} & {rss} \\\\")
    out.append("\\bottomrule")
    if missing:
        print(f"WARNING: {missing} row(s) have no memory diagnostics; those results "
              f"predate the instrumentation -- re-run scripts/exp_main.sh --force",
              file=sys.stderr)
    return "\n".join(out)
