"""Root relaxation size (tab.size3/4/5).

Variables, constraints, nonzero coefficients and on-disk size of the sBB root
relaxation. These are measured WITHOUT solving, so they are deterministic and
machine-independent -- unlike the memory table, nothing here depends on the
host. The CBF column is the model written in Conic Benchmark Format, the
"file size" a reviewer would see.

Both relaxation modes are shown, so the cost of the McCormick families is
visible alongside the bound they buy in tab.ddps*.

Produced by: scripts/exp_main.sh (DDPS+) and scripts/exp_ddps_ablation.sh (DDPS)
"""
import sys

from .common import escape, label_of, load_result

ROWS = [("RLT", "DDPS+", True), ("RLT_DDPS", "DDPS", True)]

TABLES = {f"size{m}": dict(m=m, rows=ROWS, experiment="exp_main.sh") for m in (3, 4, 5)}


def build(name, spec, ctx):
    out, missing = [], 0
    for state in ctx.states(spec["m"]):
        out.append("\\midrule")
        out.append(f"\\multirow{{{len(spec['rows'])}}}{{*}}{{{escape(ctx.name(state))}}}")
        for algo, text, emph in spec["rows"]:
            shown = label_of(algo, text, emph)
            r = load_result(ctx.result_dirs, state, algo)
            if r is None or r.get("relax_nvars") in (None, 0):
                out.append(f" & {shown} & N/A & N/A & N/A & N/A \\\\")
                missing += 1
                continue
            cbf = "-" if not r.get("relax_cbf_bytes") else f"{r['relax_cbf_bytes'] / 2**20:.2f}"
            out.append(f" & {shown} & {r['relax_nvars']} & {r['relax_ncons']} & "
                       f"{r['relax_nnz']} & {cbf} \\\\")
    out.append("\\bottomrule")
    if missing:
        print(f"WARNING: {missing} row(s) have no size diagnostics; those results "
              f"predate the instrumentation -- re-run scripts/exp_main.sh --force",
              file=sys.stderr)
    return "\n".join(out)
