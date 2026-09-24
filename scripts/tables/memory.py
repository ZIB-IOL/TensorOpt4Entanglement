"""Peak memory per algorithm (tab.mem).

Answers the referee's "estimate of the memory requirements for each ... like
NNZ, file size, literal memory usage or something". One row per algorithm and
one column per subsystem count, reporting the worst case over the instances of
that size, so the whole answer is a single small float rather than one table
per m. The root-relaxation sizes (variables, constraints, nonzeros, CBF bytes)
are folded into the caption, since they depend only on m and not on the state.

Produced by: scripts/exp_main.sh
"""
import math
import sys

from .common import ROOT, label_of, load_result, wrap_table
from .main import ROWS

SPEC = "{l|rrr|r}"
HEAD = (r"    \textbf{Algorithm} & $m=3$ & $m=4$ & $m=5$ & \textbf{SGM} \\")
SIZES = (3, 4, 5)
SHIFT = 1.0   # GiB; the ranking is unchanged for shifts 0 to 10

TABLES = {"mem": dict(m=None, rows=ROWS, experiment="exp_main.sh",
                      label="tab.mem", caption=None)}


def _peak(ctx, m, algo):
    """Worst-case peak RSS in GiB over the instances with this subsystem count."""
    vals = []
    for state in ctx.states(m):
        r = load_result(ctx.result_dirs, state, algo)
        v = (r or {}).get("peak_rss_mib")
        if v:
            vals.append(v / 1024.0)
    return max(vals) if vals else None


def _all_peaks(ctx, algo):
    """Peak RSS in GiB on every instance, any size."""
    out = []
    for m in SIZES:
        for state in ctx.states(m):
            v = (load_result(ctx.result_dirs, state, algo) or {}).get("peak_rss_mib")
            if v:
                out.append(v / 1024.0)
    return out


def _sgm(vals, shift=SHIFT):
    """Shifted geometric mean, the usual benchmarking summary."""
    if not vals:
        return None
    return math.exp(sum(math.log(v + shift) for v in vals) / len(vals)) - shift


def _relax_note(ctx):
    """The largest root relaxation, for the caption."""
    best = None
    for state in ctx.states(max(SIZES)):
        r = load_result(ctx.result_dirs, state, "DDPS+")
        if r and r.get("relax_nvars"):
            key = r["relax_nnz"]
            if best is None or key > best[2]:
                best = (r["relax_nvars"], r["relax_ncons"], r["relax_nnz"],
                        r["relax_cbf_bytes"] / 2**20)
    if best is None:
        return ""
    v, c, nz, mb = best
    return (f" The sBB root relaxation of DDPS+ has at most ${v}$ variables, "
            f"${c}$ constraints and ${nz/1e4:.1f}\\times10^4$ nonzeros "
            f"(${mb:.2f}$\\,MB in CBF) at $m={max(SIZES)}$.")


def build(name, spec, ctx):
    out, missing = [], 0
    for algo, text, emph in spec["rows"]:
        cells = []
        for m in SIZES:
            p = _peak(ctx, m, algo)
            if p is None:
                missing += 1
                cells.append("-")
            else:
                cells.append(f"{p:.1f}")
        g = _sgm(_all_peaks(ctx, algo))
        cells.append("-" if g is None else f"{g:.2f}")
        out.append(f" {label_of(algo, text, emph)} & " + " & ".join(cells) + r" \\")
    out.append("\\bottomrule")
    if missing:
        print(f"WARNING: {missing} cell(s) have no memory diagnostics; those results "
              f"predate the instrumentation -- re-run scripts/exp_main.sh --force",
              file=sys.stderr)
    caption = (r"\lid{Peak resident memory in GiB, worst case over the instances of "
               r"each size; SGM is the shifted geometric mean (shift $1$\,GiB) over all "
               r"eleven instances." + _relax_note(ctx) +
               r" Underlining as in \Cref{tab.m3}.}")
    return wrap_table("\n".join(out), SPEC, HEAD, caption, spec["label"])
