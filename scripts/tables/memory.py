"""Peak memory per algorithm (tab.mem).

Reports peak resident memory in GiB: the worst case over the instances of each
size and the shifted geometric mean over all instances, with shift 1 GiB.

Produced by: scripts/exp_main.sh
"""
import math
import sys

from .common import label_of, load_result, wrap_table
from .main import ROWS

SPEC = "{l|rrr|r}"
HEAD = (r"    \textbf{Algorithm} & $m=3$ & $m=4$ & $m=5$ & \textbf{SGM} \\")
SIZES = (3, 4, 5)
SHIFT = 1.0   # GiB; the ranking is unchanged for shifts 0 to 10

# PDGR is run by the same runner and records its peak memory too; it sits
# after Alt-SDP, as in the results tables.
MEM_ROWS = ROWS[:1] + [("PDGR", "PDGR", False)] + ROWS[1:]

TABLES = {"mem": dict(m=None, rows=MEM_ROWS, experiment="exp_main.sh",
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
    caption = r"\lid{Peak resident memory.}"
    return wrap_table("\n".join(out), SPEC, HEAD, caption, spec["label"])
