"""LADMM factorisation-size sweep (tab.m5low).

LADMM at r = 400…900 on the m = 5 instances, testing how sensitive the
heuristic is to the low-rank approximation.

Produced by: scripts/exp_lowrank.sh
"""
from .common import bounds_block, wrap_table, BOUNDS_SPEC, BOUNDS_HEAD

ROWS = [(f"LADMM_{r}", f"LADMM\\_{r}", True)
        for r in (400, 500, 600, 700, 800, 900)]

CAPTION = (r"Results of low-rank approximations for LADMM for $m=5$. Italics as "
           r"in \Cref{tab.m3}; the suffix gives the factorization size $r$.")

TABLES = {"m5low": dict(m=5, rows=ROWS, experiment="exp_lowrank.sh",
                        label="tab.m5low", caption=CAPTION)}


def build(name, spec, ctx):
    body = bounds_block(ctx, spec["m"], spec["rows"])
    return wrap_table(body, BOUNDS_SPEC, BOUNDS_HEAD, spec["caption"], spec["label"])
