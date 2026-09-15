"""LADMM factorisation-size sweep (tab.m5low).

LADMM at r = 400…900 on the m = 5 instances, testing how sensitive the
heuristic is to the low-rank approximation.

Produced by: scripts/exp_lowrank.sh
"""
from .common import bounds_block

ROWS = [(f"LDR{i}", f"LADMM\\_{r}", True)
        for i, r in enumerate((400, 500, 600, 700, 800, 900))]

TABLES = {"m5low": dict(m=5, rows=ROWS, experiment="exp_lowrank.sh")}


def build(name, spec, ctx):
    return bounds_block(ctx, spec["m"], spec["rows"])
