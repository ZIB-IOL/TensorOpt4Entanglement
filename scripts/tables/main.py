"""The three main results tables (tab.m3, tab.m4, tab.m5).

One block per state, one row per algorithm compared in the paper. The PDGR
rows come from an external implementation (liu2025unified / FrankWolfe.jl) and
are emitted as a commented placeholder so the block can be pasted in directly.
Because PDGR is absent, bolding is computed over the rows generated here, which
can differ from the paper where a PDGR value was the best.

Produced by: scripts/exp_main.sh
"""
from .common import bounds_block

# algorithm code -> (display name, emphasise?)  emphasis marks methods
# developed in this paper, matching the existing tables.
ROWS = [("A", "Alt-SDP", False), ("LD1", "LADMM", True), ("D", "CP", True),
        ("LDL", "IR", True), ("PPT", "DPS", False), ("RLT", "DDPS+", True)]

TABLES = {
    "m3": dict(m=3, rows=ROWS, experiment="exp_main.sh"),
    "m4": dict(m=4, rows=ROWS, experiment="exp_main.sh"),
    "m5": dict(m=5, rows=ROWS, experiment="exp_main.sh"),
}


def build(name, spec, ctx):
    return bounds_block(ctx, spec["m"], spec["rows"], pdgr_placeholder=True)
