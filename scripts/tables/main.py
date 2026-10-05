"""The three main results tables (tab.m3, tab.m4, tab.m5).

One block per state, one row per algorithm compared in the paper. The PDGR
rows use local runs of the bundled implementation (liu2025unified / FrankWolfe.jl)
when available, falling back to the published values in data/pdgr.csv.

Produced by: scripts/exp_main.sh
"""
from .common import bounds_block, wrap_table, BOUNDS_SPEC, BOUNDS_HEAD

# algorithm code -> (display name, emphasise?)  emphasis marks methods
# developed in this paper, matching the existing tables.
ROWS = [("Alt-SDP", "Alt-SDP", False), ("LADMM", "LADMM", True), ("CP", "CP", True),
        ("IR", "IR", True), ("DPS", "DPS", False), ("DDPS+", "DDPS+", True)]

CAP3 = (r"Experimental results for $m=3$. \lid{Underlined algorithms are "
        r"developed or adapted in this paper; the others are existing methods "
        r"used as baselines.}")
CAP = r"Experimental results for $m=%d$. Underlining as in \Cref{tab.m3}."

TABLES = {
    "m3": dict(m=3, rows=ROWS, experiment="exp_main.sh",
               label="tab.m3", caption=CAP3),
    "m4": dict(m=4, rows=ROWS, experiment="exp_main.sh",
               label="tab.m4", caption=CAP % 4),
    "m5": dict(m=5, rows=ROWS, experiment="exp_main.sh",
               label="tab.m5", caption=CAP % 5),
}


def build(name, spec, ctx):
    body = bounds_block(ctx, spec["m"], spec["rows"], pdgr_placeholder=True)
    return wrap_table(body, BOUNDS_SPEC, BOUNDS_HEAD, spec["caption"], spec["label"])
