"""DDPS vs DDPS+ ablation of the sBB oracle (tab.ddps3/4/5).

DDPS+ is not "folded into" the LMO -- it *is* the LMO's relaxation, since
initRelaxationNode and initRelaxationThreshold call the same
strengthenRelaxation. These tables pair each algorithm with its DDPS-only
counterpart so the contribution of the McCormick families is measured.

The DDPS+ rows come from exp_main.sh, the DDPS rows from
exp_ddps_ablation.sh, so no run is duplicated.
"""
from .common import bounds_block

ROWS = [("RLT_DDPS", "DDPS", True),       ("RLT", "DDPS+", True),
        ("D_DDPS", "CP (DDPS)", True),    ("D", "CP (DDPS+)", True),
        ("LDL_DDPS", "IR (DDPS)", True),  ("LDL", "IR (DDPS+)", True)]

TABLES = {f"ddps{m}": dict(m=m, rows=ROWS, experiment="exp_ddps_ablation.sh")
          for m in (3, 4, 5)}


def build(name, spec, ctx):
    return bounds_block(ctx, spec["m"], spec["rows"])
