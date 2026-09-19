"""DDPS vs DDPS+ ablation of the sBB oracle (tab.ddps3/4/5).

DDPS+ is not "folded into" the LMO -- it *is* the LMO's relaxation, since
initRelaxationNode and initRelaxationThreshold call the same
strengthenRelaxation. These tables pair each algorithm with its DDPS-only
counterpart so the contribution of the McCormick families is measured.

The DDPS+ rows come from exp_main.sh, the DDPS rows from
exp_ddps_ablation.sh, so no run is duplicated.
"""
from .common import (bounds_block, wrap_table, load_result, display_name,
                     escape, fmt, bold)

ROWS = [("DDPS", "DDPS", True),          ("DDPS+", "DDPS+", True),
        ("CP-DDPS", "CP (DDPS)", True),  ("CP", "CP (DDPS+)", True),
        ("IR-DDPS", "IR (DDPS)", True),  ("IR", "IR (DDPS+)", True)]

TABLES = {f"ddps{m}": dict(m=m, rows=ROWS, experiment="exp_ddps_ablation.sh")
          for m in (3, 4, 5)}

# The compact ablation the paper prints: one row per instance, lower bounds
# only, since that is the only column the two relaxations differ in.
PAIRS = [("DDPS", "DDPS+"), ("CP-DDPS", "CP"), ("IR-DDPS", "IR")]
SPEC = "{l|c|cc|cc|cc}"
HEAD = ("\n".join([
    r" \textbf{State} & $m$ & \multicolumn{2}{c|}{relaxation alone} & "
    r"\multicolumn{2}{c|}{CP oracle} & \multicolumn{2}{c}{IR oracle} \\",
    r"  &  & DDPS & DDPS+ & DDPS & DDPS+ & DDPS & DDPS+ \\"]))
CAPTION = (r"\lid{check: Ablation of the McCormick strengthening: lower bounds "
           r"$\lb_{\relx}$ from the plain DDPS outer approximation "
           r"\eqref{eq.dmmr} against DDPS+, which adds the scalar "
           r"\eqref{eq.mccons} and tensor \eqref{eq.psd} McCormick "
           r"inequalities. The first pair uses the relaxation on its own, the "
           r"others use it inside the sBB LMO of CP and IR. Findings in "
           r"\Cref{sec.ablation}.}")
TABLES["ddps"] = dict(m=None, rows=ROWS, experiment="exp_ddps_ablation.sh",
                      label="tab.ddps", caption=CAPTION)


def build(name, spec, ctx):
    if name != "ddps":
        return bounds_block(ctx, spec["m"], spec["rows"])
    out = []
    for m in (3, 4, 5):
        for state in ctx.states(m):
            cells = []
            for plain, plus in PAIRS:
                for algo in (plain, plus):
                    r = load_result(ctx.result_dirs, state, algo)
                    v = r["glblb"] if r else None
                    txt = "--" if v is None else fmt(v)
                    cells.append(bold(txt, algo == "DDPS+"))
            out.append(f" {escape(display_name(ctx.name(state)))} & {m} & "
                       + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    return wrap_table("\n".join(out), SPEC, HEAD, spec["caption"], spec["label"],
                      toprule=r"\midrule")
