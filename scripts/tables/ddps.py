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
SPEC = "{l|l|cc|cc|cc}"
HEAD = ("\n".join([
    r" \textbf{State} & & \multicolumn{2}{c|}{relaxation alone} & "
    r"\multicolumn{2}{c|}{CP oracle} & \multicolumn{2}{c}{IR oracle} \\",
    r"  &  & DDPS & DDPS+ & DDPS & DDPS+ & DDPS & DDPS+ \\"]))
CAPTION = (r"\lid{Ablation of the McCormick strengthening: DDPS against DDPS+.}")
TABLES["ddps"] = dict(m=None, rows=ROWS, experiment="exp_ddps_ablation.sh",
                      label="tab.ddps", caption=CAPTION)


def build(name, spec, ctx):
    if name != "ddps":
        return bounds_block(ctx, spec["m"], spec["rows"])
    out = []
    for m in (3, 4, 5):
        for state in ctx.states(m):
            recs = {a: load_result(ctx.result_dirs, state, a)
                    for p in PAIRS for a in p}
            lines = []
            # upper bound first, then lower; the standalone relaxations return
            # no upper bound at all, which prints as a dash
            # the better bound of each pair is bolded: smaller for the upper
            # bound, larger for the lower one
            for key, label, better in (("glbub", r"$\ub_{\relx}$", min),
                                       ("glblb", r"$\lb_{\relx}$", max)):
                cells = []
                for plain, plus in PAIRS:
                    vals = {}
                    for algo in (plain, plus):
                        r = recs[algo]
                        v = r[key] if r else None
                        # a zero upper bound means no upper bound was produced
                        vals[algo] = None if (v is None or (key == "glbub" and v == 0.0)) else v
                    # compare at the printed precision, so an exact tie bolds both
                    seen = [round(v, 5) for v in vals.values() if v is not None]
                    best = better(seen) if (better and seen) else None
                    for algo in (plain, plus):
                        v = vals[algo]
                        cells.append(bold("-" if v is None else fmt(v),
                                          v is not None and round(v, 5) == best))
                lines.append((label, cells))
            nm = escape(display_name(ctx.name(state)))
            out.append(r"\midrule")
            out.append(f" \\multirow{{2}}{{*}}{{{nm}}} & {lines[0][0]} & "
                       + " & ".join(lines[0][1]) + r" \\")
            out.append(f" & {lines[1][0]} & " + " & ".join(lines[1][1]) + r" \\")
    out.append(r"\bottomrule")
    # nine columns of five decimals overrun \textwidth at the default 6pt
    return wrap_table("\n".join(out), SPEC, HEAD, spec["caption"], spec["label"],
                      toprule=r"\midrule",
                      pre=r"\setlength{\tabcolsep}{4pt}")
