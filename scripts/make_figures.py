#!/usr/bin/env python3
"""Generate the paper's figures, as pgfplots/TikZ, from saved results.

The companion of make_tables.py: nothing here is drawn by hand, so a figure
cannot drift from the results behind it. Each figure writes its own data files
and a self-contained `figure` environment the paper `\\input`s.

    python3 scripts/make_figures.py                      # both, into ./figures and ./plots
    python3 scripts/make_figures.py --out <paper dir>    # write beside the paper
    python3 scripts/make_figures.py --figure bounds

  bounds       per-instance bounds + performance profile of the lower bound
  convergence  upper-bound trajectories over cutting-plane iterations, m = 5

Captions here are deliberately short and descriptive; findings belong in the
section that discusses the figure, not in its caption.

Colours are the four categorical slots validated for colour-vision deficiency
(worst all-pairs Delta E 9.2 under deuteranopia); marker shape and line style
repeat that distinction, so identity never rests on hue alone.
"""
import argparse, csv, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables.common import (ROOT, load_instances, load_result,  # noqa: E402
                           display_name, load_pdgr)

PALETTE = [("sOne", "2A78D6"), ("sTwo", "EB6834"),
           ("sThree", "4A3AA7"), ("sFour", "1BAF7A")]

PREAMBLE = r"""\pgfplotsset{
  vizbase/.style={
    width=0.86\textwidth,
    tick label style={font=\scriptsize}, label style={font=\footnotesize},
    axis line style={gray!55, line width=0.3pt},
    tick style={gray!55, line width=0.3pt},
    ymajorgrids, grid style={gray!20, line width=0.2pt},
    legend style={font=\scriptsize, draw=none, fill=none,
                  at={(0.5,1.02)}, anchor=south, legend columns=-1,
                  /tikz/every even column/.append style={column sep=7pt}},
  }
}
"""


def colours():
    return "".join(f"\\definecolor{{{n}}}{{HTML}}{{{h}}}" + ("\n" if i % 2 else "")
                   for i, (n, h) in enumerate(PALETTE))


def short(name):
    """GHZ_5_2 -> G_5, Dicke_5_1 -> D_{5,1}, Cluster_4 -> C_4."""
    p = name.split("_")
    if p[0] == "GHZ":
        return f"$\\mathrm{{G}}_{{{p[1]}}}$"
    if p[0] == "Cluster":
        return f"$\\mathrm{{C}}_{{{p[1]}}}$"
    return f"$\\mathrm{{D}}_{{{p[1]},{p[2]}}}$"


# --------------------------------------------------------------------------
def fig_bounds(out, data):
    inst = load_instances(os.path.join(ROOT, "benchmark"))
    # same ordering as the tables (Context.states): by subsystem count, then
    # display name -- the caption says the figure follows the tables, so it must
    order = sorted(inst.items(), key=lambda kv: (kv[1][1], display_name(kv[1][0])))
    dirs = [os.path.join(ROOT, "results", "main")]
    pdgr = load_pdgr()

    # DPS and DDPS+ now return the same bound on every instance (the corrected
    # DPS conversion coincides with the PPT level that DDPS imposes at the root),
    # so they share one series; plotting both would hide one marker under the
    # other. PDGR holds the best bound on three instances and gets its own
    # series, on the instances where it ran.
    rows, labels, pdgr_pts = [], [], []
    for st, (name, m) in order:
        g = lambda a, k: (load_result(dirs, st, a) or {}).get(k)
        ubs = [v for v in (g("Alt-SDP", "glbub"), g("CP", "glbub"), g("IR", "glbub"))
               if v and math.isfinite(v) and v > 0]
        lbs = {a: g(a, "glblb") for a in ("CP", "IR", "DPS", "DDPS+")}
        lbs = {a: v for a, v in lbs.items() if v is not None and math.isfinite(v)}
        if not ubs or not lbs:
            continue
        i = len(rows)
        rows.append((i, min(ubs), max(lbs.values()), lbs.get("CP")))
        labels.append(short(name))
        pl = (pdgr.get(display_name(name)) or {}).get("lb")
        if pl is not None:
            pdgr_pts.append((i, pl))
    path = os.path.join(data, "bounds_by_instance.dat")
    with open(path, "w") as fh:
        fh.write("# idx best_ub best_lb cp_lb\n")
        for i, bu, bl, cp in rows:
            fh.write(f"{i} {bu:.5f} {bl:.5f} {cp:.5f}\n")
    # Separate file: PDGR coverage may be incomplete; omit missing points
    # rather than emitting a sentinel pgfplots has to be told to skip.
    ppdgr = os.path.join(data, "pdgr_by_instance.dat")
    with open(ppdgr, "w") as fh:
        fh.write("# idx pdgr_lb\n")
        for i, v in pdgr_pts:
            fh.write(f"{i} {v:.5f}\n")

    # Dolan-More performance profile of the lower bound: for each method, the
    # fraction of instances whose bound is within a factor tau of the best bound
    # any method attained. PDGR is included, so the reference really is the best
    # bound known; the instances it did not run on count as failures, which is
    # what caps its curve. Written in the column order the figure indexes into.
    profs = ("Alt-SDP", "LADMM", "CP", "IR", "DPS", "DDPS+", "PDGR")
    ratios = {a: [] for a in profs}
    for st, (name, mm) in order:
        vals = {a: (load_result(dirs, st, a) or {}).get("glblb")
                for a in profs if a != "PDGR"}
        vals["PDGR"] = (pdgr.get(display_name(name)) or {}).get("lb")
        vals = {a: v for a, v in vals.items() if v is not None and math.isfinite(v)}
        if not vals:
            continue
        best = max(vals.values())
        for a in profs:
            v = vals.get(a)
            ratios[a].append(best / v if (v is not None and v > 0 and best > 0) else math.inf)
    ninst = max((len(v) for v in ratios.values()), default=0)
    ppath = os.path.join(data, "profile_lb.dat")
    with open(ppath, "w") as fh:
        fh.write("# tau " + " ".join(profs) + "\n")
        t = 1.0
        while t <= 3.0001:
            fh.write(f"{t:.3f} " + " ".join(
                # an exact tie must count for both methods, so compare the ratio
                # to tau with a relative tolerance rather than exactly
                f"{sum(1 for r in ratios[a] if r <= t * (1 + 1e-6)) / ninst:.4f}" if ninst else "0.0000"
                for a in profs) + "\n")
            t += 0.05

    marks = [("*", 2.3, "sOne"), ("square*", 2.2, "sTwo"), ("triangle*", 2.8, "sThree")]
    series = "\n".join(
        f"\\addplot[only marks, mark={mk}, mark size={sz}pt, {c},\n"
        f"         mark options={{draw=white, line width=0.5pt, fill={c}}}]\n"
        f"  table[x index=0, y index={k}] {{plots/bounds_by_instance.dat}};"
        for k, (mk, sz, c) in enumerate(marks, start=1))
    series += ("\n\\addplot[only marks, mark=diamond*, mark size=2.6pt, sFour,\n"
               "         mark options={draw=white, line width=0.5pt, fill=sFour}]\n"
               "  table[x index=0, y index=1] {plots/pdgr_by_instance.dat};")

    tex = f"""% generated by scripts/make_figures.py -- do not edit by hand
\\begin{{figure}}[tbp]
\\centering
{PREAMBLE}{colours()}
\\begin{{subfigure}}{{\\textwidth}}\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[vizbase, height=4.3cm,
  ylabel={{bounds on $\\opt$}}, ymin=-0.04, ymax=1.02, ytick={{0,0.25,0.5,0.75,1.0}},
  xmin=-0.6, xmax={len(rows) - 0.4}, xtick={{0,...,{len(rows) - 1}}},
  xticklabels={{{','.join(labels)}}},
  xticklabel style={{font=\\scriptsize}},
]
{series}
\\legend{{best $\\ub_{{\\relx}}$, $\\lb_{{\\relx}}$ (DDPS+ $=$ DPS), $\\lb_{{\\relx}}$ (CP), $\\lb_{{\\relx}}$ (PDGR)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{}}\\label{{fig.bounds.inst}}
\\end{{subfigure}}

\\vspace{{0.8em}}

\\begin{{subfigure}}{{\\textwidth}}\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[vizbase, height=4.1cm,
  xlabel={{factor of the best lower bound}},
  ylabel={{fraction of instances}},
  xmin=1, xmax=3, ymin=-0.04, ymax=1.09,
  xtick={{1,1.5,2,2.5,3}}, minor xtick={{1.25,1.75,2.25,2.75}},
  ytick={{0,0.25,0.5,0.75,1.0}},
  every axis plot/.append style={{mark=none, line width=1.1pt, const plot}},
]
\\addplot[sTwo]           table[x index=0, y index=6] {{plots/profile_lb.dat}};
\\addplot[sThree, dashed] table[x index=0, y index=3] {{plots/profile_lb.dat}};
\\addplot[sOne, dash pattern=on 1pt off 2pt, line width=1.4pt]
                         table[x index=0, y index=4] {{plots/profile_lb.dat}};
\\addplot[sFour, dash pattern=on 4pt off 1.5pt on 1pt off 1.5pt]
                         table[x index=0, y index=7] {{plots/profile_lb.dat}};
\\legend{{DDPS+ (=DPS), CP, IR, PDGR}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{}}\\label{{fig.prof.lb}}
\\end{{subfigure}}
\\caption{{\\lid{{Upper and Lower Bound:
\\subref{{fig.bounds.inst}} per instance; \\subref{{fig.prof.lb}} performance
profile of $\\lb_{{\\relx}}$.}}}}
\\label{{fig.profiles}}
\\end{{figure}}
"""
    p = os.path.join(out, "fig_bounds.tex")
    open(p, "w").write(tex)
    return p, len(rows)


# --------------------------------------------------------------------------
def last_run(rows):
    """Keep only the final run in a trace.

    A re-run appends to the existing file instead of replacing it, so a trace
    can hold several complete runs back to back. Within one run `is_last` only
    ever goes false->true and the state pool only grows between CP calls, so a
    row that resets `iter` to 0 *and* returns the pool to its opening size
    starts a new run. The tables report the final run, so the figure must too.
    """
    if not rows:
        return rows
    opening = rows[0].get("n_states")
    starts = [0] + [i for i, r in enumerate(rows)
                    if i and r.get("iter", "1").strip() in ("0", "0.0")
                    and r.get("n_states") == opening]
    return rows[starts[-1]:]


def running_min(pts):
    """The bound actually held after each iteration. Every CP call reopens from
    the trivial bound, which would draw as a jump back to 1; the running minimum
    is what the algorithm reports as glbub."""
    best = math.inf
    for s, v in pts:
        best = min(best, v)
        yield s, best


def decimate(pts, every=8, tol=1e-4):
    """Thin a monotone curve without rounding off its corners: keep the shoulder
    on either side of any real move, and a point at least every `every` steps,
    so a one-iteration cliff still draws as a cliff."""
    out, prev, last = [], None, None
    for p in pts:
        if last is None:
            out.append(p)
            last = p
        else:
            drop = last[1] - p[1] >= tol
            if drop and prev is not None and prev != last:
                out.append(prev)
            if drop or p[0] - last[0] >= every:
                out.append(p)
                last = p
        prev = p
    if prev is not None and prev != last:
        out.append(prev)
    return out


def fig_convergence(out, data, m=5):
    inst = load_instances(os.path.join(ROOT, "benchmark"))
    tdir = os.path.join(ROOT, "results", "main", "traces")
    panels = []
    for st, (name, mm) in sorted(inst.items(), key=lambda kv: display_name(kv[1][0])):
        if mm != m:
            continue
        series, restarts = {}, []
        for algo in ("CP", "IR"):
            src = os.path.join(tdir, f"{st}_{algo}.cp.csv")
            if not os.path.isfile(src):
                continue
            pts, bounds = [], []
            with open(src) as fh:
                for step, r in enumerate(last_run(list(csv.DictReader(fh)))):
                    try:
                        v = float(r["ub_relx"])
                    except (KeyError, ValueError):
                        continue
                    if not math.isfinite(v):
                        continue
                    # `iter` restarts at 0 on every CP call. CP makes one; IR
                    # makes several, each opened by a LADMM solve that re-seeds
                    # the state pool. Index the axis by row so a panel reads as
                    # one continuous run, and keep those boundaries to draw --
                    # they are what the shape of the IR curve is explained by.
                    if step and float(r.get("iter", 1)) == 0:
                        bounds.append(step)
                    pts.append((step, v))
            if not pts:
                continue
            if algo == "IR":
                restarts = bounds
            dst = f"conv_{name}_{algo}.dat"
            curve = decimate(list(running_min(pts)))
            with open(os.path.join(data, dst), "w") as fh:
                fh.write("# step best_ub_relx\n")
                for s, v in curve:
                    fh.write(f"{s} {v:.9f}\n")
            series[algo] = (dst, curve[-1][1])
        if series:
            panels.append((name, series, restarts))

    lo, hi = min(b for _, s, _ in panels for _, b in s.values()) - 0.004, 1.005
    body = []
    for k, (name, s, restarts) in enumerate(panels):
        plots = [f"\\addplot[{c}{st}, mark=none, line width=0.9pt] "
                 f"table[x index=0, y index=1] {{plots/{f}}};"
                 for (_, (f, _)), c, st in zip(sorted(s.items()),
                                               ("sOne", "sTwo"), (", dashed", ""))]
        plots += ["\\addplot[black!55, densely dotted, line width=0.8pt, mark=none] "
                  f"coordinates {{({x},{lo:.3f}) ({x},{hi})}};" for x in restarts]
        # A three-entry legend is half the height of a 3.9cm panel and there is
        # nowhere inside one it does not cross a curve, so it is drawn once for
        # the figure. Label the outer axes only, as small multiples usually do.
        body.append(
            f"\\begin{{subfigure}}{{0.49\\textwidth}}\\centering\n"
            f"\\begin{{tikzpicture}}\\begin{{axis}}[vizbase, width=\\textwidth, height=3.9cm,\n"
            f"  title={{\\footnotesize {display_name(name).replace(chr(95), chr(92) + chr(95))}}},\n"
            + ("  xlabel={cutting-plane iteration},\n" if k >= len(panels) - 2 else "")
            + ("  ylabel={$\\ub_{\\relx}$},\n" if k % 2 == 0 else "")
            + f"  xmin=0, ymin={lo:.3f}, ymax={hi},\n"
            f"  scaled x ticks=false, /pgf/number format/1000 sep={{}},\n]\n"
            + "\n".join(plots) + "\n"
            f"\\end{{axis}}\\end{{tikzpicture}}\n\\end{{subfigure}}")

    key = ("\\begin{tikzpicture}[baseline]\n"
           "\\draw[sOne, dashed, line width=0.9pt] (0,0) -- (0.42,0);\n"
           "\\node[anchor=west, font=\\scriptsize, inner sep=1.5pt] at (0.42,0) {CP};\n"
           "\\draw[sTwo, line width=0.9pt] (1.25,0) -- (1.67,0);\n"
           "\\node[anchor=west, font=\\scriptsize, inner sep=1.5pt] at (1.67,0) {IR};\n"
           "\\draw[black!55, densely dotted, line width=0.8pt] (2.4,-0.16) -- (2.4,0.16);\n"
           "\\node[anchor=west, font=\\scriptsize, inner sep=1.5pt] at (2.44,0) "
           "{LADMM restart};\n\\end{tikzpicture}")
    tex = ("% generated by scripts/make_figures.py -- do not edit by hand\n"
           "\\begin{figure}[tbp]\n\\centering\n" + PREAMBLE + colours() + "\n"
           + key + "\n\n\\vspace{0.4em}\n\n"
           # pair the panels two per row; rstrip() takes a character set, not a
           # suffix, so the separator is placed between panels rather than
           # appended and trimmed
           + "\n".join(p + (r"\hfill" if i % 2 == 0 else "\n\n\\vspace{0.6em}\n")
                        if i < len(body) - 1 else p
                        for i, p in enumerate(body)) + "\n"
           "\\caption{\\lid{Best $\\ub_{\\relx}$ so far against cumulative\n"
           "cutting-plane iteration, $m=5$.}}\n"
           "\\label{fig.conv.m5}\n\\end{figure}\n")
    p = os.path.join(out, "fig_convergence.tex")
    open(p, "w").write(tex)
    return p, len(panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=("bounds", "convergence", "all"), default="all")
    ap.add_argument("--out", default=ROOT, help="directory holding figures/ and plots/")
    a = ap.parse_args()
    fig = os.path.join(a.out, "figures")
    dat = os.path.join(a.out, "plots")
    os.makedirs(fig, exist_ok=True)
    os.makedirs(dat, exist_ok=True)
    if a.figure in ("bounds", "all"):
        p, n = fig_bounds(fig, dat)
        print(f"  wrote {p}  ({n} instances)")
    if a.figure in ("convergence", "all"):
        p, n = fig_convergence(fig, dat)
        print(f"  wrote {p}  ({n} panels)")


if __name__ == "__main__":
    main()
