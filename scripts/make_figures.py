#!/usr/bin/env python3
"""Generate the paper's figures, as pgfplots/TikZ, from saved results.

The companion of make_tables.py: nothing here is drawn by hand, so a figure
cannot drift from the results behind it. Each figure writes its own data files
and a self-contained `figure` environment the paper `\\input`s.

    python3 scripts/make_figures.py                      # both, into ./figures and ./plots
    python3 scripts/make_figures.py --out <paper dir>    # write beside the paper
    python3 scripts/make_figures.py --figure bounds

  bounds       per-instance bounds + performance profile of the lower bound
  convergence  LADMM objectives and CP upper bounds within IR, m = 5

Captions here are deliberately short and descriptive; findings belong in the
section that discusses the figure, not in its caption.
The convergence figure uses [H]; the manuscript must load the float package.

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
    # Same ordering as the tables (Context.states): subsystem count, then name.
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
        # the best upper bound over every method in the tables, PDGR included,
        # so the marker matches the bold entry of Tables 2-4
        pu = (pdgr.get(display_name(name)) or {}).get("ub")
        ubs = [v for v in (g("Alt-SDP", "glbub"), g("LADMM", "glbub"),
                          g("CP", "glbub"), g("IR", "glbub"), pu)
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
\\begin{{figure}}[!htbp]
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
\\caption{{\\lid{{Upper and lower bounds:
\\subref{{fig.bounds.inst}} best upper bound over all methods and lower bounds per instance;
\\subref{{fig.prof.lb}} performance profile of $\\lb_{{\\relx}}$.}}}}
\\label{{fig.profiles}}
\\end{{figure}}
"""
    p = os.path.join(out, "fig_bounds.tex")
    open(p, "w").write(tex)
    return p, len(rows)


# --------------------------------------------------------------------------
def decimate_extrema(pts, bins=250):
    """Retain endpoints and extrema in log-x bins, including upward moves.

    LADMM objectives are nonmonotone. Every retained coordinate is an actual
    recorded iterate.
    """
    if len(pts) <= 4 * bins:
        return pts
    lo, hi = math.log10(pts[0][0]), math.log10(pts[-1][0])
    groups = {}
    for i, point in enumerate(pts):
        x = point[0]
        key = min(bins - 1, int(bins * (math.log10(x) - lo) / (hi - lo)))
        groups.setdefault(key, []).append(i)
    keep = set()
    for indices in groups.values():
        keep.update((indices[0], indices[-1],
                     min(indices, key=lambda i: pts[i][1]),
                     max(indices, key=lambda i: pts[i][1])))
    return [pts[i] for i in sorted(keep)]


def ir_stage_segments(prefix):
    """Place recorded iterations on one axis using the shared IR stage log.

    Calls without iteration rows retain their place in the call order, but
    supply no plotted values. Missing stage numbers are valid when IR skips a
    LADMM call; duplicated stages instead indicate appended runs.
    """
    rows = {}
    for kind, fields in (("stages", {"stage", "algorithm", "event", "elapsed"}),
                         ("cp", {"stage", "iter", "ub_relx"}),
                         ("ladmm", {"stage", "iter", "z"})):
        path = f"{prefix}.{kind}.csv"
        if not os.path.isfile(path):
            raise ValueError(f"Ordered IR convergence requires a stage-labelled trace: {path}")
        with open(path) as fh:
            reader = csv.DictReader(fh)
            if not fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing shared stage labels or trace fields: {path}")
            rows[kind] = list(reader)

    stages, by_id = [], {}
    active, previous_elapsed, previous_stage = None, -math.inf, 0
    for row in rows["stages"]:
        stage, elapsed = int(row["stage"]), float(row["elapsed"])
        algo, event = row["algorithm"].upper(), row["event"]
        if algo not in ("CP", "LADMM") or not math.isfinite(elapsed) or elapsed < previous_elapsed:
            raise ValueError(f"Invalid IR stage event: {prefix}: {row}")
        previous_elapsed = elapsed
        if event == "start":
            if active is not None or stage <= previous_stage:
                raise ValueError(f"Overlapping or appended IR runs: {prefix}; use fresh traces")
            active = {"stage": stage, "algorithm": algo, "start": elapsed,
                      "end": None, "rows": [], "points": [], "order": len(stages)}
            stages.append(active)
            by_id[stage] = active
            previous_stage = stage
        elif event in ("end", "error"):
            if active is None or (stage, algo) != (active["stage"], active["algorithm"]):
                raise ValueError(f"Unmatched IR stage event: {prefix}: {row}")
            active["end"] = elapsed
            active = None
        else:
            raise ValueError(f"Unknown IR stage event: {prefix}: {event}")
    if not stages:
        raise ValueError(f"No IR stage events: {prefix}")

    for kind, algo, field in (("cp", "CP", "ub_relx"), ("ladmm", "LADMM", "z")):
        previous_order = -1
        for row in rows[kind]:
            stage, iteration = int(row["stage"]), int(row["iter"])
            call = by_id.get(stage)
            if call is None or call["algorithm"] != algo or call["order"] < previous_order:
                raise ValueError(f"Iteration has no matching ordered IR stage: {prefix}: {row}")
            if iteration < 0 or (call["rows"] and iteration <= call["rows"][-1][0]):
                raise ValueError(f"Repeated or unordered iteration: {prefix}: {row}")
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite IR {algo} objective: {prefix}: {row}")
            call["rows"].append((iteration, 1 - value if algo == "LADMM" else value))
            previous_order = call["order"]

    step, best = 0, math.inf
    for call in stages:
        for iteration, value in call.pop("rows"):
            step += 1
            if call["algorithm"] == "CP":
                best = min(best, value)
                value = best
            call["points"].append((step, value, iteration))
    for algo in ("CP", "LADMM"):
        if not any(call["points"] for call in stages if call["algorithm"] == algo):
            raise ValueError(f"No recorded IR {algo} iterations: {prefix}")
    return stages


def decimate_stages(calls):
    """Preserve every stage's endpoints and extrema while thinning long traces."""
    points = [point for call in calls for point in call["points"]]
    keep = {point[0] for point in decimate_extrema(points)}
    for call in calls:
        values = call["points"]
        if values:
            keep.update(point[0] for point in
                        (values[0], values[-1], min(values, key=lambda p: p[1]),
                         max(values, key=lambda p: p[1])))
    return [[point for point in call["points"] if point[0] in keep] for call in calls]


def fig_convergence(out, data, m=5):
    inst = load_instances(os.path.join(ROOT, "benchmark"))
    tdir = os.path.join(ROOT, "results", "main", "traces")
    panels = []
    for st, (name, mm) in sorted(inst.items(), key=lambda kv: display_name(kv[1][0])):
        if mm != m:
            continue
        stages = ir_stage_segments(os.path.join(tdir, f"{st}_IR"))
        series = {}
        for algo in ("CP", "LADMM"):
            calls = [call for call in stages if call["algorithm"] == algo]
            dst = f"conv_{name}_IR{'_LADMM' if algo == 'LADMM' else ''}.dat"
            with open(os.path.join(data, dst), "w") as fh:
                field = "ub_heur" if algo == "LADMM" else "best_ub_relx"
                fh.write(f"# iteration {field} stage local_iter\n")
                first = True
                for call, points in zip(calls, decimate_stages(calls)):
                    if not points:
                        continue
                    if not first:
                        fh.write("nan nan -1 -1\n")
                    for step, value, iteration in points:
                        fh.write(f"{step} {value:.12f} {call['stage']} {iteration}\n")
                    first = False
            series[algo] = dst
        ymin = min(point[1] for call in stages for point in call["points"])
        ymax = max(point[1] for call in stages for point in call["points"])
        margin = max(0.001, 0.04 * (ymax - ymin))
        panels.append((name, series, ymin - margin, ymax + margin))

    if not panels:
        raise ValueError(f"No IR convergence traces for m={m}")
    body = []
    for k, (name, series, lo, hi) in enumerate(panels):
        plots = [f"\\addplot[{style}, unbounded coords=jump, line width=0.9pt] "
                 f"table[x index=0, y index=1] {{plots/{f}}};"
                 for f, style in ((series["LADMM"], "sTwo, mark=none"),
                                  (series["CP"], "sOne, dashed, mark=*, mark size=0.65pt"))]
        # Draw one shared legend and label only the outer axes.
        body.append(
            f"\\begin{{subfigure}}{{0.49\\textwidth}}\\centering\n"
            f"\\begin{{tikzpicture}}\\begin{{axis}}[vizbase, width=\\textwidth, height=3.9cm,\n"
            f"  title={{\\footnotesize {display_name(name).replace(chr(95), chr(92) + chr(95))}}},\n"
            + ("  xlabel={\\lid{Iteration in IR order}},\n" if k >= len(panels) - 2 else "")
            + ("  ylabel={\\lid{objective / bound}},\n" if k % 2 == 0 else "")
            + f"  xmode=log, log basis x=10, xmin=1, ymin={lo:.6f}, ymax={hi:.6f},\n"
            "  scaled y ticks=false, yticklabel style={/pgf/number format/fixed,\n"
            "    /pgf/number format/precision=3},\n]\n"
            + "\n".join(plots) + "\n"
            f"\\end{{axis}}\\end{{tikzpicture}}\n\\end{{subfigure}}")

    key = ("\\begin{tikzpicture}[baseline]\n"
           "\\draw[sOne, dashed, line width=0.9pt] (0,0) -- (0.42,0);\n"
           "\\fill[sOne] (0.21,0) circle[radius=0.65pt];\n"
           "\\node[anchor=west, font=\\scriptsize, inner sep=1.5pt] at (0.42,0) {CP ($\\ub_{\\relx}$)};\n"
           "\\draw[sTwo, line width=0.9pt] (2.35,0) -- (2.77,0);\n"
           "\\node[anchor=west, font=\\scriptsize, inner sep=1.5pt] at (2.77,0) {LADMM ($\\ub_{\\heur}$)};\n"
           "\\end{tikzpicture}")
    tex = ("% generated by scripts/make_figures.py -- do not edit by hand\n"
           "\\begin{figure}[H]\n\\centering\n" + PREAMBLE + colours() + "\n"
           + key + "\n\n\\vspace{0.4em}\n\n"
           # pair the panels two per row; rstrip() takes a character set, not a
           # suffix, so the separator is placed between panels rather than
           # appended and trimmed
           + "\n".join(p + (r"\hfill" if i % 2 == 0 else "\n\n\\vspace{0.6em}\n")
                        if i < len(body) - 1 else p
                        for i, p in enumerate(body)) + "\n"
           "\\caption{\\lid{LADMM and CP trajectories within IR for $m=5$.}}\n"
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
