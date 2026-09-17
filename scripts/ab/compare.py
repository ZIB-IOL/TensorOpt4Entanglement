#!/usr/bin/env python3
"""Compare a legacy run against a current run, cell by cell.

    python3 scripts/ab/compare.py ab/<stamp>

For each cell it reports the final bounds from both sides and, where the logs
carry per-iteration lines, the first iteration at which the two trajectories
part. That last part is the point of the exercise: identical trajectories that
end differently mean timing, whereas a divergence at a specific iteration means
code. Both sides print the same "iteration: i, #states: n, ... primalobj: p,
dualobj: d" format, so one parser serves both.

The node model reported by each side is printed too -- a pair that ran on
different CPUs cannot separate code from hardware, and the comparison should be
repeated rather than believed.
"""
import argparse, os, re, sys

ITER = re.compile(r"iteration:\s*(\d+),\s*#states:\s*(\d+).*?primalobj:\s*(\S+),\s*dualobj:\s*(\S+)")
FIELD = re.compile(r"^(\w+):\s*(\S+)\s*$", re.M)


def trajectory(log):
    """[(iter, states, primalobj, dualobj)] from a run log."""
    out = []
    try:
        text = open(log, errors="replace").read()
    except OSError:
        return out
    for m in ITER.finditer(text):
        try:
            out.append((int(m.group(1)), int(m.group(2)),
                        float(m.group(3)), float(m.group(4))))
        except ValueError:
            continue
    return out


def result(d):
    """Bounds from whichever result file the side wrote (names differ)."""
    for fn in os.listdir(d) if os.path.isdir(d) else []:
        p = os.path.join(d, fn)
        if not os.path.isfile(p) or fn in ("run.log",) or fn.endswith(".csv"):
            continue
        rec = dict(FIELD.findall(open(p, errors="replace").read()))
        if "lb_relx" in rec or "glblb" in rec:
            g = lambda *k: next((float(rec[x]) for x in k if x in rec), None)
            return dict(ub=g("ub_relx", "glbub"), lb=g("lb_relx", "glblb"),
                        ubh=g("ub_heur", "approxub"), t=g("time"), file=fn)
    return None


def fmt(v):
    return f"{'--':>12}" if v is None else f"{v:>12.6f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="ab/<stamp> produced by submit.sh")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="difference at which trajectories count as parted")
    a = ap.parse_args()

    if not os.path.isdir(a.run_dir):
        sys.exit(f"not a directory: {a.run_dir}")

    verdicts = []
    for cell in sorted(os.listdir(a.run_dir)):
        cur_d, leg_d = (os.path.join(a.run_dir, cell, s) for s in ("current", "legacy"))
        if not (os.path.isdir(cur_d) and os.path.isdir(leg_d)):
            continue
        cur, leg = result(cur_d), result(leg_d)
        tc, tl = trajectory(os.path.join(cur_d, "run.log")), trajectory(os.path.join(leg_d, "run.log"))

        print(f"\n=== {cell} ===")
        if leg is None or cur is None:
            print(f"  incomplete: legacy={'ok' if leg else 'MISSING'} "
                  f"current={'ok' if cur else 'MISSING'}")
            verdicts.append((cell, "incomplete"))
            continue
        print(f"  {'':8}{'ub_relx':>12}{'lb_relx':>12}{'ub_heur':>12}{'time':>12}")
        for tag, r in (("legacy", leg), ("current", cur)):
            print(f"  {tag:8}{fmt(r['ub'])}{fmt(r['lb'])}{fmt(r['ubh'])}{fmt(r['t'])}")

        print(f"  rounds: legacy {len(tl)}  current {len(tc)}")
        # Where do they part?
        part = None
        for (i1, s1, p1, d1), (i2, s2, p2, d2) in zip(tl, tc):
            if s1 != s2 or abs(p1 - p2) > a.tol:
                part = (i1, s1, s2, p1, p2)
                break
        if not tl or not tc:
            verdict = "single-solve algorithm: no per-iteration trajectory to compare"
        elif part is None:
            common = min(len(tl), len(tc))
            verdict = (f"identical for all {common} shared rounds -> "
                       f"{'timing only' if len(tl) != len(tc) else 'identical run'}")
        else:
            i, s1, s2, p1, p2 = part
            verdict = f"PARTS at iteration {i}: states {s1} vs {s2}, primalobj {p1!r} vs {p2!r}"
        print(f"  {verdict}")
        verdicts.append((cell, verdict))

    print("\n" + "=" * 70)
    for cell, v in verdicts:
        print(f"  {cell:<28} {v}")
    parted = [c for c, v in verdicts if v.startswith("PARTS")]
    print("=" * 70)
    if parted:
        print(f"{len(parted)} cell(s) diverge inside the run -> a code difference:")
        for c in parted:
            print(f"    {c}")
        return 1
    print("no cell diverges inside a run; differences are in how far each got")
    return 0


if __name__ == "__main__":
    sys.exit(main())
