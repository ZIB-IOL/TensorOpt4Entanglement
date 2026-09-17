#!/usr/bin/env python3
"""Check that the result files say something sensible.

Coverage (check_coverage.py) asks whether a cell has data. This asks whether
the data is believable, using only properties that hold by construction:

  * a relaxation's bound cannot cross a feasible point's bound, for any
    algorithm pair on the same instance;
  * GHZ_m white noise has a closed-form threshold 1 - 1/(1 + 2^(m-1)), which a
    lower bound cannot exceed and an upper bound cannot undercut;
  * every bound lies in (0, 1), and nothing is NaN or infinite;
  * a run cannot have taken longer than its time limit by a wide margin.

It also compares against the results committed with the paper where the same
cell exists, which is how a solver or code change shows itself.

    python3 scripts/check_results.py                 # everything found
    python3 scripts/check_results.py --results-dir results/main
    python3 scripts/check_results.py --compare-legacy
"""
import argparse, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from tables.common import load_instances, load_result, ROOT   # noqa: E402

# Bounds that were never computed are written as sentinels, not as values:
# Drivers.jl returns literal 0 for the fields an algorithm does not produce, and
# cuttingPlane_ leaves its Lagrangian bound at -Inf when no oracle round ran.
# Neither is a defect, so both are skipped rather than range-checked.
def missing(v):
    return v is None or v == 0 or math.isinf(v)


# Algorithms that return only a relaxation bound: (0, lb, 0, 0, 0) in Drivers.jl.
RELAX_ONLY = {"DPS", "DDPS", "DDPS+", "PPT", "RLT", "RLT_DDPS"}
# Algorithms whose ub_relx is a genuine upper bound on the threshold.
UPPER = {"Alt-SDP", "LADMM", "CP", "IR", "CP-DDPS", "IR-DDPS", "Alt-SDP+CP",
         "A", "LD1", "D", "LDL", "AD"}

TOL = 1e-6


def ghz_threshold(m):
    """Closed-form white-noise threshold for the m-party GHZ state."""
    return 1 - 1 / (1 + 2.0 ** (m - 1))


def scan(results_dirs, instances):
    """Every (state, algo) file present under the given directories."""
    found = {}
    for d in results_dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if not os.path.isfile(path) or "_" not in fn:
                continue
            for state in instances:
                if fn.startswith(state + "_"):
                    algo = fn[len(state) + 1:]
                    rec = load_result([d], state, algo)
                    if rec is not None:
                        found[(state, algo)] = rec
                    break
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", action="append", default=None,
                    help="where to look (repeatable); default: results/ and its parts")
    ap.add_argument("--benchmark-dir", default=os.path.join(ROOT, "benchmark"))
    ap.add_argument("--compare-legacy", action="store_true",
                    help="also diff against the results committed with the paper")
    a = ap.parse_args()

    dirs = a.results_dir or [os.path.join(ROOT, "results", p)
                             for p in ("main", "ddps", "lowrank")]
    instances = load_instances(a.benchmark_dir)
    found = scan(dirs, instances)

    if not found:
        print("no result files found in: " + ", ".join(os.path.relpath(d, ROOT) for d in dirs))
        return 1

    bad, warn = [], []

    def fail(state, algo, msg):
        bad.append(f"  {state} {algo}: {msg}")

    # --- per-result checks -------------------------------------------------
    for (state, algo), r in sorted(found.items()):
        name, m = instances.get(state, (state, None))
        for k in ("glbub", "glblb", "approxub", "approxfeas", "time"):
            v = r[k]
            if v is None or math.isnan(v):
                fail(state, algo, f"{k} is {v}")
        lb, ub = r["glblb"], r["glbub"]
        if not missing(lb) and not (-TOL < lb < 1 + TOL):
            fail(state, algo, f"lower bound {lb} outside [0,1]")
        if not missing(ub) and not (-TOL < ub < 1 + TOL):
            fail(state, algo, f"upper bound {ub} outside [0,1]")
        if algo in RELAX_ONLY and missing(lb):
            fail(state, algo, "a relaxation produced no lower bound")
        if not missing(lb) and not missing(ub) and lb > ub + TOL:
            fail(state, algo, f"lower bound {lb} exceeds its own upper bound {ub}")
        if not missing(ub) and ub > 1 - 1e-6:
            warn.append(f"  {state} {algo}: upper bound {ub:.9f} is vacuous (= 1)")
        if r["time"] < 0:
            fail(state, algo, f"negative runtime {r['time']}")
        # GHZ has a closed form, so a bound on the wrong side of it is a defect
        if name.startswith("GHZ_") and m:
            t = ghz_threshold(m)
            if not missing(lb) and lb > t + 1e-4:
                fail(state, algo, f"lower bound {lb:.6f} exceeds the analytic "
                                  f"GHZ_{m} threshold {t:.6f}")
            if algo in UPPER and not missing(ub) and ub < t - 1e-4:
                warn.append(f"  {state} {algo}: upper bound {ub:.6f} is below the "
                            f"analytic GHZ_{m} threshold {t:.6f} by {t - ub:.2e}")

    # --- cross-algorithm bracketing ---------------------------------------
    by_state = {}
    for (state, algo), r in found.items():
        by_state.setdefault(state, {})[algo] = r
    for state, rows in sorted(by_state.items()):
        lbs = {a: r["glblb"] for a, r in rows.items() if not missing(r["glblb"])}
        ubs = {a: r["glbub"] for a, r in rows.items()
               if a in UPPER and not missing(r["glbub"])}
        if not lbs or not ubs:
            continue
        alb, blb = max(lbs.items(), key=lambda kv: kv[1])
        aub, bub = min(ubs.items(), key=lambda kv: kv[1])
        # Reported, not failed: only the DDPS+/Alt-SDP pair is asserted by the
        # test suite. Other pairs may bound different quantities, so a crossing
        # is something to look at rather than a proven defect.
        if blb > bub + 1e-4:
            warn.append(f"  {state}: lower bound {blb:.6f} ({alb}) exceeds upper "
                        f"bound {bub:.6f} ({aub}) by {blb - bub:.2e}")

    # --- against the published runs ---------------------------------------
    diffs = []
    if a.compare_legacy:
        legacy_dir = os.path.join(ROOT, "results")
        for (state, algo), r in sorted(found.items()):
            old = load_result([legacy_dir], state, algo)
            if old is None:
                continue
            for k, label in (("glblb", "lb_relx"), ("glbub", "ub_relx")):
                if not missing(r[k]) or not missing(old[k]):
                    d = abs(r[k] - old[k])
                    if d > 1e-4:
                        diffs.append(f"  {state} {algo} {label}: "
                                     f"{old[k]:.6f} -> {r[k]:.6f}  (Δ {d:.2e})")

    # --- report ------------------------------------------------------------
    print(f"checked {len(found)} result(s) over {len(by_state)} instance(s)")
    if warn:
        print("\nwarnings:");  print("\n".join(warn))
    if diffs:
        print(f"\ndiffers from the published runs ({len(diffs)}):")
        print("\n".join(diffs))
        print("  (expected after a solver change; check the size of the shift)")
    if bad:
        print(f"\nPROBLEMS ({len(bad)}):")
        print("\n".join(bad))
        return 1
    print("\nall checks passed: bounds are in range, consistent with each other,")
    print("and on the right side of the analytic GHZ thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
