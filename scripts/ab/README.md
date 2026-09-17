# Bound-disagreement A/B

Answers one question: when the current branch and the published legacy runs
disagree on a bound, is it the **code** or the **machine**?

The two checkouts are made to differ in source only — `setup.sh` copies this
branch's `Manifest.toml` into the legacy worktree, so both solve with the same
Mosek and the same JuMP against the same depot. Each pair is then run with
identical resources, and on Slurm under the same `--constraint`, so the CPU is
held fixed too.

```bash
bash scripts/ab/setup.sh          # legacy worktree on this branch's solver stack
bash scripts/ab/submit.sh         # both sides of every cell in cells.txt
python3 scripts/ab/compare.py ab/<stamp>
```

`--local` runs sequentially instead of submitting; `-t 600` shortens the budget
for a quick pass.

## Reading the verdict

`compare.py` parses the per-iteration lines both sides print and finds the
first round where they part:

- **identical for all shared rounds** — the code computes the same thing; any
  difference in the final numbers is how far each run got, i.e. timing.
- **PARTS at iteration N** — the two trees diverge inside the run. That is a
  code difference, and N localises it.
- **single-solve algorithm** — no trajectory exists (DPS, DDPS+); only the
  final bounds can be compared.

`cells.txt` ends with two DDPS+ controls, which agreed to 10+ digits between
branches. If a control parts, the harness is at fault, not the code under test.

## What is already known

Run locally at a short budget, before this harness existed:

- CP on GHZ_5: state counts and `primalobj` identical for 80 rounds, upper
  bounds to 1e-10.
- LADMM (`LADMM_900` vs `LDR5`) on Dicke_5_1: `ub_heur` identical to 15 digits.

So the open question is the m=5 dual-bound cluster at **full** budget, which is
what these jobs are for — a short budget puts both sides in the regime where
the bound is meaningless and nothing can be concluded.
