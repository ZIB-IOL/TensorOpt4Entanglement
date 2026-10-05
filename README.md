# TensorOpt4Entanglement

Convex optimization over the cone of PSD tensors, applied to white-noise
entanglement thresholds. Reproduces the paper's tables and figures.

## Requirements

| | version | note |
|---|---|---|
| Julia | **1.11.4** | use this version to reproduce the root `Manifest.toml` environment |
| Mosek.jl | **10.2.0** | root manifest version; installs the MOSEK library for solver-backed algorithms |
| MosekTools.jl | 0.15.5 | MOSEK/JuMP interface in the full project |

A MOSEK **licence** is needed for runs that include solver-backed algorithms.
PDGR-only runs and environment checks need no MOSEK licence. Initial
installation needs internet access to download Julia dependencies.

## Installation

Run from the repository root. The usual `runjobs.sh` command runs
`Pkg.instantiate()` and `Pkg.precompile()` before a local run or Slurm
submission, installing all dependencies from the main project.

```bash
juliaup add 1.11.4
export JULIA_BIN='julia +1.11.4'

# Optional local package cache: uncomment BOTH lines before installing.
# mkdir -p .julia_depot
# export JULIA_DEPOT_PATH="$PWD/.julia_depot:"

# Install automatically, then run one PDGR benchmark locally.
bash runjobs.sh --local main --algo PDGR --state state_13.jl -t 600 --seed 0
```

If `.julia_depot` already exists, set `JULIA_DEPOT_PATH` as above before
installation: `runjobs.sh` selects that directory automatically, whereas
direct Julia commands otherwise use the default cache.

For runs that include MOSEK-backed algorithms, also configure the licence:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic     # or port@host
bash runjobs.sh --env
```

`--env` checks the batch environment and solves a test LP. For a PDGR-only
check, run the small product-state probe instead:

```bash
bash runjobs.sh --env --algo PDGR
```

Environment checks and dry runs do not install packages. For manual setup
without starting a benchmark:

```bash
$JULIA_BIN --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run the Julia tests. MOSEK-backed integration tests are skipped when no usable
licence is available:

```bash
$JULIA_BIN --project=. -e 'using Pkg; Pkg.test()'
```

Run the driver tests, which use local Julia/Slurm stubs and submit no jobs:

```bash
python3 -m unittest discover -s test -p test_runner.py
```

### MOSEK loading on recent glibc

MOSEK 10.2's `libmosek64.so` declares `PT_GNU_STACK = RWE`, and modern glibc
refuses to make the stack executable at `dlopen`:

```
cannot enable executable stack as shared object requires: Invalid argument
```

`fix_execstack.py` clears the flag on a copy in `.mosek_bin`. The copy must be
**MOSEK 10.2**, matching Mosek.jl 10.2.0 — `Pkg.build` rejects any other
version, so a 10.1 or 11.x install fails here.

Only if the loading error above occurs, replace `/path/to/mosek` with the
installed MOSEK directory and run:

```bash
python3 scripts/fix_execstack.py --src /path/to/mosek/10.2/tools/platform/linux64x86/bin
export MOSEKBINDIR="$PWD/.mosek_bin"
$JULIA_BIN --project=. -e 'using Pkg; Pkg.build("Mosek"); Pkg.precompile()'
```

If you are not patching, leave `MOSEKBINDIR` **unset**: it overrides Mosek.jl's
own download. Clear a leftover value with:

```bash
unset MOSEKBINDIR
```

## Site settings

Edited once at the top of `runjobs.sh`; an environment value always wins.

| setting | default |
| --- | --- |
| `JULIA_BIN` | `julia` |
| `MOSEKLM_LICENSE_FILE` | `~/mosek/mosek.lic`, else `27007@solice01.zib.de` |
| `JULIA_DEPOT_PATH` | `./.julia_depot` when that directory exists |

On a cluster also set `--mem`, `--time`, `--partition`, `--constraint` in
`run.slurm`. Keep a `--constraint`: CP, IR and LADMM spend a fixed wall-clock
budget, so a slower node returns weaker bounds and results stop being
comparable.

## Running

```bash
bash runjobs.sh --env          # can this machine run the jobs?
bash runjobs.sh --dry-run      # list jobs and report have/MISSING per table
bash runjobs.sh --check        # verify every table cell would be covered
bash runjobs.sh --local        # run here, sequentially
bash runjobs.sh                # submit to Slurm
bash runjobs.sh --help         # all options
```

A job whose result file exists is **skipped**, so re-running resumes an
interrupted run. `--force` redoes everything.

Subsets — parts are `main`, `lowrank`, `ddps_ablation`:

```bash
bash runjobs.sh --size 3                                # probe: m=3 only, 30 jobs
bash runjobs.sh main                                    # one part
bash runjobs.sh --local main --algo PDGR --state state_13.jl -t 600
bash scripts/exp_main.sh --state state_13.jl --algo CP  # exactly one job
bash scripts/exp_main.sh -t 60 --results-dir /tmp/smoke # smoke test
```

| experiment | runs | results | cost |
|---|---|---|---|
| `scripts/exp_main.sh` | all × {Alt-SDP, LADMM, CP, IR, DPS, DDPS+, PDGR} | `results/main/` | ~210 CPU-h plus PDGR |
| `scripts/exp_lowrank.sh` | m=5 × LADMM, r = 400…900 | `results/lowrank/` | ~72 CPU-h |
| `scripts/exp_ddps_ablation.sh` | DDPS-only variants | `results/ddps/` | ~80 CPU-h |

## Tables and figures

Set the destination to your paper directory:

```bash
paper_dir=/path/to/paper
python3 scripts/make_tables.py  --out "$paper_dir/tables" # all tables
python3 scripts/make_figures.py --out "$paper_dir"        # all figures
python3 scripts/make_tables.py  --list                   # registered tables
python3 scripts/make_tables.py  --manifest               # raw file behind every cell
```

Tables: `m3` `m4` `m5` `m5low` `m5cp` `ddps` `ddps3` `ddps4` `ddps5` `mem`.
Figures: `bounds` (per-instance bounds + performance profile), `convergence`.

Results are searched in `results/main`, `results/lowrank`, `results/ddps`, `results/pdgr`, then
`results/`, first hit winning — so a stray smoke run in `results/main/` silently
shadows the published data. Send throwaway runs elsewhere with `--results-dir`.

matplotlib is not a dependency: the scripts emit `.dat` files that pgfplots
`\addplot table` reads directly.

## Algorithm codes

Stable contract — `results/` filenames and the analysis scripts key off them.

| code | what it runs |
|---|---|
| `Alt-SDP` | alternating SDP |
| `LADMM` | LADMM + one CP crossover |
| `LADMM_400`…`LADMM_900` | LADMM at factorisation size r (m=5 sweep) |
| `CP` | standalone cutting plane |
| `IR` | iterative refinement (LADMM + CP) |
| `DPS` | DPS hierarchy lower bound via Ket.jl |
| `PDGR` | primal-dual geometric reconstruction |
| `DDPS+` | PPT + partial trace + scalar McCormick at every tree node |
| `DDPS`, `CP-DDPS`, `IR-DDPS` | DDPS-only counterparts, for the ablation |

Pre-rename shorthand (`A`, `LD1`, `D`, `LDL`, `PPT`, `RLT`, …) is still accepted
on the command line and when reading result files.

## PDGR

The implementation in [`lib/PDGR`](lib/PDGR/) is adapted from
[EntanglementDetection.jl](https://github.com/ZIB-IOL/EntanglementDetection.jl).
It is a self-contained source folder loaded by the main project, with no
separate package installation or external checkout required. It supports all 11 benchmarks and
uses the same `-t`, `--seed`, and `--log-level` options as the other algorithms.
Time limits are cooperative, and completed bounds and certificates are saved on timeout.
Source provenance and adaptations are recorded in [`lib/PDGR/NOTICE`](lib/PDGR/NOTICE).

Run PDGR locally:

```bash
bash runjobs.sh --local main --algo PDGR
```

Show available options, including the PDGR settings:

```bash
bash runjobs.sh --help
```

## Result files

```
ub_relx / lb_relx / ub_heur / feas_heur / time     reported values
relaxation / seed / julia / host                   provenance
relax_nvars / relax_ncons / relax_nnz              root relaxation size
mem_total_* / mem_cp_* / mem_lmo_* / mem_ladmm_*   memory per level (nested)
```

`EXACTENT_TRACE` is a path prefix; the loops append `<prefix>.cp.csv`
(`iter,is_last,ub_relx,lb_relx,b_lower,n_states`) and `<prefix>.ladmm.csv`.
The experiment scripts set it automatically. To skip the extra model build:

```bash
export EXACTENT_NO_DIAGNOSTICS=1
```

## Layout

```
src/            the ExactEntanglement package (sbb/, cuttingplane/, solvers/)
lib/PDGR/       PDGR source adapted from EntanglementDetection.jl
benchmark/      input instances
scripts/        experiment drivers (exp_*.sh) and analysis (make_*.py)
results/        output
```
