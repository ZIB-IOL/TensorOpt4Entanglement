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

```bash
# 1. Julia
juliaup add 1.11.4
export JULIA_BIN='julia +1.11.4'

# 2. optional: keep packages beside the repo
mkdir -p .julia_depot && export JULIA_DEPOT_PATH="$PWD/.julia_depot:"

# 3. MOSEK licence -- needed only for solver-backed algorithms, not PDGR
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic     # or port@host

# 4. install and compile
$JULIA_BIN --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 5. verify -- solves a test LP; use --algo PDGR for a licence-free check
bash runjobs.sh --env
```

`runjobs.sh` also instantiates and precompiles before any local run or Slurm
submission, so step 4 is only needed to install without starting a benchmark.
Set `JULIA_DEPOT_PATH` before installing, or direct Julia commands use the
default cache.

Tests:

```bash
$JULIA_BIN --project=. -e 'using Pkg; Pkg.test()'         # MOSEK tests skip without a licence
python3 -m unittest discover -s test -p test_runner.py    # driver tests, no jobs submitted
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

The experiments come in three parts, each with its own script and result
directory; `bash runjobs.sh <part>` runs one of them:

| part | script | runs | results | cost |
|---|---|---|---|---|
| `main` | `scripts/exp_main.sh` | all × {Alt-SDP, LADMM, CP, IR, DPS, DDPS+, PDGR} | `results/main/` | ~210 CPU-h (PDGR ~1.4 of them) |
| `lowrank` | `scripts/exp_lowrank.sh` | m=5 × LADMM, r = 400…900 | `results/lowrank/` | ~72 CPU-h |
| `ddps_ablation` | `scripts/exp_ddps_ablation.sh` | DDPS-only variants | `results/ddps/` | ~80 CPU-h |

## Reproducing each table and figure

Each row gives the runs a table or figure reads and the command that writes it.
Add `--local` to a `runjobs.sh` command to run on this machine instead of
Slurm. Numbers are those of the revised paper; the LaTeX label is in brackets.

```bash
paper_dir=/path/to/paper
```

| paper | content | experiment | generator |
|---|---|---|---|
| Table 1 (`tab.mem`) | peak memory | `bash runjobs.sh main` | `python3 scripts/make_tables.py --table mem --out "$paper_dir/tables"` |
| Figure 1 (`fig.profiles`) | bounds per instance, lower-bound profile | `bash runjobs.sh main` | `python3 scripts/make_figures.py --figure bounds --out "$paper_dir"` |
| Table 2 (`tab.m3`) | results, m=3 | `bash runjobs.sh main --size 3` | `python3 scripts/make_tables.py --table m3 --out "$paper_dir/tables"` |
| Table 3 (`tab.m4`) | results, m=4 | `bash runjobs.sh main --size 4` | `python3 scripts/make_tables.py --table m4 --out "$paper_dir/tables"` |
| Table 4 (`tab.m5`) | results, m=5 | `bash runjobs.sh main --size 5` | `python3 scripts/make_tables.py --table m5 --out "$paper_dir/tables"` |
| Table 5 (`tab.ddps`) | DDPS vs DDPS+ ablation | `bash runjobs.sh ddps_ablation` and `bash runjobs.sh main --algo DDPS+ --algo CP --algo IR` | `python3 scripts/make_tables.py --table ddps --out "$paper_dir/tables"` |
| Table 6 (`tab.m5low`) | LADMM rank sweep, m=5 | `bash runjobs.sh lowrank` | `python3 scripts/make_tables.py --table m5low --out "$paper_dir/tables"` |
| Table 7 (`tab.m5CP`) | gap-closing CP averages, m=5 | `bash runjobs.sh main --size 5 --algo CP --algo IR` | `python3 scripts/make_tables.py --table m5cp --out "$paper_dir/tables"` |
| Figure 2 (`fig.conv.m5`) | CP and IR convergence, m=5 | `bash runjobs.sh main --size 5 --algo CP --algo IR` | `python3 scripts/make_figures.py --figure convergence --out "$paper_dir"` |

Table 5 pairs each DDPS run from `ddps_ablation` with its DDPS+ counterpart
from `main`, so it needs both. Table 7 and Figure 2 read the CP trajectories
that `main` records in `results/main/traces/`.

Everything at once:

```bash
bash runjobs.sh                                           # all three parts
python3 scripts/make_tables.py  --out "$paper_dir/tables" # all tables
python3 scripts/make_figures.py --out "$paper_dir"        # all figures
```

For debugging, a single job or a short smoke run kept apart from the
published results:

```bash
bash scripts/exp_main.sh --state state_13.jl --algo CP  # exactly one job
bash scripts/exp_main.sh -t 60 --results-dir /tmp/smoke # smoke test
```

## Notes on the generators

Results are searched in `results/main`, `results/lowrank`, `results/ddps`, `results/pdgr`, then
`results/`, first hit winning — so a stray smoke run in `results/main/` silently
shadows the published data. Send throwaway runs elsewhere with `--results-dir`.

The results tables print the exact threshold under each state for which one is
known; the values and their sources are in `data/exact_thresholds.csv`.

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
| `PDGR` | primal-dual geometric reconstruction (baseline, [`lib/PDGR`](lib/PDGR/)) |
| `DDPS+` | PPT + partial trace + scalar McCormick at every tree node |
| `DDPS`, `CP-DDPS`, `IR-DDPS` | DDPS-only counterparts, for the ablation |

Pre-rename shorthand (`A`, `LD1`, `D`, `LDL`, `PPT`, `RLT`, …) is still accepted
on the command line and when reading result files.

`PDGR` is adapted from
[EntanglementDetection.jl](https://github.com/ZIB-IOL/EntanglementDetection.jl)
and ships as a source folder loaded by the main project, so nothing extra is
installed. It takes the same `-t`, `--seed` and `--log-level` options as the
other codes; its own settings are listed by `bash runjobs.sh --help`. Bounds
found before a timeout are saved. Provenance and changes:
[`lib/PDGR/NOTICE`](lib/PDGR/NOTICE).

## Result files

```
ub_relx / lb_relx / ub_heur / feas_heur / time     reported values
relaxation / seed / julia / host                   provenance
peak_rss_mib                                       peak memory (Table 1)
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
data/           exact thresholds from the literature; published PDGR values (fallback)
test/           Julia and Python tests
scripts/        experiment drivers (exp_*.sh) and analysis (make_*.py)
results/        output
```
