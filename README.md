# TensorOpt4Entanglement — quick setup & run

Short instructions to install Julia packages used by this project and to run jobs using `jobs.sh` and a Slurm array (`run.slurm`).

## Prerequisites
- Linux machine with Julia 1.11.x (tested with 1.11.6) and Mosek.
- A working Slurm cluster.

> **Julia version matters.** Use 1.11.x. On Julia 1.12+ the benchmark loader is
> fine, but the pinned `Manifest.toml` resolves differently; if you use
> [juliaup](https://github.com/JuliaLang/juliaup):
> ```bash
> juliaup add 1.11.6
> julia +1.11.6 --project=. -e 'using Pkg; Pkg.instantiate()'
> ```

> **Mosek needs a licence file.** The Julia package installs without one, but
> every solver call then fails with `License cannot be located`. Put your
> licence at `~/mosek/mosek.lic`, or point at it explicitly:
> ```bash
> export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
> ```
> Check it works with:
> ```bash
> julia +1.11.6 --project=. -e 'using JuMP, MosekTools; m=Model(Mosek.Optimizer); set_silent(m); @variable(m,x>=1.5); @objective(m,Min,x); optimize!(m); println(termination_status(m))'
> ```
- Project directory layout:
    - `src/` — the `ExactEntanglement` package
    - `scripts/run_experiment.jl` — command-line entry point
    - `benchmark/` — input instances
    - `results/` — output directory
    - `test/` — test suite (`julia --project=. -e 'using Pkg; Pkg.test()'`)
    - `jobs.sh` — job list generator (already provided)

### Source layout

```
src/
  ExactEntanglement.jl   module: imports, includes, exports
  Types.jl               Status codes, Param, run clock
  MathUtils.jl           index maps, McCormick helpers, small numerics
  Solver.jl              Mosek setup, conic result classification
  Lift.jl                the smooth lift Psi and its analytic gradient
  sbb/                   spatial branch-and-bound (the LMO)
  cuttingplane/          cutting-plane master problem
  solvers/               lifted nonconvex solvers (LADMM, dual ALM, Alt-SDP)
  Drivers.jl             size presets, algorithm table, runEntangle
```

Naming: `lowerCamelCase` for functions, `UpperCamelCase` for types. The
`snake_case` methods (`manifold_dimension`, `retract_project!`, ...) are
interface methods whose names are fixed by ManifoldsBase/Manopt.

## Install required Julia packages
From the project root (this repo), prefer using the project environment if present:
```bash
julia +1.11.6 --project=. -e 'using Pkg; Pkg.instantiate()'
```
This will install packages listed in `Project.toml`/`Manifest.toml` if they exist.

## Prepare and run jobs locally
1. Edit the top of `jobs.sh` to set up experiments:
The algorithm codes accepted by `-a` (and used in `results/` filenames and by
`scripts/make_tables.py`) are:

| code | paper name | what it runs |
|------|------------|--------------|
| `LD` / `LD0` / `LDL` | IR | iterative refinement (LADMM + cutting plane) |
| `LD1` | LADMM | one refinement iteration = standalone LADMM plus one crossover |
| `LDR0`…`LDR5` | LADMM_r | as `LD1` with factorisation size r = 400…900 (m = 5 only) |
| `D` | CP | standalone cutting plane |
| `A` | Alt-SDP | alternating SDP |
| `AD` | Alt-SDP + CP | alternating SDP inside the refinement loop |
| `PPT` | DPS | DPS hierarchy lower bound via Ket.jl |
| `RLT` | DDPS+ | tensor-RLT lower bound at the sBB root |

```bash
# at the top of jobs.sh (example)
algorithms=("LD1" "LDR0" "LDR1" "LDR2" "LDR3" "LDR4" "LDR5" "LDL" "D" "A")
timelimit=-1
datapath="$PWD/benchmark"
resultpath="$PWD/results"
juliabin="julia"
```
Example: automatic timelimit (`-1`) and auto-detect Julia executable.

- Use `timelimit=-1` to let `jobs.sh` pick a sensible timeout per instance (based on filename/size).
- Detect the Julia binary at runtime so `juliabin` points to the actual executable found on `PATH`.

2. Run `jobs.sh` with bash from the project root to create job lists:
```bash
# simple run (uses defaults inside jobs.sh)
bash jobs.sh
# job_list.txt will be created in the project root
```

3. Set up Slurm environment (see `run.slurm`):
```bash
# Example Slurm header for distributed job array
#SBATCH --job-name=distributed_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5222   # hardware feature or tag (e.g. "Gold5222")
#SBATCH --output=/dev/null      # redirect stdout (set to a file for debugging)
#SBATCH --error=/dev/null       # redirect stderr (set to a file for debugging)
```

### Notes
- `--job-name`: human-readable job name.
- `--ntasks`: total MPI/tasks (1 for single-task jobs; increase for multi-task runs).
- `--cpus-per-task`: threads per task (set to number of threads your Julia worker uses).
- `--mem`: memory per node (or per job depending on cluster config). Adjust to workload.
- `--time`: wall-clock limit (format HH:MM:SS). Jobs exceeding this are killed.
- `--partition`: target partition/queue on the cluster.
- `--constraint`: node feature/label filter (matches nodes with this property).
- `--output` / `--error`: currently discarding logs to `/dev/null`; for debugging replace with a path like `results/%x-%j.out` and `results/%x-%j.err`.

4. Set up system environment variables used by `runjobs.sh`:
```bash
export JULIA_DEPOT_PATH=".julia_depot"
export MOSEKHOME=
export MOSEKLM_LICENSE_FILE=
```

5. Run all the experiments:
```bash
bash runjobs.sh
```


## Reproducing the paper's tables

The workflow is split in two: **bash scripts run experiments**, **python scripts
analyse the saved results**. The three experiments are disjoint — no
instance/algorithm pair is executed twice — and each writes into its own
directory, so every raw file is attributable to the run that produced it.

### Experiments (bash)

| script | what it runs | results | cost |
|---|---|---|---|
| `scripts/exp_main.sh` | all instances × {Alt-SDP, LADMM, CP, IR, DPS, DDPS+} | `results/main/` | ~210 CPU-h |
| `scripts/exp_lowrank.sh` | m=5 × LADMM with r = 400…900 | `results/lowrank/` | ~72 CPU-h |
| `scripts/exp_ddps_ablation.sh` | m=3,4 × the DDPS-only variants | `results/ddps/` | ~44 CPU-h |
| `scripts/run_all_experiments.sh` | all three, or a named subset | | |

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic

bash scripts/exp_main.sh                          # one experiment
bash scripts/run_all_experiments.sh               # all of them
bash scripts/run_all_experiments.sh main ddps_ablation

bash scripts/run_all_experiments.sh --dry-run     # list jobs, run nothing
bash scripts/run_all_experiments.sh --force       # redo every experiment
bash scripts/exp_main.sh -t 60                    # smoke test (not paper settings)
bash runjobs.sh                                  # on the cluster (see below)
bash scripts/exp_main.sh --help                   # all options
```

Finished jobs are skipped, so an interrupted run resumes. `M="3"` restricts an
experiment to one subsystem count.

### On the cluster

```bash
bash runjobs.sh                  # generate job lists and submit all three
bash runjobs.sh main             # just one experiment
bash runjobs.sh --dry-run        # generate lists, submit nothing
```

`runjobs.sh` instantiates the project, then calls the same `scripts/exp_*.sh`
with `--slurm` so they emit job lists instead of running, and submits each as a
Slurm array. **The experiment design lives in one place**: the cluster runs
exactly what a local run would, because both go through the same scripts.

```
runjobs.sh  ->  scripts/exp_*.sh --slurm  ->  job_list_<part>.txt  ->  run.slurm
```

Each line is `<instance> <algorithm> <timelimit> <results_dir>`; the Julia
binary is deliberately not in the list, since that is a property of the machine
— set `JULIA_BIN` (default `julia`). `run.slurm` accepts either an array index
(`SLURM_ARRAY_TASK_ID`) or a line number as `$1`, and `JOB_LIST` selects which
list, so the three experiments never collide.

`exp_main.sh` is the raw material for most of the analysis: it produces the
final bounds, the CP and LADMM trajectories, and the size/memory diagnostics in
one pass, so the convergence plots, the gap-closing table and the performance
profiles all derive from it rather than needing their own runs.

### Analysis (python)

| command | produces |
|---|---|
| `make_tables.py --table m3\|m4\|m5` | the three main results tables |
| `make_tables.py --table m5low` | the LADMM rank sweep |
| `make_tables.py --table m5cp` | gap-closing CP averages (from trajectories) |
| `make_tables.py --table ddps3\|ddps4\|ddps5` | DDPS vs DDPS+ ablation |
| `make_tables.py --table mem3\|mem4\|mem5` | per-level memory |
| `make_tables.py --table size3\|size4\|size5` | relaxation size and file size |
| `make_tables.py --manifest` | **the raw file behind every table cell** |
| `make_tables.py --list` | which module and experiment each table comes from |
| `performance_profile.py` | performance profiles (Dolan–Moré) |
| `summarize_traces.py` | convergence trajectories |

`make_tables.py` is a dispatcher; each table family lives in its own module,
mirroring the experiment split:

| module | tables | filled by |
|---|---|---|
| `tables/main.py` | `m3` `m4` `m5` | `exp_main.sh` |
| `tables/lowrank.py` | `m5low` | `exp_lowrank.sh` |
| `tables/gapclosing.py` | `m5cp` | `exp_main.sh` (trajectories) |
| `tables/ddps.py` | `ddps3` `ddps4` `ddps5` | `exp_ddps_ablation.sh` + `exp_main.sh` |
| `tables/memory.py` | `mem3` `mem4` `mem5` | `exp_main.sh` |
| `tables/size.py` | `size3` `size4` `size5` | `exp_main.sh`, `exp_ddps_ablation.sh` |
| `tables/common.py` | — | loading, formatting, provenance |

Adding a table means adding a module with a `TABLES` dict and a `build()`;
the dispatcher picks it up from `tables/__init__.py`.

```bash
python3 scripts/make_tables.py --table all --out tables/
python3 scripts/make_tables.py --manifest          # provenance of every cell
python3 scripts/performance_profile.py --out plots/
python3 scripts/summarize_traces.py --out plots/
```

`--manifest` answers "which raw files back this table": it lists every cell with
its file path, and marks cells with no data as `MISSING` along with the
experiment that would produce them. With `--out` it is written to
`MANIFEST.txt` so the mapping is a checked-in artifact rather than only
terminal output:

```bash
python3 scripts/make_tables.py --manifest --out tables/
```

Results are searched in `results/main`, `results/lowrank`, `results/ddps`, then
`results/` itself (the flat files published with the paper), first hit winning —
so a fresh run shadows the published one without deleting it.

> **Shadowing cuts both ways.** A short smoke run left in `results/main/` will
> silently take precedence over the published data. Send throwaway runs
> somewhere else:
> ```bash
> bash scripts/exp_main.sh -t 60 --results-dir /tmp/smoke
> ```

matplotlib is not a dependency: the plot scripts emit whitespace-separated
`.dat` files that `\addplot table` reads directly.

### Algorithm codes

The `-a` codes are a stable contract — `results/` filenames and the analysis
scripts key off them.

| code | paper name | |
|------|------------|---|
| `LD` / `LD0` / `LDL` | IR | iterative refinement |
| `LD1` | LADMM | one refinement iteration = standalone LADMM + crossover |
| `LDR0`…`LDR5` | LADMM_r | r = 400…900 (m = 5) |
| `D` | CP | cutting plane |
| `A` | Alt-SDP | alternating SDP |
| `AD` | Alt-SDP + CP | |
| `PPT` | DPS | DPS hierarchy bound via Ket.jl |
| `RLT` | DDPS+ | tensor-RLT bound at the sBB root |
| `RLT_DDPS`, `D_DDPS`, `LDL_DDPS` | — | DDPS-only counterparts, for the ablation |

### What each run records

Result files carry the bounds plus provenance and diagnostics:

```
glbub / glblb / approxub / approxfeas / time      the reported values
relaxation / seed / julia / host                   provenance
relax_nvars / relax_ncons / relax_nnz              root relaxation size
relax_cbf_bytes                                    its size on disk (CBF)
mem_total_* / mem_cp_* / mem_lmo_* / mem_ladmm_*   memory per level
```

The relaxation is measured **without solving it**, so those figures are
deterministic and machine-independent; only the memory fields depend on the
host. Memory is attributed to each algorithmic level, and the phases **nest**
(`:total` ⊃ `:cp` ⊃ `:lmo`), so the figures are inclusive and `:lmo` is
reported separately to show its share. Set `EXACTENT_NO_DIAGNOSTICS=1` to skip
the extra model build.

### DDPS vs DDPS+

DDPS+ *is* the relaxation the sBB oracle uses: `initRelaxationNode` (the oracle)
and `initRelaxationThreshold` (the `RLT` bound) call the same
`strengthenRelaxation`. `--relaxation ddps` restricts it to the DDPS outer
approximation alone — node bounds and partial-trace consistency, without the
tensor and scalar McCormick families — so the contribution of the `+` can be
measured. `ddpsplus` is the default, so existing behaviour is unchanged.

### Recording trajectories

`EXACTENT_TRACE` is a path *prefix*; the two iterative loops each append a CSV:

| file | columns |
|---|---|
| `<prefix>.cp.csv` | `iter,is_last,ub_relx,lb_relx,b_lower,n_states` |
| `<prefix>.ladmm.csv` | `iter,zeta,f,pen,residual,grad_norm,z,alm` |

`b_lower` is the sBB oracle's lower bound for that round, so
`lb_relx = ub_relx + b_lower`; `residual` is the coupling violation
‖A(z) + a − Ψ(x)‖₂, which certifies `ub_heur` once it vanishes. The experiment
scripts set this automatically. `summarize_traces.py` also reports how many CP
iterations returned **no** lower bound because the oracle terminated early.

