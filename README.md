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

Each experimental table has its own script, plus one master script that runs
them all. Finished jobs are skipped, so an interrupted run can be restarted.

| script | table | what it runs |
|---|---|---|
| `scripts/table_m3.sh` | `tab.m3` | m=3: 3 states x {Alt-SDP, LADMM, CP, IR, DPS, DDPS+}, 1 h limit |
| `scripts/table_m4.sh` | `tab.m4` | m=4: 4 states x same 6 algorithms, 2 h limit |
| `scripts/table_m5.sh` | `tab.m5` | m=5: 4 states x same 6 algorithms, 3 h limit |
| `scripts/table_m5_lowrank.sh` | `tab.m5low` | m=5: LADMM with r = 400…900 |
| `scripts/table_m5_gapclosing.sh` | `tab.m5CP` | m=5: CP and IR, averaged over gap-closing iterations |
| `scripts/run_all_tables.sh` | all of the above | |

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic

bash scripts/table_m3.sh                      # one table
bash scripts/run_all_tables.sh                # everything (~200 CPU-hours)
bash scripts/run_all_tables.sh m3 m5          # a subset

bash scripts/run_all_tables.sh --dry-run      # list the jobs, run nothing
bash scripts/run_all_tables.sh --force        # redo every experiment from scratch
bash scripts/table_m3.sh --time-limit 60      # quick smoke test (not paper settings)
bash scripts/run_all_tables.sh --slurm        # emit job lists for run.slurm
bash scripts/table_m3.sh --help               # all options
```

By default a job whose result file already exists is skipped, so an interrupted
run can simply be restarted. `--force` (`-f`) ignores existing results and
re-runs everything; it is forwarded by `run_all_tables.sh` to every table.

Options: `-f/--force`, `-n/--dry-run`, `-t/--time-limit SEC`, `--slurm`,
`--results-dir`, `--trace-dir`, `--log-dir`, `--julia`, `-h/--help`. The
equivalent environment variables (`FORCE`, `DRY_RUN`, `TIME_LIMIT`,
`USE_SLURM`, `RESULTS_DIR`, `TRACE_DIR`, `LOG_DIR`, `JULIA_BIN`,
`BENCHMARK_DIR`) still work.

Instances are selected by the `N = <m>` line inside each benchmark file, so a
new benchmark automatically joins the right table.

`TIME_LIMIT=-1` (the default) lets the code apply the paper's per-size limits
of 1/2/3 hours for m=3/4/5.

### Generating the LaTeX

```bash
python3 scripts/make_tables.py --table all          # print all table bodies
python3 scripts/make_tables.py --table m3           # just one
python3 scripts/make_tables.py --table all --out tables/
```

Two caveats:

- The **PDGR** rows come from an external implementation (FrankWolfe.jl, Liu et
  al.) and are not produced here; the generator emits a commented placeholder.
  Because of that, bolding of the best bound is computed over the rows this
  repository generates, which can differ from the paper where a PDGR value was
  the best.
- The **gap-closing table** (`tab.m5CP`) averages over CP iterations rather than
  using final values, so it is built from the trajectory files described below.

### Recording trajectories

Result files hold only the final bounds. Set `EXACTENT_TRACE` to a path
*prefix* and the two iterative loops each append a CSV trajectory:

| file | columns | written by |
|---|---|---|
| `<prefix>.cp.csv` | `iter,is_last,ub_relx,lb_relx,b_lower,n_states` | the cutting plane (`D`, `LDL`, `LD1`, `AD`) |
| `<prefix>.ladmm.csv` | `iter,zeta,f,pen,residual,grad_norm,z,alm` | LADMM (`LD1`, `LDL`, `LDR*`) |

`b_lower` is the sBB oracle's lower bound for that round, so
`lb_relx = ub_relx + b_lower`. `residual` is the coupling violation
‖A(z) + a − Ψ(x)‖₂, which certifies `ub_heur` once it vanishes; `zeta` is the
LADMM penalty ζ.

The table scripts set this automatically (into `--trace-dir`, default
`results/traces/`). To record one by hand:

```bash
EXACTENT_TRACE=/tmp/run julia --project=. scripts/run_experiment.jl -s state_13.jl -a LDL -t 600
# -> /tmp/run.cp.csv and /tmp/run.ladmm.csv
```

Tracing is off unless `EXACTENT_TRACE` is set, and the traced and untraced code
paths are otherwise identical.

## Run the Python script to parse result data

Prerequisites:
- Python 3.8+ installed.
- (Optional) Create and activate a virtual environment and install dependencies if a requirements file exists.


Basic usage (from project root):
```bash
# simple run (reads/writes paths relative to project root)
python3 scripts/make_tables.py --table all
```

This prints the tables aggregating the results.