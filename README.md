# TensorOpt4Entanglement — quick setup & run

Short instructions to install the Julia packages this project uses and to reproduce the paper's experiments, locally or on a Slurm cluster.

## Prerequisites
- Linux machine with Julia 1.11.x (tested with 1.11.6).
- A MOSEK licence. The solver itself is fetched by Mosek.jl (MOSEK 11.2).
- A working Slurm cluster.

### Setup

```bash
# 1. Julia 1.11.x -- the Manifest is resolved for it
juliaup add 1.11.6 && export JULIA_BIN='julia +1.11.6'
# no juliaup? unpack an official tarball and set JULIA_BIN to its bin/julia

# 2. MOSEK licence -- the solver itself is fetched by Mosek.jl
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic     # or port@host, or ~/mosek/mosek.lic

# 3. packages
mkdir .julia_depot                                  # optional: keep them in the repo
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# 4. check
bash runjobs.sh --env
```

`bash runjobs.sh --dry-run` doubles as a status report: it lists every cell as
`have` or `MISSING` and tallies each table, so it answers "what results do we
actually have" without running anything.

A Slurm submission runs this check itself and refuses to queue anything if it
fails (`--skip-env-check` overrides), so a broken environment costs one message
rather than a hundred failed jobs.

Notes:
- On 1.12+ the Manifest re-resolves to different package versions, so stay on 1.11.x.
- `+1.11.6` needs juliaup; a plain `julia` reads it as a filename.
- `.julia_depot/` is used only if it exists; the first install takes a few GB.
- Mosek.jl downloads MOSEK 11.2 itself, so no solver path to configure — but
  the machine that installs needs outbound internet.
- Do not set `MOSEKBINDIR`: it overrides that, and the build rejects any MOSEK
  whose version differs from Mosek.jl's. `unset MOSEKBINDIR` if one is left over.

- Project directory layout:
    - `src/` — the `ExactEntanglement` package
    - `scripts/run_experiment.jl` — command-line entry point
    - `benchmark/` — input instances
    - `results/` — output directory
    - `test/` — test suite (`julia --project=. -e 'using Pkg; Pkg.test()'`)

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

## Running the experiments

Everything goes through `runjobs.sh`; see
[Reproducing the paper's tables](#reproducing-the-papers-tables) below for the
full workflow. In short:

```bash
bash runjobs.sh --env        # can this machine run the jobs at all?
bash runjobs.sh --check      # verify every table would have data
bash runjobs.sh --dry-run    # list the jobs, run nothing
bash runjobs.sh --local      # run here, sequentially
bash runjobs.sh              # submit to Slurm
```

### Site settings

`runjobs.sh` opens with a **Site settings** block — three plain assignments you
edit once for your machine, and the only place the runs take their environment
from:

| setting | what it is | shipped default |
| --- | --- | --- |
| `JULIA_BIN` | Julia executable | `julia` |
| `MOSEKLM_LICENSE_FILE` | licence file or `port@host` | ZIB licence server |
| `JULIA_DEPOT_PATH` | package depot | `./.julia_depot` when that directory exists |

A value already in the environment wins, so a one-off override needs no edit.
`bash runjobs.sh --env` checks the settings the jobs will actually get.

`scripts/check_env.sh` checks Julia's version, the project layout, writable
output directories, disk space, the Mosek licence — by solving a test LP, not
just looking for the variable — and, where Slurm is present, that the partition
and node feature named in `run.slurm` actually exist and that `MaxArraySize` is
large enough. It reports every problem rather than stopping at the first.

On the cluster, adjust the resource requests at the top of `run.slurm`
(`--mem`, `--time`, `--partition`) to your site; the paper's runs used 10 GB
and one thread per job. There is no `--constraint`, so jobs take any node in
the partition — add one if you need a fixed CPU model for timing comparisons.

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
| `scripts/exp_ddps_ablation.sh` | m=3,4,5 × the DDPS-only variants | `results/ddps/` | ~80 CPU-h |


```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic

bash runjobs.sh --size 3               # probe: the m=3 jobs only (27 jobs)
bash runjobs.sh                        # submit all experiments to Slurm
bash runjobs.sh main                   # just one
bash runjobs.sh --local                # run here instead, sequentially
bash runjobs.sh --dry-run              # show what would happen, do nothing
bash runjobs.sh --local main -t 60     # smoke test (not paper settings)
bash runjobs.sh --force                # redo every experiment
bash runjobs.sh --check                # verify every table cell is covered
bash runjobs.sh --help                 # all options

bash scripts/exp_main.sh               # or run one experiment directly
```

**A job whose result file already exists is skipped** — in every mode, local
and Slurm alike — so an interrupted or partially failed run is resumed simply
by running the same command again; only the missing jobs are queued. `--force`
ignores existing results and redoes everything. When nothing is left to do,
nothing is submitted.

### Probing with the small instances first

The m=3 instances carry the paper's 1-hour limit, so they exercise the whole
pipeline — every algorithm, the trajectories, the diagnostics, the tables — at
a fraction of the cost. Run them before committing to the rest:

```bash
bash runjobs.sh --size 3              # 27 jobs, ~27 CPU-h worst case
bash runjobs.sh --size 3 --check      # what those 27 would complete
python3 scripts/make_tables.py --table m3
```

That fills `m3`, `ddps3`, `mem3` and `size3` completely. `--size` takes several
counts (`--size "3 4"`), and `exp_lowrank.sh` reports that it has nothing to do
unless 5 is among them, since the rank sweep is defined only at m=5.

Once it looks right:

```bash
bash runjobs.sh                        # the remaining 96 jobs; m=3 is skipped
```

Nothing is re-run: jobs with a result file are skipped, so the full submission
picks up exactly where the probe left off.

### Running one part

The three parts are named after their scripts — `main`, `lowrank`,
`ddps_ablation` — and `runjobs.sh` discovers them by globbing `scripts/exp_*.sh`,
so `--help` always lists what actually exists.

```bash
bash runjobs.sh main                      # one part
bash runjobs.sh lowrank ddps_ablation     # two
bash scripts/exp_lowrank.sh               # or call the part script directly
```

Every mode takes a part subset: `--local`, `--dry-run`, `--force` and `--check`
all accept one. An unknown name is rejected with the list of valid ones.

> `--check` is a **global** audit — it asks "if I submit this, will every table
> have data?". With a part subset it will therefore report the tables the other
> parts would have filled, and exit non-zero. That is the intended answer; run
> it without a subset for a meaningful pass.

### Running part of an experiment

Every level of granularity, from the whole suite down to one job:

```bash
bash runjobs.sh                                    # everything
bash runjobs.sh main                               # one experiment
M=3 bash scripts/exp_main.sh                       # one subsystem count
bash scripts/exp_main.sh --algo RLT                # one algorithm, all instances
bash scripts/exp_main.sh --state state_13.jl       # one instance, all algorithms
bash scripts/exp_main.sh --state state_13.jl --algo D   # exactly one job
```

`--state` and `--algo` are repeatable and can be combined. Asking for an
algorithm the experiment does not contain is an error rather than a silent
no-op, so a typo cannot look like a completed run. Use these rather than
calling `scripts/run_experiment.jl` directly: they set `EXACTENT_RESULTS_DIR`
and `EXACTENT_TRACE` for you, so the result lands in the right part directory
and its trajectory is recorded.

On the cluster, one line of a job list is one job:

```bash
JOB_LIST=job_list_main.txt sbatch --array=7        run.slurm   # just line 7
JOB_LIST=job_list_main.txt sbatch --array=7,12,30  run.slurm   # a few
JOB_LIST=job_list_main.txt sbatch run.slurm 7                  # without an array
```

That is how you re-run the handful of array elements that failed, without
resubmitting the rest — though simply re-running `runjobs.sh` does the same
thing, since completed jobs are skipped.

### One entry point, two destinations

`runjobs.sh --check` audits the generated job lists against every table and
fails if any cell would be left without data, so a missing experiment is caught
before cluster time is spent rather than after.

`runjobs.sh` decides only *where* the experiments run; the design lives in
`scripts/exp_*.sh`. So the cluster and a local machine execute exactly the same
thing, and the available parts are discovered from the scripts that exist
rather than listed a second time.

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

| code | what it runs |
|------|--------------|
| `Alt-SDP` | alternating SDP |
| `LADMM` | one refinement iteration = LADMM + one CP crossover |
| `LADMM_400`…`LADMM_900` | LADMM at factorisation size r (m = 5 sweep) |
| `CP` | standalone cutting plane |
| `IR` | iterative refinement (LADMM + CP) |
| `DPS` | DPS hierarchy lower bound via Ket.jl |
| `DDPS+` | tensor-RLT lower bound at the sBB root |
| `DDPS`, `CP-DDPS`, `IR-DDPS` | DDPS-only counterparts, for the ablation |
| `Alt-SDP+CP`, `IR-nolazy`, `IR-clear`, `DualALM` | variants not named in the paper |

The codes **are** the paper's algorithm names, so a table row and the file that
backs it carry the same label.

The shorthand used before the rename — `A`, `LD1`, `D`, `LDL`, `PPT`, `RLT`,
`LDR0`…`LDR5` — is still accepted on the command line and when reading result
files, so the runs published with the paper still load without being renamed.
New runs are written under the canonical name, so `-a RLT` produces
`state_133.jl_DDPS+`.

### What each run records

Result files carry the bounds plus provenance and diagnostics:

```
ub_relx / lb_relx / ub_heur / feas_heur / time     the reported values
relaxation / seed / julia / host                   provenance
relax_nvars / relax_ncons / relax_nnz              root relaxation size
relax_cbf_bytes                                    its size on disk (CBF)
mem_total_* / mem_cp_* / mem_lmo_* / mem_ladmm_*   memory per level
```

Field names are the paper's symbols. Runs published before the rename used
`glbub` / `glblb` / `approxub` / `approxfeas` / `approxweights`; readers accept
either, so the published data still loads.

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

