# TensorOpt4Entanglement — quick setup & run

Short instructions to install Julia packages used by this project and to run jobs using `jobs.sh` and a Slurm array (`run.slurm`).

## Prerequisites
- Linux machine with Julia 1.11.x installed (the project was tested with Julia 1.11.6) and Mosek.
- A working Slurm cluster.
- Project directory layout:
    - `benchmark/` — input instances
    - `results/` — output directory
    - `jobs.sh` — job list generator (already provided)

## Install required Julia packages
From the project root (this repo), prefer using the project environment if present:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This will install packages listed in `Project.toml`/`Manifest.toml` if they exist.

## Prepare and run jobs locally
1. Edit the top of `jobs.sh` to set up experiments:
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


## Run the Python script to parse result data

Prerequisites:
- Python 3.8+ installed.
- (Optional) Create and activate a virtual environment and install dependencies if a requirements file exists.


Basic usage (from project root):
```bash
# simple run (reads/writes paths relative to project root)
python3 ./exerpiementanaylysis.py
```

This prints the tables aggregating the results.