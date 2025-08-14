#!/bin/bash
#SBATCH --job-name=distributed_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5122
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PARAM=$(sed -n "$(($SLURM_ARRAY_TASK_ID + 1))p" job_list.txt)

# Split PARAM into variables
read instance algorithm timelimit <<< "$PARAM"

# julia --project=. src/main.jl benchmarks/test testg.txt none CPLEX 20 LEFPA Small
echo "julia" "--project=."  "src/main.jl" $instance $algorithm $timelimit
srun julia --project=. src/main.jl -s "$instance" -a "$algorithm" -t "$timelimit" >  outputs/output_${instance}_${algorithm}.out
