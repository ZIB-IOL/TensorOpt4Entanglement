#!/bin/bash
#SBATCH --job-name=distributed_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=05:58:00
#SBATCH --partition=opt_int
#SBATCH --constraint=Gold5222 # 5122
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# 接收行号作为参数
LINENUM=$1

# 获取对应参数行
PARAM=$(sed -n "${LINENUM}p" job_list.txt)

# 拆分参数
read juliabin instance algorithm timelimit <<< "$PARAM"

echo "$juliabin" "--project=."  "scripts/run_experiment.jl" $instance $algorithm $timelimit
srun "$juliabin" --project=. scripts/run_experiment.jl -s "$instance" -a "$algorithm" -t "$timelimit" > outputs/output_${instance}_${algorithm}.out
