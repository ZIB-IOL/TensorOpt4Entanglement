bash jobs.sh
#rm -r results/**/*
#if [ -d "outputs" ]; then
#  rm -r outputs/*
#fi
lines=$(wc -l < job_list.txt)
echo "Number of jobs: $lines"
export LC_ALL=C
export JULIA_DEPOT_PATH=".julia_depot"
export MOSEKHOME="/software/mosek/10.2"
export MOSEKLM_LICENSE_FILE=27007@solice01.zib.de
/home/htc/lxu/.julia/juliaup/julia-1.11.4+0.x64.linux.gnu/bin/julia --project=.  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

total_lines=$(wc -l < job_list.txt)

for ((i = 1; i <= total_lines; i++)); do
    sbatch run.slurm $i
done
