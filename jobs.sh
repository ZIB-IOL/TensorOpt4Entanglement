#!/bin/bash
# Set variables
timelimit=-1
algorithms=("RLT") #("RLT" "LD1" "LDR0" "LDR1" "LDR2" "LDR3" "LDR4" "LDR5" "LDL" "D" "A")
datapath="$PWD/benchmark"
resultpath="$PWD/results"
juliabin="/home/lxu/software/julia-1.10.2/bin/julia"


# Generate all job combinations and save to file
rm -f job_list.txt
for instance in $(ls $datapath)
do
  for algorithm in "${algorithms[@]}"
  do
      #if ! grep -q "N = 5" "$datapath/$instance"; then
      #  continue
      #fi
      echo  "-s" "$instance" "-a" "$algorithm" "-t" "$timelimit"
      echo "$juliabin" "$instance" "$algorithm" "$timelimit">> job_list.txt
  done
done

