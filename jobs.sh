#!/bin/bash
# Set variables
timelimit=-1
algorithms=("LDL" "D") #("LDL" "D" "A" "LD1")
datapath="$PWD/benchmark"
resultpath="$PWD/results"
juliabin="julia" #"/home/lxu/software/julia-1.10.2/bin/julia"


# Generate all job combinations and save to file
rm -f job_list.txt
for instance in $(ls $datapath)
do
  for algorithm in "${algorithms[@]}"
  do
    if [ "$algorithm" == "LDL" ]; then
      if ! grep -q "N = 5" "$datapath/$instance"; then
        continue
      fi
      echo "Processing instance $datapath/$instance with algorithm LDL"
    fi
      echo  "-s" "$instance" "-a" "$algorithm" "-t" "$timelimit"
      echo "$instance" "$algorithm" "$timelimit">> job_list.txt
  done
done

