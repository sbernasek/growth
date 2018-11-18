#!/bin/bash
cd /Users/Sebi/Documents/grad_school/research/growth/simulations/Sweep_181118_144549 

echo "Starting all batches at `date`"
while read P; do
echo "Processing batch ${P}"
python ./scripts/run_batch.py ${P} -s 1
done < ./batches/index.txt 
echo "Completed all batches at `date`"
exit
