#!/bin/bash
cd /Users/Sebi/Documents/grad_school/research/growth/simulations/Sweep_181118_150116 

while IFS=$'\t' read P
do
b_id=$(echo $(basename ${P}) | cut -f 1 -d '.')
   JOB=`msub - << EOJ

#! /bin/bash
#MSUB -A p30653 
#MSUB -q normal 
#MSUB -l walltime=10:00:00 
#MSUB -m abe 
#MSUB -o ./log/${b_id}/outlog 
#MSUB -e ./log/${b_id}/errlog 
#MSUB -N ${b_id} 
#MSUB -l nodes=1:ppn=1 
#MSUB -l mem=1gb 

module load python/anaconda3.6
source activate ~/pythonenvs/growth_env

cd /Users/Sebi/Documents/grad_school/research/growth/simulations/Sweep_181118_150116 

python ./scripts/run_batch.py ${P} -s 0
EOJ
`

done < ./batches/index.txt 
echo "All batches submitted as of `date`"
exit
