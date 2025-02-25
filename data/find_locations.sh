#!/bin/bash
#BSUB -J find_locations
#BSUB -q hpc
#BSUB -W 04:00
#BSUB -R "rusage[mem=8192MB]"
#BSUB -R "select[model==XeonGold6142]"
#BSUB -n 1         
#BSUB -R "span[hosts=1]"   
#BSUB -o data/batch_out/sleeper_%J.out
#BSUB -e data/batch_out/sleeper_%J.err 
#BSUB -N  

source '../special_course/bin/activate'

python data/find_location.py
