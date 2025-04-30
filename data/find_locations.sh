#!/bin/bash
#BSUB -J findlocations
#BSUB -q hpc
#BSUB -W 40:00
#BSUB -R "rusage[mem=8192MB]"
#BSUB -n 1         
#BSUB -R "span[hosts=1]"   
#BSUB -o data/batch_out/sleeper_%J.out
#BSUB -e data/batch_out/sleeper_%J.err  

source '../special_course/bin/activate'
python data/find_location.py
