#!/bin/bash
#BSUB -J findlocs
#BSUB -q hpc
#BSUB -W 45:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 7         
#BSUB -R "span[hosts=1]"   
#BSUB -o data/batch_out/find_locations_%J.out
#BSUB -e data/batch_out/find_locations_%J.err  

source '../special_course/bin/activate'
python data/find_location.py
