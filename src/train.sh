#!/bin/bash
#BSUB -J train
#BSUB -q hpc
#BSUB -W 20
#BSUB -R "rusage[mem=8192MB]"
#BSUB -n 2         
#BSUB -R "span[hosts=1]"   
#BSUB -o src/batch_out/sleeper_%J.out
#BSUB -e src/batch_out/sleeper_%J.err 
#BSUB -N  

source '../special_course/bin/activate'

python src/train.py
