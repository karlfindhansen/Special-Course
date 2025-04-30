#!/bin/bash
#BSUB -J wandb
#BSUB -q gpuv100
#BSUB -W 900
#BSUB -R "rusage[mem=4096MB]"
#BSUB -n 4         
#BSUB -R "span[hosts=1]"   
#BSUB -o src/batch_out/sleeper_%J.out
#BSUB -e src/batch_out/sleeper_%J.err 

source '../special_course/bin/activate'
wandb agent karl-find-hansen-technical-university-of-denmark/greenland-bedmap-generation/43odwsqx
