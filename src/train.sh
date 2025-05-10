#!/bin/bash
#BSUB -J train
#BSUB -q gpua100
#BSUB -W 300
#BSUB -R "rusage[mem=5GB]"
#BSUB -n 3         
#BSUB -R "span[hosts=1]"   
#BSUB -o src/batch_out/train_%J.out
#BSUB -e src/batch_out/train_%J.err 

source '../special_course/bin/activate'

python src/train.py
