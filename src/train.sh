#!/bin/bash
#BSUB -J train_k
#BSUB -q gpua100
#BSUB -W 800
#BSUB -R "rusage[mem=12GB]"
#BSUB -n 14         
#BSUB -R "span[hosts=1]"   
#BSUB -o src/batch_out/trainK_%J.out
#BSUB -e src/batch_out/trainK_%J.err 

source '../special_course/bin/activate'

python src/train.py
