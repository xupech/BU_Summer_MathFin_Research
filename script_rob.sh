#!/bin/bash -l
#$ -m ae
#$ -l h_rt=24:00:00
#$ -pe omp 28

module load python3/3.6.5

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute Final report.ipynb