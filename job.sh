#!/bin/bash
#$ -M msiddiq3@nd.edu
#$ -m abe
#$ -q long           # Specify queue
#$ -N nlbse       # Specify job name

module load conda
source activate nlbse
python3 code.py