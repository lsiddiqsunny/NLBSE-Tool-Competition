#!/bin/bash
#$ -M msiddiq3@nd.edu
#$ -m abe
#$ -q gpu-debug
#$ -l gpu=1
#$ -pe smp 1
#$ -N nlbse_tf 

module load tensorflow
conda activate tf
python3 ~/Public/NLBSE-Tool-Competition/classify_text_with_bert.py
