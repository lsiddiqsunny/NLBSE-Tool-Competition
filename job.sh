#!/bin/bash
#$ -M msiddiq3@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu=1
#$ -pe smp 1
#$ -N nlbse 
conda activate nlbse
python3 ~/Public/NLBSE-Tool-Competition/code.py