#!/bin/bash
#$ -M msiddiq3@nd.edu
#$ -m abe
#$ -q long
#$ -N nlbse 
module load conda
source activate nlbse
python3 ~/Public/NLBSE-Tool-Competition/code.py