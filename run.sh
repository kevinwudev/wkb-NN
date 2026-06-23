#!/bin/bash
#SBATCH --account=ACD114087
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=10:00:00
#SBATCH --output=General-job-%j-output.log

uv run fig_nn.py
