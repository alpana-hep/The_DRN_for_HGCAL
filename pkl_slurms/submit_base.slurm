#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rothmans@umn.edu
#SBATCH -p cms-1,v100
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="grid_search"

cd 

export PYTHONUNBUFFERED=1
