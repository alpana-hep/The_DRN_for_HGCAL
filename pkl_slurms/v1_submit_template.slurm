#!/bin/bash -l

#hopefully most of the fields here are self-explanatory 
#lines starting with "#SBATCH" are instructions for the job scheduler 
#available partitions (ie different node configurations) are summaries at:
#   https://www.msi.umn.edu/partitions
#MSI also has a tutorial on job submission here:
#   https://www.msi.umn.edu/content/job-submission-and-scheduling-slurm 
#you submit this script with the command:
#   sbatch submit_template.slurm

#SBATCH --time=16:00:00
#SBATCH --ntasks=8
#SBATCH --mem=450g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asirohi@umn.edu
#SBATCH -p  amd2tb
#SBATCH --job-name="job name"

cd /home/rusack/asirohi/public/finalGNN/
#whatever you want to run here... 
module load cuda/10.1
module load cmake
module load gcc
module load graphviz

./prepareHGCAL_v1
#./train HGCAL_semi HGCAL_TestBeam/Test_alps --idx_name all --best_arch --semi_dscb --target trueE

