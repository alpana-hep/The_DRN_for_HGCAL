#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=300g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asirohi@umn.edu
#SBATCH -p cms-1
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="DNN_trainign"

folder="/home/rusack/asirohi/public/finalGNN/output_training/v1"

export PYTHONUNBUFFERED=1
module load cuda/10.1
module load cmake
module load gcc
module load graphviz
source activate torch1.8
./train $folder HGCAL_TestBeam/Test_alps/0To1M --idx_name all  --best_arch --semi_dscb --valid_batch_size 100 --train_batch_size 250 --acc_rate 2 &>> $folder/training.log
#train $folder <DATA FOLDER> --arguments... &>> $folder/training.log

#<DATA FOLDER> options include:
#2018_UL_Particle_MC_30M: latest large dataset for electrons
#2018_UL_Photon_Refined: latest large dataset for photons
#2018_UL_photon_Refined/test: test version of the above with only ~8k particles
