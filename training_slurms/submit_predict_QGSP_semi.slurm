#!/bin/bash -l

#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --mem=350g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asirohi@umn.edu
#SBATCH -p cms-1,v100
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="lr3e4QGSP"

folder="/home/rusack/asirohi/public/finalGNN/output_training/withFulldataset/lr1e04/copy/epoch28_onwards/QGSP"
#v2->trueE, no semi
#v1->semi_dscb, loss fun= dscb

export PYTHONUNBUFFERED=1
module load cuda/10.1
module load cmake
module load gcc
module load graphviz
source activate torch1.8
./train $folder HGCAL_TestBeam/Test_alps/sim_QGSP_DiscreteEnergy --idx_name all --target trueE  --best_arch --nosemi --valid_batch_size 300  --lr_sched Const --max_lr 0.0001   --predict_only &>> $folder/training.log
#train $folder <DATA FOLDER> --arguments... &>> $folder/training.log

#<DATA FOLDER> options include:
#2018_UL_Particle_MC_30M: latest large dataset for electrons
#2018_UL_Photon_Refined: latest large dataset for photons
#2018_UL_photon_Refined/test: test version of the above with only ~8k particles
