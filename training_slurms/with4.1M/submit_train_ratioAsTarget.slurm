#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=100g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asirohi@umn.edu
#SBATCH -p cms-1,v100
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="allinoneratiolr1e03"

folder="/home/rusack/shared/pickles/HGCAL_TestBeam/Training_results/1M_EdgeCheck_ratio/"
#v2->trueE, no semi
#v1->semi_dscb, loss fun= dscb

export PYTHONUNBUFFERED=1
module load cuda/10.1
module load cmake
module load gcc
module load graphviz
source activate torch1.8
./../../train $folder HGCAL_TestBeam/Test_alps/2To3M --idx_name all --nosemi --target trueE --in_layers 3 --mp_layers 2 --out_layers 2 --pool max --lr_sched Const --max_lr 0.0001 --valid_batch_size 250  --agg_layers 2 --train_batch_size 400 --acc_rate 10   &>> $folder/training.log
#train $folder <DATA FOLDER> --arguments... &>> $folder/training.log
#<DATA FOLDER> options include:
#2018_UL_Particle_MC_30M: latest large dataset for electrons
#2018_UL_Photon_Refined: latest large dataset for photons
#2018_UL_photon_Refined/test: test version of the above with only ~8k particles
