#!/bin/bash

#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -o /scratch/htc/ashestak/cluster_logs/resnet18_3d_%j.out
#SBATCH --gres=gpu:1
#SBATCH --job-name=r18_3d
#SBATCH --mem=24000
#SBATCH --nodelist=htc-gpu001

export http_proxy=http://squid.zib.de:8080
export https_proxy=https://squid.zib.de:8080

# ENVIRONMENT DEFINITION
export scratch="/scratch/htc/ashestak"
export CONDA_bin="$scratch/miniconda/bin"

#ENVIRONMENT ACTIVATION
source $CONDA_bin/activate
conda activate metr

# RUN SCRIPT
exec python train_backbone.py 