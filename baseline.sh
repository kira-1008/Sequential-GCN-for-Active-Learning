#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=2layerkn
#SBATCH --error=2layerknCore.%J.err
#SBATCH --output=2layerknCore.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python3.7/3.7
module load conda-python/3.7
module load cuda/10.0
module load cuDNN/cuda_9.2_7.2.1

source activate baseline 
python main.py -m CoreGCN -d cifar10 -c 10 -layers 2

source deactivate

