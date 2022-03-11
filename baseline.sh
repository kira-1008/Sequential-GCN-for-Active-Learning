#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=2layer
#SBATCH --error=2layer_gcnii_uncertaingcn.%J.err
#SBATCH --output=2layer_gcnii_uncertaingcn.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python3.7/3.7
module load conda-python/3.7
module load cuda/10.0
module load cuDNN/cuda_9.2_7.2.1

source activate baseline 
python main.py -m UncertainGCN -d cifar10 -c 10 -layers 2
source deactivate

