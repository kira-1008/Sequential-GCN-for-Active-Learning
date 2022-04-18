#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=8layer
#SBATCH --error=8ayer_gcnii_UncertainGcn_cifar100.%J.err
#SBATCH --output=8ayer_gcnii_UncertainGcn_cifar1-0.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python3.7/3.7
module load conda-python/3.7
module load cuda/10.0
module load cuDNN/cuda_9.2_7.2.1

source activate baseline 
python main.py -m UncertainGCN -d cifar100 -c 10 -layers 8
source deactivate

