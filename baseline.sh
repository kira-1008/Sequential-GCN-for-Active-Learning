#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=cifar10_4layer_pn
#SBATCH --error=cifar10_4layer_pn_uncertain.%J.err
#SBATCH --output=cifar10_4layer_pn_uncertain.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python3.7/3.7
module load conda-python/3.7
module load cuda/10.0
module load cuDNN/cuda_9.2_7.2.1

source activate baseline 
python main.py -m UncertainGCN -d cifar10 -c 10 -layers 4 --norm_mode PN

source deactivate

