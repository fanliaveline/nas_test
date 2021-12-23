#!/bin/bash

#SBATCH --account=cuixiaohui
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=240:00:00
#SBATCH --output=search-han

module load nvidia/cuda/9.0
nvidia-smi

cd $SLURM_SUBMIT_DIR

python  MGPU_search_arch.py --gpu_ids 0,1

nvidia-smi

