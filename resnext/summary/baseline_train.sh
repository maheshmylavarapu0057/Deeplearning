#!/bin/bash
#SBATCH -n 10
#SBATCH -A nlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=9
#SBATCH --mem-per-cpu=3000

#SBATCH --time=4-00:00:00

#SBATCH --mail-type=END



module add cuda/10.0

module add cudnn/7.6-cuda-10.0 
python train.py 0 model  model log.out 0
echo "Train completed"

echo "Test noise series are impulse_noise.npy, speckle_noise.npy, gaussian_noise.npy, shot_noise.npy"


 
